#!/usr/bin/env python3
#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright 2023 The vLLM team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
# Adapted from https://github.com/vllm-project/vllm/tree/main/tools
#
"""Review gate: the machine-checkable subset of the review rule table.

Layered review model (see DESIGN.md 6.7):

- **pre-commit (this script)**: only rules that can be judged statically and
  deterministically -- seconds, offline, no model, fail fast.
- **pre-push / CI**: the full review (cross-file semantics, design intent,
  governance) run by the Reviewer sub-agent.

Deliberate design choices:

- Only **high-precision** checks are enabled. A noisy gate gets bypassed with
  `--no-verify`, which is worse than no gate. Checks are validated against the
  existing tree before being enabled.
- Every finding prints the rule id from
  `.claude/skills/reviewer/reference/review-rules.md` so the reason is
  traceable (rule: "every exclusion/conclusion must be explainable").

Checks (all calibrated to zero false positives on the existing tree):

- R1-1 / R10-3: bare `except: pass` (catches everything, including
  `KeyboardInterrupt` / `SystemExit`).
- R1-11: `os.environ[...] = <non-string literal>` raises `TypeError` at runtime.
- R1-7: dead code guarded by `if False:` / `if 0:`.
- R1-2: vendor hardcoding via `is_rocm(` / `is_cuda(` call.

Calibration is a hard requirement, not a nicety: a gate that fires on
legitimate code gets bypassed with `--no-verify` and becomes worthless. Rules
whose precision depends on context (e.g. the general "silent except" rule, magic
numbers, bare `assert`) are intentionally left to the Reviewer agent.
"""

import ast
import subprocess
import sys

RULE_TABLE = ".claude/skills/reviewer/reference/review-rules.md"
PY_GLOB = "vllm_ascend/**/*.py"


def _silent_except(filepath: str, tree: ast.AST) -> list[str]:
    """Flag only bare `except: pass` -- the unambiguously wrong form.

    Calibration note: the broader rule "any `except ...: pass`" (R1-1/R10-3) is
    deliberately NOT used here. Checking the existing tree showed it cannot be
    judged statically at high precision -- legitimate patterns include
    `except queue.Empty: pass` around `queue.get`, `except ImportError: pass`
    optional-dependency guards, and handlers followed by an explicit fallback
    path. Those need judgement, so they belong to the Reviewer agent, not to a
    fail-fast pre-commit gate. Only the bare `except:` form (which also swallows
    KeyboardInterrupt / SystemExit) is flagged here.
    """
    out = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Try, getattr(ast, "TryStar", ast.Try))):
            continue
        for handler in node.handlers:
            if handler.type is not None:
                continue
            real = [s for s in handler.body if not _is_noop_stmt(s)]
            if not real:
                out.append(
                    f"{filepath}:{handler.lineno}: bare `except: pass` "
                    "(R1-1/R10-3); catches everything including "
                    "KeyboardInterrupt -- catch a specific exception and "
                    "raise / log / return explicitly"
                )
    return out


def _is_noop_stmt(node: ast.AST) -> bool:
    if isinstance(node, ast.Pass):
        return True
    if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant):
        return node.value.value is Ellipsis
    return False


def _environ_nonstr(filepath: str, tree: ast.AST) -> list[str]:
    out = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if not _is_environ_subscript(target):
                continue
            val = node.value
            if isinstance(val, ast.Constant) and not isinstance(val.value, str):
                out.append(
                    f"{filepath}:{node.lineno}: `os.environ[...] = {val.value!r}` "
                    "(R1-11); os.environ only accepts `str` values"
                )
    return out


def _is_environ_subscript(node: ast.AST) -> bool:
    if not isinstance(node, ast.Subscript):
        return False
    value = node.value
    return (
        isinstance(value, ast.Attribute)
        and value.attr == "environ"
        and isinstance(value.value, ast.Name)
        and value.value.id == "os"
    )


def _if_false(filepath: str, tree: ast.AST) -> list[str]:
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.If) and _is_falsey_constant(node.test):
            out.append(f"{filepath}:{node.lineno}: dead code guarded by a constant-false condition (R1-7); remove it")
    return out


def _is_falsey_constant(node: ast.AST) -> bool:
    return isinstance(node, ast.Constant) and node.value in (False, 0)


def _vendor_call(filepath: str, tree: ast.AST) -> list[str]:
    if "rocm" in filepath.lower():
        return []
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id in ("is_rocm", "is_cuda"):
                out.append(
                    f"{filepath}:{node.lineno}: vendor check `{node.func.id}()` "
                    "(R1-2); must not silently route NPU to the wrong branch"
                )
    return out


CHECKS = (
    ("R1-1/R10-3", _silent_except),
    ("R1-11", _environ_nonstr),
    ("R1-7", _if_false),
    ("R1-2", _vendor_call),
)


def check_source(filepath: str, source: str) -> list[str]:
    try:
        tree = ast.parse(source, filename=filepath)
    except SyntaxError:
        return []
    out = []
    for _rule, fn in CHECKS:
        out.extend(fn(filepath, tree))
    return out


def check_file(filepath: str) -> list[str]:
    try:
        with open(filepath, encoding="utf-8") as fh:
            source = fh.read()
    except (OSError, UnicodeDecodeError):
        return []
    return check_source(filepath, source)


def _all_python_files() -> list[str]:
    result = subprocess.run(
        ["git", "ls-files", PY_GLOB],
        capture_output=True,
        text=True,
        check=False,
    )
    return [line for line in result.stdout.splitlines() if line]


def main() -> int:
    args = sys.argv[1:]
    if args == ["--all"]:
        targets = _all_python_files()
    elif args:
        targets = args
    else:
        print("Usage: check_review_rules.py <file> ... | --all", file=sys.stderr)
        return 1

    violations = []
    for filepath in targets:
        violations.extend(check_file(filepath))

    if violations:
        print(f"Review gate violations (rule table: {RULE_TABLE}):\n")
        for v in violations:
            print(f"  {v}")
        print(f"\n{len(violations)} violation(s). Fix them, or bypass with `git commit --no-verify`.")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
