#!/usr/bin/env python3
"""Reviewer 脚本 B：影响面 + 缺失测试 + 风险排序——只产线索，不做判定。

用法：
    impact_scan.py [--repo DIR] [--base REF] [--head REF] [--out DIR]
                   [--depth 2] [--max-nodes 200] [--max-symbols 60]
                   [--files PATH ...] [--preview]

设计约束（见 DESIGN.md 6.2 / 11 A3）：
- **只读**：不改任何源码，只写 `<out>/review/impact.json`；
- 选文件逻辑**复用脚本 A**（`import review_scope`），保证 A/B 看到同一个改动集；
- 调用者反查**限深（--depth，默认 2）限量（--max-nodes）**，避免在全仓爆炸；
- 风险打分必须**可解释**：每一项加分都带理由字符串，排序决定深审顺序。

调用者反查的取数口径：`git grep` 打**被评审的 ref**（不是工作区——否则复跑历史 PR 时
行号取自当前 tree，与 ref 的符号区间错位，归因会指到无关函数），并排除 **import 行**与
定义行（import 只是引入，不是调用）。

静态分析盲区（DESIGN.md 12）：以**数据表**（下方 `GAPS`）逐条声明，并把说明绑到它
实际影响的**那个查询**上——`ast` 解析不到的动态导入 / 装饰器注册 / C++ 侧，一律交人工
判定，不猜、不静默略过。纪律：某项能力一旦实现，**必须从 `GAPS` 删除对应条目**（表里
留着的必须是今天仍然成立的盲区，否则表本身就成了误导）。
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import sys
import warnings
from collections import deque
from dataclasses import dataclass
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import review_scope  # noqa: E402  （同目录，复用选文件与 gate 链）

PY_PATHSPEC = "*.py"
TEST_PATHSPEC = "tests/"
_HUNK = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@")

# --------------------------------------------------------------- 盲区声明
#
# 盲区写成**数据**而不是散落的散文说明，理由同 crg 的 uncertainty.py：一条盲区必须
# 绑到它实际影响的**那个查询**上（C++ 解析不了影响的是 callers / tests，不影响
# 「改了哪些符号」），否则会把无关的查询也一并说得不可信。
QUERY_CALLERS = "callers"
QUERY_TESTS = "tests"
MAX_GAP_CHARS = 220


@dataclass(frozen=True)
class Gap:
    """一处已确认的静态分析盲区：适用范围（路径前缀，"" = 全仓）× 受影响查询。"""

    prefix: str
    affects: tuple[str, ...]
    note: str


GAPS: tuple[Gap, ...] = (
    Gap(
        prefix="csrc/",
        affects=(QUERY_CALLERS, QUERY_TESTS),
        note="C++ 侧不参与反查：ast 只解析 Python，C++ 算子/内核的调用与测试关系需人工追。",
    ),
    Gap(
        prefix="vllm_ascend/patch/",
        affects=(QUERY_CALLERS,),
        note="patch 在 import 时直接替换上游符号：谁受该 patch 影响，文本反查看不出来。",
    ),
    Gap(
        prefix="",
        affects=(QUERY_CALLERS,),
        note="注册表式调用（register_* / 查表分派）与 getattr / importlib 动态取用不含静态函数名。",
    ),
    Gap(
        prefix="",
        affects=(QUERY_TESTS,),
        note="仅按名字命中 tests/，不沿调用链传递：改动可能间接影响只测下游用例的测试。",
    ),
)


def gaps_for(query: str, path: str) -> list[str]:
    """该文件在该查询下适用的盲区说明（合并成一句，长度有上限）。"""
    notes = [gap.note for gap in GAPS if query in gap.affects and (not gap.prefix or path.startswith(gap.prefix))]
    return [" ".join(notes)[:MAX_GAP_CHARS]] if notes else []


# --------------------------------------------------------------- 符号提取


@dataclass
class Symbol:
    file: str
    name: str
    qualname: str
    lineno: int
    end_lineno: int
    kind: str
    owner: str | None = None

    @property
    def is_public(self) -> bool:
        return not self.name.startswith("_")


def collect_symbols(path: str, source: str) -> list[Symbol]:
    """提取函数 / 方法 / 类（含嵌套），带行号区间。"""
    try:
        with warnings.catch_warnings():
            # 反查会解析大量「恰好引用了同名符号」的无关文件，其中不少带着
            # SyntaxWarning（如正则里的无效转义）。那是被引用方的历史问题，不是
            # 本次改动的问题，不能让它污染评审输出。
            warnings.simplefilter("ignore", SyntaxWarning)
            tree = ast.parse(source, filename=path)
    except SyntaxError:
        return []
    out: list[Symbol] = []

    def walk(body, prefix: str, owner: str | None) -> None:
        for node in body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                qualname = f"{prefix}{node.name}"
                if isinstance(node, ast.AsyncFunctionDef):
                    kind = "async_function"
                else:
                    kind = "method" if owner else "function"
                end = getattr(node, "end_lineno", None) or node.lineno
                out.append(Symbol(path, node.name, qualname, node.lineno, end, kind, owner))
                walk(node.body, f"{qualname}.", owner)
            elif isinstance(node, ast.ClassDef):
                qualname = f"{prefix}{node.name}"
                end = getattr(node, "end_lineno", None) or node.lineno
                out.append(Symbol(path, node.name, qualname, node.lineno, end, "class", owner))
                walk(node.body, f"{qualname}.", qualname)

    walk(tree.body, "", None)
    return out


# --------------------------------------------------------------- 索引缓存

_SYMBOL_CACHE: dict[tuple[str, str], list[Symbol]] = {}
_SOURCE_CACHE: dict[tuple[str, str], str | None] = {}


def source_of(repo: str, ref: str, path: str) -> str | None:
    """从 ref 取文件内容（`git show <ref>:<path>`），而非读工作区。

    评审对象可能不是当前 checkout（如复跑历史 PR head），读磁盘会把「被评审的
    版本」和「本地当前版本」悄悄混起来——`new_signature` / 行号都会失准。统一
    以 ref 为准。
    """
    key = (ref, path)
    if key not in _SOURCE_CACHE:
        text = review_scope._git(repo, "show", f"{ref}:{path}", check=False)
        _SOURCE_CACHE[key] = text or None
    return _SOURCE_CACHE[key]


def symbols_of(repo: str, ref: str, path: str) -> list[Symbol]:
    key = (ref, path)
    if key not in _SYMBOL_CACHE:
        src = source_of(repo, ref, path)
        _SYMBOL_CACHE[key] = collect_symbols(path, src) if src else []
    return _SYMBOL_CACHE[key]


def enclosing_symbol(repo: str, ref: str, path: str, lineno: int) -> Symbol | None:
    """包含该行的最内层符号（取区间最小者）。"""
    best = None
    for sym in symbols_of(repo, ref, path):
        if sym.lineno <= lineno <= sym.end_lineno:
            if best is None or (sym.end_lineno - sym.lineno) < (best.end_lineno - best.lineno):
                best = sym
    return best


# --------------------------------------------------------------- 改动行 / 引用


def added_ranges(repo: str, base: str, head: str, path: str) -> list[tuple[int, int]]:
    """该文件在新版本中的新增行区间（`git diff -U0` 的 `+c,d`）。"""
    out = review_scope._git(repo, "diff", "--unified=0", f"{base}...{head}", "--", path)
    ranges = []
    for line in out.splitlines():
        match = _HUNK.match(line)
        if not match:
            continue
        start = int(match.group(1))
        count = int(match.group(2) or 1)
        if count > 0:
            ranges.append((start, start + count - 1))
    return ranges


_BINDING_LINES_CACHE: dict[tuple[str, str, str], set[int]] = {}


def binding_lines(repo: str, ref: str, path: str, name: str) -> set[int]:
    """该文件里 `name` 的**非调用**出现行：import 语句区间 ∪ 同名定义行。

    - import 行（`from m import f` / `import m`）只是把名字引进来，不是调用；
    - 同名定义行（`def f(...)` / `class F:`）是另一处定义（同名符号可能在多个文件
      各定义一次），更不是调用。

    不过滤这两类，它们会被算进「直接调用者 N 处」——既抬高风险分，也把评审者
    引向不存在的调用点（实测：5 处里有 3 处是 import 或定义）。
    """
    key = (ref, path, name)
    if key not in _BINDING_LINES_CACHE:
        lines: set[int] = set()
        src = source_of(repo, ref, path)
        if src:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", SyntaxWarning)
                    tree = ast.parse(src, filename=path)
                for node in ast.walk(tree):
                    if isinstance(node, (ast.Import, ast.ImportFrom)):
                        end = getattr(node, "end_lineno", None) or node.lineno
                        lines.update(range(node.lineno, end + 1))
                    elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                        if node.name == name:
                            lines.add(node.lineno)
            except SyntaxError:
                pass
        _BINDING_LINES_CACHE[key] = lines
    return _BINDING_LINES_CACHE[key]


def grep_refs(repo: str, ref: str, name: str, pathspec: str) -> list[tuple[str, int, str]]:
    """在**被评审的 ref** 上按词精确反查（不是工作区）。

    行号必须与 `enclosing_symbol` / `source_of` 取自同一个 ref，否则「命中行」与
    「符号区间」来自两个版本，归因会把评论指到无关函数上。
    """
    out = review_scope._git(repo, "grep", "-n", "-w", "-e", name, ref, "--", pathspec, check=False)
    refs = []
    for line in out.splitlines():
        # 带 rev 时 git 会加前缀：`<ref>:<path>:<line>:<text>`（不带 rev 才是
        # `<path>:<line>:<text>`）。两种都接住，免得前缀把整行判成非法格式而全丢。
        parts = line.split(":", 3)
        if len(parts) >= 4 and parts[2].isdigit():
            ref_path, ref_line, text = parts[1], parts[2], parts[3]
        elif len(parts) >= 3 and parts[1].isdigit():
            ref_path, ref_line, text = parts[0], parts[1], parts[2]
        else:
            continue
        refs.append((ref_path, int(ref_line), text.strip()))
    return refs


def collect_callers(repo: str, ref: str, seed: Symbol, max_depth: int, budget: dict) -> tuple[list[dict], list[dict]]:
    """自 seed 起反查引用（BFS，限深限量）。

    返回 `(direct, transitive)`：`direct` 是直接引用（depth=1，用于风险打分与
    「影响面在哪」），`transitive` 是多跳扩散（depth≥2，只用于看传播范围）。
    两者分开是因为：把多跳引用也算进「被调用次数」，会把风险分推到所有公共
    热点上（某函数的一个直接调用者若有上千个上层调用者，分数会失真）。
    """
    direct: list[dict] = []
    transitive: list[dict] = []
    seen: set[tuple[str, int]] = {(seed.file, seed.lineno)}
    queue = deque([(seed.name, seed.file, seed.lineno, seed.end_lineno, 0)])
    while queue and budget["left"] > 0:
        name, path, lo, hi, depth = queue.popleft()
        if depth >= max_depth:
            continue
        for ref_path, ref_line, _ in grep_refs(repo, ref, name, PY_PATHSPEC):
            if ref_path == path and lo <= ref_line <= hi:
                continue
            if ref_line in binding_lines(repo, ref, ref_path, name):
                continue
            mark = (ref_path, ref_line)
            if mark in seen:
                continue
            seen.add(mark)
            budget["left"] -= 1
            if budget["left"] <= 0:
                budget["truncated"] = True
                return direct, transitive
            caller = enclosing_symbol(repo, ref, ref_path, ref_line)
            item = {
                "file": ref_path,
                "qualname": caller.qualname if caller else None,
                "line": ref_line,
                "depth": depth + 1,
            }
            (direct if depth == 0 else transitive).append(item)
            if caller is not None and caller.name != name and depth + 1 < max_depth:
                queue.append((caller.name, caller.file, caller.lineno, caller.end_lineno, depth + 1))
    return direct, transitive


# --------------------------------------------------------------- 风险打分


def score_symbol(
    sym: Symbol, is_patch: bool, new_signature: bool, direct: list[dict], test_refs: list[dict]
) -> tuple[int, list[str]]:
    """可解释的风险打分：每一项加分都带理由（DESIGN.md L6）。"""
    score, reasons = 0, []
    uniq = {(c["file"], c["qualname"]) for c in direct}
    if uniq:
        add = min(len(uniq), 5) * 2
        score += add
        reasons.append(f"直接调用者 {len(uniq)} 处(+{add})")
    if sym.is_public:
        score += 3
        reasons.append("公共 API(+3)")
    if is_patch:
        score += 3
        reasons.append("位于 patch/**(+3)")
    if not test_refs:
        score += 4
        reasons.append("无测试引用(+4)")
    if new_signature:
        score += 2
        reasons.append("新增/改动签名行(+2)")
    return score, reasons


def impact_summary(transitive: list[dict], cap: int = 50) -> dict:
    """多跳扩散范围：只留计数与去重文件（避免 impact.json 被热点函数撑爆）。"""
    files = sorted({c["file"] for c in transitive})
    return {
        "transitive_refs": len(transitive),
        "files": files[:cap],
        "files_truncated": len(files) > cap,
    }


# --------------------------------------------------------------- 主流程


def build_impact(
    repo: str, base: str, head: str, files=None, max_depth: int = 2, max_nodes: int = 200, max_symbols: int = 60
) -> dict:
    scope = review_scope.build_scope(repo, base, head, files)
    kept_py = [k["path"] for k in scope["kept"] if k["path"].endswith(".py")]

    seeds: list[dict] = []
    skipped: list[dict] = []
    for path in kept_py:
        src = source_of(repo, head, path)
        if src is None:
            skipped.append({"file": path, "reason": "unreadable_or_missing"})
            continue
        ranges = added_ranges(repo, base, head, path)
        for sym in symbols_of(repo, head, path):
            if not any(lo <= line <= hi for lo, hi in ranges for line in (sym.lineno, sym.end_lineno)):
                continue
            seeds.append(
                {
                    "symbol": sym,
                    "is_patch": path.startswith("vllm_ascend/patch/"),
                    "new_signature": any(lo <= sym.lineno <= hi for lo, hi in ranges),
                }
            )

    # 深审顺序：patch → 公共 API → 定义行靠前；测试文件内的符号不做调用者反查
    seeds.sort(
        key=lambda s: (
            s["symbol"].file.startswith("tests/"),
            not s["is_patch"],
            not s["symbol"].is_public,
            s["symbol"].file,
            s["symbol"].lineno,
        )
    )

    budget = {"left": max_nodes, "truncated": False}
    changed_symbols, missing_tests = [], []
    for index, item in enumerate(seeds):
        sym: Symbol = item["symbol"]
        entry = {
            "file": sym.file,
            "qualname": sym.qualname,
            "kind": sym.kind,
            "lineno": sym.lineno,
            "end_lineno": sym.end_lineno,
            "is_public": sym.is_public,
            "is_patch": item["is_patch"],
            "new_signature": item["new_signature"],
        }
        if index >= max_symbols:
            entry.update(
                {
                    "scan": "skipped",
                    "callers": [],
                    "caller_count": 0,
                    "impact": impact_summary([]),
                    "test_refs": [],
                    "risk": 0,
                    "reasons": [],
                }
            )
            skipped.append({"file": sym.file, "qualname": sym.qualname, "reason": f"超出 --max-symbols={max_symbols}"})
            changed_symbols.append(entry)
            continue
        if sym.file.startswith("tests/"):
            entry.update(
                {
                    "scan": "skipped(test file)",
                    "callers": [],
                    "caller_count": 0,
                    "impact": impact_summary([]),
                    "test_refs": [],
                    "risk": 0,
                    "reasons": [],
                }
            )
            changed_symbols.append(entry)
            continue
        if sym.name.startswith("__") and sym.name.endswith("__"):
            # 魔术方法没有独立的调用点：`git grep -w __init__` 会命中全仓所有构造
            # 调用，把节点预算瞬间烧光且没有区分度。影响面由所属类的条目承载。
            entry.update(
                {
                    "scan": "skipped(dunder)",
                    "callers": [],
                    "caller_count": 0,
                    "impact": impact_summary([]),
                    "test_refs": [],
                    "risk": 0,
                    "reasons": [],
                    "note": "魔术方法无独立调用点，影响面见所属类条目",
                }
            )
            changed_symbols.append(entry)
            continue
        direct, transitive = collect_callers(repo, head, sym, max_depth, budget)
        # test_refs 不过滤 import 行：这里问的是「测试有没有引用它」，import 也算引用；
        # 过滤会制造假的「缺测试」。而 callers 必须过滤——那边问的是「谁调用了它」。
        test_refs = [
            {"file": ref_path, "line": ref_line}
            for ref_path, ref_line, _ in grep_refs(repo, head, sym.name, TEST_PATHSPEC)
        ]
        score, reasons = score_symbol(sym, item["is_patch"], item["new_signature"], direct, test_refs)
        entry.update(
            {
                "scan": "full",
                "callers": direct,
                "caller_count": len({(c["file"], c["qualname"]) for c in direct}),
                "impact": impact_summary(transitive),
                "test_refs": test_refs,
                "risk": score,
                "reasons": reasons,
            }
        )
        # 空结果才附盲区说明（同 crg：提示只挂在空结果上，有结果时保持原样）；
        # 「0 个调用者」不等于「没人用」，这话必须紧跟在负结论旁边，不能只写在文末。
        if not direct:
            entry["caller_gaps"] = gaps_for(QUERY_CALLERS, sym.file)
        if not test_refs:
            entry["test_gaps"] = gaps_for(QUERY_TESTS, sym.file)
            missing_tests.append(
                {
                    "file": sym.file,
                    "qualname": sym.qualname,
                    "risk": score,
                    "gaps": entry["test_gaps"],
                }
            )
        changed_symbols.append(entry)

    risk_order = [
        e["file"] + "::" + e["qualname"] for e in sorted(changed_symbols, key=lambda e: -e["risk"]) if e["risk"] > 0
    ]
    missing_tests.sort(key=lambda m: -m["risk"])

    notes = [
        "脚本输出仅作线索，结论以源码为准（DESIGN.md 7.4 / 11 A7）。",
        "callers / test_refs 为文本反查（限 Python）：已排除 import 行与定义行；"
        "命中为空时，该条目会附 caller_gaps / test_gaps 说明「为什么这个空可能是假象」"
        "（盲区表见脚本内 GAPS）。",
    ]
    if budget["truncated"]:
        notes.append(
            f"调用者反查触发节点上限 --max-nodes={max_nodes}，结果被截断；可先跑脚本 A 收窄范围后重跑，或调大上限。"
        )

    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "diff_base": base,
        "head": head,
        "scope_coverage": scope["coverage"],
        "limits": {"max_depth": max_depth, "max_nodes": max_nodes, "max_symbols": max_symbols},
        "truncated": budget["truncated"],
        "changed_symbols": changed_symbols,
        "missing_tests": missing_tests,
        "risk_order": risk_order,
        "skipped": skipped,
        "notes": notes,
    }


def render_preview(impact: dict, top: int = 15) -> str:
    lines = [
        f"影响面扫描  diff base {impact['diff_base'][:12]} → {impact['head']}",
        f"改动符号 {len(impact['changed_symbols'])}："
        f"风险排序 {len(impact['risk_order'])} / 缺测试 {len(impact['missing_tests'])}"
        + ("（已截断）" if impact["truncated"] else ""),
        f"深审顺序（Top {min(top, len(impact['risk_order']))}）：",
    ]
    by_key = {e["file"] + "::" + e["qualname"]: e for e in impact["changed_symbols"]}
    for key in impact["risk_order"][:top]:
        entry = by_key[key]
        reasons = "；".join(entry["reasons"])
        lines.append(f"  [{entry['risk']:>2}] {key}  → {reasons}")
    if impact["missing_tests"]:
        lines.append("缺失测试清单（负结论含盲区说明，见 impact.json 的 gaps）：")
        for item in impact["missing_tests"][:top]:
            lines.append(f"  [{item['risk']:>2}] {item['file']}::{item['qualname']}")
    if impact["skipped"]:
        lines.append(f"未扫描 {len(impact['skipped'])} 项（见 impact.json 的 skipped）")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Reviewer 脚本 B：影响面 + 缺失测试 + 风险排序（只读）",
    )
    parser.add_argument("--repo", default=None, help="仓根（默认自动上溯定位）")
    parser.add_argument("--base", default=None, help="diff base（默认 merge-base main <head>）")
    parser.add_argument("--head", default="HEAD", help="评审对象 ref（默认 HEAD）")
    parser.add_argument("--out", default=None, help="<输出根目录>（默认取 .day0/*/tracker.md 所在目录）")
    parser.add_argument("--files", nargs="+", default=None, help="显式指定改动文件")
    parser.add_argument("--depth", type=int, default=2, help="调用者反查深度（默认 2）")
    parser.add_argument("--max-nodes", type=int, default=200, help="调用者节点上限（默认 200）")
    parser.add_argument("--max-symbols", type=int, default=60, help="参与逐符号分析的上限（默认 60）")
    parser.add_argument("--preview", action="store_true", help="只打印摘要，不写 impact.json")
    args = parser.parse_args()

    repo = os.path.abspath(args.repo) if args.repo else review_scope._repo_root()
    base = args.base or review_scope.resolve_base(repo, args.head)
    impact = build_impact(repo, base, args.head, args.files, args.depth, args.max_nodes, args.max_symbols)

    if args.preview:
        print(render_preview(impact))
        return 0

    out_root = review_scope.resolve_out_root(repo, args.out)
    out_dir = os.path.join(out_root, "review")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "impact.json")
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(impact, fh, ensure_ascii=False, indent=2)
    print(render_preview(impact))
    print(f"\n已写入 {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
