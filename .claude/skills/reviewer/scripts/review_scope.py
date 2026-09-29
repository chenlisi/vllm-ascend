#!/usr/bin/env python3
"""Reviewer 脚本 A：确定性选文件 + 规则匹配 + 覆盖记账——只产证据，不做判定。

用法：
    review_scope.py [--repo DIR] [--base REF] [--head REF] [--out DIR]
                    [--files PATH ...] [--preview]

设计约束（见 DESIGN.md 6.2 / 11 A2）：
- **只读**：不改任何源码，只写 `<out>/review/scope.json`；
- **preview 与正式评审共用同一实现**（同一个 `build_scope()`），从根上杜绝
  「预览与实际不一致」的漂移——这是本脚本存在的首要理由；
- 每个被排除的文件都带**结构化理由（枚举）**，不是自由文本，便于覆盖矩阵
  核对「已排除 Y（附理由）」。

排除 gate 链的顺序即优先级，首个命中即判定（DESIGN.md 6.2：生成物 → 本地化
文件 → 测试数据 → 纯文档 → 纯资源）。每个 gate 的判定都刻意收窄：评审的价值
在「不漏」，gate 过宽会把该审的文件悄悄刷掉，那正是本设计要修掉的「覆盖不
完整」。因此只排除「确定产生不了评审信号」的文件。

与规则表的边界（同样收窄）：
- `docs/**` 与教程 `*.md` 是 R12 的评审对象，**不排除**；只排除仓库根部的
  元文档（LICENSE / CHANGELOG / NOTICE ...），它们不承载产品评审信号。
- `tests/**/*.py`（R11）与 `tests/**/*.yaml|json`（E 项 E2E 配置、R11-3 nightly
  注册）**保留**，只排除数据块（权重 / 数据集 / 二进制 fixture）。
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import subprocess
import sys
from datetime import datetime

RULE_TABLE_REL = ".claude/skills/reviewer/reference/review-rules.md"

# --------------------------------------------------------------- 排除 gate 链

GATE_GENERATED = "generated"
GATE_LOCALIZED = "localized"
GATE_TEST_DATA = "test_data"
GATE_PURE_DOC = "pure_doc"
GATE_ASSET = "asset"
GATE_DELETED = "deleted"

REASON = {
    GATE_GENERATED: "generated_artifact",
    GATE_LOCALIZED: "localized_mirror",
    GATE_TEST_DATA: "test_data_blob",
    GATE_PURE_DOC: "meta_doc",
    GATE_ASSET: "binary_asset",
    GATE_DELETED: "deleted_file",
}

_GENERATED_RE = re.compile(
    r"(^|/)(__pycache__|\.pytest_cache|\.mypy_cache|\.ruff_cache|\.code-review-graph"
    r"|build|dist|node_modules|\.vllm_ascend|csrc/output|csrc/third_party)(/|$)"
    r"|\.py[co]$|\.egg-info(/|$)|\.so$|\.orig$|\.rej$"
)

_LOCALIZED_RE = re.compile(r"(^|/)zh(/|$)|\.zh(_cn)?\.|_zh\.|\.po$|\.mo$|\.pot$")

# 数据块（二进制 / 数据集），不含 yaml|json|toml 配置——后者是 R11-3、E 项的
# 评审对象，不能排除。
_DATA_EXTS = (
    ".bin",
    ".safetensors",
    ".pt",
    ".pth",
    ".npy",
    ".npz",
    ".ckpt",
    ".jsonl",
    ".csv",
    ".parquet",
    ".arrow",
    ".h5",
    ".onnx",
    ".pkl",
    ".pickle",
    ".zip",
    ".tar",
    ".gz",
    ".tgz",
    ".7z",
    ".db",
    ".sqlite",
)

# 仓库根部的元文档：不承载产品评审信号（产品文档在 docs/**，走 R12）。
_META_DOC_STEMS = (
    "changelog",
    "license",
    "notice",
    "contributing",
    "code_of_conduct",
    "security",
    "authors",
    "governance",
    "third_party",
)

_ASSET_EXTS = (
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".svg",
    ".ico",
    ".bmp",
    ".webp",
    ".pdf",
    ".woff",
    ".woff2",
    ".ttf",
    ".otf",
    ".mp4",
    ".mov",
    ".avi",
)


def _is_generated(path: str) -> bool:
    return bool(_GENERATED_RE.search(path))


def _is_localized(path: str) -> bool:
    return bool(_LOCALIZED_RE.search(path.lower()))


def _is_test_data(path: str) -> bool:
    if not path.startswith("tests/"):
        return False
    return path.endswith(_DATA_EXTS) or "/data/" in path


def _is_pure_doc(path: str) -> bool:
    if "/" in path:
        return False
    stem = path.rsplit(".", 1)[0].lower()
    return any(stem == s or stem.startswith(s) for s in _META_DOC_STEMS)


def _is_asset(path: str) -> bool:
    return path.lower().endswith(_ASSET_EXTS)


# 顺序即优先级
GATES = (
    (GATE_GENERATED, _is_generated),
    (GATE_LOCALIZED, _is_localized),
    (GATE_TEST_DATA, _is_test_data),
    (GATE_PURE_DOC, _is_pure_doc),
    (GATE_ASSET, _is_asset),
)


def gate_of(path: str) -> str | None:
    """返回该文件命中的排除 gate；None 表示保留。"""
    for gate, predicate in GATES:
        if predicate(path):
            return gate
    return None


# --------------------------------------------------------------- 规则表解析

_GROUP_ROW = re.compile(r"^\|\s*(R\d+)\s*\|\s*(.+?)\s*\|\s*(.+?)\s*\|\s*$")
_RULE_ROW = re.compile(r"^\|\s*(R\d+-\d+)\s*\|\s*(.+?)\s*\|\s*(阻断|重要|建议)\s*\|\s*(.+?)\s*\|\s*$")
_SECTION = re.compile(r"^##\s+(R\d+)\s+(.+?)\s*$")
_LOCALIZABLE_EXT = (".py", ".txt", ".md", ".json", ".yaml", ".yml", ".cfg", ".toml", ".sh", ".jsonl", ".ini")


def _globs_from_cell(cell: str) -> list[str]:
    """从 glob 表单元格里抽真正的路径模式。

    同一个目录在规则表里既可能写全路径（`vllm_ascend/patch/**`）也可能写简写
    （`patch/**`），两者语义相同；含 `/` 或 `*` 的 token 视为模式，纯文件名
    （`setup.py` 等）也保留；「配置类」「KV 传输与内存相关」「跨组」这类散文
    说明性 token 丢弃（它们靠 R14 人工判定兜底）。
    """
    tokens = re.findall(r"`([^`]+)`", cell)
    if not tokens:
        tokens = re.split(r"[、,，]", cell)
    out = []
    for token in tokens:
        token = token.strip()
        if not token:
            continue
        if "/" in token or "*" in token or token.endswith(_LOCALIZABLE_EXT):
            out.append(token)
    return out


def parse_rule_table(text: str) -> tuple[dict, dict]:
    """解析 `review-rules.md` → (groups, rules)。

    groups: gid → {title, globs, rules, manual}
    rules:  rid → {group, severity, text}
    """
    groups: dict[str, dict] = {}
    rules: dict[str, dict] = {}
    for raw in text.splitlines():
        line = raw.rstrip()
        section = _SECTION.match(line)
        if section:
            gid = section.group(1)
            groups.setdefault(gid, {"title": section.group(2), "globs": [], "rules": [], "manual": True})
            groups[gid]["title"] = section.group(2)
            continue
        row = _RULE_ROW.match(line)
        if row:
            rid, body, severity = row.group(1), row.group(2), row.group(3)
            gid = rid.split("-")[0]
            rules[rid] = {"group": gid, "severity": severity, "text": body}
            groups.setdefault(gid, {"title": "", "globs": [], "rules": [], "manual": True})
            groups[gid]["rules"].append(rid)
            continue
        row = _GROUP_ROW.match(line)
        if row:
            gid = row.group(1)
            globs = _globs_from_cell(row.group(2))
            entry = groups.setdefault(gid, {"title": row.group(3), "globs": [], "rules": [], "manual": True})
            entry["globs"] = globs
            if not entry["title"]:
                entry["title"] = row.group(3)
            continue
    for gid, entry in groups.items():
        entry["manual"] = not entry["globs"]
        entry["rules"] = sorted(entry["rules"], key=_rule_sort_key)
    return groups, rules


def _rule_sort_key(rid: str) -> tuple[int, int]:
    gid, idx = rid.split("-")
    return int(gid[1:]), int(idx)


def load_rule_table(repo: str) -> tuple[dict, dict]:
    path = os.path.join(repo, RULE_TABLE_REL)
    with open(path, encoding="utf-8") as fh:
        groups, rules = parse_rule_table(fh.read())
    if not rules:
        raise RuntimeError(f"规则表解析为空：{path}")
    return groups, rules


# --------------------------------------------------------------- glob 匹配

_REGEX_CACHE: dict[str, re.Pattern] = {}


def _glob_regex(pattern: str) -> re.Pattern:
    """把 glob 编译为正则：`**/` 匹配零或多层目录，`*` 不跨 `/`。"""
    cached = _REGEX_CACHE.get(pattern)
    if cached is not None:
        return cached
    out = ["^"]
    i, n = 0, len(pattern)
    while i < n:
        char = pattern[i]
        if char == "*":
            if i + 1 < n and pattern[i + 1] == "*":
                j = i + 2
                if j < n and pattern[j] == "/":
                    out.append("(?:[^/]+/)*")
                    i = j + 1
                else:
                    out.append(".*")
                    i = j
            else:
                out.append("[^/]*")
                i += 1
        elif char == "?":
            out.append("[^/]")
            i += 1
        else:
            out.append(re.escape(char))
            i += 1
    out.append("$")
    compiled = re.compile("".join(out))
    _REGEX_CACHE[pattern] = compiled
    return compiled


def glob_match(path: str, pattern: str) -> bool:
    """路径是否命中模式（以仓根为基准）。

    额外允许一种简写展开：规则表里同一目录既可能写全路径
    （`vllm_ascend/patch/**`）也可能写简写（`patch/**`，见 R9），二者语义相同。
    这里只做**包内简写**的展开（去掉 `vllm_ascend/` 前缀后再试一次），不做任意
    后缀匹配——否则 `tests/ut/patch/...` 会误命中 `patch/**`，把 R9 的规则噪声
    加到测试文件上。
    """
    rx = _glob_regex(pattern)
    if rx.match(path):
        return True
    prefix = "vllm_ascend/"
    return path.startswith(prefix) and bool(rx.match(path[len(prefix) :]))


def groups_for_path(path: str, groups: dict) -> list[str]:
    return [gid for gid, entry in groups.items() if entry["globs"] and any(glob_match(path, g) for g in entry["globs"])]


def rules_for_path(path: str, groups: dict) -> list[str]:
    out: set[str] = set()
    for gid in groups_for_path(path, groups):
        out.update(groups[gid]["rules"])
    return sorted(out, key=_rule_sort_key)


# --------------------------------------------------------------- git / 输入


def _repo_root() -> str:
    """从脚本位置向上找仓根（同时含 .git 与 vllm_ascend/）。"""
    here = os.path.dirname(os.path.abspath(__file__))
    while True:
        if os.path.isdir(os.path.join(here, "vllm_ascend")) and os.path.isdir(os.path.join(here, ".git")):
            return here
        parent = os.path.dirname(here)
        if parent == here:
            raise RuntimeError("无法定位 vllm-ascend 仓根（向上未找到同时含 .git 与 vllm_ascend/ 的目录）。")
        here = parent


def _git(repo: str, *args: str, check: bool = True) -> str:
    result = subprocess.run(
        ["git", "-C", repo, *args],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if check and result.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} 失败（{result.returncode}）：{result.stderr.strip()}")
    return result.stdout


def resolve_base(repo: str, head: str) -> str:
    """diff base = 与 main 的 merge-base（DESIGN.md 6.2）。"""
    for ref in ("main", "origin/main"):
        out = _git(repo, "merge-base", ref, head, check=False).strip()
        if out:
            return out
    raise RuntimeError(
        f"无法用 `git merge-base main {head}` 求出 diff base（main / origin/main 均不可得）。 请用 --base 显式指定。"
    )


def enumerate_changes(repo: str, base: str, head: str) -> list[dict]:
    """`git diff --name-status` → [{path, status}]（比 --name-only 多带状态）。

    多带状态是为了不让删除静默消失：`--diff-filter=ACMR` 会把删除过滤掉，
    覆盖矩阵里就少一行、看不出「这里被跳过了」——这正是覆盖记账要避免的。
    """
    out = _git(repo, "diff", "--name-status", f"{base}...{head}")
    changes = []
    for line in out.splitlines():
        parts = line.split("\t")
        if len(parts) < 2:
            continue
        status, path = parts[0][0], parts[-1]
        changes.append({"path": path, "status": status})
    return changes


def build_scope(repo: str, base: str, head: str, files: list[str] | None = None) -> dict:
    """确定性选文件 + 规则匹配 + 覆盖记账（preview 与正式评审共用）。"""
    groups, rules = load_rule_table(repo)

    if files:
        changes = [{"path": p, "status": "M"} for p in files]
    else:
        changes = enumerate_changes(repo, base, head)

    kept, excluded = [], []
    for item in changes:
        path, status = item["path"], item["status"]
        if status == "D":
            excluded.append({"path": path, "gate": GATE_DELETED, "reason": REASON[GATE_DELETED]})
            continue
        gate = gate_of(path)
        if gate:
            excluded.append({"path": path, "gate": gate, "reason": REASON[gate]})
            continue
        group_ids = groups_for_path(path, groups)
        rule_ids = rules_for_path(path, groups)
        kept.append({"path": path, "groups": group_ids, "rules": rule_ids})

    by_gate: dict[str, int] = {}
    for item in excluded:
        by_gate[item["gate"]] = by_gate.get(item["gate"], 0) + 1

    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "diff_base": base,
        "head": head,
        "rule_table": RULE_TABLE_REL,
        "coverage": {
            "changed": len(changes),
            "kept": len(kept),
            "excluded": len(excluded),
            "by_gate": by_gate,
            "kept_without_rules": sum(1 for k in kept if not k["rules"]),
        },
        "changed_files": changes,
        "kept": kept,
        "excluded": excluded,
        "manual_groups": sorted(
            [gid for gid, entry in groups.items() if entry["manual"]],
            key=lambda g: int(g[1:]),
        ),
        "groups": groups,
        "rules": rules,
    }


# --------------------------------------------------------------- 输出


def resolve_out_root(repo: str, explicit: str | None) -> str:
    """`<输出根目录>` = 唯一命中的 `.day0/*/tracker.md` 所在目录。"""
    if explicit:
        return os.path.abspath(explicit)
    matches = sorted(glob.glob(os.path.join(repo, ".day0", "*", "tracker.md")))
    if len(matches) == 1:
        return os.path.dirname(matches[0])
    if not matches:
        raise RuntimeError(
            "未找到 <输出根目录>：`.day0/*/tracker.md` 无命中。"
            " 评审判定需要 tracker 作为流程锚点，请用 --out 显式指定输出根目录。"
        )
    raise RuntimeError(
        f"`.day0` 下命中 {len(matches)} 个 tracker.md，无法判定唯一输出根目录：\n  "
        + "\n  ".join(matches)
        + "\n请用 --out 显式指定。"
    )


def render_preview(scope: dict) -> str:
    lines = [
        f"评审范围预览  diff base {scope['diff_base'][:12]} → {scope['head']}",
        f"改动文件 {scope['coverage']['changed']}："
        f"保留 {scope['coverage']['kept']} / 排除 {scope['coverage']['excluded']}",
    ]
    if scope["excluded"]:
        lines.append("排除明细：")
        for item in scope["excluded"]:
            lines.append(f"  [{item['gate']}] {item['path']}  ({item['reason']})")
    lines.append("保留文件（规则组 / 规则条数）：")
    for item in scope["kept"]:
        groups = ",".join(item["groups"]) or "-"
        lines.append(f"  {item['path']}  → {groups} / {len(item['rules'])} 条")
    if scope["coverage"]["kept_without_rules"]:
        lines.append(
            f"提醒：{scope['coverage']['kept_without_rules']} 个保留文件未命中任何规则组"
            "（无现成规则，按 DESIGN.md 8.3 交人工判定）。"
        )
    lines.append("人工判定组（无 glob，靠 Reviewer 知识）：" + (",".join(scope["manual_groups"]) or "无"))
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Reviewer 脚本 A：确定性选文件 + 规则匹配 + 覆盖记账（只读）",
    )
    parser.add_argument("--repo", default=None, help="仓根（默认自动上溯定位）")
    parser.add_argument("--base", default=None, help="diff base（默认 merge-base main <head>）")
    parser.add_argument("--head", default="HEAD", help="评审对象 ref（默认 HEAD）")
    parser.add_argument("--out", default=None, help="<输出根目录>（默认取 .day0/*/tracker.md 所在目录）")
    parser.add_argument("--files", nargs="+", default=None, help="显式指定改动文件（跳过 git diff，便于复现）")
    parser.add_argument("--preview", action="store_true", help="只打印预览，不写 scope.json（与正式评审共用同一实现）")
    args = parser.parse_args()

    repo = os.path.abspath(args.repo) if args.repo else _repo_root()
    base = args.base or resolve_base(repo, args.head)
    scope = build_scope(repo, base, args.head, args.files)

    if args.preview:
        print(render_preview(scope))
        return 0

    out_root = resolve_out_root(repo, args.out)
    out_dir = os.path.join(out_root, "review")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "scope.json")
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(scope, fh, ensure_ascii=False, indent=2)
    print(render_preview(scope))
    print(f"\n已写入 {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
