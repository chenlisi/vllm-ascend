#!/usr/bin/env python3
"""Reviewer 脚本 C：行号定位 + findings 批量回填——精确优先、模糊兜底、失败显式标记。

用法（单条定位）：
    locate_comment.py FILE (--text TEXT | --regex RE | --from-line N)
                     [--repo DIR] [--ref REF] [--base REF]
                     [--threshold 0.85] [--limit 10] [--json]

用法（批量回填，对应 G9 的结构化输出契约）：
    locate_comment.py --findings review/findings.json
                     [--repo DIR] [--ref REF] [--base REF]
                     [--out review/findings.json] [--json]

设计约束（见 DESIGN.md 6.2 / 7.6 G4・G9 / 11 A4）：
- **只读**：不写源码；批量模式的产物只写 `--out` 指定的 findings 文件；
- **分级定位**（对齐 `ocr` 的 `internal/diff/resolver.go`）：先限定在**本次改动的
  新增行**内做归一化精确匹配（hunk 新侧优先），再退到全文精确，最后才是滑窗模糊。
  这样「同一文本在文件里还有一处老代码」时，会把评论钉在**改动处**而不是老位置；
- **变体兜底**：若片段是从 `git diff` 直接粘来的（行首带 `+`/`-` 标记），主变体
  匹配不到时自动去标记重试——仅在**同一级内**回退，不跨级降级，避免把锚点锚错；
- **唯一命中才认定**：多处命中一律标 `ambiguous` 且不给单一行号——报告里的行号
  要么可核对，要么显式认输交人工回定位。

退出码：0 = 全部唯一命中；1 = 存在未定位（ambiguous / not_found / error）；2 = 用法错误。
"""

from __future__ import annotations

import argparse
import difflib
import json
import os
import re
import subprocess
import sys
from datetime import datetime

SCHEMA_VERSION = "1"
SEVERITIES = ("阻断", "重要", "建议")
# locate() 的 status → 汇总计数的键（`unique` 在汇总层叫 `located`）
COUNT_KEY = {"unique": "located", "ambiguous": "ambiguous", "not_found": "not_found", "error": "error"}
_WS = re.compile(r"\s+")
_HUNK = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@")

# 分级定位的优先级：前一级只要有命中就定案，不跨级继续找
PHASES = ("hunk_new", "exact", "fuzzy")


# --------------------------------------------------------------- 文本规范化


def _norm_line(line: str) -> str:
    return _WS.sub(" ", line).strip()


def _norm_lines(lines: list[str]) -> str:
    return "\n".join(_norm_line(line) for line in lines).strip()


def _strip_marker(line: str) -> str:
    """去掉 diff 行首标记（`+`/`-`），保留缩进。

    `-1` / `-0.5` 这类负数字面量不是标记，不剥——否则会把代码里的负数当 diff 标记。
    """
    stripped = line.lstrip()
    if not stripped or stripped[0] not in "+-":
        return line
    rest = stripped[1:]
    if rest[:1].isdigit() or rest[:1] == ".":
        return line
    return line[: len(line) - len(stripped)] + rest


def query_variants(query: list[str]) -> list[tuple[str, list[str]]]:
    variants = [("raw", list(query))]
    stripped = [_strip_marker(line) for line in query]
    if stripped != list(query):
        variants.append(("diff-marker-stripped", stripped))
    return variants


# --------------------------------------------------------------- 读取与 hunk


def read_source(repo: str, path: str, ref: str | None) -> list[str]:
    if ref:
        result = subprocess.run(
            ["git", "-C", repo, "show", f"{ref}:{path}"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        if result.returncode != 0:
            raise RuntimeError(f"`git show {ref}:{path}` 失败：{result.stderr.strip()}")
        return result.stdout.splitlines()
    full = path if os.path.isabs(path) else os.path.join(repo, path)
    with open(full, encoding="utf-8") as fh:
        return fh.read().splitlines()


def added_ranges(repo: str, base: str, head: str, path: str) -> list[tuple[int, int]]:
    """文件在 head 中的新增行区间（`git diff -U0` 的 `+c,d`）。"""
    result = subprocess.run(
        ["git", "-C", repo, "diff", "--unified=0", f"{base}...{head}", "--", path],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    ranges = []
    for line in result.stdout.splitlines():
        match = _HUNK.match(line)
        if not match:
            continue
        start, count = int(match.group(1)), int(match.group(2) or 1)
        if count > 0:
            ranges.append((start, start + count - 1))
    return ranges


def _window_in_added(start: int, size: int, ranges: list[tuple[int, int]] | None) -> bool:
    """窗口（0 基 start，长度 size）是否完整落在**同一个**新增区间内。"""
    if ranges is None:
        return True
    lo_line, hi_line = start + 1, start + size
    return any(lo <= lo_line and hi_line <= hi for lo, hi in ranges)


# --------------------------------------------------------------- 匹配


def exact_candidates(lines: list[str], query: list[str], ranges: list[tuple[int, int]] | None = None) -> list[dict]:
    """规范化空白后逐字符相等的窗口。"""
    target = _norm_lines(query)
    size = len(query)
    out = []
    for start in range(0, len(lines) - size + 1):
        if not _window_in_added(start, size, ranges):
            continue
        if _norm_lines(lines[start : start + size]) == target:
            out.append(_candidate(start, size, 1.0, lines))
    return out


def fuzzy_candidates(
    lines: list[str], query: list[str], threshold: float, ranges: list[tuple[int, int]] | None = None
) -> list[dict]:
    """滑窗模糊匹配；窗口高度在 query 行数附近浮动，容忍多/少一行。"""
    target = _norm_lines(query)
    best: dict[int, float] = {}
    sizes = sorted({max(1, len(query) + delta) for delta in (-1, 0, 1)})
    for size in sizes:
        for start in range(0, len(lines) - size + 1):
            if not _window_in_added(start, size, ranges):
                continue
            ratio = difflib.SequenceMatcher(None, _norm_lines(lines[start : start + size]), target).ratio()
            if ratio >= threshold and ratio > best.get(start, 0.0):
                best[start] = ratio
    out = [_candidate(start, len(query), score, lines) for start, score in best.items()]
    return sorted(out, key=lambda c: (-c["score"], c["line"]))


def _candidate(start: int, size: int, score: float, lines: list[str]) -> dict:
    text = " / ".join(lines[start : start + size]).strip()
    return {
        "line": start + 1,
        "end_line": start + size,
        "score": round(score, 4),
        "text": text[:200],
    }


def locate(
    lines: list[str],
    query: list[str],
    regex: str | None,
    threshold: float,
    limit: int,
    ranges: list[tuple[int, int]] | None = None,
) -> dict:
    """分级定位：hunk_new → exact → fuzzy；同级内主变体失败才试去标记变体。"""
    if regex:
        pattern = re.compile(regex)
        hits = [
            {"line": i + 1, "end_line": i + 1, "score": 1.0, "text": lines[i].strip()[:200]}
            for i, line in enumerate(lines)
            if pattern.search(line)
        ]
        return _result(hits, "regex", None, limit)

    variants = query_variants(query)
    for phase in PHASES:
        if phase == "hunk_new" and not ranges:
            continue
        for label, candidate_query in variants:
            if phase == "fuzzy":
                hits = fuzzy_candidates(lines, candidate_query, threshold)
            else:
                hits = exact_candidates(lines, candidate_query, ranges if phase == "hunk_new" else None)
            if hits:
                return _result(hits, phase, label, limit)
    return _result([], "exact", variants[0][0], limit)


def _result(hits: list[dict], match_type: str, variant: str | None, limit: int) -> dict:
    if len(hits) == 1:
        status, line, end_line = "unique", hits[0]["line"], hits[0]["end_line"]
        note = None
    elif len(hits) > 1:
        status, line, end_line = "ambiguous", None, None
        note = (
            f"{len(hits)} 处命中，无法唯一认定——请按上下文人工回定位"
            "（脚本不挑「最像的一条」，避免把报告行号钉在错处）。"
        )
    else:
        status, line, end_line = "not_found", None, None
        note = "未命中：片段可能与目标文件不一致，或该文本本就来自其它文件。"
    return {
        "status": status,
        "match_type": match_type,
        "variant": variant,
        "line": line,
        "end_line": end_line,
        "candidates": hits[:limit],
        "candidate_count": len(hits),
        "note": note,
    }


# --------------------------------------------------------------- 批量回填


def locate_findings(repo: str, data: dict, ref: str | None, base: str | None, threshold: float, limit: int) -> dict:
    """对 findings 逐条回填 `start_line` / `end_line` 与定位状态（只产结构化产物）。"""
    findings = data.get("findings") or []
    use_hunks = bool(base and ref)
    ranges_cache: dict[str, list[tuple[int, int]]] = {}
    located, counts = [], {"located": 0, "ambiguous": 0, "not_found": 0, "error": 0}
    unlocated_by_severity = {sev: 0 for sev in SEVERITIES}

    for index, raw in enumerate(findings):
        entry = dict(raw)
        path = entry.get("path") or ""
        anchor = entry.get("existing_code") or entry.get("text") or ""
        entry.setdefault("id", f"F{index + 1}")
        if not path or not anchor:
            _mark(entry, "error", None, None, None, 0, "缺少 path 或 existing_code/text，无法定位。")
            counts["error"] += 1
        else:
            try:
                lines = read_source(repo, path, ref)
                if use_hunks and path not in ranges_cache:
                    ranges_cache[path] = added_ranges(repo, base, ref, path)
                result = locate(lines, anchor.splitlines(), None, threshold, limit, ranges_cache.get(path))
                _mark(
                    entry,
                    result["status"],
                    result["match_type"],
                    result["variant"],
                    result["line"],
                    result["candidate_count"],
                    result["note"],
                )
                if result["status"] == "unique" and result["candidates"]:
                    entry["end_line"] = result["candidates"][0]["end_line"]
                counts[COUNT_KEY[result["status"]]] += 1
            except (OSError, RuntimeError) as exc:
                _mark(entry, "error", None, None, None, 0, f"读取失败：{exc}")
                counts["error"] += 1
        if entry["locate_status"] != "unique":
            severity = entry.get("severity")
            if severity in unlocated_by_severity:
                unlocated_by_severity[severity] += 1
        located.append(entry)

    return {
        "schema_version": data.get("schema_version", SCHEMA_VERSION),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "ref": ref,
        "diff_base": base,
        "hunk_preference": "enabled" if use_hunks else "disabled（需同时给 --base 与 --ref）",
        "findings": located,
        "summary": {
            "total": len(located),
            **counts,
            "unlocated_by_severity": unlocated_by_severity,
        },
    }


def _mark(
    entry: dict,
    status: str,
    match_type: str | None,
    variant: str | None,
    line: int | None,
    candidates: int,
    note: str | None,
) -> None:
    entry["locate_status"] = status
    entry["locate_match_type"] = match_type
    entry["locate_variant"] = variant
    entry["locate_candidates"] = candidates
    entry["locate_note"] = note
    entry["start_line"] = line
    if status != "unique":
        entry["end_line"] = None


# --------------------------------------------------------------- 输出


def render_text(path: str, query_preview: str, result: dict) -> str:
    head = f"{path}  query=「{query_preview}」  ({result['match_type']}"
    if result.get("variant") and result["variant"] != "raw":
        head += f", {result['variant']}"
    head += ")"
    if result["status"] == "unique":
        return f"unique  {head}\n  → {path}:{result['line']}"
    if result["status"] == "ambiguous":
        lines = [f"定位失败（ambiguous）  {head}", f"  {result['note']}"]
        for cand in result["candidates"]:
            lines.append(f"  候选 {path}:{cand['line']}  score={cand['score']}  {cand['text']}")
        return "\n".join(lines)
    return f"定位失败（not_found）  {head}\n  {result['note']}"


def render_batch(payload: dict) -> str:
    summary = payload["summary"]
    lines = [
        f"findings 回填  ref={payload['ref'] or '(工作区)'}  hunk 侧别匹配：{payload['hunk_preference']}",
        f"共 {summary['total']} 条：定位成功 {summary['located']} / "
        f"歧义 {summary['ambiguous']} / 未命中 {summary['not_found']} / "
        f"错误 {summary['error']}",
    ]
    pending = {k: v for k, v in summary["unlocated_by_severity"].items() if v}
    if pending:
        lines.append("未定位按严重度：" + "、".join(f"{k} {v} 条" for k, v in pending.items()))
    for entry in payload["findings"]:
        mark = "✓" if entry["locate_status"] == "unique" else "✗"
        position = f":{entry['start_line']}" if entry["start_line"] else ""
        lines.append(
            f"  {mark} [{entry.get('severity', '?')}] {entry['id']} "
            f"{entry.get('path', '?')}{position}  ({entry['locate_status']}"
            f"/{entry['locate_match_type']})"
        )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Reviewer 脚本 C：行号定位 + findings 批量回填（只读）",
    )
    parser.add_argument("file", nargs="?", help="待定位的文件（仓内相对路径）")
    parser.add_argument("--text", help="待定位片段（可多行）")
    parser.add_argument("--regex", help="正则定位（与 --text 二选一）")
    parser.add_argument("--from-line", type=int, default=None, help="以该行（1 起）的原文作为片段")
    parser.add_argument("--findings", help="批量回填模式：读入 findings JSON")
    parser.add_argument("--out", default=None, help="批量回填模式：写出回填后的 JSON")
    parser.add_argument("--repo", default=None, help="仓根（默认自动上溯定位）")
    parser.add_argument(
        "--ref", default=None, help="从该 git ref 读取文件（默认读工作区）；与 --base 同时给出才启用 hunk 侧别匹配"
    )
    parser.add_argument("--base", default=None, help="diff base；与 --ref 同时给出才启用 hunk 侧别匹配")
    parser.add_argument("--threshold", type=float, default=0.85, help="模糊匹配相似度阈值")
    parser.add_argument("--limit", type=int, default=10, help="每条最多列出多少候选")
    parser.add_argument("--json", action="store_true", help="以 JSON 输出（便于串到报告里）")
    args = parser.parse_args()

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import review_scope  # noqa: E402

    repo = os.path.abspath(args.repo) if args.repo else review_scope._repo_root()

    if args.findings:
        try:
            with open(args.findings, encoding="utf-8") as fh:
                data = json.load(fh)
        except (OSError, json.JSONDecodeError) as exc:
            print(f"ERROR: 读取 {args.findings} 失败：{exc}", file=sys.stderr)
            return 2
        payload = locate_findings(repo, data, args.ref, args.base, args.threshold, args.limit)
        if args.out:
            os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
            with open(args.out, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, ensure_ascii=False, indent=2)
        if args.json:
            print(json.dumps(payload, ensure_ascii=False, indent=2))
        else:
            print(render_batch(payload))
            if args.out:
                print(f"\n已写入 {args.out}")
        pending = payload["summary"]["total"] - payload["summary"]["located"]
        return 0 if pending == 0 else 1

    if not args.file:
        print("ERROR: 需给出 FILE（单条定位）或 --findings（批量回填）", file=sys.stderr)
        return 2

    try:
        lines = read_source(repo, args.file, args.ref)
    except (OSError, RuntimeError) as exc:
        print(f"ERROR: 读取 {args.file} 失败：{exc}", file=sys.stderr)
        return 2

    if args.from_line is not None:
        if not 1 <= args.from_line <= len(lines):
            print(f"ERROR: --from-line {args.from_line} 超出文件行数 {len(lines)}", file=sys.stderr)
            return 2
        query = [lines[args.from_line - 1]]
    elif args.text:
        query = args.text.splitlines() or [args.text]
    elif not args.regex:
        print("ERROR: 需给出 --text / --regex / --from-line 之一", file=sys.stderr)
        return 2
    else:
        query = []

    ranges = None
    if args.base and args.ref:
        ranges = added_ranges(repo, args.base, args.ref, args.file)
    result = locate(lines, query, args.regex, args.threshold, args.limit, ranges)
    query_preview = (args.regex and f"regex:{args.regex}") or _WS.sub(" ", " ".join(query)).strip()[:60]

    if args.json:
        print(json.dumps({"file": args.file, "query": query_preview, **result}, ensure_ascii=False, indent=2))
    else:
        print(render_text(args.file, query_preview, result))

    return 0 if result["status"] == "unique" else 1


if __name__ == "__main__":
    sys.exit(main())
