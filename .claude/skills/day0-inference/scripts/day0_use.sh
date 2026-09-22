#!/usr/bin/env bash
# 切换 .day0/.current 指针到指定 run——恢复旧 run 或高频迭代时手动定位用；
# 平时由 init_day0_dir.sh 自动维护，无需手工调用。
# 用法：day0_use.sh <run 目录> | --latest
# 注意：SessionStart hook 只在会话启动时读指针，切换后须 /clear 或新开会话才注入新环境变量。
set -euo pipefail

[ -d vllm_ascend ] || { echo "须从 vllm-ascend 仓根运行" >&2; exit 1; }
BASE=".day0"

target="${1:?用法: day0_use.sh <run 目录 | --latest>}"
if [ "$target" = "--latest" ]; then
  latest="" newest=0 mt
  shopt -s nullglob
  for t in "$BASE"/*/tracker.md; do
    mt=$(stat -f %m "$t" 2>/dev/null || stat -c %Y "$t" 2>/dev/null || echo 0)
    if [ "$mt" -gt "$newest" ]; then newest=$mt; latest=${t%/tracker.md}; fi
  done
  shopt -u nullglob
  [ -n "$latest" ] || { echo "错误：$BASE 下没有 run" >&2; exit 1; }
  target="$latest"
fi
[ -f "$target/tracker.md" ] || { echo "错误：$target 下无 tracker.md（不是有效 run 目录）" >&2; exit 1; }

ABS=$(cd "$target" && pwd)
printf '%s\n' "$ABS" > "$BASE/.current.tmp" && mv "$BASE/.current.tmp" "$BASE/.current"
echo "$ABS"
