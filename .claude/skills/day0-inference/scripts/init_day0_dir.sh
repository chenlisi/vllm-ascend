#!/usr/bin/env bash
# Day0 立项脚本：创建本次 Day0 的输出根目录并打印路径。
# 用法：init_day0_dir.sh <模型输入路径（含 config.json 的目录）>
# 命名规则：<arch>_<yyyymmdd>_<num>
#   arch —— config.json 的 architectures 首项（非法字符替换为 _）
#   num  —— 从 1 开始；同日同模型已有目录时取最大 num + 1
set -euo pipefail

MODEL_PATH="${1:?用法: init_day0_dir.sh <模型输入路径>}"
CONFIG="$MODEL_PATH/config.json"
[ -f "$CONFIG" ] || { echo "错误：$CONFIG 不存在" >&2; exit 1; }

ARCH=$(python3 -c "import json,sys; a=json.load(open(sys.argv[1]))['architectures']; print(a[0] if isinstance(a,list) else a)" "$CONFIG")
ARCH=$(printf '%s' "$ARCH" | tr -c 'A-Za-z0-9_-' '_')

BASE=".day0"
DATE=$(date +%Y%m%d)
PREFIX="${ARCH}_${DATE}_"
mkdir -p "$BASE"

MAX=0
for d in "$BASE/${PREFIX}"*; do
  [ -d "$d" ] || continue
  n="${d##"$BASE/$PREFIX"}"
  case "$n" in *[!0-9]*) continue;; esac
  [ "$n" -gt "$MAX" ] && MAX=$n
done

DIR="$BASE/${PREFIX}$((MAX + 1))"
mkdir -p "$DIR"
echo "$DIR"
