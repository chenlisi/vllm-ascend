#!/usr/bin/env bash
# Day0 立项脚本：创建本次 Day0 的输出根目录并打印路径。
# 用法：init_day0_dir.sh <模型输入路径（含 config.json 的目录）>
# 命名规则：<arch>_<yyyymmdd>_<num>
#   arch —— config.json 的 architectures 首项（非法字符替换为 _）
#   num  —— 从 1 开始；同日同模型已有目录时取最大 num + 1
set -euo pipefail

[ -d vllm_ascend ] || { echo "须从 vllm-ascend 仓根运行" >&2; exit 1; }

MODEL_PATH="${1:?用法: init_day0_dir.sh <模型输入路径>}"
CONFIG="$MODEL_PATH/config.json"
[ -f "$CONFIG" ] || { echo "错误：$CONFIG 不存在" >&2; exit 1; }

ARCH=$(python3 -c "
import json, sys
cfg = json.load(open(sys.argv[1]))
a = cfg.get('architectures')
arch = a[0] if isinstance(a, list) and a else (a if isinstance(a, str) else None)
if not arch:
    sys.exit('错误：' + sys.argv[1] + ' 缺 architectures 键或首项为空')
print(arch)
" "$CONFIG") || exit 1
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
# 必须输出绝对路径：调用方以 `export ASCENDBOT_FILE_PATH=$(...)` 捕获，全流程据此引用
# 产物目录。相对路径会让 cwd 不同的子代理解析到不同位置，导致产物散落。
echo "$(cd "$DIR" && pwd)"
