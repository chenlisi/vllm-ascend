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

# 解释器：优先推理环境 venv（与 preflight 同一套环境纪律，见 AGENTS.md），
# 再退 uv 能找到的解释器，最后才系统 python3。只读 stdlib json，任一可用即可。
# 注意：只用 `uv python find`（只读查询）——`uv run python` 会在仓根自动建 .venv，
# 探测解释器不该产生写盘副作用。
PY="${DAY0_PYTHON:-}"
if [ -z "$PY" ] && [ -x .venv/bin/python ]; then
  PY=.venv/bin/python
fi
if [ -z "$PY" ] && command -v uv >/dev/null 2>&1; then
  PY=$(uv python find 2>/dev/null || true)
fi
if [ -z "$PY" ]; then
  command -v python3 >/dev/null 2>&1 && PY=python3
fi
[ -n "$PY" ] || {
  echo "错误：找不到可用 python（试过 \$DAY0_PYTHON、.venv/bin/python、uv python find、python3）" >&2
  exit 1
}

ARCH=$($PY -c "
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
shopt -s nullglob
for d in "$BASE/${PREFIX}"*; do
  [ -d "$d" ] || continue
  n="${d##"$BASE/$PREFIX"}"
  case "$n" in *[!0-9]*) continue;; esac
  [ "$n" -gt "$MAX" ] && MAX=$n
done
shopt -u nullglob

DIR="$BASE/${PREFIX}$((MAX + 1))"
mkdir -p "$DIR"

# 输出根目录的绝对路径——它注入三段链路，缺一不可：
#   ① stdout：主控捕获后写进 Task prompt（子代理靠 prompt 传参，不继承环境）
#   ② .day0/.current：本目录下的一个文件，供 SessionStart hook（day0-env.sh）定位
#      → 注入 ASCENDBOT_FILE_PATH 等环境变量给后续每条 Bash 命令
#   ③ 主控回填 tracker「环境信息」块的「输出根目录」行（持久真值）
# 必须是绝对路径：相对路径会被 cwd 不同的子代理解析到不同位置，导致产物散落。
ABS=$(cd "$DIR" && pwd)
printf '%s\n' "$ABS" > "$BASE/.current"
echo "$ABS"
