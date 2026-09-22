#!/usr/bin/env bash
# Day0 立项脚本：环境清理 + 安装 vllm/vllm-ascend + 探针实测 + 创建输出根目录。
# 用法：init_day0_dir.sh <模型输入路径（含 config.json 的目录）> <venv 根路径> <vllm 源码路径> [served-name]
# 硬门禁语义：安装或探针失败 → 非零退出，不建目录、不动 .day0/.current——环境不对就立不了项。
# 命名规则：<arch>_<yyyymmdd>_<num>
#   arch —— config.json 的 architectures 首项（非法字符替换为 _）
#   num  —— 从 1 开始；同日同模型已有目录时取最大 num + 1
set -euo pipefail

[ -d vllm_ascend ] || { echo "须从 vllm-ascend 仓根运行" >&2; exit 1; }
VLLM_ASCEND_ROOT=$(pwd)

MODEL_PATH="${1:?用法: init_day0_dir.sh <模型输入路径> <venv 根路径> <vllm 源码路径> [served-name]}"
VENV="${2:?缺少 venv 根路径（装好 torch_npu 的推理环境）}"
VLLM_SRC="${3:?缺少 vllm 源码路径（立项参数 \$VLLM）}"
SERVED_NAME="${4:-$(basename "$MODEL_PATH")}"
PY="$VENV/bin/python"
PIP="$VENV/bin/pip"
[ -x "$PY" ] || { echo "错误：$PY 不存在或不可执行" >&2; exit 1; }
[ -d "$VLLM_SRC/vllm" ] || { echo "错误：$VLLM_SRC 不是 vllm 源码树" >&2; exit 1; }

CONFIG="$MODEL_PATH/config.json"
[ -f "$CONFIG" ] || { echo "错误：$CONFIG 不存在" >&2; exit 1; }

# ① 清理残留：残留 serve 进程持有旧代码且占用 8000 端口，不杀会使后续 readiness 假阳性
pkill -f "vllm serve.*${SERVED_NAME}" 2>/dev/null || true

# ② 安装正确版本（editable + --no-deps：不动依赖解析；一律用 venv 的 pip，禁止裸 pip）
"$PIP" uninstall -y 'vllm*' || true
( cd "$VLLM_SRC" && "$PIP" install setuptools-rust \
  && VLLM_TARGET_DEVICE=empty "$PIP" install -v -e . --no-build-isolation --no-deps )
( cd "$VLLM_ASCEND_ROOT" && "$PIP" install --no-build-isolation -v -e . --no-deps )

# ③ 探针：实测解释器实际加载哪两棵树（原文落盘，禁止凭记录转述）
PROBE_OUT=$("$PY" -c "import vllm, vllm_ascend; print(vllm.__file__); print(vllm_ascend.__file__)")
VLLM_COMMIT=$(git -C "$VLLM_SRC" rev-parse HEAD)

# ④ 命名与建目录
ARCH=$("$PY" -c "
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
mkdir -p "$DIR/preflight"

# ⑤ 落安装记录（tracker 环境信息块「环境安装记录」四字段的数据源，实例化 tracker 时逐字抄入）
printf '%s\n' "$PROBE_OUT" > "$DIR/preflight/install_probe.txt"
{
  printf -- '- 安装时间：%s\n' "$(date '+%Y-%m-%d %H:%M:%S')"
  printf -- '- $VLLM：%s（commit %s）\n' "$VLLM_SRC" "$VLLM_COMMIT"
  printf -- '- 安装校验输出（原文）：\n'
  printf '%s\n' "$PROBE_OUT" | sed 's/^/  /'
  printf -- '- 状态：已安装\n'
} > "$DIR/install_record.md"

# 输出根目录的绝对路径——它注入三段链路，缺一不可：
#   ① stdout：主控捕获后写进 Task prompt（子代理靠 prompt 传参，不继承环境）
#   ② .day0/.current：供 SessionStart hook（day0-env.sh）定位 → 注入环境变量
#   ③ 主控回填 tracker「环境信息」块（持久真值）
# 必须是绝对路径：相对路径会被 cwd 不同的子代理解析到不同位置，导致产物散落。
ABS=$(cd "$DIR" && pwd)

# ⑥ 旧 run 封存：「重新跑一律新建目录」的另一半——切换指针前给旧 run 的 tracker
#    追加封存记录，旧目录自此退为只读历史证据源（中断恢复不调本脚本，不受影响）
OLD=$(head -n 1 "$BASE/.current" 2>/dev/null || true)
if [ -n "$OLD" ] && [ "$OLD" != "$ABS" ] && [ -f "$OLD/tracker.md" ]; then
  printf '| %s | 封存 | 任务重跑，由 %s 承接；本目录退为只读历史证据源 | — |\n' \
    "$(date '+%Y-%m-%d %H:%M')" "$ABS" >> "$OLD/tracker.md"
  echo "已封存旧 run：$OLD" >&2
fi

# 原子写：先写临时文件再 mv，避免并发/中断留下半截指针
printf '%s\n' "$ABS" > "$BASE/.current.tmp" && mv "$BASE/.current.tmp" "$BASE/.current"
echo "$ABS"
