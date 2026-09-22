#!/usr/bin/env bash
# Day0 环境变量注入（SessionStart hook）
#
# 目的：把 Day0 流程的四个路径注入每条 Bash 命令，供主控与子代理直接使用——
#   ASCENDBOT_FILE_PATH  Day0 输出根目录（tracker.md / 各 Phase 产物所在）
#   VENV                推理环境 venv 根路径（含 bin/，解释器 = $VENV/bin/python）
#   VLLM_ASCEND         vllm-ascend 仓根
#   VLLM                上游 vLLM 源码路径
#
# 背景：Bash 工具每条命令独立进程，脚本里的 export 出不了 subshell，也不跨调用
# 持久。SKILL.md 原先靠「主控把字面绝对路径写进每条命令与 Task prompt」传递，
# 真值只活在对话上下文里——上下文压缩或换会话即丢失。本 hook 把它落到文件上。
#
# 用法：由 .claude/settings.json 的 SessionStart 注册，无需手工调用。
#   - 未立项（.day0 下无任何 run）→ 静默退出，不阻断会话
#   - 定位优先级：DAY0_DIR 显式钉住 > .current 指针 > tracker.md 最新修改的 run 兜底
#   - 变量未回填（仍是模板占位 <path>）→ 该变量跳过，不注入垃圾值
set -uo pipefail

[ -n "${CLAUDE_ENV_FILE:-}" ] || exit 0

root="${CLAUDE_PROJECT_DIR:-.}"
state="$root/.day0/.current"

# 定位优先级：
#   ① DAY0_DIR 环境变量显式钉住（恢复旧 run 时用，会话启动前 export）
#   ② .day0/.current（init_day0_dir.sh / day0_use.sh 维护的指针）
#   ③ 兜底：tracker.md 最新修改的 run 目录——配合「重新跑一律新目录」，
#      最新 run 即活跃 run；指针失效时不再静默退出导致环境变量全空。
locate_day0_dir() {
  local d
  if [ -n "${DAY0_DIR:-}" ] && [ -f "${DAY0_DIR%/}/tracker.md" ]; then
    printf '%s' "${DAY0_DIR%/}"
    return 0
  fi
  if [ -f "$state" ]; then
    d=$(head -n 1 "$state")
    [ -n "$d" ] && [ -f "$d/tracker.md" ] && {
      printf '%s' "$d"
      return 0
    }
  fi
  local latest="" newest=0 mt
  shopt -s nullglob
  for d in "$root"/.day0/*/tracker.md; do
    mt=$(stat -f %m "$d" 2>/dev/null || stat -c %Y "$d" 2>/dev/null || echo 0)
    if [ "$mt" -gt "$newest" ]; then newest=$mt; latest=${d%/tracker.md}; fi
  done
  shopt -u nullglob
  [ -n "$latest" ] || return 1
  printf '%s' "$latest"
}

DIR=$(locate_day0_dir) || exit 0
[ -f "$DIR/tracker.md" ] || exit 0

# 解析 tracker「环境信息」块的一行。取值形态三态：
#   - `- \`$VLLM\`（…）：\`/abs/path\`   → 填值，取反引号内容
#   - `- \`$VLLM\`（…）：/abs/path      → 填值，取冒号后全文
#   - `- work-dir（…）：<path>`         → 模板占位，跳过
read_field() {
  local key="$1" line val
  line=$(grep -m1 -E "^[[:space:]]*-[[:space:]]+[$]?${key}([^A-Za-z0-9_]|$)" \
    "$DIR/tracker.md" 2>/dev/null) || return 1
  [ -n "$line" ] || return 1
  val=${line##*：}
  val=${val##*:}
  if [[ "$val" == *'`'*'`'* ]]; then
    val=${val#*\`}
    val=${val%%\`*}
  fi
  val=$(printf '%s' "$val" | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//' \
    -e 's/^[`"'"'"']//' -e 's/[`"'"'"']$//')
  [ -n "$val" ] || return 1
  case "$val" in
    "" | "<"*">" | *"<path>"*) return 1 ;; # 模板占位或未回填
  esac
  printf '%s' "$val"
}

emit() {
  local name="$1" val="$2"
  [ -n "$val" ] || return 0
  # 路径含空格时 export 必须带引号，否则被拆成多个参数
  printf "export %s='%s'\n" "$name" "$val" >>"$CLAUDE_ENV_FILE"
}

# 输出根目录：以 tracker 所在目录为准（.current 的文件名规则可能失效，
# 目录位置不会）。已存在于 shell 的 CLAUDE_ENV_FILE 内容按 >> 追加，不覆盖。
emit ASCENDBOT_FILE_PATH "$DIR"

# VENV / $VLLM / $VLLM_ASCEND 均为立项参数，tracker 实例化时填入环境信息块
# （Phase 0 §1 对 $VLLM / $VLLM_ASCEND 做安装一致性校验）；未填时为空
# （当次会话仍需用字面路径，见 SKILL.md 步骤 1）。
emit VENV "$(read_field venv || true)"
emit VLLM_ASCEND "$(read_field VLLM_ASCEND || true)"
emit VLLM "$(read_field VLLM || true)"

exit 0
