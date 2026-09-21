#!/usr/bin/env bash
# Day0 环境变量注入（SessionStart hook）
#
# 目的：把 Day0 流程的三个路径注入每条 Bash 命令，供主控与子代理直接使用——
#   ASCENDBOT_FILE_PATH  Day0 输出根目录（tracker.md / 各 Phase 产物所在）
#   VLLM_ASCEND         vllm-ascend 仓根
#   VLLM                上游 vLLM 源码路径
#
# 背景：Bash 工具每条命令独立进程，脚本里的 export 出不了 subshell，也不跨调用
# 持久。SKILL.md 原先靠「主控把字面绝对路径写进每条命令与 Task prompt」传递，
# 真值只活在对话上下文里——上下文压缩或换会话即丢失。本 hook 把它落到文件上。
#
# 用法：由 .claude/settings.json 的 SessionStart 注册，无需手工调用。
#   - 未立项（无 .day0/.current）→ 静默退出，不阻断会话
#   - .current 指向不存在的 tracker → 按文件名兜底（无歧义 / 有 [1] 优先）
#   - 变量未回填（仍是模板占位 <path>）→ 该变量跳过，不注入垃圾值
set -uo pipefail

[ -n "${CLAUDE_ENV_FILE:-}" ] || exit 0

root="${CLAUDE_PROJECT_DIR:-.}"
state="$root/.day0/.current"

# 第 1 跳：.day0/.current 存的是 `init_day0_dir.sh` 打印的绝对路径。
# 第 2 跳：.current 不在（未立项，或目录被移动导致绝对路径失效）时按文件名兜底。
#         `.day0/<name>_<name>_<num>/` 这种重复前缀来自变量为空的 shell 展开，
#         属历史遗留，按目录名排序取 [1] 可复现，也视为唯一命中。
# 多命中且无规律 → 不猜，静默退出。
locate_day0_dir() {
  local d
  if [ -f "$state" ]; then
    d=$(head -n 1 "$state")
    [ -n "$d" ] && [ -f "$d/tracker.md" ] && {
      printf '%s' "$d"
      return 0
    }
  fi
  local -a cands=()
  shopt -s nullglob
  for d in "$root"/.day0/*/tracker.md; do
    cands+=("${d%/tracker.md}")
  done
  shopt -u nullglob
  [ ${#cands[@]} -eq 0 ] && return 1
  [ ${#cands[@]} -eq 1 ] && {
    printf '%s' "${cands[0]}"
    return 0
  }
  IFS=$'\n' read -r -d '' -a sorted < <(
    printf '%s\n' "${cands[@]}" | LC_ALL=C sort
  ) || true
  case "${sorted[0]}" in
    */*_*/*) printf '%s' "${sorted[0]}" ;; # 重复前缀的确定性选择
    *) return 1 ;;
  esac
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

# $VLLM_ASCEND / $VLLM 由 golden_flow Phase 0 §1 采集后回填 tracker 环境信息块；
# 未回填时为空（当次会话仍需用字面路径，见 SKILL.md 步骤 1）。
emit VLLM_ASCEND "$(read_field VLLM_ASCEND || true)"
emit VLLM "$(read_field VLLM || true)"

exit 0
