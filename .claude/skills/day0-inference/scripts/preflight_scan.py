#!/usr/bin/env python3
"""Day0 Phase 0 原始证据采集脚本——只产信号，不做判定。

用法：.venv/bin/python preflight_scan.py <模型输入路径（含 config.json）> <输出 md 路径>
采集结果写入输出文件，供编排者（LLM）阅读后做五维同构度判定与路径裁决。
所有采集项失败时记录「不可得」，不中断。
"""
import glob
import json
import os
import re
import subprocess
import sys
from datetime import datetime

TORCH_NPU_PROBES = [
    "npu_fused_infer_attention_score", "npu_mla_prolog_v3",
    "npu_sparse_flash_attention", "npu_swiglu", "npu_rotary_mul",
    "npu_grouped_matmul", "npu_moe_distribute_dispatch",
    "npu_moe_distribute_combine", "npu_recurrent_gated_delta_rule",
    "npu_causal_conv1d", "npu_chunk_gated_delta_rule",
    "float4_e2m1fn_x2", "float8_e8m0fnu", "npu_dynamic_mx_quant",
    "npu_quant_matmul",
]
MAGIC_KEYS = [
    "hidden_size", "num_attention_heads", "num_key_value_heads",
    "n_routed_experts", "num_experts", "num_experts_per_tok",
    "kv_lora_rank", "qk_rope_head_dim", "qk_nope_head_dim",
    "v_head_dim", "moe_intermediate_size", "first_k_dense_replace",
    "index_topk",
]


def run(cmd, timeout=30):
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return (r.stdout + r.stderr).strip() or "(无输出)"
    except Exception as e:
        return f"不可得（{type(e).__name__}: {e}）"


def load_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def _repo_root() -> str:
    """vllm-ascend 仓根：从本脚本位置向上查找仓根标记。

    不能依赖 os.getcwd()——调用方可能从任意目录启动，导致仓内相对路径
    （.github/vllm-main-verified.commit、vllm_ascend/utils.py）全部读不到，
    且失败是静默的（§3/§9 退化为空结果，看起来像「真的什么都没有」）。
    也不按固定层数上溯——脚本被移动或仓路径变化都会失准；改为向上找标记。
    """
    d = os.path.dirname(os.path.abspath(__file__))
    while True:
        if os.path.isdir(os.path.join(d, "vllm_ascend")) and os.path.exists(
            os.path.join(d, ".git")
        ):
            return d
        parent = os.path.dirname(d)
        if parent == d:  # 到文件系统根仍未找到
            raise RuntimeError(
                "无法定位 vllm-ascend 仓根（向上未找到同时含 .git 与 vllm_ascend/ 的目录）。"
                "请确认本脚本位于 vllm-ascend 仓内。"
            )
        d = parent


def main():
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    model_path = os.path.abspath(sys.argv[1])
    out_path = os.path.abspath(sys.argv[2])
    repo = _repo_root()
    os.chdir(repo)  # 后续所有仓内相对路径以仓根为准
    cfg = load_json(os.path.join(model_path, "config.json")) or {}
    # 多模态 config 的关键字段嵌在 text_config 里（如 GLM-5.3-Flash），展平后
    # 再取信号，否则 §2 只会输出 3 条基础信号、五维判定拿不到注意力/cache 字段。
    if isinstance(cfg.get("text_config"), dict):
        cfg = {**cfg, **cfg["text_config"]}
    L = []  # 输出行
    w = L.append

    # 环境自检：探不到 vllm 源码时 §3/§6/§8 会全部退化为空。这必须显式阻断——
    # 「空结果」与「真的什么都没有」在下游判定里含义完全相反（前者是环境问题，
    # 后者是「该架构上游未合入」），静默产出会让 G0 路径门禁判错。
    vllm_dir_probe = run([sys.executable, "-c", "import vllm, os; print(os.path.dirname(vllm.__file__))"])
    vllm_missing = not vllm_dir_probe.startswith("/")
    if vllm_missing:
        print(
            "ERROR: 无法定位 vllm 源码（import vllm 失败），§3/§6/§8 将不可用。\n"
            "  §3 上游支持状态是 G0 路径门禁的唯一输入，缺失会导致路径判定错误。\n"
            "  请先确保 vllm 已安装（或 PYTHONPATH 指向 vllm 源码树），再重跑本脚本。\n"
            f"  探测结果：{vllm_dir_probe}",
            file=sys.stderr,
        )
        return 2

    w(f"# Phase 0 原始证据 — {model_path}")
    w(f"\n> 生成时间 {datetime.now().isoformat(timespec='seconds')}；本文件只产信号不做判定，判定见 golden_flow.md Phase 0。\n")

    # §1 环境定位
    w("## §1 环境定位\n")
    vllm_dir = vllm_dir_probe  # 复用启动自检的探测结果
    vllm_ver = run([sys.executable, "-c", "import vllm; print(vllm.__version__)"])
    w(f"- vllm 源码路径：`{vllm_dir}`（版本 {vllm_ver}）")
    w(f"- vllm-ascend 仓根（本脚本推定）：`{repo}`")
    w(f"- 上游对齐基线：`{run(['cat', '.github/vllm-main-verified.commit'])}`")
    w(f"- npu-smi info（前 5 行）：```\n{chr(10).join(run(['npu-smi', 'info']).splitlines()[:5]) or '不可得'}\n```")
    w(f"- torch_npu 版本：`{run([sys.executable, '-c', 'import torch_npu; print(torch_npu.__version__)'])}`\n")

    # §2 config.json
    w("## §2 config.json 关键信号\n")
    if cfg:
        w(f"- architectures：`{cfg.get('architectures')}`；model_type：`{cfg.get('model_type')}`；torch_dtype：`{cfg.get('torch_dtype')}`")
        for k in ["max_position_embeddings", "num_hidden_layers", "hidden_size",
                  "num_attention_heads", "num_key_value_heads", "head_dim",
                  "qk_rope_head_dim", "qk_nope_head_dim", "kv_lora_rank", "v_head_dim",
                  "rope_theta", "rope_scaling", "n_routed_experts", "num_experts",
                  "num_experts_per_tok", "n_group", "topk_group", "scoring_func",
                  "n_shared_experts", "first_k_dense_replace", "moe_intermediate_size",
                  "index_topk", "num_nextn_predict_layers", "quantization_config"]:
            if k in cfg:
                w(f"- `{k}` = `{json.dumps(cfg[k], ensure_ascii=False)[:200]}`")
        w("\n<details><summary>config.json 全量</summary>\n\n```json")
        w(json.dumps(cfg, ensure_ascii=False, indent=1)[:8000])
        w("```\n</details>\n")
    else:
        w("- config.json 不可得\n")

    arch = (cfg.get("architectures") or [""])[0]
    family = re.sub(r"(ForCausalLM|ForConditionalGeneration|Model)$", "", arch).lower()

    # §3 上游支持状态
    w("## §3 上游支持状态（注册表 grep 原始结果）\n")
    if vllm_dir.startswith("/"):
        w("```")
        w(run(["grep", "-n", f'"{arch}"', f"{vllm_dir}/model_executor/models/registry.py"]))
        w(run(["bash", "-c", f"ls {vllm_dir}/model_executor/models/ | grep -i '{family}'"]))
        w("```\n")
    else:
        w("- vllm 源码不可得，跳过\n")

    # §4 量化格式
    w("## §4 量化格式\n")
    w(f"- `quant_model_description.json`：{'存在' if os.path.exists(os.path.join(model_path, 'quant_model_description.json')) else '不存在'}")
    w(f"- `quantization_config.quant_method`：`{(cfg.get('quantization_config') or {}).get('quant_method')}`\n")

    # §5 权重名
    w("## §5 权重名前缀（safetensors index）\n")
    idx_path = os.path.join(model_path, "model.safetensors.index.json")
    idx = load_json(idx_path)
    if idx and "weight_map" in idx:
        prefixes = sorted({k.rsplit(".", 1)[0] for k in idx["weight_map"]})
        w(f"- 权重条目 {len(idx['weight_map'])} 个，去重前缀 {len(prefixes)} 个（样本 60 条）：\n```")
        w("\n".join(prefixes[:60]))
        w("```\n")
    elif os.path.exists(idx_path):
        w("- `model.safetensors.index.json` 存在但无法解析为 JSON——若为 Git LFS 指针"
          "（文件头 `version https://git-lfs`），说明权重未实际拉取，权重名前缀证据不可得，"
          "标「待环境实测」（需拉取真实权重后重跑或改读 safetensors 文件头）\n")
    else:
        n = len(glob.glob(os.path.join(model_path, "*.safetensors")))
        w(f"- 无 index 文件（共 {n} 个 safetensors 分片），需直接读文件头\n")

    # §6 modeling 结构
    w("## §6 modeling 文件结构\n")
    py_files = glob.glob(os.path.join(model_path, "*.py"))
    if py_files:
        for f in py_files:
            w(f"### `{os.path.basename(f)}`\n```")
            w(run(["grep", "-nE", r"^class |self\.[a-z_]+\s*=\s*nn\.", f], timeout=10)[:4000])
            w("```\n")
    else:
        w("- 无自带 .py（未用 trust_remote_code）\n")

    # §7 chat template 形态
    w("## §7 chat template 形态\n")
    jinja = glob.glob(os.path.join(model_path, "*.jinja"))
    tc = load_json(os.path.join(model_path, "tokenizer_config.json")) or {}
    w(f"- .jinja 文件：`{[os.path.basename(j) for j in jinja] or '无'}`")
    w(f"- tokenizer_config.json 含 chat_template 字段：{'是' if 'chat_template' in tc else '否'}")
    w("- 两者皆无 → 疑似程序化 prompt 编码（要求内置 tokenizer-mode）\n")

    # §8 parser 注册表可用项
    w("## §8 上游 parser 注册表可用项\n")
    if vllm_dir.startswith("/"):
        names = set()
        for d in [f"{vllm_dir}/entrypoints/openai/tool_parsers", f"{vllm_dir}/reasoning"]:
            for f in glob.glob(f"{d}/*.py"):
                try:
                    names |= set(re.findall(r"register(?:_lazy)?_module\(\s*[\"']([^\"']+)", open(f).read()))
                except Exception:
                    pass
        w("- " + ("、".join(f"`{n}`" for n in sorted(names)) if names else "未提取到") + "\n")
    else:
        w("- vllm 源码不可得，跳过\n")

    # §9 OOT 注册表粗比对
    w("## §9 OOT 注册表粗比对（类名精确匹配，仅供参考——同名不同义需 LLM 甄别）\n")
    utils_py = os.path.join(repo, "vllm_ascend/utils.py")
    reg = set()
    try:
        src = open(utils_py).read()
        for m in re.finditer(r"REGISTERED_ASCEND_OPS\s*(?:=\s*\{|\.update\(\s*\{)(.*?)\}", src, re.S):
            reg |= set(re.findall(r"[\"'](\w+)[\"']", m.group(1)))
        reg |= set(re.findall(r"REGISTERED_ASCEND_OPS\[[\"'](\w+)[\"']\]", src))
    except Exception:
        pass
    model_classes = set()
    for f in py_files:
        try:
            model_classes |= set(re.findall(r"^class (\w+)", open(f).read(), re.M))
        except Exception:
            pass
    w(f"- `REGISTERED_ASCEND_OPS` 共 {len(reg)} 项")
    if model_classes:
        hit = sorted(model_classes & reg)
        w(f"- 模型类名 {len(model_classes)} 个，精确命中 {len(hit)} 个：{hit or '无'}\n")
    else:
        w("- 无模型类名可比（无自带 modeling 文件）\n")

    # §10 torch_npu 符号探测
    w("## §10 torch_npu 符号探测\n")
    probe = "import torch_npu\n" + "\n".join(f"print('{s}', hasattr(torch_npu, '{s}'))" for s in TORCH_NPU_PROBES)
    w("```")
    w(run([sys.executable, "-c", probe]))
    w("```\n")

    # §11 魔法数字粗扫
    w("## §11 魔法数字粗扫（命中≠踩坑，需逐个甄别）\n")
    for k in MAGIC_KEYS:
        v = cfg.get(k)
        if isinstance(v, int) and v > 2:
            hits = run(["bash", "-c",
                        f"grep -rln --include='*.py' -w '{v}' vllm_ascend/ 2>/dev/null | head -10"])
            if hits and "不可得" not in hits:
                w(f"- `{k}={v}` 命中：\n```\n{hits}\n```")
    w("")

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w") as f:
        f.write("\n".join(L))
    print(out_path)


if __name__ == "__main__":
    sys.exit(main())
