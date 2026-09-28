from vllm import ModelRegistry


def register_model():
    # Kimi-K3 (VL checkpoint, text path only in Stage 1). The top-level arch is
    # a VL shell; the language side is `KimiLinearForCausalLM`, already known to
    # upstream. Registered lazily (string target) so importing this module does
    # not pull in the model implementation.
    ModelRegistry.register_model(
        "KimiK3ForConditionalGeneration",
        "vllm_ascend.models.kimi_k3.model:KimiK3ForConditionalGeneration",
    )
    ModelRegistry.register_model("DeepseekV4ForCausalLM", "vllm_ascend.models.deepseek_v4:AscendDeepseekV4ForCausalLM")
    ModelRegistry.register_model("DeepSeekV4MTPModel", "vllm_ascend.models.deepseek_v4_mtp:DeepSeekV4MTP")
    ModelRegistry.register_model(
        "DSparkDraftModel",
        "vllm_ascend.models.deepseek_v4_dspark:DSparkDeepseekV4ForCausalLM",
    )
    ModelRegistry.register_model(
        "LlamaForCausalLMVwnEagle3", "vllm_ascend.models.llama_eagle3_vwn:Eagle3VwnLlamaForCausalLM"
    )
    ModelRegistry.register_model("Qwen3DSparkModel", "vllm_ascend.models.qwen3_dspark:AscendQwen3DSparkForCausalLM")
