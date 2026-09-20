# 新模型 NPU 适配实现模板（类型 0-5 代码骨架）

> 配套 `.claude/skills/day0-inference/reference/ascend-oot.md`（六类适配类型）与 `.claude/skills/day0-inference/reference/adapt/golden-adapter/golden-worker-adapter.md`（逐 module 判定）使用。每个模板给出最小可用骨架 + 仓内真实参考实现路径。骨架中的 `<占位>` 必须按设计文档的逐 module 判定表替换；参考实现是判型后的第一阅读材料，**不要凭骨架空想实现细节**。
>
> 通用纪律（所有模板适用）：
> - 先 `forward_native` / torch 等价实现跑通精度，再换融合算子（先正确后性能）；
> - 所有兜底 `else` 分支显式 `raise NotImplementedError`；
> - 激活/路由/norm 的中间计算用 fp32；门控参数（`dt_bias`/`A_log` 等）float32 参数 + float32 运算；
> - 新自定义算子必须注册 meta 实现，否则无法被 ACLGraph 捕获。

## 模板 A：类型 1 — 新增 CustomOp OOT

适用：上游是 `CustomOp` 子类（有 `forward_cuda`/`forward_native`），Ascend 未注册。

参考实现：`vllm_ascend/ops/activation.py` `AscendSiluAndMulWithClamp`（最简，仅覆写 `forward_oot`）。

```python
# vllm_ascend/ops/<module>.py
import torch
import torch_npu
from vllm.model_executor.layers.<upstream_module> import <UpstreamLayer>


class Ascend<UpstreamLayer>(<UpstreamLayer>):
    """默认只覆写 forward_oot，__init__ 交给基类。"""

    def forward_oot(self, x: torch.Tensor) -> torch.Tensor:
        # 第一版：torch 等价实现（对齐 forward_native 语义）跑通精度
        # 第二版：换 torch_npu / aclnn 融合算子
        return torch_npu.<npu_fused_op>(x)
```

需覆写 `__init__` 的两种例外（(a) 要从 `vllm_config` 读开关；(b) 有上游基类不存在的构造参数）：

```python
class Ascend<UpstreamLayer>(<UpstreamLayer>):
    def __init__(self, *args, <new_param>: float = 1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.<new_param> = <new_param>  # forward_oot 从 self 读取

    def forward_oot(self, x: torch.Tensor) -> torch.Tensor:
        ...
```

注册（否则类写了也不会生效）：把映射加入 `vllm_ascend/utils.py` 的 `REGISTERED_ASCEND_OPS`（`"<UpstreamLayer>": Ascend<UpstreamLayer>`），启动日志须出现 `Instantiating custom op: <UpstreamLayer> using <AscendImpl>`。

## 模板 B：类型 2 — 新增 PluggableLayer OOT

适用：上游是 `PluggableLayer` 子类（`__init__` 组合子模块），Ascend 未注册。只覆写 3 类方法，其余全继承。

参考实现：`vllm_ascend/ops/bailing_moe_linear_attn.py` `AscendBailingMoELinearAttention`（文件头注释即范例说明）。

```python
from vllm.model_executor.models.<upstream_model> import <UpstreamAttention>
from vllm.v1.attention.backend import AttentionMetadata


class Ascend<UpstreamAttention>(<UpstreamAttention>):
    """OOT PluggableLayer：只覆写平台相关的 3 类方法，
    __init__ / forward / 权重加载 / state shape 全部继承上游。"""

    # (1) 计算路径
    def _forward(self, hidden_states, output, positions):
        forward_context = get_forward_context()
        attn_metadata: AttentionMetadata = forward_context.attn_metadata
        ...  # 调 Ascend 侧 kernel（ops/triton/ 或 torch_npu）

    # (2) 若涉及 KV/state cache：state 形状与 dtype
    # def get_state_shape(self) -> ...: ...
    # def get_state_dtype(self) -> ...: ...

    # (3) 若需要专属 backend
    # def get_attn_backend(self) -> type["AttentionBackend"]: ...
```

注册后启动日志须出现 `Instantiating pluggable layer: <name> using <impl>`（与 CustomOp 文案不同，自检时两种都要匹配）。

## 模板 C：类型 3 — monkey patch（工厂函数两处 binding）

适用：上游无注册装饰器，或目标是工厂函数。先过 patch 决策树（`.claude/skills/day0-inference/reference/ascend-oot.md` §5），通过后才用本模板，并同步在 `vllm_ascend/patch/__init__.py` 完成四段式登记。

参考实现：`vllm_ascend/patch/platform/patch_fused_moe.py`（MoE 工厂重定向，含版本巷道口防御）。

要点：**必须改两处 binding**（包 `__init__` 和 layer 模块），否则部分模型拿到未 patch 版本；worker patch 必须在任何模型模块 import 之前执行。

```python
# vllm_ascend/patch/<platform|worker>/patch_<target>.py
import vllm.<pkg> as _pkg
import vllm.<pkg>.<layer_module> as _layer_mod

# 1. 先捕获真实原始符号（在本模块代码运行前，上游模块级代码可能已做替换）
_original_factory = _layer_mod.<FactoryName>


def _ascend_factory(*args, **kwargs):
    """Ascend 替换实现：只覆盖不兼容路径，其余委托原实现。"""
    if <npu_compatible_condition>:
        return _original_factory(*args, **kwargs)
    return Ascend<Runner>(*args, **kwargs)


# 2. 两处 binding 都改：包 __init__ + layer 模块
_layer_mod.<FactoryName> = _ascend_factory
_pkg.<FactoryName> = _ascend_factory

# 3. 防御性校验：上游签名一旦变化即显式失败，而非静默错位
from inspect import signature
_EXPECTED = ["<arg1>", "<arg2>"]  # 上游函数的关键参数名快照
_actual = list(signature(_original_factory).parameters)
assert all(name in _actual for name in _EXPECTED), \
    f"upstream <FactoryName> signature changed: {_actual}"
```

版本巷道防御（上游多版本并存时）：按 `vllm_version_is("x.y.z")` 或符号存在性选择 patch 目标，参考 `patch_fused_moe.py` 对 `FusedMoE`/`FusedMoEFactory` 双名的处理。

## 模板 D：类型 5 — 新 attention backend

适用：新注意力形态，`(use_mla, use_sparse, use_compress)` 三元组未命中 `get_attn_backend_cls` 任何查表键。

参考实现：`vllm_ascend/attention/sfa_v1.py`（继承 `MLACommonMetadataBuilder` 复用 MLA 骨架）、`vllm_ascend/attention/dsa_v1.py`（完全独立体系）。

```python
# vllm_ascend/attention/<name>_v1.py
class Ascend<X>MetadataBuilder(MLACommonMetadataBuilder):  # 或 AttentionMetadataBuilder
    """负责把 per-layer metadata 下沉到 device；
    必须正确输出五态状态机（PrefillNoCache / PrefillCacheHit /
    DecodeOnly / ChunkedPrefill / SpecDecoding）以驱动图模式选择。"""
    ...


class Ascend<X>Backend(AttentionBackend):
    @staticmethod
    def get_name() -> str: return "<name>_v1"
    @staticmethod
    def get_impl_cls() -> type: return Ascend<X>Impl
    @staticmethod
    def get_builder_cls() -> type: return Ascend<X>MetadataBuilder
    # 图支持级别如实声明：稀疏/潜变量注意力只承诺 UNIFORM_BATCH


class Ascend<X>Impl(AttentionImpl):
    def forward(self, layer, query, key, value, kv_cache, attn_metadata, ...):
        ...  # 调模板 E 的 kernel 或 torch_npu 融合算子
```

同时在 `NPUPlatform.get_attn_backend_cls` 的查表中扩展分派键（或新增 selector 探测函数），并同步注册模板 F 的 cache spec。

## 模板 E：类型 5 — 新 Triton kernel

适用：`torch_npu` 无融合算子、组合现有算子也不可行。长期高性能路径是 AscendC（`csrc/<op>/{op_host, op_kernel}` + `torch.ops._C_ascend.*` 注册），Triton 是快速跑通的过渡路径。

参考实现：`vllm_ascend/ops/triton/`（如 `activation/swiglustep.py`——融合 kernel + `HAS_TRITON` 探测 + native 回退的标准形态）。

```python
# vllm_ascend/ops/triton/<family>/<op>.py
import triton
import triton.language as tl


@triton.jit
def _<op>_kernel(x_ptr, y_ptr, N: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(x_ptr + offs, mask=mask)
    tl.store(y_ptr + offs, <compute>(x), mask=mask)


def <op>_forward(x: torch.Tensor) -> torch.Tensor:
    y = torch.empty_like(x)
    N = x.numel()
    grid = (triton.cdiv(N, 1024),)
    _<op>_kernel[grid](x, y, N, BLOCK=1024)
    return y
```

调用侧标准形态（能力探测 + 回退，见 `AscendSwigluStepAndMul`）：

```python
from vllm.triton_utils import HAS_TRITON

if HAS_TRITON:
    from vllm_ascend.ops.triton.<family>.<op> import <op>_forward
    return <op>_forward(x)
return <native_torch_equivalent>(x)  # 无 triton 时的正确性保底
```

## 模板 F：类型 5 — 新 KV cache spec

适用：新 cache 形态（indexer 独立缓存 / recurrent state / 压缩 latent 等），既有 spec 子类无法描述。**spec 必须先于一切性能工作定稿**（`.claude/skills/day0-inference/reference/adapt/golden-adapter/golden-schedule-adapter.md`）。

参考实现：`vllm_ascend/core/kv_cache_interface.py`（`AscendSFAIndexerCacheSpec` 的 `page_size_bytes`/`merge()` 写法与 `register_ascend_kv_cache_specs()` 注册点）。

```python
# vllm_ascend/core/kv_cache_interface.py
@dataclass(frozen=True, kw_only=True)
class Ascend<X>CacheSpec(<ClosestBaseSpec>):  # 选语义最近的基类
    <extra_field>: int = 1

    @property
    def page_size_bytes(self) -> int:
        ...  # 单页字节数：跨 cache group 必须一致，警惕被 mamba padding 撑大

    @classmethod
    def merge(cls, specs: list[Self]) -> Self:
        ...  # 同组各层 spec 合并；字段不一致时 assert 显式失败
```

注册（engine-core 规划 KV cache 时调用，**放 worker 组补丁无效**，见 `register_ascend_kv_cache_specs()`）：

```python
KVCacheSpecRegistry.register(
    kvcache_spec_cls=Ascend<X>CacheSpec,
    manager_class=<FullAttentionManager | SlidingWindowManager | 自定义>,
    uniform_type_base_spec=<归并用的基 spec>,
)
```

> **Q1' standalone 变体**：整层 import 即崩、需 standalone 重写时，参考 `vllm_ascend/models/deepseek_v4.py`（复用平台中立基类 + ascend 自有 kernel），优先继承而非从零设计。
