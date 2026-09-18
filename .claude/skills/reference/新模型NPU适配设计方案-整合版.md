# 新模型 NPU 适配设计方案（整合版）

> 适用范围：一个新模型（含厂商自带 `modeling_*.py`）需要在 Ascend NPU 上跑起来时，如何系统性地判断"哪些 module 需要适配、怎么适配、EngineCore 侧要动什么"。
>
> ⚠️ **行号防腐声明**：全文 `文件:行号` 引用基于上述基线版本。随 vllm-ascend 版本更新，行号会漂移——**函数名/类名是锚点，行号是辅助**。行号失效时按函数名 grep 即可重新定位。
>
> ⚠️ **单一事实源声明**：每个事实只在一处详述，他处用引用。速查表只写判定结论，详述见对应 E 项或类型节；反之亦然。改事实时只改一处。

---

**目录**：第一部分 总框架 · 第二部分 逐 module 判定 · 第三部分 六类适配 · 第四部分 EngineCore（含 §4.0 加载期权重映射检查）· 第五部分 工作流 · 附录 A 算子清单 / B 陷阱 / C 自检 / D 枚举完整性

---

## 第一部分：适配总框架

### 1.1 三个层次（判断顺序不能颠倒）

vllm-ascend 是 vLLM 的 out-of-tree 硬件插件。一个新模型要跑在 NPU 上，工作量分布在三个层次：

```
┌─ L1  模型定义层（vLLM 侧）──────────────────────────────────┐
│  上游 vLLM 是否已有这个模型的实现？                          │
│  ├─ 有 → L1 零工作量，直接进入 L2                            │
│  └─ 无 → 需要写 vLLM 风格的模型定义（并行/量化/KV cache 感知）│
└──────────────────────────────────────────────────────────┘
                          ↓
┌─ L2  算子层（vllm-ascend 侧）─────────────────────────────┐
│  模型用到的每个 module，NPU 上是否已有实现？                 │
│  ├─ 已在 REGISTERED_ASCEND_OPS 里 → 零工作量                │
│  ├─ 上游是 CustomOp/PluggableLayer 但 Ascend 未注册 → 新增 OOT 实现 │
│  └─ 上游硬编码 CUDA 分支 → monkey patch                     │
└──────────────────────────────────────────────────────────┘
                          ↓
┌─ L3  框架配置层（EngineCore / Platform / Worker）──────────┐
│  KV cache 形态、注意力后端、调度、并行、图模式、量化识别      │
│  这一层不改，模型能构造出来但跑不起来或跑错                  │
└──────────────────────────────────────────────────────────┘
```

**关键认知**：L1 往往已经完成。上游 vLLM 有 `vllm/models/<model>/{nvidia,amd}/` 的厂商分支约定（见 `vllm/model_executor/models/registry.py` 中 `"vllm.models.*"` 路径的条目）。这种情况下 vllm-ascend 的工作是**补一个 ascend 覆盖实现**，而不是从零实现。

**三套分类体系的关系**：各回答不同问题，不要混用——L1/L2/L3 回答"工作量在**哪里**"，P0/P1/P2 回答"走**哪条路**"，类型 0-5 回答"**怎么适配**"。使用顺序：L1/L2/L3 定位 → P0/P1/P2 决策 → 类型 0-5 查模板。

### 1.1.1 模型级前置检查：平台分派覆盖（Q0）

在进入逐 module 判定**之前**，先做这个**模型级**检查（同一模型只问一次，不是 per-module 的）：

读上游 `vllm/models/<model>/__init__.py` 的分派逻辑。若只按 `is_rocm()` 二分（`if not current_platform.is_rocm(): from .nvidia... else: from .amd...`），NPU 会静默落入 nvidia 分支，import CUDA 专属代码直接崩——**必须在 `vllm_ascend/models/__init__.py::register_model()` 注册覆盖实现（→ E1），且确保 ascend 分支不 import 上游 nvidia 模块**。

**Q0 命中后：选哪个上游分支作基线**

上游若有多个厂商分支，**不要默认选 nvidia**。逐个评估后选耦合最低的，选错会让工作量翻倍：

```bash
# ① 查各分支的模块级厂商 import（NPU 上 import 即崩的那种）
grep -nE "^(from|import).*(cute|cutlass|aiter|hip|rocm|nvshmem)" <branch>/*.py

# ② 对比各分支用的是「可注册层」还是「私有类」
grep -rn "PluggableLayer.register\|CustomOp.register" <该分支引用的层>
```

判据（优先级从高到低）：

| 观察 | 含义 |
|---|---|
| 模块级 import 厂商专属 kernel（如 `cute_dsl`） | ❌ 该分支不可用，import 即崩 |
| 用带 `@PluggableLayer.register` 的**共享层** | ✅ 可走类型 2（注册替换） |
| 用**私有类**（无注册装饰器） | ⚠️ 只能走类型 3（monkey patch），成本更高 |
| 行数更少 | ✅ 通常意味着厂商特化更少 |

> 实践中曾出现 nvidia 分支不可用而 amd 分支干净的情况，务必逐个查。

通过 Q0 后，才进入第二部分逐 module 判定。

### 1.2 三条适配路径（成本递增，优先选成本低的）

| 路径 | 机制 | 触发点 | 成本 | 适用 |
|---|---|---|---|---|
| **P0 零适配** | 复用 vLLM 模型代码 + 已注册的 Ascend CustomOp | `CustomOp.__new__` 自动拦截 | 0 | 模块结构与已支持模型一致 |
| **P1 补丁/小新增** | `patch/worker/patch_<model>.py`，或新增单个 CustomOp/PluggableLayer OOT 实现 | `NPUWorker.__init__` → `adapt_patch()`；`register_ascend_customop()` | 低 | 少数方法有 CUDA 假设 / 缺个别算子 |
| **P2 重写** | `models/<model>/` 完整实现 + `ModelRegistry.register_model` | `vllm.general_plugins` | 高 | 架构级差异 / 需自带算子 / 上游分派不含本平台 |

**判据**：
- 改动 ≤ 3 个方法且不涉及层间数据流 → **P1**
- 需要新算子内核、新 attention backend、改变层组装方式、或 vLLM 上游无该架构 → **P2**

**规模分诊**（第二部分逐 module 判定完成后，按 P2 的个数决定落地形态）：

> **前置规则**：**Q0 命中（上游分派不含 NPU）→ P2 至少为 1**（覆盖注册本身
> 是 P2 工作量），再叠加下表的类型 5 module 计数决定总规模（§1.3：类型 5 即原 P2 路径）。判 P2 ≠ 全量重写——
> 若选定的基线分支大部分可经注册表复用，`models/<model>/` 可以只是一个**薄覆盖层**。
> Q0 未命中时表中数值即为总规模。

| 类型 5 module 数量 | 落地形态 |
|---|---|
| 0 个 | 纯补丁：`patch/worker/patch_<model>.py` |
| 1-2 个 | 补丁 + 局部新增：patch + 若干 `ops/<op>.py` |
| ≥ 3 个 | 建 `models/<model>/` 完整目录，走 MiniMax-M3 路线 |

### 1.3 路径与适配类型的对应

第二部分的决策树会给出更细的**六类适配类型**（类型 0-5，其中类型 0 为零适配），与三条路径的对应关系：

| 类型 | 含义 | 路径 |
|---|---|---|
| 类型 0 | 零适配（已注册，自动替换） | P0 |
| 类型 1 | 新增 CustomOp OOT 实现（覆写 `forward_oot`） | P1 |
| 类型 2 | 新增 PluggableLayer OOT 实现（覆写 `forward`） | P1 |
| 类型 3 | monkey patch（无法走注册表） | P1 |
| 类型 4 | 扩展现有 Ascend 实现（参数/分支被静默丢弃） | P1~P2 |
| 类型 5 | 全新结构（算子 + 可能的 backend + KV cache spec） | P2 |

> **Q1' standalone 重写**：决策树 Q1' 判定"上游层不可 import"时，虽上游有对应层但无法继承——行为等价于类型 5，按类型 5 的模板处理，但可参考上游的平台中立基类设计。参照 `vllm_ascend/models/deepseek_v4.py`。

---

## 第二部分：逐 module 判定

### 2.0 快通道：标准 module 速查表

新模型到手，**先对照这张表逐行打勾**。全部命中"默认判定 P0"时可直接跳过 2.1 的决策树，只走 EngineCore 清单（第四部分）。

「机制」列决定了适配时**该覆写哪个方法**（见 2.2 判定 B）：CustomOp → `forward_oot`，PluggableLayer → `forward`。**写错不报错，只静默不生效。**

| # | Module | 机制 | vllm-ascend 现状 | 默认判定 | 需要适配的触发条件 |
|---|---|---|---|---|---|
| 1 | Embedding（`VocabParallelEmbedding`） | PluggableLayer | ✅ 已 OOT | **P0** | 词表并行切分非标准 / 多模态 embedding 融合 |
| 2 | RMSNorm / LayerNorm | CustomOp | ✅ 已 OOT（`AscendRMSNorm`、`GemmaRMSNorm`、`RMSNormGated`） | **P0** | 归一化公式变体（加 scale/offset）；**norm 位置改变**（QK-norm、sandwich norm）可能连带影响 attention backend |
| 3 | QKV Linear（`QKVParallelLinear`） | PluggableLayer | ✅ 已 OOT | **P0** | QKV 融合方式不同（如 MLA 双低秩） |
| 4 | Rotary Embedding | CustomOp | ✅ 已 OOT（标准/M-RoPE/YaRN/Deepseek-scaling） | **P0** | 新位置编码算法（3D-RoPE、可学习插值）。**NoPE**（不做旋转）**适配位置在模型层而非 rotary 算子**——上游共享层（如 `MultiHeadLatentAttentionWrapper`）不含 `use_nope` 参数，由模型侧自有 attention 类处理；但需确认 Ascend MLA 在 nope 路径下不会去取 rope cos/sin cache（`ops/mla.py` 接收 `rotary_emb`） |
| 5 | Attention 核心 | backend 分发 | ✅ 5 个后端（Dense/MLA/SFA/DSA/FA3） | **P0** 若匹配 / **P2** 若不匹配 | 新注意力算法（稀疏、gating、output gate）。注：**层封装**（如 `MultiHeadLatentAttentionWrapper`）走 PluggableLayer 替换（见附录 A），**backend** 走 `get_attn_backend_cls()` 分发，两者是不同层次 |
| 6 | **KV Cache 布局** | spec 注册 | ✅ 标准 + MLA 系（3 个 spec）<br>⚠️ 线性态复用上游 `MambaSpec` + `patch_mamba_*` | **P0** 标准/MLA<br>**P1** 与 Mamba spec 一致<br>**P2** 需新增 spec | 非常规 cache。判据见 E4。page 对齐见 4.2 E5 |
| 7 | MLP / FFN | PluggableLayer | ✅ Linear 全系已 OOT | **P0** | 新激活函数 |
| 8 | 激活函数 | CustomOp | ✅ SiluAndMul / QuickGELU / SiluAndMulClamp | **P0** 若命中 / **P1** 若新增 | 新激活（加 CustomOp）。⚠️ **MoE 路径需单独加分支**，否则落 `else` 兜底被当 SwiGLU 静默算错 |
| 9 | MoE 路由 + 专家 | 工厂函数 patch | ✅ `ops/fused_moe/` + EPLB（noaux_tc 经 sigmoid + correction_bias 组合支持，语义级非字符串级；sigmoid / grouped_topk 原生支持） | **P0** | 路由算法变体、共享专家、**latent MoE**（专家维度 ≠ hidden_size，需确认 transform 被调用） |
| 10 | LM Head / Logits | PluggableLayer | ✅ 已 OOT | **P0** | 词表裁剪、多头输出 |
| 11 | **量化** | quant config | ✅ ModelSlim / compressed-tensors / FP8 / W8A8 系<br>⚠️ MXFP4 有**两道门**（版本门 + A5 融合门），详见 E2 | **P0** 若格式已支持**且两道门均通过** | 新量化格式或新 ignore 规则 |
| 12 | 多模态 ViT | 混合 | ⚠️ 部分（`MMEncoderAttention` 已 OOT） | 先查 `patch/worker/` 有无同名塔的 patch：<br>**有 → P1**（验证签名/逻辑一致后可降为 P0），无 → **P1** | ViT 位置编码、patch merger。即使同名类，不同模型的 ViT 架构（如 2D vs 3D patch embed）可能签名不同，**不可直接判 P0** |
| 13 | 投机解码 / MTP | 模型注册 | ✅ `spec_decode/` | **P0/P1** | MTP 层结构特殊 |
| 14 | **线性注意力 / Mamba / GDN** | PluggableLayer | ✅ `AscendGatedDeltaNetAttention`、`AscendBailingMoELinearAttention`；Triton kernel 在 `ops/triton/{fla,mamba,kda}/` | **P1** 若为已有变体<br>**P2** 若为新算法 | 混合架构已是主流。注意：**kernel 存在 ≠ 已接线**（示例：KDA kernel 已有但无层调用）；上游层若无 `@PluggableLayer.register` 则只能 patch 或 standalone 重写（见决策树 Q1'） |
| 15 | **跨层状态传递** | — | ❌ 无通用支持 | **P1~P2** | decoder layer 间传递 `hidden_states` 以外的张量（如块级残差累积）。**直接阻塞 ACL Graph 捕获**，且影响 PP 切分（见 4.2 E7） |
| 16 | 归一化位置变体 | CustomOp | ✅ 算子已有 | **P0** 若仅位置变化<br>**P1** 若需融合 | QK-norm、sandwich norm。算子复用无问题，但可能改变 attention backend 的融合假设 |
| 17 | **层类型混合派发** | 模型骨架 | ✅ 有先例（Qwen3-Next full+linear 混合） | **P2** | 同模型内按 `layer_idx` 派发多种 attention/层类型（示例：24 层 MLA + 69 层 KDA）。**连带计数**：层类型混合本身 P2，且几乎必然带来混合 KV cache（#6，+1 P2）和/或新 attention backend（#5，+1 P2），实际 P2 ≥ 2。若同时有跨层状态传递，见决策树"特殊"分支 |

### 2.1 判定决策树（速查表未全命中时，对每个 module 走这棵树）

> 前提：§1.1.1 前置检查 Q0（平台分派覆盖）已完成。以下决策树是**逐 module** 的。

```
Q1. 上游 vLLM 有没有语义等价的层？
│
├─ 有 → 先过 Q1'：
│  │
│  Q1'. 上游层所在模块在 NPU 上能否 import 成功？
│  │    （检查模块顶层 import 链是否含 vllm._custom_ops / cute_dsl /
│  │      flash_attn / deep_gemm 等 CUDA 专属依赖——有则 NPU 上
│  │      import 即崩，"继承上游类覆写方法"的路径根本不可用）
│  │
│  │    检查命令（传递依赖，单层 grep 抓不到）：
│  │    # 最可靠：直接试 import
│  │    python -c "import vllm.models.<model>.<branch>.model" 2>&1 | tail -3
│  │    # 静态查一层传递
│  │    grep -h "^from\|^import" <branch>/*.py | grep "^from \." | \
│  │      sed 's/from \.\([a-z_]*\).*/\1/' | sort -u | \
│  │      xargs -I{} grep -l "cute_dsl\|deep_gemm\|flash_attn" <branch>/{}.py
│  │    ├─ 能 → 继续 Q2-Q4
│  │    └─ 不能 → standalone 重写：复用上游平台中立基类 + ascend 自有
│  │              kernel，参照 vllm_ascend/models/deepseek_v4.py 模式；
│  │              重写后各算子仍按 Q2-Q4 判定复用方式
│  │
│  ├─ 是 CustomOp 子类
│  │  └─ Q2. Ascend 是否已在 REGISTERED_ASCEND_OPS 注册？
│  │     ├─ 是 → Q2b. Ascend 实现是否覆盖新模型的全部参数/分支？
│  │     │       ├─ 是 → 【类型 0】零适配，沿用现有结构
│  │     │       └─ 否 → 【类型 4】扩展现有实现（参数被静默丢弃/分支缺失）
│  │     └─ 否 → 【类型 1】新增 CustomOp OOT 实现（覆写 forward_oot）
│  │
│  ├─ 是 PluggableLayer 子类
│  │  └─ Q3. Ascend 是否已注册？
│  │     ├─ 是 → Q3b.（同 Q2b）Ascend 实现是否覆盖新模型的全部参数/分支？
│  │     │       ├─ 是 → 【类型 0】零适配
│  │     │       └─ 否 → 【类型 4】扩展现有实现
│  │     └─ 否 → 【类型 2】新增 PluggableLayer OOT 实现（覆写 forward）
│  │
│  ├─ 上游实现里硬编码了 CUDA/ROCm 分支，且**无注册装饰器**
│  │  └─ 【类型 3】monkey patch（无法走注册表）
│  │     注：有注册装饰器 + 仅函数内硬编码 → 走类型 2（继承后覆写该方法）
│  │     注：工厂函数已 patch 但新模型有额外参数差异（如 latent MoE 的
│  │         routed_expert_hidden_size）→ 在类型 3 patch 基础上叠加类型 4 扩展
│  │
│  └─ 参数/语义有差异（如多了 gate、少了 RoPE）
│     └─ Q4. 差异能否用上游已有的可选参数表达？
│        ├─ 能 → 【类型 1/2】按上面处理，但要确认 Ascend 实现没丢弃该参数
│        └─ 不能 → 【类型 4】扩展 Ascend 实现 + 可能需要改 attention backend
│
└─ 完全没有对应层（全新结构）
   └─ 【类型 5】全新实现：算子 + 可能的 attention backend + KV cache spec

（以上为逐 module 判定。以下为非 module 级的特殊情况：）

┌─ 特殊：不是单个 module，而是层间数据流
│  （如 block residual / 跨层状态累积 / 前缀和传递）
│  └─ 【类型 5】全新实现 + E7 图模式 piecewise（跨层状态直接阻塞
│     ACL Graph 全图捕获，需将状态传递点加入 splitting_ops）
│     默认判类型 5（对应 P2）。若跨层状态仅是简单的标量门控
│     （如 1 个 Linear + sigmoid），可按类型 3 处理（对应 P1）。
│     参考：某些混合模型的块级残差机制（如 attn_res_block_size）（示例）
└─
```

### 2.2 三个判定的具体操作

**判定 A：上游有没有等价层**

```bash
# 按类名找
grep -rn "^class <ModuleName>" $VLLM/vllm/model_executor/
# 按功能找（如激活函数）
grep -rn "@CustomOp.register\|@PluggableLayer.register" $VLLM/vllm/model_executor/layers/
```

**判定 B：是 CustomOp 还是 PluggableLayer**

两者的 `register_oot` 写进**同一个全局字典** `op_registry_oot`（`vllm/model_executor/custom_op.py:22`），注册方式一样，但**实现时覆写的方法不同**：

| | CustomOp | PluggableLayer |
|---|---|---|
| 抽象粒度 | 算子（无状态） | 层（有参数、组合子模块） |
| 替换时机 | `__new__`（实例化时） | `__new__`（实例化时） |
| **要覆写的方法** | **`forward_oot`** | **`forward`** |
| forward 分发 | 有（`dispatch_forward`） | 无 |
| `custom_ops` 开关 | 受控 | 不受控 |

> ⚠️ **写错不报错，只静默不生效**。给 PluggableLayer 子类实现 `forward_oot`，那个方法永远不会被调用。这是最隐蔽的坑。

确认方法：顺着基类往上查，直到撞见 `CustomOp` 或 `PluggableLayer`。

**判定 C：Ascend 是否已注册**

```bash
grep -n "\"<RegisterName>\"" $VLLM_ASCEND/vllm_ascend/utils.py
```

注册表在 `vllm_ascend/utils.py:813`（`REGISTERED_ASCEND_OPS`）。注意 **key 是类名**（`"RMSNorm"`），不是上游的注册名（`"rms_norm"`）——因为 `__new__` 里用的是 `cls.__name__`。

### 2.3 逐 module 判定表模板

对新模型做适配评估时，产出一张判定表，列定义：**Module（厂商代码）| 上游等价层 | 机制 | Ascend 现状 | 判定（类型 0-5）| 工作量**。P 级只用于模型级路径判定（§1.2 与 workflow Phase 0），不进 module 判定表——module 行的机制信息已由类型 0-5 经 §1.3 映射承载。

> ⚠️ **「上游等价层」以 §1.1.1 Q0-② 选定的基线分支为准**。不同厂商分支的同一 module 可能落在不同类型——基线选错，整张判定表跟着错。典型：某分支用带 `@PluggableLayer.register` 的**共享层**（→ 类型 2，注册替换），另一分支用**私有类**（→ 类型 3，monkey patch）。填表前先在表头注明基线分支名。

---

## 第三部分：六类适配逻辑（类型 0-5）

### 类型 0：零适配（沿用现有结构）

**判定依据**：上游有等价层 + Ascend 已在 `REGISTERED_ASCEND_OPS` 注册。

**什么都不用做**。上游模型代码写 `RMSNorm(hidden_size)`，`CustomOp.__new__` 查表返回 `AscendRMSNorm`，自动调 `torch_npu` 融合算子。

**唯一要确认的**：`compilation_config.custom_ops` 是否为 `["all"]`（在 `NPUPlatform.check_and_update_config()` 中对非 310P 设置）。设成 `"none"` 时类照样被替换，但 forward 会退回 `forward_native` 纯 PyTorch 实现。

**典型例子**：RMSNorm、SiluAndMul、RotaryEmbedding、各类 ParallelLinear、VocabParallelEmbedding。

### 类型 1：新增 CustomOp OOT 实现

**判定依据**：上游是 `CustomOp` 子类（有 `forward_cuda`/`forward_native`），Ascend 未注册。

**要点**：
- 默认只覆写 `forward_oot`，**不要覆写 `__init__`**，让基类处理参数
- **例外：需覆写 `__init__` 的两种场景**：
  - (a) 需要从 `config` 读额外配置（如 `vllm_config` 中的开关）
  - (b) 新算子/新激活有**上游基类不存在的构造参数**（如 `beta`、`linear_beta`）——此时必须同时覆写 `__init__`（接收新参数并赋给 `self`）和 `forward_oot`（从 `self` 读取使用）
- 先用 `forward_native` 的 torch 等价实现跑通精度，再换融合算子
- 精度敏感的（如带 tanh/sigmoid 的激活）注意中间计算用 fp32
- **精度细节检查（新算子/新激活，昇腾上最典型的精度不达标来源）**：
  - 激活/路由/norm 的中间计算用 **fp32**（sigmoid/tanh/softmax/softplus），不要 bf16 一路算到底
  - 门控参数 `dt_bias`/`A_log`/`beta` 用 **float32 参数 + float32 运算**（对照同族层 GDN/KDA 写法）
  - 低秩分解（q_lora/kv_lora）的展开顺序与 dtype 与厂商实现等价
  - 形状/布局：conv1d weight 是否需 `unsqueeze(1)`；状态矩阵 layout（dim-first vs dim-last）与 kernel 一致
  - 无法在无 NPU 环境验证的，标注「待真实权重门验证」，不阻塞开发

**参考模板**：`vllm_ascend/ops/activation.py:41` `AscendSiluAndMulWithClamp`——结构最简单，7 行覆写。代码骨架见 `新模型NPU适配实现模板.md` 模板 A。

### 类型 2：新增 PluggableLayer OOT 实现

**判定依据**：上游是 `PluggableLayer` 子类（`__init__` 里组合子模块），Ascend 未注册。

通常只覆写 3 类方法，其余全继承：(1) `forward`/`_forward`（计算路径）；(2) `get_state_shape`/`dtype`（若涉及 KV/state cache）；(3) `get_attn_backend`（若需要专属 backend）。

**参考模板**：`vllm_ascend/ops/bailing_moe_linear_attn.py:42`——文件头注释明确写了"只覆写 3 个平台相关方法，其余全继承"，是最干净的范例。代码骨架见 `新模型NPU适配实现模板.md` 模板 B。

### 类型 3：monkey patch

**判定依据**（满足任一）：
- 上游层**没有** `@CustomOp.register` / `@PluggableLayer.register` 装饰器 → 不在注册表里，无法 OOT 替换
- 上游实现内部**硬编码** `if current_platform.is_rocm(): ... else: <CUDA>`，**且无注册装饰器** → 没有 NPU 分支又无法替换
- 目标不是类而是**工厂函数**（如 `FusedMoE`）

> **与类型 2 的边界**：上游层**有**注册装饰器、模块可正常 import、只是某个方法内部
> 硬编码厂商分支 → 走**类型 2**（继承后覆写该方法，自带 NPU 分派），不必 patch。
> 三者的区分：模块 import 即崩 → standalone 重写（Q1'）；有装饰器 + 函数内硬编码 →
> 类型 2；无装饰器 → 类型 3。

**分支选择**：改引擎/调度层 → `patch/platform/`（`pre_register_and_update()` 生效）；改模型/算子层 → `patch/worker/`（`NPUWorker.__init__` 生效）。

**时序铁律**：worker patch 必须在**任何模型模块被 import 之前**执行。`NPUWorker.__init__` 的顺序是：`① adapt_patch() → ② from vllm_ascend import ops → ③ register_ascend_customop() → ④ 模型加载`。

**工厂函数替换**：必须改**两处 binding**（包 `__init__` 和 layer 模块），否则部分模型拿到未 patch 版本。代码骨架（含 FusedMoE 示例）见 `新模型NPU适配实现模板.md` 模板 C。

> ⚠️ AGENTS.md 要求：新增 patch 必须经架构评审，且有上游回贡计划。patch 是技术债，注册表才是目的地。

#### patch 决策树（强制走查，顺序不可颠倒）

1. 模型数学/权重问题 → 改上游模型文件或在 `vllm_ascend/models/` 注册新架构；模型主体适配代码禁止以 patch 形式进入 vllm-ascend（RFC #7539）。
2. 能用既有机制（CustomOp 分派、继承、fusion pass、composition）就不用 patch。
3. 动代码前先走 fallback ladder 定位（复现 → `--enforce-eager` → `TORCHDYNAMO_DISABLE=1` → 关多模态）。
4. 仅当框架行为在 NPU 上错误且无插件钩子时，允许框架级最小 patch（只覆盖不兼容路径），并强制四段式登记（Why / How / Related PR / Future Plan + 移除条件），写入 `vllm_ascend/patch/__init__.py`；防御性写法：对上游函数做签名级校验，变更即 RuntimeError。
5. 「不打 patch 模型就不能跑」→ 停止并提 issue 分析根因，而不是加 patch。

### 类型 4：扩展现有 Ascend 实现

**判定依据**：上游有等价层且 Ascend 已实现，但**新模型用到了 Ascend 实现未覆盖的可选参数/路径**。

这是最容易被漏判的一类——表面看"已支持"，实际跑起来参数被静默丢弃或走错分支。

| 模式 | 症状 | 检查方法 |
|---|---|---|
| 参数被丢弃 | Ascend 实现构造子模块时没传某个 optional 参数 | 对比上游 `__init__` 的参数列表与 Ascend 实现实际使用的 |
| 分支缺失 | 枚举新增了成员，Ascend 的 if-elif 链没有对应分支，落到 `else` 兜底 | grep 该枚举在 Ascend 侧的所有使用点 |
| 无条件假设 | Ascend 实现无条件调用某个前提操作（如无条件取 RoPE cache） | 找 Ascend 实现里的无条件调用，对照新模型是否满足前提 |
| 算法别名 | 按名称查"缺失"，实际语义已被支持（如 `noaux_tc` 就是 `scoring_func="sigmoid"` + `e_score_correction_bias` 的组合）；反向更危险：名义支持、参数语义不同 | 名称查无此项时，**先核对数学定义再判缺失**；对名称命中的也要核对参数语义 |

**最危险的是"分支缺失"**：新的激活函数枚举落到 `else` 分支被当成 SwiGLU 算——不报错，结果全错。

**适配逻辑**：
1. 先定位差异点（用上表的检查方法）
2. 加显式分支或条件开关，**不要改默认行为**
3. 兜底 `else` 分支改成显式 `raise NotImplementedError`，把静默错误变成早期失败

**检查命令**（对应上表三种症状）：
```bash
# ① 参数被丢弃：对比上游 __init__ 签名 vs Ascend 实际使用
diff <(grep -A30 "def __init__" $VLLM/<upstream>.py | grep -oE "^\s+\w+:" | tr -d ' :') \
     <(grep -oE "\w+=\w+\." $VLLM_ASCEND/<ascend>.py | cut -d= -f1)

# ② 分支缺失：枚举成员数 vs Ascend 侧分支数
grep -c "MoEActivation\.\|ActivationType\." $VLLM_ASCEND/ops/fused_moe/moe_mlp.py

# ③ 无条件假设：找无 if 保护的前提调用（如无条件取 RoPE cache）
grep -n "get_cos_and_sin\|rope_single\|apply_rotary" $VLLM_ASCEND/attention/mla_v1.py
```

### 类型 5：全新结构

**判定依据**：上游完全没有对应层。

**算子实现**三选一：有 `torch_npu` 融合算子 → 直接调；无但可组合 → 组合现有算子；都不行 → 写 Triton kernel（`ops/triton/`）或 C++ 自定义算子（`csrc/`）。

后续步骤（KV cache spec → attention backend → 模型层组装）的依赖关系与实施顺序见第五部分 Phase 3；代码骨架见 `新模型NPU适配实现模板.md` 模板 D/E/F。

**参考模板**：
- 新 attention backend：`vllm_ascend/attention/sfa_v1.py`（继承 `MLACommonMetadataBuilder`，复用 MLA 骨架）或 `dsa_v1.py`（完全独立体系）
- 新 Triton kernel：`vllm_ascend/ops/triton/`
- 新 KV cache spec：`vllm_ascend/core/kv_cache_interface.py:213`
- standalone 重写：`vllm_ascend/models/deepseek_v4.py`（复用平台中立基类 + ascend 自有 kernel）

> **Q1' standalone 变体**：Q1' 判定"不可 import"时，虽属类型 5 但可参考上游的平台中立基类（而非从零设计）。

---

## 第四部分：EngineCore 侧适配清单

模型层适配完，还有一层框架配置。**这一层不改，模型能构造出来但跑不起来或跑错。**

### 4.0 加载期权重映射检查（新，第三部分与第四部分之间）

> ⚠️ 本部分覆盖 **checkpoint 权重名 → vLLM 层参数结构**的映射。前面所有 module 判定都是**运行期**视角（这个层怎么算、用什么 backend），而权重映射是**加载期**视角（checkpoint 里的名字怎么落到 vLLM 的参数上）。两者独立——一个 module 判定为"类型 0 零适配"，但它的权重名如果对不上，照样加载失败。**跳过本节的典型后果：`safe_open` 报 missing/unexpected keys，或权重静默加载错误导致精度错。**

**必查命令**（对每个判定为"需加载权重"的 module 都要做，重点是新注意力类）：

```bash
# ① 厂商 checkpoint 权重名（safetensors index 是权威，不读 modeling 文件的字符串）
python -c "
from safetensors import safe_open
import json
with open('<model>/model.safetensors.index.json') as f:
    idx = json.load(f)
for k in sorted(idx['weight_map']):
    print(k)
" | grep -E "self_attn|mlp|moe|conv|gate|proj" | head -60

# ② vLLM 期望的权重名（看 load_weights 的映射 / _stacked_params_mapping / 参数名）
grep -n "weights_mapping\|params_mapping\|_load\|def load_weights\|name_mapping" $VLLM/vllm/model_executor/models/<model>.py

# ③ 直接 diff 两份清单
python3 - <<'EOF'
# 厂商名集合 vs vLLM 层参数名集合 → missing / unexpected 两组
EOF
```

**四类映射差异**（按危险程度排序）：

| 类型 | 示例（真实案例） | 后果 | 处理 |
|---|---|---|---|
| **融合打包** | 厂商三套 `q_proj/k_proj/v_proj` → vLLM packed `in_proj_qkvgfab` | 名字对不上 → missing keys | 写 `_stacked_params_mapping` 或 loader 重排 |
| **子模块并入** | 厂商 `kv_b_proj` 被 vLLM 在加载后吸收成 `W_UK_T`/`W_UV`（DeepSeek 系 MLA） | vLLM 参数名 ≠ 厂商名 | 确认 `process_weights_after_loading` 正确处理 |
| **旧版兼容** | 厂商 `A_log` 存 4D `(1,1,H,1)`，vLLM 期望 1D `[H]` | 形状断言失败 | loader 里做 `.view()`（vLLM 已有 `a_log_weight_loader`） |
| **命名拼写** | `conv1d.weight` 是否需要 `.unsqueeze(1)`；`dt_bias` 初始化差异 | 加载成功但权重形状错 → 精度错 | 对照 vLLM 同族层（GDN/Mamba）的 loader |

**判定规则**：把「厂商权重名集合」与「vLLM 层参数名集合」的 **missing / unexpected 两组**列进设计文档的判定表，作为每个 module 的一行。**不允许出现"名字看起来对、没实际验证"**——用 §2.0 速查表的 `❌/⚠️/✅` 语义标记。

**自检命令（加载期）**：`vllm serve <model> --load-format safetensors 2>&1 | grep -E "not initialized|size mismatch|shape mismatch"` —— 出现任一项都是阻断项，回 Developer 修复 loader 再放行。匹配文案随 vLLM 版本变化（当前版本实测缺失输出为 "Following weights were not initialized from"），**以当前安装版本实测校准**；`Unexpected extra config keys` 属配置项校验，与权重缺失无关，不作阻断项。

---

EngineCore 侧的适配几乎全部通过 `NPUPlatform.check_and_update_config()`（`platform.py:408`）和 `patch/platform/` 完成。

### 4.1 检查清单

| # | 检查项 | 触发条件 | 适配位置 |
|---|---|---|---|
| E1 | **模型注册** | 上游没有该模型，或需要覆盖上游实现 | `vllm_ascend/models/__init__.py::register_model()` |
| E2 | **量化方法识别** | 模型带 `quantization_config` | `quantization/utils.py::maybe_auto_detect_quantization`；新格式需在 `AscendCompressedTensorsConfig._detect_quant_type` 加分支。⚠️ MXFP4 落地有**两道门**：(1) `torch_npu` 符号可用性（`mxfp_compat.py:60` `ensure_mxfp4_*`，**版本门**，与 SoC 无关）；(2) 动态 MX 量化融合算子的 SoC 门（`mxfp_compat.py:25`，**仅 A5**）。符号缺失时需在 `check_and_update_config` 中早期报错或回退 W8A16。速查表 #11 有交叉引用 |
| E3 | **Attention backend 选择** | 用了 MLA / 稀疏 / 新注意力 | `NPUPlatform.get_attn_backend_cls()` 的 `(use_mla, use_sparse, use_compress)` 分发表 |
| E4 | **KV cache spec** | 新的 KV 形态（MLA 变体、线性注意力 state） | **先查上游 spec 体系（FullAttention / MLA / Mamba / SlidingWindow）能否表达**——线性态通常直接复用上游 `MambaSpec`（GDN 先例）。Ascend 已有 3 个 spec：`AscendMLAAttentionSpec`、`AscendSFAIndexerCacheSpec`、`AscendSlidingWindowMLASpec`。**判据**：若 state 形状/生命周期与 Mamba spec 完全一致（按 seq 分配、不按 token 增长），复用 `MambaSpec`；若 state 有额外的 gate 调制/conv 混合/非标准 page 计算，则需在 `core/kv_cache_interface.py::register_ascend_kv_cache_specs()` 新增 spec + manager |
| E5 | **混合 KV cache** | 模型是 hybrid（部分层 full attn + 部分层 linear） | `patch/platform/patch_kv_cache_{coordinator,utils}.py`、`patch_mamba_config.py`（**page size 对齐断言**）、`patch_mamba_manager.py` |
| E6 | **block_size / page 对齐** | 新 state 形态改变了 page 大小 | `patch_mamba_config.py` 的对齐断言；`utils.py::refresh_block_size` |
| E7 | **图模式** | 模型有跨层状态传递、动态控制流 | `check_and_update_config` 里的 cudagraph_mode 收敛；必要时 `enforce_eager` 或加 splitting_ops |
| E8 | **MoE 通信方式** | MoE 模型 | `ascend_forward_context.py::select_moe_comm_method`（按 SoC + 专家数 + EP size 分发） |
| E9 | **并行约束校验** | TP/EP/DP/PP 有特殊要求 | `check_and_update_config` 加 assert，早期失败优于运行时崩 |
| E10 | **多模态处理器** | 多模态模型 | 类似 `patch/hunyuan_vl_processor_compat.py` 的 processor 兼容 patch（该文件在 `patch/` 根目录，不在 `patch/platform/` 或 `patch/worker/` 下） |
| E11 | **Worker 类** | 需要定制 worker 行为 | `parallel_config.worker_cls`（默认 `NPUWorker`） |
| E12 | **投机解码 / MTP** | 模型带 MTP 或 draft model | `patch_speculative_config.py`、MTP 模型注册 |

**关键原则**：模型层适配放 worker 组补丁或 `models/`；**配置与调度适配必须放 platform 组**（`pre_register_and_update` / `check_and_update_config`），因为它们要在配置定型前生效，且 engine-core 进程也要看到。例如 KV cache spec 注册由 engine-core 在规划 KV cache 时调用——那时 worker 还没起来，放 worker 组补丁完全无效。

### 4.2 三个最容易漏的

**E5 混合 KV cache 的 page 对齐**

`patch_mamba_config.py` 里有强制的对齐断言：
```
attn_single_token_k_page_size * attn_block_size == ssm_block_page_size
```
新模型的 linear attention state 形状（`num_heads × head_dim × head_dim`）一旦和 attention 侧的 page 大小对不上，这里会直接断言失败。**这是混合模型最先崩的地方。**

**E7 图模式与跨层状态**

如果模型的 decoder layer 之间传递额外状态（不只是 `hidden_states`），ACL Graph 捕获会出问题：
- 静态图要求每层输入输出形状固定
- 跨层累积的张量在 PP 切分时需要额外的通信

处理方式：先 `enforce_eager` 跑通正确性，再评估能否 piecewise 捕获（把状态传递点加进 `splitting_ops`）。

**E8 MoE 通信与专家数**

`select_moe_comm_method` 按 SoC 分发，各代次约束不同：

> ⚠️ 下表是 `ascend_forward_context.py::_select_a2/a3/a5_moe_comm_method` 的逻辑快照，**以源码为准**。代码更新后表可能过时，使用前应 grep 确认。

| SoC | MC2 条件 | 说明 |
|---|---|---|
| A2 | `num_experts / ep_world_size <= 24` 且 `ep_world_size >= 16` 且 `num_tokens <= mc2_tokens_capacity` | 大专家数模型需要很大的 EP 才能走 MC2，否则掉回 ALLGATHER（性能差） |
| A3 | `enable_fused_mc2` 时 EP≤64 走 FUSED_MC2（需 CANN mega_moe 支持）；EP≤32 走 dispatch_ffn_combine | 不看专家数，只看 EP size |
| A5 | **首选 MC2**：`num_tokens <= mc2_tokens_capacity` 且 `world_size > 1`<br>fallback：`world_size <= num_experts_per_tok` → ALLGATHER，否则 ALLTOALL | 与 A2/A3 不同，A5 不看专家数 |
| 310P | 固定 ALLGATHER | |

**部署前必算**：用目标 EP size 代入公式，确认落在哪个分支。注意 A2/A3/A5 的 MC2 路径还需满足 `num_tokens <= mc2_tokens_capacity`（token 量超出容量时 A2 掉回 ALLGATHER、A3 掉回 ALLTOALL）。

### 4.3 调度层适配清单（KVCacheSpec 是首要适配物）

KVCacheSpec 必须先于一切性能工作定稿：block size、分组、prefix 命中语义全部由其导出，事后更改会级联推翻调度与图模式配置。

1. **cache 形态枚举**：逐层确认注意力形态（GQA/Full / MLA / SFA / DSA / GDN/KDA 线性注意力），是否出现 indexer 独立缓存、recurrent state、多 cache group。
2. **KVCacheSpec 选型**：优先复用既有 spec 子类（`FullAttentionSpec` / `AscendMLAAttentionSpec` / `AscendSFAIndexerCacheSpec` / `MambaSpec` 等）；确需新增时经 `KVCacheSpecRegistry.register()` 注册。选型错误有显存量级代价（V3.2 曾按 2 倍 MLA page size 分配，浪费 38.8%）。
3. **跨 cache group page size 一致性**：统一内存池强制各组 page size 相同；警惕 mamba padding 把 attention block size 撑大（实测有 784 token 案例）——评估对短请求与 prefix 命中率的副作用。
4. **投机解码调度**：`num_lookahead_tokens` 生效确认；draft 层 KV group 归属（non-causal draft 须单独分组）；rejected 回滚路径（paged KV 覆写即可，SSM/GDN state 需快照方案）。
5. **prefix cache / PD 兼容**：线性注意力组须 `mamba_cache_mode=align`（state 仅在 block 边界物化）并查与 MTP 的互斥项；PD 分离 connector 须实现 `SupportsHMA`（null block 跳过、命中位置 LCM 对齐、传输期间 block pin 住）。

---

## 第五部分：标准工作流

```
Phase 0  情报收集
├─ 读厂商 config.json + modeling_*.py，列出所有 nn.Module
├─ grep 上游 vLLM 是否已有该模型（registry.py + vllm/models/）
│   └─ 若有：检查上游分派是否覆盖 NPU（§1.1.1 前置检查 Q0）
│       及上游模块顶层 import 链能否在 NPU 上 import（决策树 Q1'）
├─ grep vllm-ascend 自己的 kernel 库是否已有同族算子：
│   grep -rn "<算子族关键词>" $VLLM_ASCEND/vllm_ascend/ops/triton/
│   （示例：kda/delta/gated → 查 KDA/GDN 族 triton kernel；
│     kernel 存在 ≠ 已接线，但存在即大幅降低工作量评估）
└─ 确认目标硬件代次（决定量化/通信可用性）

Phase 1  逐 module 判定
├─ 先走 2.0 速查表快通道；未全命中的 module 走 2.1 决策树
├─ 产出判定表（类型 0-5 + 工作量）
├─ 按 1.2 的「规模分诊」表，依类型 5 个数决定落地形态
└─ 标记阻塞项 vs 非阻塞项

Phase 2  EngineCore 清单核对
└─ 走第四部分 E1-E12，标记需要改的

Phase 3  实现（依赖关系：①② 可并行，③ 依赖 ①，⑤ 必须最后）
├─ ① 算子层（类型 1/2/5）—— 先 native 保正确，再融合提性能
├─ ② attention backend（若需要）        ← 通常与 ① 无依赖，可并行；
│                                          若 backend 复用 ① 的 kernel
│                                          （如 KDA 复用 triton/kda/）则依赖 ①
├─ ③ KV cache spec + 混合管理（若需要）  ← 依赖 ① 的 state 形状确定
├─ ④ 模型组装 + 注册
└─ ⑤ EngineCore 配置                    ← 最后做：需前四步的实际形态才能定配置

Phase 4  验证（贯穿）
├─ ⓪ 适配自检（见附录 C）：确认所有 OOT 注册**实际生效**再往下走
│     两种机制日志文案不同，务必同时匹配 custom op / pluggable layer
│     跳过这步 → 精度不对时无法区分「算法错」还是「根本没替换成功」
├─ 单算子精度：tests/ut/ops/
├─ 端到端精度：先 enforce_eager，对齐厂商参考实现输出
├─ 图模式：再开 ACL Graph，对比 eager 结果
└─ 性能：benchmarks/
```

### 实现顺序的铁律

1. **先正确后性能**：所有新算子先用 torch 等价实现跑通精度，再换融合算子
2. **先 eager 后图**：`enforce_eager=True` 跑通再开 ACL Graph
3. **先小规模后并行**：单卡跑通再上 TP/EP/DP
4. **静默错误要变成显式失败**：所有兜底 `else` 分支加 `raise NotImplementedError`

### 与 Day0 workflow 的映射

本文档的 Phase 0-4 是判定与实现方法论；流程编排层（`.claude/skills/day0-inference`，四 Stage 渐进叠加：golden → 并行量化 → 特性叠加 → 验收）的 **Stage 1**（`flows/golden_flow.md`）将其映射为以下执行阶段与门禁（Stage 2-4 的映射随各 flow 实现后补充）：

| 本文档 | workflow 阶段 | 门禁 |
|---|---|---|
| （新增）依赖就绪扫描 | Phase 0 | G0 路径门禁（P0/P1/P2 + 排期模板） |
| 第一~四部分（判定） | Phase A Designer | 设计完整性检查 |
| 第三、五部分（实现） | Phase B Developer | G1 实现门禁（UT + OOT 自检 + patch 台账） |
| §4.0 加载期 + Phase 4 | Phase C1/C2 Tester | G2 冒烟（含捕获计数）/ G3 精度 |
| 附录 F | Phase C3 Tester | G4 图模式/性能 |
| 附录 E + patch 台账 | Phase D Reviewer | G5 发布 |

依赖阻塞（上游未合入 / CANN 算子缺口）不是实现缺陷，不进修复回路：启动并行预案（外挂算子包 / 上游 pre-release 分支 / Triton 过渡），无回退路径则停止并提 issue。

---

## 第六部分：服务层适配清单

服务层错配多为静默失败——HTTP 200 ≠ 部署正确。2026 代模型的关键变化是程序化 prompt 编码取代 Jinja 模板，`--tokenizer-mode` 从「tokenizer 实现选择」升级为「会话协议实现选择」。

### 6.1 parser 三件套同名同代对齐

- `--tokenizer-mode` / `--tool-call-parser` / `--reasoning-parser` 必须同名同代成套配置；错代次 parser 名（如对 K3 用 `kimi_k2`）可能合法但输出全错（思考文本落入 content、tool_calls 解析失败）。
- chat template 两形态：自带 Jinja（GLM/Qwen）vs 程序化编码（DSV4 的 encoding/、K3 的 Python renderer）——后者要求内置 tokenizer-mode，输入 encoding 与输出 parsing 缺一不可。

### 6.2 reasoning_effort 与 checkpoint 代次

- effort 档位命名各厂商分裂（low/high/max 与 low/medium/xhigh 并存），需网关归一化。
- **checkpoint 代次是显式配置维度**：effort 映射等前端行为须按 checkpoint 特征字段分支；对齐新代次时不得回归旧代次（DSV4 的 0731 effort 补丁曾回归 Preview checkpoint）。
- 强制思考类模型（GLM-5.3）：`thinking.disabled` 直接 400，迁移时必须改写请求。

### 6.3 服务矩阵验证（验收用）

1. **render 验证**：`/v1/chat/completions/render` 端点比对渲染 token_ids，验证模板与 effort 前缀生效（不同档位应渲染不同长度前缀）。
2. **tool_choice 组合**：tool_choice ∈ {auto, required, named, none} × {流式, 非流式} × {有无 reasoning_effort}，校验 `finish_reason` 与 arguments JSON 可解析性。
3. **多轮 reasoning 回归**：重点第 3 轮（think 泄漏进 content 是已知失败模式）；enable_thinking=False 时 reasoning parser 应 identity 回落。
4. **长上下文泄漏回归**：长上下文省略特殊 token 导致原文泄漏为先例，须覆盖。

---

## 附录 A：现有 CustomOp 覆盖清单

`REGISTERED_ASCEND_OPS`（`vllm_ascend/utils.py:813`）当前 30 项基础注册 + 条件项。**新模型用到这些结构时零适配**。（下表按组归纳，含条件项与说明项；基础注册表为 30 项，不要数表格行数；行号会漂移，以 `grep -n "REGISTERED_ASCEND_OPS = {" vllm_ascend/utils.py` 实测为准。）

| 组 | 注册名 | 机制 |
|---|---|---|
| 归一化 | `RMSNorm` `GemmaRMSNorm` `RMSNormGated` `FusedRMSNormGated` | CustomOp → `forward_oot` |
| 激活 | `SiluAndMul` `SiluAndMulClamp` `QuickGELU` | CustomOp → `forward_oot` |
| 位置编码 | `RotaryEmbedding` `MRotaryEmbedding` `YaRNScalingRotaryEmbedding` `DeepseekScalingRotaryEmbedding` `ApplyRotaryEmb` | CustomOp → `forward_oot` |
| 线性层 | `ColumnParallelLinear` `RowParallelLinear` `MergedColumnParallelLinear` `QKVParallelLinear` `ReplicatedLinear` `GateLinear` | PluggableLayer → `forward` |
| 词表/输出 | `VocabParallelEmbedding` `ParallelLMHead` `LogitsProcessor` | PluggableLayer → `forward` |
| 注意力 | `MultiHeadLatentAttentionWrapper` `RelPosAttention` | PluggableLayer → `forward` |
| 注意力 | `MMEncoderAttention` | CustomOp → `forward_oot` |
| 线性注意力 | `GatedDeltaNetAttention` `BailingMoELinearAttention` | PluggableLayer → `forward` |
| MoE | `MoERunner` `RoutedExperts` | 注册表替换（与 `patch_fused_moe` 工厂重定向并存） |
| 其他 | `CustomQwen2Decoder` | PluggableLayer → `forward` |
| 其他 | `Conv3dLayer`（基类 `ConvLayerBase`） | CustomOp → `forward_oot` |
| 条件项 | `KimiK3MultiHeadLatentAttentionWrapper`（vLLM ≠ 0.28.0 时注册） | PluggableLayer |
| 310P 覆盖 | 11 项 `*310` 变体 | — |

> ⚠️ **310P 差异不止这 11 项**：310P 不设 `custom_ops = ["all"]`，算子退回 `forward_native`（机制见类型 0），需单独验证执行路径。

**不在表里的重要结构**：
- `FusedMoE` —— 走 `patch/platform/patch_fused_moe.py`（工厂函数重定向）；注意 **MoE 已是双机制并存**：工厂入口仍靠 patch 重定向，而 `MoERunner`/`RoutedExperts` 已进注册表、可经注册表替换——判定 MoE 类 module 时先查注册表再考虑 patch
- Attention 主体 —— 走 `get_attn_backend_cls()` 分发
- 线性注意力 state cache —— 走上游 `MambaSpec` + `patch_mamba_*`（Ascend 无对应 spec）

---

## 附录 B：常见陷阱

| # | 陷阱 | 后果 | 规避 |
|---|---|---|---|
| 1 | 给 PluggableLayer 实现 `forward_oot` | 静默不生效 | 先确认基类，见 2.2 判定 B |
| 2 | 新激活枚举落到 `else` 兜底 | **静默算错**，不报错 | grep 枚举所有使用点；兜底改 raise |
| 3 | `REGISTERED_ASCEND_OPS` 用注册名而非类名 | 替换不生效 | key 用 `cls.__name__` |
| 4 | patch 时序晚于模型 import | patch 不生效 | worker patch 放 `NPUWorker.__init__` 第一步 |
| 5 | 工厂函数只改一处 binding | 部分模型拿到未 patch 版本 | 包 `__init__` 和 layer 模块都要改 |
| 6 | Ascend 实现丢弃上游 optional 参数 | 功能静默缺失 | 对比 `__init__` 参数列表 |
| 7 | 混合模型 page size 不对齐 | 启动断言失败 | 先算 state 形状与 page 大小 |
| 8 | 大专家数 MoE 未算 EP 门槛 | 掉回慢通信路径 | 部署前代入 `select_moe_comm_method` 公式 |
| 9 | `tensor.item()` 在热路径 | 性能骤降（NPU 同步） | 见 AGENTS.md NPU-Specific Considerations |
| 10 | 精度敏感算子未用 fp32 中间计算 | 精度不达标 | 激活/路由/norm 的中间量用 fp32 |
| 11 | **上游分派只按 `is_rocm()` 二分，NPU 静默落入 nvidia 分支** | import CUDA 专属代码（如 CUTLASS DSL）直接崩 | §1.1.1 前置检查 Q0；在 vllm-ascend 注册覆盖实现，且确保 ascend 分支不 import 上游 nvidia 模块 |
| 12 | 上游层所在模块顶层 import CUDA 专属依赖（`_custom_ops` / cute_dsl / deep_gemm 等） | NPU 上 **import 即崩**，"继承上游类覆写方法"路径不可用 | 决策树 Q1'；standalone 重写（`models/deepseek_v4.py` 模式），只复用平台中立基类 |

---

## 附录 C：适配自检（静默失败的可主动检测）

附录 B 列了 4 条"静默失败"陷阱（#1 写错方法、#2 else 兜底、#3 注册名错、#6 丢参数）。其中 **#1/#3 可用下列命令①②检测；#2/#6 的检查方法见「类型 4 → 检查命令」**。陷阱表告诉你"会静默失败"，但没告诉你怎么知道自己踩了：

```bash
# ① OOT 替换是否真的生效（两种机制日志文案不同，需同时匹配）
#    CustomOp 分支: "Instantiating custom op: <name> using <impl>"
#    PluggableLayer 分支: "Instantiating pluggable layer: <name> using <impl>"
VLLM_LOGGING_LEVEL=DEBUG vllm serve <model> 2>&1 | grep -E "Instantiating (custom op|pluggable layer)"

# ② 注册表实际内容（确认你的 op 在里面、key 是类名）
python -c "from vllm.model_executor.custom_op import op_registry_oot; print(sorted(op_registry_oot))"

# ③ attention backend 实际选中的（确认没走错 backend，示意调用）
python -c "from vllm_ascend.platform import NPUPlatform; print(NPUPlatform.get_attn_backend_cls(...))"  # 签名需补 selected_backend/attn_selector_config，此处仅示意

# ④ MXFP4 符号可用性（确认 torch_npu 有所需符号）
python -c "import torch_npu; print([s for s in ['float4_e2m1fn_x2','float8_e8m0fnu','npu_dynamic_mx_quant','npu_quant_matmul'] if hasattr(torch_npu, s)])"
```

这能把 4 条"靠经验避免"的静默失败变成"可主动检测"。

---

## 附录 D：module 枚举完整性交叉验证（防"漏列没进判定表"）

> ⚠️ 最大剩余风险不是"module 判定错"，而是"**module 根本没被枚举进判定表**"——漏列了，后面所有类型 0-5 判定都覆盖不到它。附录 C 能检测"替换没生效"，但**检测不了"没被列进表的东西"**。本节把"漏列"变成可交叉验证的检查。

**从两个独立来源枚举 module，交叉比对去重**：

```bash
# 来源① config.json 的模块清单（模型配置声明的结构）
python3 - <<'EOF'
import json
cfg = json.load(open('<model>/config.json'))
def walk(o, path=""):
    if isinstance(o, dict):
        for k, v in o.items():
            walk(v, f"{path}.{k}" if path else k)
    elif isinstance(o, list) and o:
        print(f"{path}: list[{len(o)}]")
walk(cfg)
EOF

# 来源② modeling_*.py 的 nn.Module 类清单（实现声明的结构）
grep -nE "class |nn\.(Linear|Module|Embedding|RMSNorm|LayerNorm)|def forward|self\.[a-z_]+\s*=\s*" <model>/modeling_*.py | head -100

# 来源③（vLLM 侧对照）上游等价层的 load_weights / 参数名
grep -n "def load_weights\|stacked_params\|_load_" $VLLM/vllm/model_executor/models/<model>.py
```

**交叉验证规则**：
1. config 里的**每个模块**必须在 modeling 文件里有对应类（缺 → 说明实现漏了，或 config 声明了未实现的结构）。
2. modeling 文件里的**每个 `self.<attr> = nn.Xxx`** 必须能在 config 里找到对应配置字段（缺 → 说明有 config 没暴露的隐藏结构，如 K3 的 `attn_res_block_size`）。
3. **两个来源都有的模块**才进判定表；**只在一个来源出现的**标 `⚠️` 单独审查，不能直接忽略。
4. 对**非主流字段**（config 里名字不带 `num_`/`hidden`/`head` 的，如 `attn_res_block_size`、`e_score_correction_bias`、`routed_expert_hidden_size`、`mla_use_output_gate`、`use_full_rank_gate`）**逐个**追问"这个字段在 forward 里被用了吗？在哪个分支？"——这是角落结构（attention residual、latent MoE、输出门）最可能藏身的地方。

**把交叉验证结果写进设计文档**：判定表加一列「枚举来源(config/modeling)」，标注每个 module 来自哪个来源、是否两源一致。附录 C 的检查能保证"替换生效"，附录 D 能保证"枚举完整"——两个合起来才构成完整自检。

---

## 附录 E：组合矩阵回归清单（长尾高发区）

历史已知问题几乎全部位于叠加组合而非基线。Day0 验收须按 Designer 产出的清单覆盖以下笛卡尔积的相关子集：

- **量化**：BF16 / W8A8 / W8A8C8 / W4A8MXFP / MXFP4 等该模型实际发布的权重族；
- **图模式**：eager / PIECEWISE / FULL_DECODE_ONLY（稀疏与线性注意力只承诺 UNIFORM_BATCH）；
- **投机解码**：无 / MTP / eagle / DSpark，verify 步 1+k 与 capture size 对齐；
- **CP/PD**：DCP / PCP / PD 分离（含 connector 形态）。

每个启用组合至少一条 E2E 用例，配置落在 `tests/e2e/models/configs/<Model>.yaml`。已知交叉项事故示例：MTP × prefix cache 静默输出损坏、DSpark × 混合 KV 分组 padding 膨胀、量化 draft × BF16 target page-size 统一失败。

---

## 附录 F：ACLGraph 兼容清单

### 六类不可入图操作（新模型 forward 路径静态扫描清单）

1. stream 同步及隐含同步 memcpy（回放上下文同步 rtMemcpy 崩溃，错误码 107030）；
2. event 状态查询；
3. aclop 算子（捕获时申请显存）；
4. host 侧 tiling 依赖算子（attention 即属此类，是 piecewise 把 attention 排除出图的根本原因）；
5. 控制流分支；
6. full-graph 区域内任何 Python 副作用（含 logger.debug——曾致全部 TP worker 崩溃）。

另：新自定义算子必须注册 meta 实现才可被捕获。

### 逐级开图门禁

顺序强制 eager → PIECEWISE → FULL_DECODE_ONLY，每级独立验证，`--enforce-eager` 作为定界手段。feature-first：EP + ACLGraph + FlashComm + MTP 默认全开，失败项保留证据而非默认关闭。

> **交叉引用（G2 冒烟门禁证据，在 C1 阶段采集，此处仅备查）**：ACLGraph 捕获计数期望，对齐真实断言 `tests/e2e/pull_request/two_card/aclgraph/test_aclgraph_capture_replay.py`——
> `warmup_runs = 1 + 2 × 捕获 batch size 个数`（A3 且 DeepSeek 系额外 +1，MC2 warmup）；`padding_runs = ⌈total_steps/32⌉×32 − total_steps`（32 步全局对齐空跑）；`expected = (warmup_runs + padding_runs) × dp_size`。三项易漏：DP 倍乘、对齐 padding、A3+DeepSeek 额外 warmup。

---

## 参考

> 以下路径中，`vllm_ascend/` 与 `AGENTS.md` 位于 **vllm-ascend 仓**，`vllm/` 位于 **vLLM 仓**。

**vllm-ascend 仓**
- `AGENTS.md`（仓根目录）—— 贡献规范、patch 评审要求、NPU 特有注意事项
- `vllm_ascend/utils.py:765` `register_ascend_customop()` —— 算子注册入口，`:813` 为 `REGISTERED_ASCEND_OPS`
- `vllm_ascend/platform.py:408` `check_and_update_config()` —— EngineCore 配置中枢
- `vllm_ascend/core/kv_cache_interface.py:213` `register_ascend_kv_cache_specs()` —— KV cache spec 注册
- `vllm_ascend/device/mxfp_compat.py:60` —— MXFP4 的符号门禁（`ensure_mxfp4_*` 系列，按 `torch_npu` 符号而非 SoC）；`:25` 为 RMSNorm+MX 融合算子的 A5 独占检查
- `vllm_ascend/ascend_forward_context.py:344` `select_moe_comm_method()` —— MoE 通信方式分发

**vLLM 仓**
- `vllm/model_executor/custom_op.py` —— CustomOp / PluggableLayer 双轨机制（`:22` 共用的 `op_registry_oot`）
- `vllm/model_executor/models/registry.py` —— 模型注册表，含 `"vllm.models.*"` 外部包路径约定
