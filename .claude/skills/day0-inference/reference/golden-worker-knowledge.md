# Worker 层机制知识库（控制面 + 数据面）

> **定位**：Worker 层机制与实现知识库（不驱动流程）。两个消费方：**Designer**——Phase 1 判定 Worker 层适配需求的依据，**本文每一章即一类适配场景**；**Developer**——Phase 2 按 Worker 层 design spec 落地时回查机制细节。判定逻辑与 spec 契约见设计侧规范 `.claude/skills/day0-inference/reference/design/golden-designer/golden-worker-designer.md`；落地流程见 `.claude/skills/day0-inference/reference/adapt/golden-adapter/golden-worker-adapter.md`；类型 0-5 代码模板见 `.claude/skills/day0-inference/reference/adapter-templates.md`。
>
> **核心命题：最大剩余风险不是「module 判定错」，而是「module 根本没被枚举进判定表」。** 双源枚举交叉验证（§2.1.7）+ OOT 替换自验（ascend-oot.md 附录 C）合起来才构成完整自检。
>
> 本文覆盖两面：**控制面**（第一章）：管线与执行流、并行策略约束、权重加载——决定模型「如何被实例化与驱动」；**数据面**（第二章）：逐 module 判定流水线、attention backend 分派、各注意力形态的 NPU 实现要点、MoE 与缺算子处置、ACLGraph 图模式兼容——决定每个算子「落在哪条 NPU 实现路径上」。
>
> 文档族其他成员：
>
> - `.claude/skills/day0-inference/reference/adapt/golden-adapter/golden-adapter.md` —— 分层适配总纲（索引）
> - `.claude/skills/day0-inference/reference/adapt/golden-adapter/golden-service-adapter.md` —— 服务层落地流程（机制知识库见 `.claude/skills/day0-inference/reference/golden-service-knowledge.md`）
> - `.claude/skills/day0-inference/reference/golden-schedule-knowledge.md` —— 调度层机制知识库（含 EngineCore E1-E12 配置清单；落地流程 `.claude/skills/day0-inference/reference/adapt/golden-adapter/golden-schedule-adapter.md`）
> - `.claude/skills/day0-inference/reference/ascend-oot.md` —— OOT 机制与六类适配类型总述
> - `.claude/skills/day0-inference/reference/adapter-templates.md` —— 类型 0-5 代码模板 A-F
> - `.claude/skills/day0-inference/flows/golden_flow.md` —— Stage 1 流程与 G0-G4 门禁
>
> ⚠️ **行号防腐声明**：文中 `文件:行号` 引用基于撰写时的基线版本，随版本更新行号会漂移——**函数名/类名是锚点，行号只是辅助**。行号失效时按函数名 grep 重新定位。
>
> ⚠️ **术语纪律**：**P0/P1/P2 只用于模型级路径判定**（Q0 前置规则与规模分诊）；**module 级只用类型 0-5**。两者通过 §2.1.2 的对应表关联，不混用、不并列标注在同一 module 行上。
>
> ⚠️ **Stage 范围**：§2.5 ACLGraph 属 **Stage 3**，Stage 1 不加载。
>
> ⚠️ **单一事实源**：每个事实只在本文档详述一次；与文档族其他成员重叠的内容一律用引用，不复制整段。

---

## 一、控制面：管线、并行与权重加载

### 1.1 四级管线与执行流

vllm-ascend 的执行路径为四级结构，**初始化顺序严格为 Platform → Worker → Runner**：

1. `NPUPlatform`：配置校验、全局 patch 与 CustomOp 注册（dispatch key 为 `PrivateUse1`）——机制详述见 `.claude/skills/day0-inference/reference/ascend-oot.md`。
2. `NPUWorker`：每设备初始化、模型加载与执行委托。其启动序列为：
   `adapt_patch()`（兼容性 patch）→ `register_ascend_customop()`（自定义算子注册，**含图捕获所需的 dummy 融合算子**）→ CPU 绑核（A2/310P/A5 按 PCIe 拓扑的 `TOPO_AFFINITY_MODE`，A3 用全局均分的 `GLOBAL_SLICE_MODE`）→ CaMemAllocator 睡眠管理。
3. `NPUModelRunner`（继承上游 `GPUModelRunner`）：`execute_model()` 六步流程——
   ① `_prepare_inputs()` 组织连续 buffer → ② `_sync_metadata_across_dp()` 做 DP 对齐 → ③ 构建 `AscendCommonAttentionMetadata` 并由各 backend 的 `MetadataBuilder` 下沉 per-layer metadata → ④ `set_ascend_forward_context()` 选定 MoE 通信方式等运行时参数 → ⑤ `_model_forward()`（图模式下由 `ACLGraphWrapper` 拦截重放）→ ⑥ `sample_tokens()` 由 `AscendSampler` 完成采样。

**新 backend 的验收点**：新模型的 attention `MetadataBuilder` 必须正确输出五态状态机（`PrefillNoCache` / `PrefillCacheHit` / `DecodeOnly` / `ChunkedPrefill` / `SpecDecoding`）以驱动图模式选择。五态输出错误不会让模型立刻崩，但会让图模式选错捕获档——这是新 attention backend 落地时必须显式验证的一项。

### 1.2 `_dummy_run` 三职责

`_dummy_run` 在 Runner 上承担三类职责：

1. **profile_run**：测峰值激活显存，供 KV cache 预算反推；
2. **编译触发**：触发 torch.compile 编译路径；
3. **图捕获**：按 capture batch size 列表逐个捕获 ACLGraph。

其调用计数是可机器校验的（计数不符即意味着捕获/回放链路被意外修改），**计数三要素（warmup 公式、DP 对齐 padding、A3+DeepSeek 额外 warmup）的唯一详述处在 `.claude/agents/performance.md`（Stage 3 图模式验收素材）**，本文档不复制公式（§2.5 仅保留一段备查引用）。图模式验收不在 Stage 1 范围（Stage 1 全程 eager）。

### 1.3 显存预算陷阱

NPUWorker 历史上**未把图捕获显存计入 KV cache 预算**：图内存与 KV cache 静默竞争同一 `gpu_memory_utilization` 配额，可能导致 OOM 或 KV cache 小于预期（RFC #8260）。v0.21.0rc1 起已在 KV cache 分配前加入 ACL graph 内存预估。

**Stage 3 开图前验证纪律**：必须显式确认该预估逻辑对本模型的 capture sizes 生效——预估按捕获档计算，新模型若引入新的 capture 形态（如投机解码 1+k 查询长度），预估漏档会直接表现为 KV cache 容量缩水。（Stage 1 全程 eager，无图捕获显存，不涉及本项。）

### 1.4 并行策略约束（fail-fast 校验族）

并行策略的适配本质是一组**整除与互斥约束的 fail-fast 校验**——不合法组合在配置期直接报错，属预期行为：

- **DCP**（Decode Context Parallel，KV cache 沿序列维分片）：
  - MLA 模型：要求 `tensor_parallel_size >= decode_context_parallel_size` 且整除；
  - GQA 模型：走 head-sharding，要求 `num_q_per_kv % dcp_size == 0`，且 `max_dcp_size = TP // kv_heads`；
  - KV 传输场景：须设 `cp_kv_cache_interleave_size == block_size`。
- **PCP**（Prefill Context Parallel）：**仅 ModelRunner V2 支持**（v0.26.0rc1 已从 MRV1 移除）；与 DCP 互斥；PD 分离中仅 prefill 引擎可开；PCP 下投机解码组合受限——草稿采样必须 greedy、全图仅限 `FULL_DECODE_ONLY`。
- **EPLB**（Expert Parallelism Load Balancer）接入新 MoE 模型的三件事：
  1. 在 `vllm_ascend/eplb/adaptor/vllm_adaptor.py` 注册模型类型（标准化 `num_dense_layers`、`global_expert_num` 等策略参数）；
  2. 通过 expert map 与专家权重一致性校验；
  3. 量化格式白名单：W4A16 / W4A8MXFP 被拒绝；A5（Ascend 950）不支持 MXFP4 系 EPLB；A2 不支持冗余专家。
- **已移除的旧优化路径**：v0.26.0rc1 移除了 layer sharding、FlashComm2、weight prefetch，新模型适配**不得依赖**这些开关。

### 1.5 权重加载适配（加载期视角）

> ⚠️ **运行期 vs 加载期之辨**：第二章的逐 module 判定全是**运行期**视角（这个层怎么算、用什么 backend）；本节是**加载期**视角（checkpoint 里的权重名怎么落到 vLLM 的参数上）。两者独立——一个 module 判定为「类型 0 零适配」，它的权重名如果对不上，照样加载失败。**跳过本节的典型后果：`safe_open` 报 missing/unexpected keys，或权重静默加载错误导致精度错。**

#### 1.5.1 必查命令三件套

对每个需要加载权重的 module 都要做（重点是新注意力类）：

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

**权威原则**：权重名以 safetensors index 为准，不以 modeling 文件里的字符串为准——后者可能只是拼字符串的中间形态。

#### 1.5.2 四类映射差异（按危险程度排序）

| 类型 | 示例（真实案例） | 后果 | 处理 |
|---|---|---|---|
| **融合打包** | 厂商三套 `q_proj/k_proj/v_proj` → vLLM packed `in_proj_qkvgfab` | 名字对不上 → missing keys | 写 `_stacked_params_mapping` 或 loader 重排 |
| **子模块并入** | 厂商 `kv_b_proj` 被 vLLM 在加载后吸收成 `W_UK_T`/`W_UV`（DeepSeek 系 MLA） | vLLM 参数名 ≠ 厂商名 | 确认 `process_weights_after_loading` 正确处理 |
| **旧版兼容** | 厂商 `A_log` 存 4D `(1,1,H,1)`，vLLM 期望 1D `[H]` | 形状断言失败 | loader 里做 `.view()`（vLLM 已有 `a_log_weight_loader`） |
| **命名拼写** | `conv1d.weight` 是否需要 `.unsqueeze(1)`；`dt_bias` 初始化差异 | 加载成功但权重形状错 → 精度错 | 对照 vLLM 同族层（GDN/Mamba）的 loader |

**判定规则**：把「厂商权重名集合」与「vLLM 层参数名集合」的 **missing / unexpected 两组**列进设计文档的判定表，作为每个 module 的一行。**不允许出现"名字看起来对、没实际验证"**——用 §2.1.3 速查表的 `❌/⚠️/✅` 语义标记。

#### 1.5.3 自检口径（加载期）

```bash
vllm serve <model> --load-format safetensors 2>&1 | grep -E "not initialized|size mismatch|shape mismatch"
```

出现任一项都是阻断项，回 Developer 修复 loader 再放行。匹配文案随 vLLM 版本变化（当前版本实测缺失输出为 `Following weights were not initialized from`），**以当前安装版本实测校准**；`Unexpected extra config keys` 属配置项校验，与权重缺失无关，**不作阻断项**。该口径即 golden_flow.md Phase 4 / G3 精度门禁的加载证据要求（见第三章）。

#### 1.5.4 vLLM 侧映射实现要点

- **`packed_modules_mapping`**：`load_weights` 必须正确实现合并映射——`qkv_proj`（q/k/v 合一）、`gate_up_proj`（gate/up 合一）、`experts`（逐专家参数并入 `w13_weight`/`w2_weight`）。**量化模型还需把 `model_type` 加入 `vllm_ascend/quantization/modelslim_config.py` 的 `packed_modules_model_mapping`**，保证分片一致——漏掉这步的典型症状是分片维度错位、加载不报错但精度全错。
- **量化格式自动检测**：checkpoint 存在 `quant_model_description.json` 即判定为 ModelSlim 体系，显式 `--quantization` 优先级最高（PR #6645 起不再要求显式指定）——机制详述见 `.claude/skills/day0-inference/reference/ascend-oot.md`。
- **NZ 布局约束**：NPU 高性能 matmul 需要 `FRACTAL_NZ` 布局。A5（Ascend 950）的 MX 量化（W8A8 MXFP8 / W4A4 MXFP4）**强制要求**权重经 transpose + contiguous + `npu_format_cast` 转 NZ；经 SFA 融合预处理（MLAPO / PROLOG_V3）的层须**避免二次 NZ 转换**。
- **弃用路径**：v0.26.0rc1 对 W4A8 linear、W4A8 MoE per-group、W8A8 PDMix MoE 路径加了弃用警告，新模型量化不应再走这些路径。

---

## 二、数据面：逐 module 落地（判定流水线）

### 2.1 执行方式总述

数据面不给静态结论——它的执行方式是：**对模型用到的每个 module 走一遍判定流水线**，逐一定型、逐一落到类型 0-5 的适配动作上。流水线共六步，按序执行：

```
① 模型级路径判定（P0/P1/P2 + Q0 前置规则）
→ ② 路径与类型 0-5 的对应
→ ③ 速查表快通道（17 类标准 module 打勾）
→ ④ 决策树（速查表未全命中时逐 module 走树）
→ ⑤ 判定 A/B/C 操作命令（找层 / 认机制 / 查注册）
→ ⑥ 产出判定表 + 枚举完整性交叉验证
```

#### 2.1.1 三条适配路径（成本递增，优先选成本低的）

| 路径 | 机制 | 触发点 | 成本 | 适用 |
|---|---|---|---|---|
| **P0 零适配** | 复用 vLLM 模型代码 + 已注册的 Ascend CustomOp | `CustomOp.__new__` 自动拦截 | 0 | 模块结构与已支持模型一致 |
| **P1 补丁/小新增** | `patch/worker/patch_<model>.py`，或新增单个 CustomOp/PluggableLayer OOT 实现 | `NPUWorker.__init__` → `adapt_patch()`；`register_ascend_customop()` | 低 | 少数方法有 CUDA 假设 / 缺个别算子 |
| **P2 重写** | `models/<model>/` 完整实现 + `ModelRegistry.register_model` | `vllm.general_plugins` | 高 | 架构级差异 / 需自带算子 / 上游分派不含本平台 |

**判据**：

- 改动 ≤ 3 个方法且不涉及层间数据流 → **P1**；
- 需要新算子内核、新 attention backend、改变层组装方式、或 vLLM 上游无该架构 → **P2**。

**Q0 前置规则**（模型级平台分派检查，同一模型只问一次）：若上游 `vllm/models/<model>/__init__.py` 的分派只按 `is_rocm()` 二分，NPU 会静默落入 nvidia 分支、import CUDA 专属代码直接崩——**Q0 命中（上游分派不含 NPU）→ P2 至少为 1**（覆盖注册本身是 P2 工作量）。但**判 P2 ≠ 全量重写**：若选定的基线分支大部分可经注册表复用，`models/<model>/` 可以只是一个**薄覆盖层**。Q0 的基线分支选择方法（逐个评估厂商分支、不默认选 nvidia）见 `.claude/skills/day0-inference/reference/ascend-oot.md`。

**规模分诊**（逐 module 判定完成后，按类型 5 module 的个数决定落地形态；Q0 未命中时表中数值即为总规模）：

| 类型 5 module 数量 | 落地形态 |
|---|---|
| 0 个 | 纯补丁：`patch/worker/patch_<model>.py` |
| 1-2 个 | 补丁 + 局部新增：patch + 若干 `ops/<op>.py` |
| ≥ 3 个 | 建 `models/<model>/` 完整目录，走 MiniMax-M3 路线 |

#### 2.1.2 路径与适配类型的对应

| 类型 | 含义 | 路径 |
|---|---|---|
| 类型 0 | 零适配（已注册，自动替换） | P0 |
| 类型 1 | 新增 CustomOp OOT 实现（覆写 `forward_oot`） | P1 |
| 类型 2 | 新增 PluggableLayer OOT 实现（覆写 `forward`） | P1 |
| 类型 3 | monkey patch（无法走注册表） | P1 |
| 类型 4 | 扩展现有 Ascend 实现（参数/分支被静默丢弃） | P1~P2 |
| 类型 5 | 全新结构（算子 + 可能的 backend + KV cache spec） | P2 |

> **Q1' standalone 重写**：决策树 Q1' 判定「上游层不可 import」时，虽上游有对应层但无法继承——行为等价于类型 5，按类型 5 的模板处理，但可参考上游的平台中立基类设计。参照 `vllm_ascend/models/deepseek_v4.py`。

类型 0-5 的代码骨架（模板 A-F）见 `.claude/skills/day0-inference/reference/adapter-templates.md`。

#### 2.1.3 快通道：标准 module 速查表

新模型到手，**先对照这张表逐行打勾**。全部命中「默认判定 类型 0」时可直接跳过 §2.1.4 的决策树，只走 EngineCore 清单（E1-E12 见 `.claude/skills/day0-inference/reference/golden-schedule-knowledge.md`）。

「机制」列决定适配时**该覆写哪个方法**（见 §2.1.5 判定 B）：CustomOp → `forward_oot`，PluggableLayer → `forward`。**写错不报错，只静默不生效。**

> 本表的「默认判定」列按术语纪律以类型 0-5 标注（映射关系：P0↔类型 0、P1↔类型 1-4、P2↔类型 5，见 §2.1.2）。

| # | Module | 机制 | vllm-ascend 现状 | 默认判定 | 需要适配的触发条件 |
|---|---|---|---|---|---|
| 1 | Embedding（`VocabParallelEmbedding`） | PluggableLayer | ✅ 已 OOT | **类型 0** | 词表并行切分非标准 / 多模态 embedding 融合 |
| 2 | RMSNorm / LayerNorm | CustomOp | ✅ 已 OOT（`AscendRMSNorm`、`GemmaRMSNorm`、`RMSNormGated`） | **类型 0** | 归一化公式变体（加 scale/offset）；**norm 位置改变**（QK-norm、sandwich norm）可能连带影响 attention backend |
| 3 | QKV Linear（`QKVParallelLinear`） | PluggableLayer | ✅ 已 OOT | **类型 0** | QKV 融合方式不同（如 MLA 双低秩） |
| 4 | Rotary Embedding | CustomOp | ✅ 已 OOT（标准/M-RoPE/YaRN/Deepseek-scaling） | **类型 0** | 新位置编码算法（3D-RoPE、可学习插值）。**NoPE**（不做旋转）**适配位置在模型层而非 rotary 算子**——上游共享层（如 `MultiHeadLatentAttentionWrapper`）不含 `use_nope` 参数，由模型侧自有 attention 类处理；但需确认 Ascend MLA 在 nope 路径下不会去取 rope cos/sin cache（`ops/mla.py` 接收 `rotary_emb`） |
| 5 | Attention 核心 | backend 分发 | ✅ 5 个后端（Dense/MLA/SFA/DSA/FA3） | **类型 0** 若命中既有查表键 / **类型 5** 若不匹配 | 新注意力算法（稀疏、gating、output gate）。注：**层封装**（如 `MultiHeadLatentAttentionWrapper`）走 PluggableLayer 替换（OOT 覆盖清单见 `.claude/skills/day0-inference/reference/ascend-oot.md`），**backend** 走 `get_attn_backend_cls()` 分发，两者是不同层次 |
| 6 | **KV Cache 布局** | spec 注册 | ✅ 标准 + MLA 系（3 个 spec）<br>⚠️ 线性态复用上游 `MambaSpec` + `patch_mamba_*` | 标准/MLA → **类型 0**<br>与 Mamba spec 一致 → **类型 1~4**<br>需新增 spec → **类型 5** | 非常规 cache。选型判据见 golden-schedule-knowledge.md 的 E4；page 对齐见其 E5 |
| 7 | MLP / FFN | PluggableLayer | ✅ Linear 全系已 OOT | **类型 0** | 新激活函数 |
| 8 | 激活函数 | CustomOp | ✅ SiluAndMul / QuickGELU / SiluAndMulClamp | 命中 → **类型 0** / 新增 → **类型 1** | 新激活（加 CustomOp）。⚠️ **MoE 路径需单独加分支**，否则落 `else` 兜底被当 SwiGLU 静默算错 |
| 9 | MoE 路由 + 专家 | 工厂函数 patch | ✅ `ops/fused_moe/` + EPLB（noaux_tc 经 sigmoid + correction_bias 组合支持，语义级非字符串级；sigmoid / grouped_topk 原生支持） | **类型 0** | 路由算法变体、共享专家、**latent MoE**（专家维度 ≠ hidden_size，需确认 transform 被调用） |
| 10 | LM Head / Logits | PluggableLayer | ✅ 已 OOT | **类型 0** | 词表裁剪、多头输出 |
| 11 | **量化** | quant config | ✅ ModelSlim / compressed-tensors / FP8 / W8A8 系<br>⚠️ MXFP4 有**两道门**（版本门 + A5 融合门），详见 golden-schedule-knowledge.md 的 E2 | 格式已支持**且两道门均通过** → **类型 0** | 新量化格式或新 ignore 规则 |
| 12 | 多模态 ViT | 混合 | ⚠️ 部分（`MMEncoderAttention` 已 OOT） | 先查 `patch/worker/` 有无同名塔的 patch：**有 → 类型 3**（验证签名/逻辑一致后可降为类型 0）；**无 → 按决策树判定（类型 1~3）** | ViT 位置编码、patch merger。即使同名类，不同模型的 ViT 架构（如 2D vs 3D patch embed）可能签名不同，**不可直接判类型 0** |
| 13 | 投机解码 / MTP | 模型注册 | ✅ `spec_decode/` | **类型 0** / 层结构特殊 → **类型 1~4** | MTP 层结构特殊 |
| 14 | **线性注意力 / Mamba / GDN** | PluggableLayer | ✅ `AscendGatedDeltaNetAttention`、`AscendBailingMoELinearAttention`；Triton kernel 在 `ops/triton/{fla,mamba,kda}/` | 已有变体 → **类型 1~4** / 新算法 → **类型 5** | 混合架构已是主流。注意：**kernel 存在 ≠ 已接线**（示例：KDA kernel 已有但无层调用）；上游层若无 `@PluggableLayer.register` 则只能 patch 或 standalone 重写（见决策树 Q1'） |
| 15 | **跨层状态传递** | — | ❌ 无通用支持 | **类型 3~5** | decoder layer 间传递 `hidden_states` 以外的张量（如块级残差累积）。**直接阻塞 ACL Graph 捕获**，且影响 PP 切分——机制说明见 §2.1.4 决策树特殊分支，配置落点见 golden-schedule-knowledge.md 的 E7 |
| 16 | 归一化位置变体 | CustomOp | ✅ 算子已有 | 仅位置变化 → **类型 0** / 需融合 → **类型 1** | QK-norm、sandwich norm。算子复用无问题，但可能改变 attention backend 的融合假设 |
| 17 | **层类型混合派发** | 模型骨架 | ✅ 有先例（Qwen3-Next full+linear 混合） | **类型 5** | 同模型内按 `layer_idx` 派发多种 attention/层类型（示例：24 层 MLA + 69 层 KDA）。**连带计数**：层类型混合本身类型 5，且几乎必然带来混合 KV cache（#6，+1 个类型 5）和/或新 attention backend（#5，再 +1），实际类型 5 ≥ 2。若同时有跨层状态传递，见决策树「特殊」分支 |

#### 2.1.4 判定决策树（速查表未全命中时，对每个 module 走这棵树）

> 前提：模型级前置检查 Q0（平台分派覆盖）已完成。以下决策树是**逐 module** 的。

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
│  └─ 【类型 5】全新实现 + 图模式 piecewise（跨层状态直接阻塞
│     ACL Graph 全图捕获，需将状态传递点加入 splitting_ops；
│     配置落点见 golden-schedule-knowledge.md 的 E7）
│     默认判类型 5（对应模型级 P2）。若跨层状态仅是简单的标量门控
│     （如 1 个 Linear + sigmoid），可按类型 3 处理（对应模型级 P1）。
│     参考：某些混合模型的块级残差机制（如 attn_res_block_size）（示例）
└─
```

**跨层状态为何特殊**（E7 机制说明）：静态图要求每层输入输出形状固定，跨层累积的张量破坏了这一点；且在 PP 切分时需要额外的跨 stage 通信。处理顺序：先 `enforce_eager` 跑通正确性，再评估能否 piecewise 捕获（把状态传递点加进 `splitting_ops`，见 §2.5）。

#### 2.1.5 判定 A/B/C 的操作命令

**判定 A：上游有没有等价层**

```bash
# 按类名找
grep -rn "^class <ModuleName>" $VLLM/vllm/model_executor/
# 按功能找（如激活函数）
grep -rn "@CustomOp.register\|@PluggableLayer.register" $VLLM/vllm/model_executor/layers/
```

**判定 B：是 CustomOp 还是 PluggableLayer**

两者的 `register_oot` 写进**同一个全局字典** `op_registry_oot`（上游 `vllm/model_executor/custom_op.py:22`，行号会漂移，以字典名为锚点），注册方式一样，但**实现时覆写的方法不同**：

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

注册表在 `vllm_ascend/utils.py` 的 `REGISTERED_ASCEND_OPS`（撰写时约 `:813`，行号会漂移，以变量名为锚点）。注意 **key 是类名**（`"RMSNorm"`），不是上游的注册名（`"rms_norm"`）——因为 `__new__` 里用的是 `cls.__name__`。

#### 2.1.6 逐 module 判定表模板

对新模型做适配评估时，产出一张判定表，列定义：**Module（厂商代码）| 上游等价层 | 机制 | Ascend 现状 | 判定（类型 0-5）| 工作量**。

- **P 级不进表**：P0/P1/P2 只用于模型级路径判定（§2.1.1 与 golden_flow.md 的 Phase 0），module 行的路径信息已由类型 0-5 经 §2.1.2 的对应关系承载。
- ⚠️ **「上游等价层」以 Q0-② 选定的基线分支为准**。不同厂商分支的同一 module 可能落在不同类型——基线选错，整张判定表跟着错。典型：某分支用带 `@PluggableLayer.register` 的**共享层**（→ 类型 2，注册替换），另一分支用**私有类**（→ 类型 3，monkey patch）。**填表前先在表头注明基线分支名。**

#### 2.1.7 枚举完整性交叉验证（防「漏列没进判定表」）

> ⚠️ 最大剩余风险不是「module 判定错」，而是「**module 根本没被枚举进判定表**」——漏列了，后面所有类型 0-5 判定都覆盖不到它。OOT 自检（`Instantiating custom op / pluggable layer` 日志）能检测「替换没生效」，但**检测不了「没被列进表的东西」**。

**从三个独立来源枚举 module，交叉比对去重**：

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
4. 对**非主流字段**（config 里名字不带 `num_`/`hidden`/`head` 的，如 `attn_res_block_size`、`e_score_correction_bias`、`routed_expert_hidden_size`、`mla_use_output_gate`、`use_full_rank_gate`）**逐个**追问「这个字段在 forward 里被用了吗？在哪个分支？」——这是角落结构（attention residual、latent MoE、输出门）最可能藏身的地方。

**把交叉验证结果写进设计文档**：判定表加一列「枚举来源（config/modeling）」，标注每个 module 来自哪个来源、是否两源一致。OOT 替换自验保证「替换生效」，枚举完整性保证「枚举完整」——两个合起来才构成完整自检。

### 2.2 Attention backend 分派

`NPUPlatform.get_attn_backend_cls` 以 `(use_mla, use_sparse, use_compress)` 三元组为键查表分派；用户显式指定的 `attention_config.backend` 在 Ascend 上会被强制重置为插件后端。

**Attention backend 分派表（vllm-ascend main）**：

| 分派键 (use_mla, use_sparse, use_compress) | 后端类 | 适用模型 | 图支持级别 |
|---|---|---|---|
| (False, False, False) | `AscendAttentionBackend`（attention_v1） | Llama/Qwen 等 GQA/MQA | ALWAYS |
| (True, False, False) | `AscendMLABackend`（mla_v1） | DeepSeek-V2/V3/V3.1、Kimi K2 | UNIFORM_BATCH |
| (True, True, False) | `AscendSFABackend`（sfa_v1） | DeepSeek-V3.2、GLM-5.x | UNIFORM_BATCH |
| (True, False, True) | `AscendDSABackend`（dsa_v1） | DeepSeek-V4（DSA+compressor） | UNIFORM_BATCH |
| 分支：FA3 | `AscendFABackend`（fa3_v1） | 训推一致 RL（需 `flash_attn_npu_v3` + rl_config 开关） | — |
| 分支：310P | `AscendAttentionBackend310` | COMPATIBILITY 硬件档案（MLA/SFA 未支持） | — |
| 分支：PCP | `pcp_backend_map`（同四类后端） | `use_pcp=True` 时经 `get_builder_cls()/get_impl_cls()` 动态解析 CP 变体 | 不支持组合直接 NotImplementedError |

读表要点：

- **图支持级别是关键读数**：分派键越复杂，图支持越保守。GQA 基线为 `ALWAYS`；MLA/SFA/DSA 三类潜变量/稀疏后端仅声明 `UNIFORM_BATCH`——只有均匀 decode batch 可进全图，**混合 batch 自动降级 piecewise**。
- **特征键分派而非按模型名分派**：特征键由上游 attention selector 从模型 config 推导（如 hf_config 含 `index_topk` 即 `use_sparse=True`）。直接收益：新模型若其注意力特征组合已落在既有键上（MLA + `index_topk` 自动命中 SFA 键），backend 层**零代码**——这正是 GLM-5 零代码适配的机制基础。
- **新注意力形态的扩展步骤**：在查表中扩展键或新增 selector 探测函数（如 `model_uses_sfa_sparse`），并**同步注册新 cache spec**（spec 选型与注册见 `.claude/skills/day0-inference/reference/golden-schedule-knowledge.md` 的 E4）。
- **310P 盲区**：310P 分支仅支持标准 attention，稀疏/潜变量注意力在低端硬件档案上是适配盲区，立项时须显式排除。

### 2.3 各注意力形态的 NPU 实现要点

**MLA**：采用 absorbed 路径——加载后 `process_weights_after_loading` 把 `kv_b_proj` 拆为 `W_UK_T`/`W_UV` 并做 NZ 转换，decode 用 FIA v2 直接在 latent cache 上计算；MLAPO（`npu_mla_prolog_v3`）把 q 下投影、RMSNorm、RoPE、KV 写 cache、上投影吸收融合为一次调用，W8A8 模型默认开启。两个变体要点：Kimi K3 的 MLA 为 NoPE（无 RoPE），对应 `_npu_mla_prolog_v3_no_rope` 路径与 96 头去 padding 等专项；`qk_rope_head_dim==0` 的模型因 FIA 要求 rope 维度为 64，实现上喂入全零 rope 操作数变通。

**SFA**：在 MLA 基础上增加 Lightning Indexer，indexer 作为独立 KV cache spec 拥有独立 k_cache，一次调用完成 k 路径计算 → cache 写入 → top-k 选择。稀疏 kernel 按硬件分代：A5 走 `sparse_flash_mla` + 设备端 metadata 构建（避免 host 同步），A2/A3 走 `npu_sparse_flash_attention`。GLM-5.2 的 IndexShare（约每 4 层共享一个 indexer；另一口径为 78 层中 21 层计算、57 层复用——两种计数口径并存，机制一致）由 `skip_topk` + `topk_indices_buffer` 机制承接，shared 层不持运行时 indexer cache。

**DSA**（DeepSeek V4）：compressor metadata 由 `torch.ops._C_ascend.compressor_metadata` 计算，并在 forward context 上 memoize——**跨层只算一次**（这是图模式下的跨层状态正面案例，对比 §2.1.4 特殊分支的阻塞形态：memoize 在 capture 前完成即可入图）。

**GDN/KDA**（线性注意力）：NPU 映射依赖一组新增算子——GDN 侧有 `causal_conv1d_fn`、`npu_recurrent_gated_delta_rule`、950 定制的 `fused_gdn_gating`；KDA（Kimi K3，69 KDA + 24 NoPE MLA 混合）侧有 stride-aware `recurrent_kda`、`fused_norm_gate`、KDA QKV 投影融合与 AscendC `AttnResFwd` 等。

### 2.4 MoE 与缺算子处置

#### 2.4.1 MoE 通信方式（EngineCore E8 的详述处）

MoE 通信由 `select_moe_comm_method()`（`vllm_ascend/ascend_forward_context.py`，撰写时约 `:344`，行号会漂移）按 batch 特征在四种 `MoECommType` 间选择：**AllGather**（小 EP 通用）、**All2All**（DP>1 更优）、**MC2**（EP≥16 融合通信，`npu_moe_distribute_dispatch/combine`）、**FusedMC2**（dispatch+FFN+combine 单 kernel）。

按 SoC 的分发逻辑（⚠️ 下表是 `_select_a2/a3/a5_moe_comm_method` 的逻辑快照，**以源码为准**，代码更新后表可能过时，使用前应 grep 确认）：

| SoC | MC2 条件 | 说明 |
|---|---|---|
| A2 | `num_experts / ep_world_size <= 24` 且 `ep_world_size >= 16` 且 `num_tokens <= mc2_tokens_capacity` | 大专家数模型需要很大的 EP 才能走 MC2，否则掉回 ALLGATHER（性能差） |
| A3 | `enable_fused_mc2` 时 EP≤64 走 FUSED_MC2（需 CANN mega_moe 支持）；EP≤32 走 dispatch_ffn_combine | 不看专家数，只看 EP size |
| A5 | **首选 MC2**：`num_tokens <= mc2_tokens_capacity` 且 `world_size > 1`<br>fallback：`world_size <= num_experts_per_tok` → ALLGATHER，否则 ALLTOALL | 与 A2/A3 不同，A5 不看专家数 |
| 310P | 固定 ALLGATHER | |

**部署前必算**：用目标 EP size 代入公式，确认落在哪个分支。注意 A2/A3/A5 的 MC2 路径还需满足 `num_tokens <= mc2_tokens_capacity`（token 量超出容量时 A2 掉回 ALLGATHER、A3 掉回 ALLTOALL）。（调度层配置清单的 E8 行指向本节，见 `.claude/skills/day0-inference/reference/golden-schedule-knowledge.md`。）

#### 2.4.2 新路由形态：hash 路由与 swiglu_limit

DeepSeek V4 的 hash 路由（sqrtsoftplus + `tid2eid` 表）由新 AscendC 算子 `moe_gating_top_k_hash` 承接（PR #9228 随 DSV4 算子包引入）；视觉/hash 混合路由另有 `bias_vl` 动态路径。routed SwiGLU limit 映射为 `gate.clamp_(max=limit)`、`up.clamp_(±limit)` 后接 `npu_swiglu`——**精度先例**：v0.26.0rc1 修复了 DSV4 Flash/Pro 该路径的精度问题（#14397），新模型用到 clamp 类路由后处理时应以该 PR 的口径对齐数值。

#### 2.4.3 缺算子处置：三级技术下沉 + L4 工程拆分纪律

| 级别 | 路径 | 触发条件 | 实证案例 | 代价与限制 |
|---|---|---|---|---|
| L1（技术下沉） | 能力探测 + torch/aclnn 回退 | NPU 融合算子约束不满足（如 renorm 组合、group 配置越界） | `check_npu_moe_gating_top_k()` 不满足时回落上游 PyTorch topk | 性能损失但正确性保底；须保留双路径开关 |
| L2（技术下沉） | Triton 过渡 kernel | aclnn 无对应算子，需快速跑通通用路径 | KDA decode、swiglustep；v0.26.0rc1 中 Triton SwiGLuStep 反向替代旧 AscendC `fused_gdn_gating` | 依赖 triton-ascend；性能不及原生 AscendC |
| L3（技术下沉） | 新 AscendC 算子 | 性能敏感且长期存在的新结构 | `csrc/<op>/{op_host, op_kernel}` 结构 + `torch.ops._C_ascend.*` 注册（`moe_gating_top_k_hash` 等） | 必须注册 meta 实现才能被 ACLGraph 捕获（见 §2.5） |
| L4（工程拆分纪律） | 算子先行、模型随后 | 算子包与模型集成可解耦时 | DSV4.1 算子 PR #16422 与模型集成 PR #16423 拆分合入 | 需要 RFC 级排期；Day0 可先靠外挂算子包跑通 |

结构本质：L1–L3 是「正确性优先、性能下沉」的三级技术路径——L1 保证任何新模型都有可运行的基线，L2/L3 逐级把热点算子下沉到更贴近硬件的实现；L4 不是又一级技术路径，而是**工程拆分纪律**，把算子就绪度从模型集成的关键路径上拆出（三次 Day0——V3.2、V4、K3——的阻塞点均在 CANN/torch_npu 算子缺口而非插件代码本身）。

**排期铁律：算子兼容性扫描必须先于一切模型代码工作**；纯 CUDA 且无回退路径的算子直接阻塞立项，向上游提 issue，而不是带病开工。

### 2.5 ACLGraph 图模式兼容（⚠️ 属 Stage 3，Stage 1 不加载）

ACLGraph 是捕获+回放机制。以下六类操作**不可入图**，构成新模型 forward 路径的静态扫描清单：

1. **stream 同步及隐含同步 memcpy**——典型案例：FULL_DECODE_ONLY + MTP 下 `aclnnKvRmsNormRopeCache` 在回放上下文做同步 `rtMemcpy` 直接崩溃，错误码 **107030**；
2. **event 状态查询**；
3. **aclop 算子**（捕获时申请显存）；
4. **host 侧 tiling 依赖算子**——attention 即属此类，这是 piecewise 把 attention 排除出图的根本原因；
5. **控制流分支**；
6. **full-graph 区域内任何 Python 副作用**——含 `logger.debug`：曾有一行日志导致 full-graph dynamo graph break、全部 TP worker 崩溃的先例（issue #10328）。

另：**新自定义算子必须注册 meta 实现才可被捕获**——否则问题到图模式阶段才暴露，回退成本最高。

**piecewise 模式**：按 `splitting_ops` 切图（`vllm::mla_forward`、`vllm::dsa_forward` 已在默认列表），attention 段 eager 执行、其余子图捕获重放。

**逐级开图门禁**：顺序强制 **eager → PIECEWISE → FULL_DECODE_ONLY**，每级独立验证，`--enforce-eager` 作为定界手段。**feature-first 原则**：EP + ACLGraph + FlashComm + MTP 默认全开验证，失败项保留证据而非默认关闭。eager 豁免口径（定界中以 `--enforce-eager` 起服务时捕获计数检查豁免）见 `.claude/agents/performance.md`（Stage 3 图模式验收素材）。

**新模型特有风险两则**：

- DeepSeek V4 的 PIECEWISE/breakable cudagraph 尚不支持，平台层强制禁用（**单一 issue 证据，后续版本可能变化**，使用前应核实当前版本状态）；
- MTP 多 token 输出受 FIA TND layout **每请求 ≤16 token** 约束，且 `actual_seq_lengths` 末元素必须等于 num_tokens。

> **交叉引用（备查，权威详述在 `.claude/agents/performance.md` 的 Stage 3 图模式验收素材）**：ACLGraph 捕获计数期望，对齐真实断言 `tests/e2e/pull_request/two_card/aclgraph/test_aclgraph_capture_replay.py`——
> `warmup_runs = 1 + 2 × 捕获 batch size 个数`（A3 且 DeepSeek 系额外 +1：MC2 warmup）；`padding_runs = ⌈total_steps/32⌉×32 − total_steps`（32 步全局对齐空跑）；`expected = (warmup_runs + padding_runs) × dp_size`。
> 三项易漏：DP 倍乘、对齐 padding、A3+DeepSeek 额外 warmup。本段仅备查，采集时点、日志 pattern 与豁免口径以 performance.md 为准。

---

## 三、与流程的衔接

| 流程阶段 | 本文对应内容 | 落点 |
|---|---|---|
| Phase 1 场景判定 | 本文每一章即一类适配场景：控制面（§1.1-1.5）+ 数据面判定流水线（§2.1-2.4）逐章判定，产出逐 module 判定表（§2.1.6，含加载期 missing/unexpected 两组与枚举来源列）；**§2.5 属 Stage 3，本阶段不判定** | Worker 层 design spec（`./.day0/<model>/design/worker-design-spec.md`） |
| Phase 2 落地实现 | 按 spec 判定表回查机制：类型 0-5 的实现动作（§2.1.2）与覆写点机制（§2.1.5）；代码骨架见 `.claude/skills/day0-inference/reference/adapter-templates.md` 模板 A-F | 实现产物落 `./.day0/<model>/impl/` |
| Phase 4 / G3 精度门禁 | 权重加载自检（§1.5.3 的 grep 口径） | 加载证据 |
| Stage 3 特性叠加 | 图模式兼容（§2.5：六类清单、meta 实现、逐级开图） | `.claude/agents/performance.md` 为权威详述；Stage 1 不做图模式验证 |
| 跨层联动 | MoE 通信分发表（§2.4.1）、cache spec 注册联动（§2.2）是调度层知识库 E8 / E4（`.claude/skills/day0-inference/reference/golden-schedule-knowledge.md`）的详述出处 | 调度层 design spec 的引用 |

依赖阻塞（上游未合入 / CANN 算子缺口）不是实现缺陷，不进修复回路：按 golden_flow.md Phase 0 的规定启动并行预案（外挂算子包 / 上游 pre-release 分支 / Triton 过渡），无回退路径则停止并提 issue。
