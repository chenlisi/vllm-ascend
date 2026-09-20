# 调度层适配方法论（golden-schedule-adapter）

> 分层适配总纲见 `.claude/skills/day0-inference/reference/adapt/golden-adapter/golden-adapter.md`；同族文档：服务层 `.claude/skills/day0-inference/reference/adapt/golden-adapter/golden-service-adapter.md`、Worker 层 `.claude/skills/day0-inference/reference/adapt/golden-adapter/golden-worker-adapter.md`、OOT 机制 `.claude/skills/day0-inference/reference/ascend-oot.md`。
>
> **本文档范围**：调度层（Scheduler / KV Cache 管理 / 投机解码调度 / PD 分离）的适配方法，分管理面（spec 定义、装配与配置校验）与数据面（运行时调度语义）两部分。
>
> ⚠️ **行号防腐**：文中 `文件:行号` 引用以函数名/类名为锚点，行号随版本漂移，失效时按函数名 grep 重新定位；不确定的行号不写。
>
> ⚠️ **单一事实源**：每个事实只在本文档详述一次；与其他新文档重叠的内容一律用引用，不复制整段。

调度层是四层适配（服务层 / 调度层 / Worker 层 / 算子层）中「定结构」的一层：服务层参数先行验证协议正确性，**KVCacheSpec 决定调度可行性**，Worker 层完成实例化，算子层收口性能与正确性。模型层适配完，调度层不改，模型能构造出来但跑不起来或跑错。

---

## 一、管理面：spec 定义、装配与配置校验

### 1. KVCacheSpec 是首要适配物

vLLM V1 调度层的核心抽象是**统一内存池 + KV Cache Group**：所有层类型共享同一 page size 的物理 block 池，按注意力类型分组（FullAttention / SlidingWindow / Mamba / MLA 等），**组内共享 block id、跨组做 prefix 命中求交**。每种新 cache 形态由一个 `KVCacheSpec` 子类描述，子类职责是声明 `page_size_bytes`、`type_id`、`merge()` 等；Ascend 特有 spec 经 `KVCacheSpecRegistry.register(kvcache_spec_cls, manager_class, uniform_type_base_spec)` 注册，注册入口为 `vllm_ascend/core/kv_cache_interface.py::register_ascend_kv_cache_specs()`。

**Ascend 已有 spec 清单**（复用优先级从高到低）：

- 上游 `FullAttentionSpec`（GQA/Full 基线）；
- `AscendMLAAttentionSpec`（MLA，支持 scale_dim/C8/DCP 分片）；
- `AscendSFAIndexerCacheSpec`（SFA indexer 独立缓存，与主 cache 共享 block id）；
- `AscendSlidingWindowMLASpec`（DSA+compressor 滑窗形态）；
- 上游 `MambaSpec`（线性注意力 recurrent state，GDN 先例，Ascend 侧配套 `patch_mamba_*` 系列）。

**复用上游 `MambaSpec` 的判据**：若 state 形状/生命周期与 Mamba 完全一致（按 seq 分配、不按 token 增长）→ 直接复用；若 state 有额外的 gate 调制 / conv 混合 / 非标准 page 计算 → 在 `register_ascend_kv_cache_specs()` 新增 spec + manager。

**选型错误的代价是显存量级的**：DeepSeek V3.2 的 KV cache spec 最初按 `FullAttentionSpec` 承接 DSA，按 2 倍 MLA page size 分配，浪费 38.8% 显存；PR #6610 重构为 `MLAAttentionSpec` 后，单机 A3 上 KV cache 容量从 15,872 tokens 提升到 25,984 tokens。

> **硬约束：KVCacheSpec 定稿前禁止进入任何性能工作。** block size、分组、prefix 命中语义全部由 spec 导出，事后更改 spec 会级联推翻调度与图模式配置（capture size、verify 步对齐、page 对齐断言全部要重来）。

### 2. 注意力形态 × cache 构成 × spec 承接 × 调度约束 × 投机附加项（全表）

新模型到手，先逐层确认注意力形态（GQA/Full / MLA / SFA / DSA / GDN/KDA 线性注意力），是否出现 indexer 独立缓存、recurrent state、多 cache group，再对照下表定位每一层的承接方式。2026 代模型把 KV cache 从「一种 tensor」变成「多种 state 的组合」：MLA 每 token 仅存 576 元素压缩 latent，GDN/KDA 持固定大小循环 state 不随序列增长，DeepSeek V4 单层同时带 c4a/c128a 压缩 KV、indexer cache、compressor state、滑窗 KV 五种 cache。

| 注意力形态 | cache 构成 | spec 承接（vllm-ascend） | 调度层约束 | 投机解码附加项 |
|---|---|---|---|---|
| GQA/Full | 每 token K+V | 上游 `FullAttentionSpec` | 基线；Ascend block_size 默认 128 | lookahead 预分配 + rejected 覆写即可 |
| MLA | 压缩 latent（576/token） | `AscendMLAAttentionSpec`（scale_dim/C8/DCP 分片） | 单头组，page 显著小于 GQA | verify 步 query 长度 1+k 进 capture key |
| GDN/KDA | 固定大小 conv+SSM/delta state | 上游 `MambaSpec` | `mamba_cache_mode=align` 才可 prefix cache；与 MTP 互斥项需显式检查（见 §二.6） | `num_speculative_blocks=k`，每请求多 k 个 state 槽 |
| SFA（DSV3.2/GLM-5.x） | MLA latent + indexer K/scale cache | `AscendSFAIndexerCacheSpec`（与主 cache 共享 block id） | indexer 组按 UniformType 归并共享 block id；IndexShare 组不跨 PP stage | MTP 层 `compute_topk=False` 复用共享索引 |
| DSA+compressor（DSV4） | 单层 5 种 cache | `AscendSlidingWindowMLASpec` 等 | 逻辑 block 固定 256 native token；block_size∈{32,64,128} | DSpark block size=5，`num_speculative_tokens<5` 非法（见 §二.7） |

**读法自下而上**：cache 形态越复杂，调度层自由度越小。选型定稿是 Phase 1 设计文档「调度层设计」章节的必备产出（见 §三）。

### 3. 跨 cache group page size 一致性（E5/E6）

统一内存池强制各组 page size 相同。混合模型中为对齐 mamba state，attention 侧的 block size 会被撑大：Qwen3.6-27B 实测统一 block size 达 784 token、Kimi K3 TP8 为 768、Qwen3.5-0.8B 为 544。必须评估这一撑大的副作用：

- **短请求**：block 变大后短请求浪费的 block 尾部空间变大；
- **prefix 命中率**：block size 撑大后短前缀永远够不到边界（配合 §二.6 的 align 语义，意味着 prefix cache 对短前缀静默失效）。

DeepSeek V4 的对策是「逻辑 block 固定为 256 个 native token，物理存储各 cache 自定」，把五种 cache 归并到少数 page-size bucket——新模型若 cache 形态多，优先考虑同类归并策略，而不是让统一 page size 被最大的 state 拖走。

**E6 对齐断言**：`patch/platform/patch_mamba_config.py` 里有强制的对齐断言——

```
attn_single_token_k_page_size * attn_block_size == ssm_block_page_size
```

新模型的 linear attention state 形状（`num_heads × head_dim × head_dim`）一旦和 attention 侧的 page 大小对不上，这里直接断言失败——**这是混合模型最先崩的地方**。相关配置收敛还涉及 `vllm_ascend/utils.py::refresh_block_size`。混合 KV cache 的配套 patch 为 `patch/platform/patch_kv_cache_{coordinator,utils}.py`、`patch_mamba_config.py`、`patch_mamba_manager.py`。

### 4. 调度器装配

旧的 `AscendScheduler`（V0 风格：prefill-first、水位线准入、chunked prefill 默认关闭）**已删除**，main 分支 `vllm_ascend/core/` 下已无 scheduler.py。当前默认使用上游 V1 `Scheduler`/`AsyncScheduler`，并派生一组专用调度器经 `scheduler_cls` 装配：

| 调度器 | 适用部署形态 | 机制要点 |
|---|---|---|
| `RecomputeScheduler` | 仅 PD 分离 D 节点 | HBM 不足时抢占 decode 请求就地重算，配合 `RecomputeCPUOffloadConnector` 把被抢占 KV 暂存 CPU |
| `DyntraLBScheduler` | D 节点 + DP>1 单节点 | 跨 rank KV 负载均衡 |
| `ShortRequestFirstScheduler` | FCFS 下短请求优先 | — |
| `BatchJobAwareScheduler` | 批任务感知 | — |
| `ProfilingChunkScheduler` | 动态 chunked PP | — |

各调度器有**互斥矩阵**，在 `check_and_update_config` 阶段（`init_ascend_config` 校验调度扩展互斥）fail-fast，不允许带病启动。已知限制示例：Qwen3.5 + async scheduling 在 recompute CPU offload 场景尚不支持，需关 async scheduling。

> **边界声明：调度器选择是部署形态（PD 分离与否、DP 规模、吞吐策略）的函数，而非模型结构的函数。** 唯一例外是混合线性注意力模型——mamba 组的 block 数上限会约束 `max_num_seqs`，此时模型结构反噬调度配置。

### 5. EngineCore 配置中枢清单（E1-E12）

EngineCore 侧的适配几乎全部通过 `NPUPlatform.check_and_update_config()`（`vllm_ascend/platform.py`，函数名是锚点）和 `patch/platform/` 完成。下表为 E1-E12 全量清单，「归属层」列指明各项的详述位置——本文档只详述调度层归口项，其余为指针。

| # | 检查项 | 触发条件 | 适配位置 | 归属层 |
|---|---|---|---|---|
| E1 | 模型注册 | 上游没有该模型，或需要覆盖上游实现 | `vllm_ascend/models/__init__.py::register_model()` | → `.claude/skills/day0-inference/reference/ascend-oot.md` |
| E2 | 量化方法识别 | 模型带 `quantization_config` | `quantization/utils.py::maybe_auto_detect_quantization`；新格式需在 `AscendCompressedTensorsConfig._detect_quant_type` 加分支。⚠️ MXFP4 有两道门：torch_npu 符号版本门（`mxfp_compat.py::ensure_mxfp4_*`，与 SoC 无关）+ 动态 MX 量化融合算子的 A5 SoC 门；符号缺失时在 `check_and_update_config` 早期报错或回退 W8A16 | → `.claude/skills/day0-inference/reference/ascend-oot.md` |
| E3 | Attention backend 选择 | 用了 MLA / 稀疏 / 新注意力 | `NPUPlatform.get_attn_backend_cls()` 的 `(use_mla, use_sparse, use_compress)` 分发表 | → `golden-worker-adapter.md` 数据面 |
| E4 | KV cache spec | 新的 KV 形态（MLA 变体、线性注意力 state） | 先查上游 spec 体系（FullAttention / MLA / Mamba / SlidingWindow）能否表达；确需新增时 `core/kv_cache_interface.py::register_ascend_kv_cache_specs()` | **本文档 §一.1-2** |
| E5 | 混合 KV cache | 模型是 hybrid（部分层 full attn + 部分层 linear） | `patch/platform/patch_kv_cache_{coordinator,utils}.py`、`patch_mamba_config.py`、`patch_mamba_manager.py` | **本文档 §一.3** |
| E6 | block_size / page 对齐 | 新 state 形态改变了 page 大小 | `patch_mamba_config.py` 的对齐断言；`utils.py::refresh_block_size` | **本文档 §一.3** |
| E7 | 图模式 | 模型有跨层状态传递、动态控制流 | `check_and_update_config` 里的 cudagraph_mode 收敛；必要时 `enforce_eager` 或加 splitting_ops。跨层状态直接阻塞 ACL Graph 全图捕获，需将状态传递点加入 `splitting_ops` | → `golden-worker-adapter.md` 数据面 |
| E8 | MoE 通信方式 | MoE 模型 | `ascend_forward_context.py::select_moe_comm_method`（按 SoC + 专家数 + EP size 分发） | → `golden-worker-adapter.md` 数据面 |
| E9 | 并行约束校验 | TP/EP/DP/PP 有特殊要求 | `check_and_update_config` 加 assert，早期失败优于运行时崩 | → `golden-worker-adapter.md` 控制面 |
| E10 | 多模态处理器 | 多模态模型 | 类似 `patch/hunyuan_vl_processor_compat.py` 的 processor 兼容 patch（该文件在 `patch/` 根目录） | → `golden-service-adapter.md` |
| E11 | Worker 类 | 需要定制 worker 行为 | `parallel_config.worker_cls`（默认 `NPUWorker`）。加载期权重映射检查（厂商权重名 vs vLLM 参数名的 missing/unexpected 审计）同属 Worker 控制面 | → `golden-worker-adapter.md` 控制面 |
| E12 | 投机解码 / MTP | 模型带 MTP 或 draft model | `patch_speculative_config.py`、MTP 模型注册 | **本文档 §二.7** |

> **关键原则**：模型层适配放 worker 组补丁或 `models/`；**配置与调度适配必须放 platform 组**（`pre_register_and_update` / `check_and_update_config`），因为它们要在配置定型前生效，且 engine-core 进程也要看到。例如 KV cache spec 注册由 engine-core 在规划 KV cache 时调用——那时 worker 还没起来，放 worker 组补丁完全无效。

---

## 二、数据面：运行时调度语义

### 6. prefix cache 运行时语义（线性注意力组）

线性注意力组开 prefix cache 的前置是 `mamba_cache_mode=align`——recurrent state 仅在 block 边界物化（上游仍标注 experimental），且强制依赖 chunked prefill。同时必须**显式检查与 MTP 的互斥项**。三个已实证的坑：

1. **`prefix_cache_retention_interval` 默认 0** 导致边界状态不被哈希，命中率静默为低（上游 vllm#53595）——不报错，只是 prefix cache 看起来「没效果」。
2. **chunked prefill 边界必须停在 mamba block 边界**（`_mamba_block_aligned_split`）。2026-08 的上游 PR #51113 才修掉越界导致的 prefix cache 投毒 bug——在 MTP+prefix caching 场景造成约 20% 准确率下降。
3. block size 被 mamba padding 撑大后（§一.3），短前缀永远够不到边界，命中率进一步劣化——page 对齐决策与 prefix cache 语义是同一件事的两面。

正向先例：上游为 Kimi K3 重做了「物理 state block 与 prefix 匹配粒度解耦 + copy-on-extend」机制，明确惠及所有混合线性模型——第二个同族模型的这部分边际成本骤降。

### 7. 投机解码调度

投机解码对调度层的改动集中在四点：`num_lookahead_tokens=k` 流入 `allocate_slots()` 为 verify 预分配槽位；runner 采样后 out-of-band 回传 draft token 写入 `request.spec_token_ids`；每步为带 draft 的请求调度 `1+len(spec_token_ids)` 个 token；被拒 token 回滚 `num_computed_tokens`。

**回滚成本的分化是线性注意力投机内存开销的根源**：paged KV 的被拒 token 只需覆写槽位，而 SSM/GDN state 需要快照/回滚支持（每请求多 k 个 state 槽，见 §一.2 表）。另需确认 draft 层 KV group 归属——non-causal draft 须单独分组。

**DSpark**（DeepSeek V4 / Kimi K3 / GLM-5.2 的并行草稿 + 轻量 Markov/置信度头方案）：在 vLLM 走 `mtp` 配置管线，但 vllm-ascend 的 `AscendDSparkProposer` 执行形态接近 DFlash——target hidden state 预填 draft K/V，一个 anchor-first query 块一次产出全部投机 token。DSV4 发布 checkpoint 的 DSpark block size=5，`num_speculative_tokens<5` 非法，推荐配置为 `{"method":"dspark","num_speculative_tokens":7}`。调度层另有 `dynamic_spec_config` 按置信度动态调整每请求 verify budget（`initial_verify_budget_per_req`、`budget_update_interval` 等）。

**组合回归高发区**（与图模式/并行叠加的交叉项，验收时必须专项覆盖）：

- verify 步 query 长度为 1+k，图模式的 uniform decode **capture key 必须覆盖该长度**；
- 多模块 MTP drafter 在 chunked-prefill 边界消费 lookahead 槽位，兼容性按模型/后端而异（Qwen3-Next MTP + chunked prefill 崩溃为公开案例）；
- DP 场景下 DSpark 曾出现 drafter token padding 形状不等崩溃，DP + spec decode 组合需专项回归。

### 8. PD 分离与混合注意力

PD 分离对混合模型的最大障碍是 **connector 的多 KV group 支持**：上游历史上 connector 只支持单 KV group，上游 PR #25712 引入 `SupportsHMA` 接口解除限制。实现该接口的语义要求：

- remote key 带 `kv_cache_group_id`；
- 跳过 null block；
- 命中位置须被各组 block_size 的**最小公倍数**整除；
- 异步发送期间 pin 住线性注意力 block。

vllm-ascend 侧 `MooncakeHybridConnector` 已实现该接口，AscendStore/KV Pool 经 RFC #9508 支持 align 模式线性注意力。

**layerwise KV Pool**（逐层传输与注意力计算流水重叠）的收益量级：DSV3.2 TP16 下 64k 输入 TTFT 从 312.5s 降至 191.3s。但限制同样明确：目前**仅支持 memcache 后端与 MLA/SFA backend，不支持混合 KV group**（多 group 直接抛 `NotImplementedError`）——混合模型只能走 bulk 路径。

---

## 三、与流程的衔接

本文档结论在 `.claude/skills/day0-inference/flows/golden_flow.md`（Stage 1）中的落点：

| 本文档内容 | golden_flow.md 落点 |
|---|---|
| KVCacheSpec 选型与定稿（§一.1-2）、page 对齐评估（§一.3）、调度器装配（§一.4） | **Phase 1** Designer 设计文档的「调度层设计」章节——KVCacheSpec 选型是必备产出，定稿前禁止进入性能工作 |
| E1-E12 清单核对（§一.5） | **Phase 1** 设计文档的 E1-E12 标记（归属其他层的项按 §一.5 指针分发） |
| prefix cache / 投机解码 / PD 的组合回归高发项（§二.6-8） | **G4 发布门禁**：E2E 回归配置 `tests/e2e/models/configs/<Model>.yaml`，组合矩阵（量化 × 图 × 投机 × CP/PD）按 Designer 清单显式纳入——历史已知问题几乎全部位于叠加组合而非基线 |
