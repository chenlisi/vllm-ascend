# vllm-ascend OOT 机制参考：适配代码挂在哪里、怎么挂

> 本文档是 vllm-ascend out-of-tree（OOT）插件机制的参考手册，回答一个问题：**一段适配代码应该挂在哪个机制上、以什么形式挂上去**。覆盖插件注册、NPUPlatform 接口、三大替换机制（模型注册 / OOT 分派 / 工厂重定向）、monkey patch 治理、配置中枢与六类适配类型（类型 0-5）。
>
> ⚠️ **行号防腐声明**：文中 `文件:行号` 引用基于撰写时的基线版本（vllm-ascend main，约 v0.26.0rc1）。随版本更新行号会漂移——**函数名/类名是锚点，行号是辅助**。行号失效时按函数名 grep 即可重新定位；不确定的行号一律不写。
>
> ⚠️ **术语纪律**：P0/P1/P2 只用于**模型级路径判定**（见 `.claude/skills/day0-inference/reference/adapt/golden-adapter/golden-adapter.md` 与 `golden-worker-adapter.md`）；本文档第 7 节的「类型 0-5」是 **module 级**适配类型，两者回答不同问题，不混用、不同表。

## 1. 文档定位与阅读地图

本文档是**机制参考**，只讲机制本身（是什么、怎么触发、有什么坑），不讲流程编排与逐 module 判定。文档族分工：

| 文档 | 回答的问题 |
|---|---|
| 本文档（`.claude/skills/day0-inference/reference/ascend-oot.md`） | 适配代码挂在哪里、怎么挂：注册机制、替换机制、patch 治理、配置中枢、类型 0-5 目录 |
| `.claude/skills/day0-inference/flows/golden_flow.md` | 流程编排：Stage 1 的阶段划分与 G0–G4 门禁 |
| `.claude/skills/day0-inference/reference/adapt/golden-adapter/golden-adapter.md` | 分层适配总纲（索引）：服务层 / 调度层 / Worker 层的职责切分 |
| `.claude/skills/day0-inference/reference/adapt/golden-adapter/golden-service-adapter.md` | 服务层适配（parser 三件套、chat template、服务矩阵验证） |
| `.claude/skills/day0-inference/reference/adapt/golden-adapter/golden-schedule-adapter.md` | 调度层适配（KVCacheSpec、投机解码、EngineCore E1–E12 配置清单） |
| `.claude/skills/day0-inference/reference/adapt/golden-adapter/golden-worker-adapter.md` | Worker 层适配：逐 module 判定树、速查表、控制面 + 数据面落地 |
| `.claude/skills/day0-inference/reference/adapter-templates.md` | 类型 0-5 的代码骨架（模板 A–F） |

一条机制主线贯穿全文：**新模型 Day0 适配 = 架构注册（ModelRegistry）+ OOT 分派（CustomOp/PluggableLayer）+ 后端查表（attention backend），monkey patch 作为末位兜底**。第 2–3 章讲插件如何被 vLLM 发现并接管平台职责，第 4 章讲三大替换机制，第 5 章讲 patch 兜底及其治理，第 6 章讲配置中枢，第 7 章把上述机制组织成 module 级的六类适配类型目录，附录给出注册清单、陷阱表与自检命令。

## 2. 插件注册机制

### 2.1 两类 entry point 组与探测规则

vLLM 的硬件插件体系源于 2024 年 12 月的 RFC「Hardware pluggable」（vllm-project/vllm#11162），目标是硬件后端不改 vLLM 代码、pip 安装即接入。vLLM 官方插件文档定义两类 entry point 组：

- **`vllm.platform_plugins`**：注册 OOT 平台。探测函数在不支持时返回 `None`、否则返回平台类全限定名。
- **`vllm.general_plugins`**：注册 OOT 模型等通用扩展，**每个 vLLM 进程启动时都会调用**。

上游侧在 `resolve_current_platform_cls_qualname()` 中加载所有 `vllm.platform_plugins`，与内置插件（cuda/rocm/tpu/xpu/cpu）一起逐个探测：**恰好 1 个 OOT 激活则选用；≥2 个同时激活直接 `RuntimeError`**。

### 2.2 vllm-ascend 的注册声明

注册声明在 `setup.py` 的 `entry_points` 中（版本号由 setuptools-scm 管理，故不用 pyproject.toml）：

- `vllm.platform_plugins` 组声明 `ascend = vllm_ascend:register`。对应的 `register()` **不做硬件检测、总返回 `"vllm_ascend.platform.NPUPlatform"`**——安装即激活 NPU 平台。
- `vllm.general_plugins` 组声明 4 个入口：`ascend_kv_connector`、`ascend_model_loader`、`ascend_service_profiling`、`ascend_model`；另有 `ms_service_metric.providers` 组的 1 个可观测性入口。
- connector/loader/profiling 三类注册函数会先调用 `_ensure_global_patch()`，保证 engine-core 子进程也打上全局 patch；`register_model()` 则转调 `vllm_ascend.models.register_model()` 完成模型注册（见 §4.1）。

### 2.3 版本对齐策略

vllm-ascend 与上游 vLLM 实行严格的版本号 1:1 对齐：

- 自 vLLM 0.7.x 起按 PEP 440 发布同号版本，`vX.Y.Z` final 严格对齐上游同名 final，`.postN` 承载 bugfix；兼容矩阵逐行锁定 vLLM/Python/CANN/PyTorch/torch_npu 的精确组合（如 v0.26.0rc1 对应 vLLM v0.26.0、CANN 9.1.0、torch/torch_npu 2.10.x）。
- main 分支开发要求 checkout `.github/vllm-main-verified.commit` 指定的上游 commit，而非任意 tag；CI 承诺覆盖上游 main 与最近 release tag。
- 为兼容最近 1–2 个上游 release，代码内用 `vllm_version_is("x.y.z")` 做版本门控分支；版本解析失败时允许用 `VLLM_VERSION` 环境变量手动指定。

**对 Day0 适配的直接含义**：适配开始前必须先确认目标模型在当前对齐的上游 commit 上是否已有已合入实现——这是流程层 Phase 0 / G0 的输入之一（见 `.claude/skills/day0-inference/flows/golden_flow.md`）。

## 3. NPUPlatform 接口职责

`NPUPlatform(Platform)` 继承上游 `vllm.platforms.Platform`，其类属性即设备身份声明：

| 类属性 | 值 / 作用 |
|---|---|
| `_enum` | `PlatformEnum.OOT`，标识 out-of-tree 平台 |
| `dispatch_key` | `"PrivateUse1"`，使 NPU 走 PyTorch PrivateUse1 分发后端 |
| `device_control_env_var` | `"ASCEND_RT_VISIBLE_DEVICES"`，等价于 `CUDA_VISIBLE_DEVICES` |
| `supported_quantization` | 量化方法白名单（ascend、compressed-tensors、fp8 等 6 项），供上游 `verify_quantization` 校验 |

在此身份之上，NPUPlatform 的接口方法按调用时机分为四类：

| 时机分组 | 关键接口 | 职责与新模型适配的相关性 |
|---|---|---|
| 注册期 | `pre_register_and_update(parser)` | 依次做 4 件事：`adapt_patch(is_global_patch=True)` 应用全部 platform patch；注册 DeepSeek-V4 vision config 转换器；向 CLI `--quantization` choices 追加 `"ascend"`；按硬件 profile 导入量化配置类完成注册 |
| 配置期 | `apply_config_platform_defaults(vllm_config)` | 注入平台默认值，如按 `max_num_seqs * decode_query_len`（上限 512）推导 `max_cudagraph_capture_size`，有意去掉上游面向 CUDA 的尾部 `*2` |
| 配置期 | `check_and_update_config(vllm_config)` | 平台层最重的钩子，10 步可归为五组（见 §3.1） |
| 配置期 | `register_custom_kv_cache_specs(vllm_config)` | 注册 Ascend 自定义 KVCacheSpec（如 `AscendMLAAttentionSpec`），详见 `.claude/skills/day0-inference/reference/adapt/golden-adapter/golden-schedule-adapter.md` |
| 后端选择 | `get_attn_backend_cls(...)` | 按 `(use_mla, use_sparse, use_compress)` 三元组查表分派（见 §3.2） |
| 后端选择 | `get_device_communicator_cls` / `get_compile_backend` / `get_static_graph_wrapper_cls` 等 | 分别返回 `NPUCommunicator`（HCCL）、`AscendCompiler`、`ACLGraphWrapper`（CUDA Graph 的 Ascend 对应实现）；`import_kernels()` 刻意惰性初始化算子环境，避免提前初始化 CANN RTS 致 `ASCEND_RT_VISIBLE_DEVICES` 失效 |
| 能力声明 | `get_device_capability()` 与布尔能力族 | `get_device_capability()` 返回 `None`（NPU 无 CUDA 式 major/minor），这使上游 `has_device_capability` 对 NPU 一律为 False、个别上游代码走错分支需 patch 兜底；布尔声明族（`is_sleep_mode_available`、`use_custom_op_collectives` 等）被上游用作特性门控 |
| 前向上下文 | `set_additional_forward_context(...)` | 每个 forward step 前计算 Ascend 专属字段：MoE 通信方式选择、DP/PCP token 数对齐、动态 MX 量化 scale 算法等 |

### 3.1 `check_and_update_config`：十步五组的防御性配置治理

10 步可归为五组：

1. **日志与并行配置校验**：indexer PP、draft DCP、PCP/KVPP 合法性；
2. **量化自动检测**（见 §6.2）；
3. **GPU/ROCm 参数降级**：`_fix_incompatible_config` 把 flashinfer/trtllm/cudnn 系列 flag、cascade attention 等十余项 GPU/ROCm 专属参数静默降级并告警，同时把 `att_config.backend` 置 None **强制由插件后端接管**——保证新模型的上游默认配置不会在 NPU 上隐式走错路径；
4. **`init_ascend_config` 构建配置单例**并校验调度扩展互斥（见 §6.1）；
5. **编译与运行时设置**：编译模式仅支持 NONE/VLLM_COMPILE；piecewise 时向 `splitting_ops` 追加 `vllm::mla_forward`/`vllm::dsa_forward`；worker/scheduler 类选择与 `refresh_block_size`；SFA+DCP/SP 约束；`PYTORCH_NPU_ALLOC_CONF`。

### 3.2 `get_attn_backend_cls`：特征键查表而非按模型名分派

按 `(use_mla, use_sparse, use_compress)` 三元组查表，分派到 AscendAttention / MLA / SFA / DSA 四种后端；另有 RL 训推一致的 FA3 分支、310P 的 COMPATIBILITY 分支与 PCP（MRV2）独立映射，不支持的组合直接 `NotImplementedError`。逐后端的适用模型、图支持级别与新增形态扩展方法见 `.claude/skills/day0-inference/reference/adapt/golden-adapter/golden-worker-adapter.md`。

机制要点：**特征键由上游 attention selector 从模型 config 推导**（如 hf_config 含 `index_topk` 即 `use_sparse=True`），因此一个 MLA + 稀疏索引的新模型会自动落到 SFA 后端，平台层零改动——这正是 GLM-5 零代码适配的机制基础，也是模型级同构度判定的理论依据（判定方法见 `.claude/skills/day0-inference/reference/adapt/golden-adapter/golden-worker-adapter.md`）。

## 4. 三大替换机制

「替换机制」回答：模型代码里的某个类/函数，如何在不改上游源码的前提下落到 NPU 实现。共三种：模型注册（整个架构类）、OOT 分派（单个 CustomOp/PluggableLayer）、工厂函数重定向（非类符号）。

### 4.1 模型注册（ModelRegistry）

**注册链路**：general_plugins 的 `ascend_model` 入口在每个 vLLM 进程（含 engine core/worker 子进程）启动时调用，转调 `vllm_ascend.models.register_model()`，后者在 `vllm_ascend/models/__init__.py` 中对每个架构调用 `ModelRegistry.register_model(arch_name, "vllm_ascend.models.xxx:ClassName")`。

**两个约束必须遵守**：

1. **arch 名严格匹配**：第一参数（架构名）必须与 checkpoint `config.json` 的 `architectures` 字段匹配。同名覆盖语义——若架构名与上游已有架构同名则**覆盖**上游实现：新架构用唯一名，改造架构用原名。
2. **第二参数一律惰性字符串**：用 `"module:Class"` 形式，worker 进程按需导入，避免父进程提前 import 触碰设备。

**当前约 24 个注册架构的类型分布**（main 分支）：

- **新架构自持**：`KimiLinearForCausalLM`（Kimi K3）、`MiniMaxM3`、`Glm5Next`——后者因上游 PR #53906 未合并而下游自持，需配套 `_CONFIG_REGISTRY`/`is_deepseek_mla` patch 才能走 MLA 路径；
- **同名覆盖扩展**：`DeepseekV4ForCausalLM` 及其多模态版；
- **投机解码草稿模型**：MTP/Eagle3/DSpark/DFlash2 几乎全部是新注册的架构名。

**自持模型并非全部重写**：glm5next 大量复用上游积木（`MergedColumnParallelLinear`、`FusedMoEFactory`——此符号已被 platform patch 重定向到 `AscendMoERunner`，模型代码透明落到 NPU 实现——`make_layers`、`AutoWeightsLoader`），仅对自定义 attention、KDA 线性注意力、`load_weights` 反量化融合映射等 NPU 特异部分重写。

与「注册新模型」互补的是「零代码复用」路径：当新模型的架构已在上游 `ModelRegistry`、且其计算图只由已覆盖的通用层组成时，插件侧零模型代码即可运行——GLM-5（上游 `GlmMoeDsaForCausalLM` 直接继承 `DeepseekV2ForCausalLM` 无 override）与 Qwen3.x 即以这种方式落地，支持矩阵中标记「扩展兼容」的模型（Qwen2/Qwen2.5/Llama3 等）同属此类。该路径依赖下一节的 OOT 分派机制。

### 4.2 OOT CustomOp / PluggableLayer 透明分派

上游 vLLM 的 CustomOp 与 PluggableLayer 是双轨注册机制：两者的 `register_oot` 写进**同一个全局字典** `op_registry_oot`（`vllm/model_executor/custom_op.py:22`），替换时机都是 `__new__`（实例化时查表返回 OOT 实现类），但**实现时覆写的方法不同**：

| | CustomOp | PluggableLayer |
|---|---|---|
| 抽象粒度 | 算子（无状态） | 层（有参数、组合子模块） |
| 替换时机 | `__new__`（实例化时） | `__new__`（实例化时） |
| **要覆写的方法** | **`forward_oot`** | **`forward`** |
| forward 分发 | 有（`dispatch_forward`） | 无 |
| `custom_ops` 开关 | 受控 | 不受控 |

> ⚠️ **写错不报错，只静默不生效**。给 PluggableLayer 子类实现 `forward_oot`，那个方法永远不会被调用——这是最隐蔽的坑（附录 B 陷阱 #1）。确认方法：顺着基类往上查，直到撞见 `CustomOp` 或 `PluggableLayer`。

vllm-ascend 侧经 `register_ascend_customop()`（`vllm_ascend/utils.py:765`）把 30+ 种上游层名注册进 `REGISTERED_ASCEND_OPS`（同文件 `:813`；完整清单见附录 A）。注意 **key 是类名**（`cls.__name__`，如 `"RMSNorm"`），不是上游的注册名（如 `"rms_norm"`）——因为 `__new__` 里用的是 `cls.__name__`。

**`custom_ops=["all"]` 开关**：`NPUPlatform.check_and_update_config()` 对非 310P 设置 `compilation_config.custom_ops = ["all"]`。设成 `"none"` 时类照样被替换，但 forward 会退回 `forward_native` 纯 PyTorch 实现——310P 即处于该状态，需单独验证执行路径。

**MoE 双机制并存**：MoE 已有两条接管路径——`MoERunner`/`RoutedExperts` 已进注册表、可经注册表替换；工厂入口仍靠 `patch_fused_moe` 重定向（见 §4.3）。判定 MoE 类 module 时**先查注册表再考虑 patch**。

#### 模型级前置检查 Q0：平台分派覆盖

在进入任何 module 级判定**之前**，先做一次模型级检查（同一模型只问一次）：读上游 `vllm/models/<model>/__init__.py` 的分派逻辑。若只按 `is_rocm()` 二分（`if not current_platform.is_rocm(): from .nvidia... else: from .amd...`），**NPU 会静默落入 nvidia 分支**，import CUDA 专属代码直接崩——此时必须在 `vllm_ascend/models/__init__.py::register_model()` 注册覆盖实现，且确保 ascend 分支不 import 上游 nvidia 模块。

**Q0 命中后：选哪个上游分支作基线**。上游若有多个厂商分支，**不要默认选 nvidia**，逐个评估后选耦合最低的，选错会让工作量翻倍：

```bash
# ① 查各分支的模块级厂商 import（NPU 上 import 即崩的那种）
grep -nE "^(from|import).*(cute|cutlass|aiter|hip|rocm|nvshmem)" <branch>/*.py

# ② 对比各分支用的是「可注册层」还是「私有类」
grep -rn "PluggableLayer.register\|CustomOp.register" <该分支引用的层>
```

判据（优先级从高到低）：

| 观察 | 含义 |
|---|---|
| 模块级 import 厂商专属 kernel（如 `cute_dsl`） | 该分支不可用，import 即崩 |
| 用带 `@PluggableLayer.register` 的**共享层** | 可走注册替换（类型 2） |
| 用**私有类**（无注册装饰器） | 只能走 monkey patch（类型 3），成本更高 |
| 行数更少 | 通常意味着厂商特化更少 |

> 实践中曾出现 nvidia 分支不可用而 amd 分支干净的情况，务必逐个查。Q0 之后的逐 module 判定树见 `.claude/skills/day0-inference/reference/adapt/golden-adapter/golden-worker-adapter.md`。

### 4.3 工厂函数重定向

目标不是类而是**工厂函数**时（典型：`FusedMoE`），注册表机制够不着，只能 monkey patch：`patch/platform/patch_fused_moe.py` 在模型 import 前把 `FusedMoEFactory` 重定向到 `AscendMoERunner`。

**两处 binding 铁律**：工厂函数替换必须同时改**包 `__init__` 和 layer 模块**两处绑定，否则部分模型拿到未 patch 版本（附录 B 陷阱 #5）。代码骨架（含 FusedMoE 示例与防御性校验）见 `.claude/skills/day0-inference/reference/adapter-templates.md` 模板 C。

## 5. Monkey patch 机制与治理

### 5.1 加载机制：import 副作用式替换与两处触发点

当 platform plugin 的标准接口覆盖不到上游的某些位置时，兜底手段是 monkey patch。其生效方式不是注册-调用，而是 **import 模块时模块级代码直接赋值替换上游符号**（例如 `vllm.distributed.parallel_state.destroy_model_parallel = patch_destroy_model_parallel`），触发入口只有一个函数 `adapt_patch(is_global_patch)`。该函数仅在两处被调用：

- **主进程侧**：`NPUPlatform.pre_register_and_update()` 调用 `adapt_patch(is_global_patch=True)`，导入 `vllm_ascend.patch.platform`。时机：online 在 CLI 参数解析时，offline 在 `EngineArgs.create_engine_config` 时。
- **worker 侧**：`NPUWorker.__init__` 调用 `adapt_patch(is_global_patch=False)`，导入 `vllm_ascend.patch.worker`。

**EngineCore 进程**默认不可直接 patch：当前 main 通过 `patch_engine_core.py` 把 engine-core 级 patch 收敛为对 `run_engine_core` 的单一 wrapper，利用 spawn 子进程 unpickle 时重新 import 模块来**确定性重放** patch。

**时序铁律**：worker 的 MoE factory patch 必须在**任何模型文件 import 之前**完成，否则模型拿到的是未 patch 的绑定。落实到 `NPUWorker.__init__` 的四步序：`① adapt_patch() → ② from vllm_ascend import ops → ③ register_ascend_customop() → ④ 模型加载`。

### 5.2 目录结构与五类作用对象

当前 main 分支（约 v0.26.0rc1）的 patch 目录为**扁平结构**：`platform/` 下 22 个 `patch_*.py`（主进程生效），`worker/` 下约 18 个，另有 `worker/patch_v2/` 约 12 个针对上游 v2 model runner 路径（故「30+ 条 worker patch」与「18+12」两种口径实为同一事实）。早期 v0.9.x–v0.10.x 曾按上游版本再分三目录，v0.11.0 起因多版本维护成本过高废弃，改为扁平结构加 `__init__.py` 内条件 import（按 `HAS_TRITON`、硬件 profile、环境变量决定）。

按作用对象，这批 patch 覆盖五类目标：

1. **调度器与 engine core**：如 `patch_balance_schedule` 的 DP 均衡调度、engine-core 单一 wrapper；
2. **KV cache 管理**：如 `patch_kv_cache_coordinator` 修 PD 分离 + 混合 Mamba 的前缀命中问题、`patch_mamba_*` 系列的 block size 修正；
3. **算子与模型层入口**：如 `patch_fused_moe` 在模型 import 前把 `FusedMoEFactory` 重定向到 `AscendMoERunner`、`patch_triton` 重绑定 CUDA triton kernel；
4. **采样器**：`patch_rejection_sampler` 的 NPU top-k/top-p；
5. **config 与注册白名单**：`patch_glm5next_config` 注册 `_CONFIG_REGISTRY` 并扩展 `is_deepseek_mla` 判定。

`vllm_ascend/patch/__init__.py` 本身不含代码，是约 71KB 的注释形式登记册。

### 5.3 治理制度：四层约束

patch 机制表面是 hack，实际已被治理为一套严格的技术债台账。官方文档给出三条原则：**Less is more**（patch 不是唯一手段就不用）、写明移除计划、鼓励清理存量。配套制度有四层：

1. **强制四段式登记**：每条新 patch 必须在 `patch/__init__.py` 登记 Why / How / Related PR / Future Plan + 移除条件。登记册中大量 Future Plan 是「等上游合入 PR #xxxx 后删除」，升级支持版本时按链接批量核销。
2. **AGENTS.md 架构评审**：评审目标组件正确性、最小性、性能影响与上游贡献长期计划，checklist 含「Patching pattern used correctly」与「No direct model file additions」。
3. **测试要求**：每个 patch 配套 UT + E2E 测试。
4. **防御性写法**：如 `patch_fused_moe` 对上游 router `_apply_eplb_mapping` 做签名字节级校验，上游一改即 RuntimeError 而非静默错位。

### 5.4 RFC #7539 分界与 patch 决策树

社区 RFC「vllm-ascend-model-adapter」（issue #7539）给出硬性分界：**模型主体适配代码禁止进入 vllm-ascend**（原文 "Model-specific files and patches must never be introduced in vllm-ascend"）。模型数学/权重问题应改上游模型文件或在 `vllm_ascend/models/` 注册新架构；若模型不打 patch 就无法运行，正确动作是提 issue 分析根因而非加 patch。允许新增的只有**框架级最小 patch**——上游框架模块（scheduler、attention backend 选择、sampler、weight loader、worker）含 Ascend 不兼容逻辑且无扩展点时，仅覆盖不兼容路径并附移除计划。

**patch 决策树（强制走查，顺序不可颠倒）**：

1. 模型数学/权重问题 → 改上游模型文件或在 `vllm_ascend/models/` 注册新架构；模型主体适配代码禁止以 patch 形式进入 vllm-ascend（RFC #7539）。
2. 能用既有机制（CustomOp 分派、`AscendSampler` 继承、torch.compile fusion pass、composition）就不用 patch。
3. 动代码前先走 fallback ladder 定位：复现 → `--enforce-eager` → `TORCHDYNAMO_DISABLE=1` → 关多模态。
4. 仅当框架行为在 NPU 上错误且无插件钩子时，允许框架级最小 patch（只覆盖不兼容路径），并强制四段式登记（Why / How / Related PR / Future Plan + 移除条件）写入 `vllm_ascend/patch/__init__.py`，配套 UT/E2E 与架构评审；防御性写法：对上游函数做签名级校验，变更即 RuntimeError；移除条件为上游 PR 合入即核销。
5. 「不打 patch 模型就不能跑」→ 停止并提 issue 分析根因，而不是加 patch。

**反向指标**：patch 台账的密度实质上是上游插件接口成熟度的反向度量——上游每提供一个新的 dispatch/hook，就有一批 patch 核销。故 patch 不应是「遇到问题的默认动作」，而是决策树的最后一个分支。

## 6. 配置中枢：AscendConfig 与量化自动检测

### 6.1 AscendConfig

vLLM 提供的自由字典 `additional_config` 被 vllm-ascend 用来承载全部 Ascend 专属选项；`check_and_update_config` 中的 `init_ascend_config(vllm_config)` 构建 `AscendConfig` 并存为进程级单例，运行时各处经 `get_ascend_config()` 读取。

`AscendConfig` 已迁移为 `@config` pydantic dataclass：

- **`extra="forbid"`**：拒绝未知 key，实现拼写错误 fail-fast。曾因拒绝下游 vLLM-Omni 自有 key 引发 issue #15418，现对 vllm_omni 仅告警放行。
- **宽松 coercion**：bool/int 宽松转换，修掉 `bool("false")` 陷阱。
- **`derive_and_validate(vllm_config)`**：依赖 vllm_config 的派生与互斥检查集中于此。

配置项规模约 60 项，与模型行为强相关的包括：MLA 系的 `enable_mlapo`；稀疏注意力缓存 `enable_sparse_sfa_c8`/`enable_sparse_li_c8`；`enable_dsa_cp`（需 hf_config 带 `index_topk`，自动联动 FlashComm）；MoE 通信 `mc2_comm_alg`、`enable_fused_mc2`；权重布局 `weight_nz_mode`；以及 `scheduler_config`/`eplb_config`/`rl_config` 等子配置。多数项带有模型级适用约束（如 `enable_mlapo` 仅 MLA 模型），不合法组合在 `derive_and_validate` 或 `check_and_update_config` 阶段直接报错，属预期行为。配置项全表见官方文档 `docs/source/user_guide/configuration/additional_config.md`。

### 6.2 量化自动检测

量化方法的选择已被自动化。`check_and_update_config` 第 3 步 `maybe_auto_detect_quantization` **仅读 JSON 文件、不碰 safetensors**：

1. checkpoint 存在 `quant_model_description.json` → 判定为 `"ascend"`（ModelSlim 体系）；
2. `config.json` 的 `quantization_config.quant_method` 为 `compressed-tensors`/`fp8` → 取对应方法；
3. 用户显式 `--quantization` 优先级最高（与检测结果不一致时告警）。

两点注意事项：

- **文档滞后**：官方 quantization guide 仍要求 ModelSlim 量化模型显式指定 `--quantization ascend`，但 main 源码自 PR #6645 起已支持自动检测——以源码为准，显式指定仍兼容。
- **白名单校验**：检测得到的量化方法必须在 `NPUPlatform.supported_quantization` 白名单内，否则上游 `verify_quantization` 报错。

新量化格式与新 ignore 规则的落地位置（`quantization/utils.py`、`AscendCompressedTensorsConfig._detect_quant_type`、MXFP4 的两道门）见 `.claude/skills/day0-inference/reference/adapt/golden-adapter/golden-schedule-adapter.md` 的 EngineCore 清单。

## 7. 六类适配类型目录（类型 0-5）

本节是 module 级适配类型的机制目录：每种类型给出判定依据、动作要点与代码模板索引。**类型与模型级路径（P0/P1/P2）的映射、逐 module 判定树（Q1/Q1'/Q2-Q4）、标准 module 速查表与规模分诊，均属 `.claude/skills/day0-inference/reference/adapt/golden-adapter/golden-worker-adapter.md`，此处不重复。**

| 类型 | 一句话定义 | 覆写/动作 | 代码模板 |
|---|---|---|---|
| 类型 0 | 零适配（上游等价层 + Ascend 已注册） | 无 | —（见 §7.1） |
| 类型 1 | 新增 CustomOp OOT 实现 | `forward_oot` | 模板 A |
| 类型 2 | 新增 PluggableLayer OOT 实现 | `forward`（+至多 2 类方法） | 模板 B |
| 类型 3 | monkey patch（无法走注册表） | 视目标而定 | 模板 C |
| 类型 4 | 扩展现有 Ascend 实现（参数/分支被静默丢弃） | 加显式分支，不改默认行为 | —（见 §7.5） |
| 类型 5 | 全新结构（算子 + 可能的 backend + KV cache spec） | 全新实现 | 模板 D/E/F |

模板 A–F 全文见 `.claude/skills/day0-inference/reference/adapter-templates.md`。

### 7.1 类型 0：零适配（沿用现有结构）

**判定依据**：上游有等价层 + Ascend 已在 `REGISTERED_ASCEND_OPS` 注册。

**什么都不用做**。上游模型代码写 `RMSNorm(hidden_size)`，`CustomOp.__new__` 查表返回 `AscendRMSNorm`，自动调 `torch_npu` 融合算子。典型例子：RMSNorm、SiluAndMul、RotaryEmbedding、各类 ParallelLinear、VocabParallelEmbedding。

**唯一要确认的**：`compilation_config.custom_ops` 是否为 `["all"]`（`NPUPlatform.check_and_update_config()` 对非 310P 设置）。设成 `"none"` 时类照样被替换，但 forward 退回 `forward_native` 纯 PyTorch 实现（机制见 §4.2）。

### 7.2 类型 1：新增 CustomOp OOT 实现

**判定依据**：上游是 `CustomOp` 子类（有 `forward_cuda`/`forward_native`），Ascend 未注册。

要点：

- 默认只覆写 `forward_oot`，**不要覆写 `__init__`**，让基类处理参数。
- **例外——需覆写 `__init__` 的两种场景**：(a) 需要从 `config` 读额外配置（如 `vllm_config` 中的开关）；(b) 新算子/新激活有**上游基类不存在的构造参数**（如 `beta`、`linear_beta`）——此时必须同时覆写 `__init__`（接收新参数并赋给 `self`）和 `forward_oot`（从 `self` 读取使用）。
- 先用 `forward_native` 的 torch 等价实现跑通精度，再换融合算子。
- **精度细节检查清单**（新算子/新激活是昇腾上最典型的精度不达标来源）：
  - 激活/路由/norm 的中间计算用 **fp32**（sigmoid/tanh/softmax/softplus），不要 bf16 一路算到底；
  - 门控参数 `dt_bias`/`A_log`/`beta` 用 **float32 参数 + float32 运算**（对照同族层 GDN/KDA 写法）；
  - 低秩分解（q_lora/kv_lora）的展开顺序与 dtype 与厂商实现等价；
  - 形状/布局：conv1d weight 是否需 `unsqueeze(1)`；状态矩阵 layout（dim-first vs dim-last）与 kernel 一致；
  - 无法在无 NPU 环境验证的，标注「待真实权重门验证」，不阻塞开发。

**参考模板**：`vllm_ascend/ops/activation.py:41` `AscendSiluAndMulWithClamp`——结构最简单，7 行覆写。代码骨架见 `adapter-templates.md` 模板 A。

### 7.3 类型 2：新增 PluggableLayer OOT 实现

**判定依据**：上游是 `PluggableLayer` 子类（`__init__` 里组合子模块），Ascend 未注册。

通常只覆写 3 类方法，其余全继承：(1) `forward`/`_forward`（计算路径）；(2) `get_state_shape`/`dtype`（若涉及 KV/state cache）；(3) `get_attn_backend`（若需要专属 backend）。

**参考模板**：`vllm_ascend/ops/bailing_moe_linear_attn.py:42`——文件头注释明确写了「只覆写 3 个平台相关方法，其余全继承」，是最干净的范例。代码骨架见 `adapter-templates.md` 模板 B。

### 7.4 类型 3：monkey patch

**判定依据**（满足任一）：

- 上游层**没有** `@CustomOp.register` / `@PluggableLayer.register` 装饰器 → 不在注册表里，无法 OOT 替换；
- 上游实现内部**硬编码** `if current_platform.is_rocm(): ... else: <CUDA>`，**且无注册装饰器** → 没有 NPU 分支又无法替换；
- 目标不是类而是**工厂函数**（如 `FusedMoE`，见 §4.3）。

**与类型 2 的边界**——三者区分必须记住：模块 import 即崩 → standalone 重写（决策树 Q1'，按类型 5 处理）；有注册装饰器 + 仅函数内硬编码 → **类型 2**（继承后覆写该方法，自带 NPU 分派），不必 patch；无装饰器 → **类型 3**。

**分支选择**：改引擎/调度层 → `patch/platform/`（`pre_register_and_update()` 生效）；改模型/算子层 → `patch/worker/`（`NPUWorker.__init__` 生效）。

**时序铁律**：worker patch 必须在任何模型模块被 import 之前执行（`NPUWorker.__init__` 四步序，见 §5.1）。

**治理约束**：新增 patch 必须走 §5.4 的决策树与四段式登记；AGENTS.md 要求架构评审与上游回贡计划——patch 是技术债，注册表才是目的地。代码骨架见 `adapter-templates.md` 模板 C。

### 7.5 类型 4：扩展现有 Ascend 实现

**判定依据**：上游有等价层且 Ascend 已实现，但**新模型用到了 Ascend 实现未覆盖的可选参数/路径**。这是最容易被漏判的一类——表面看「已支持」，实际跑起来参数被静默丢弃或走错分支。

四种静默模式：

| 模式 | 症状 | 检查方法 |
|---|---|---|
| 参数被丢弃 | Ascend 实现构造子模块时没传某个 optional 参数 | 对比上游 `__init__` 的参数列表与 Ascend 实现实际使用的 |
| 分支缺失 | 枚举新增了成员，Ascend 的 if-elif 链没有对应分支，落到 `else` 兜底 | grep 该枚举在 Ascend 侧的所有使用点 |
| 无条件假设 | Ascend 实现无条件调用某个前提操作（如无条件取 RoPE cache） | 找 Ascend 实现里的无条件调用，对照新模型是否满足前提 |
| 算法别名 | 按名称查「缺失」，实际语义已被支持（如 `noaux_tc` 就是 `scoring_func="sigmoid"` + `e_score_correction_bias` 的组合）；反向更危险：名义支持、参数语义不同 | 名称查无此项时，**先核对数学定义再判缺失**；对名称命中的也要核对参数语义 |

**最危险的是「分支缺失」**：新的激活函数枚举落到 `else` 分支被当成 SwiGLU 算——不报错，结果全错。

**适配逻辑**：

1. 先定位差异点（用上表的检查方法）；
2. 加显式分支或条件开关，**不要改默认行为**；
3. 兜底 `else` 分支改成显式 `raise NotImplementedError`，把静默错误变成早期失败。

**检查命令**（对应上表前三种症状）：

```bash
# ① 参数被丢弃：对比上游 __init__ 签名 vs Ascend 实际使用
diff <(grep -A30 "def __init__" $VLLM/<upstream>.py | grep -oE "^\s+\w+:" | tr -d ' :') \
     <(grep -oE "\w+=\w+\." $VLLM_ASCEND/<ascend>.py | cut -d= -f1)

# ② 分支缺失：枚举成员数 vs Ascend 侧分支数
grep -c "MoEActivation\.\|ActivationType\." $VLLM_ASCEND/ops/fused_moe/moe_mlp.py

# ③ 无条件假设：找无 if 保护的前提调用（如无条件取 RoPE cache）
grep -n "get_cos_and_sin\|rope_single\|apply_rotary" $VLLM_ASCEND/attention/mla_v1.py
```

### 7.6 类型 5：全新结构

**判定依据**：上游完全没有对应层。

**算子实现三选一**：有 `torch_npu` 融合算子 → 直接调；无但可组合 → 组合现有算子；都不行 → 写 Triton kernel（`ops/triton/`）或 C++ 自定义算子（`csrc/`）。

**参考模板**：

- 新 attention backend：`vllm_ascend/attention/sfa_v1.py`（继承 `MLACommonMetadataBuilder`，复用 MLA 骨架）或 `dsa_v1.py`（完全独立体系）→ 模板 D；
- 新 Triton kernel：`vllm_ascend/ops/triton/` → 模板 E；
- 新 KV cache spec：`vllm_ascend/core/kv_cache_interface.py:213` `register_ascend_kv_cache_specs()` → 模板 F（调度层视角见 `.claude/skills/day0-inference/reference/adapt/golden-adapter/golden-schedule-adapter.md`）；
- standalone 重写：`vllm_ascend/models/deepseek_v4.py`（复用平台中立基类 + ascend 自有 kernel）。

**Q1' standalone 重写变体**：决策树 Q1' 判定「上游层所在模块在 NPU 上不可 import」时，虽上游有对应层但无法继承——行为等价于类型 5，按类型 5 的模板处理，但可参考上游的平台中立基类设计（而非从零设计）。后续步骤（KV cache spec → attention backend → 模型层组装）的依赖关系与实施顺序见 `.claude/skills/day0-inference/reference/adapt/golden-adapter/golden-worker-adapter.md`。

**特殊分支「层间数据流」**：不是单个 module，而是 decoder layer 间传递 `hidden_states` 以外的张量（block residual / 跨层状态累积 / 前缀和传递）。默认按类型 5 处理：**跨层状态直接阻塞 ACL Graph 全图捕获**，需将状态传递点加入 `splitting_ops` 走 piecewise，且影响 PP 切分；若跨层状态仅是简单的标量门控（如 1 个 Linear + sigmoid），可按类型 3 处理。参考：某些混合模型的块级残差机制（如 `attn_res_block_size`）。

## 8. 附录

### 附录 A：CustomOp 覆盖清单

`REGISTERED_ASCEND_OPS`（`vllm_ascend/utils.py:813`）当前 30 项基础注册 + 条件项。**新模型用到这些结构时零适配**（类型 0）。下表按组归纳，含条件项与说明项；基础注册表为 30 项，不要数表格行数——行数与表项数不一一对应，以 `grep -n "REGISTERED_ASCEND_OPS = {" vllm_ascend/utils.py` 实测为准。

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
| MoE | `MoERunner` `RoutedExperts` | 注册表替换（与 `patch_fused_moe` 工厂重定向并存，见 §4.2） |
| 其他 | `CustomQwen2Decoder` | PluggableLayer → `forward` |
| 其他 | `Conv3dLayer`（基类 `ConvLayerBase`） | CustomOp → `forward_oot` |
| 条件项 | `KimiK3MultiHeadLatentAttentionWrapper`（vLLM ≠ 0.28.0 时注册） | PluggableLayer |
| 310P 覆盖 | 11 项 `*310` 变体 | — |

> ⚠️ **310P 差异不止这 11 项**：310P 不设 `custom_ops = ["all"]`，算子退回 `forward_native`（机制见 §4.2 与 §7.1），需单独验证执行路径。

**不在表里的重要结构**（看到它们不要按「未注册」误判）：

- **`FusedMoE`** —— 走 `patch/platform/patch_fused_moe.py`（工厂函数重定向，见 §4.3）；注意 MoE 已是双机制并存：工厂入口仍靠 patch 重定向，而 `MoERunner`/`RoutedExperts` 已进注册表——判定 MoE 类 module 时先查注册表再考虑 patch；
- **Attention 主体** —— 走 `get_attn_backend_cls()` 分发（见 §3.2），不在 CustomOp 注册表；
- **线性注意力 state cache** —— 走上游 `MambaSpec` + `patch_mamba_*`（Ascend 无对应 spec，详见 `.claude/skills/day0-inference/reference/adapt/golden-adapter/golden-schedule-adapter.md`）。

### 附录 B：常见陷阱 12 条

| # | 陷阱 | 后果 | 规避 |
|---|---|---|---|
| 1 | 给 PluggableLayer 实现 `forward_oot` | 静默不生效 | 先确认基类，见 §4.2 双轨对比表 |
| 2 | 新激活枚举落到 `else` 兜底 | **静默算错**，不报错 | grep 枚举所有使用点；兜底改 raise（§7.5） |
| 3 | `REGISTERED_ASCEND_OPS` 用注册名而非类名 | 替换不生效 | key 用 `cls.__name__`（§4.2） |
| 4 | patch 时序晚于模型 import | patch 不生效 | worker patch 放 `NPUWorker.__init__` 第一步（§5.1） |
| 5 | 工厂函数只改一处 binding | 部分模型拿到未 patch 版本 | 包 `__init__` 和 layer 模块都要改（§4.3） |
| 6 | Ascend 实现丢弃上游 optional 参数 | 功能静默缺失 | 对比 `__init__` 参数列表（§7.5） |
| 7 | 混合模型 page size 不对齐 | 启动断言失败 | 先算 state 形状与 page 大小（详见 `.claude/skills/day0-inference/reference/adapt/golden-adapter/golden-schedule-adapter.md`） |
| 8 | 大专家数 MoE 未算 EP 门槛 | 掉回慢通信路径 | 部署前代入 `select_moe_comm_method` 公式（详见 golden-worker-adapter.md） |
| 9 | `tensor.item()` 在热路径 | 性能骤降（NPU 同步） | 见仓根 `AGENTS.md` NPU-Specific Considerations |
| 10 | 精度敏感算子未用 fp32 中间计算 | 精度不达标 | 激活/路由/norm 的中间量用 fp32（§7.2 检查清单） |
| 11 | **上游分派只按 `is_rocm()` 二分，NPU 静默落入 nvidia 分支** | import CUDA 专属代码（如 CUTLASS DSL）直接崩 | §4.2 前置检查 Q0；注册覆盖实现且确保 ascend 分支不 import 上游 nvidia 模块 |
| 12 | 上游层所在模块顶层 import CUDA 专属依赖（`_custom_ops` / cute_dsl / deep_gemm 等） | NPU 上 **import 即崩**，「继承上游类覆写方法」路径不可用 | 决策树 Q1'；standalone 重写（`models/deepseek_v4.py` 模式，§7.6），只复用平台中立基类 |

### 附录 C：适配自检命令

附录 B 列了 4 条「静默失败」陷阱（#1 写错方法、#2 else 兜底、#3 注册名错、#6 丢参数）。其中 **#1/#3 可用下列命令①②检测；#2/#6 的检查方法见 §7.5「检查命令」**。陷阱表告诉你「会静默失败」，下列命令告诉你「怎么知道自己踩了」：

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

这能把 4 条「靠经验避免」的静默失败变成「可主动检测」。跳过这步的后果：精度不对时无法区分「算法错」还是「根本没替换成功」。
