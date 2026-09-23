# Golden Worker Designer — Stage 1 Worker 层设计规范

> 本文件由 `.claude/skills/day0-inference/reference/design/golden-designer/golden-designer.md` 的**步骤 3（Worker 层设计）**加载执行，不独立触发。
> 产出：**Worker 层 design spec**——它是 Worker 层的唯一交接物，经 golden_flow Phase 1 设计完整性检查后串联传递至 Phase 2，Developer 按落地流程 `.claude/skills/day0-inference/reference/adapt/golden-adapter/golden-worker-adapter.md` 依据本 spec 落地开发（代码写法参考知识库 `.claude/skills/day0-inference/reference/golden-worker-knowledge.md`，代码模板查 `.claude/skills/day0-inference/reference/adapter-templates.md`）。
> ⚠️ 行号防腐：引用代码以函数名/类名/注册项名为锚点，不写 `文件:行号`。

## 执行逻辑（五步）

### 步骤 1：通读模型代码——双源枚举

判定之前先建立事实基础：**`config.json` + `modeling_*.py` 双源枚举全部 module**，交叉比对去重（机制与命令查知识库 `golden-worker-knowledge.md` §2.1.7）；并对照 vLLM 侧上游等价层（§2.1.7 来源③）、grep `vllm_ascend/ops/triton/` 同族算子。禁止仅凭 Phase 0 证据摘要下结论；复核 Phase 0 五维扫描的结构维结论，不一致时以你通读代码后的判定为准，并在 spec 中显式修正。

### 步骤 2：Q0 前置检查（模型级，只问一次）

上游 `vllm/models/<model>/__init__.py` 的平台分派是否覆盖 NPU（`is_rocm()` 二分陷阱——命中则注册覆盖实现）；多厂商分支时逐个评估、选定基线分支，不默认选 nvidia（机制在总纲 `golden-adapter.md` §4）。**判定表表头注明所选基线分支名**——基线选错，整张判定表跟着错。

### 步骤 3：逐 module 判定——是否落在知识库既有章节内

Worker 层适配场景的全集 = **知识库 `golden-worker-knowledge.md` 的全部章节**（⚠️ §2.5 ACLGraph 属 Stage 3，本阶段不加载、不判定；全集随知识库扩章而扩展，不写死数量）。对每个枚举出的 module 走判定流水线（机制查知识库 §2.1）：**速查表快通道（§2.1.3）→ 未全命中走决策树 Q1/Q1'/Q2-Q4（§2.1.4）→ 定型到类型 0-5（§2.1.2）**，判定 A/B/C 的操作命令见 §2.1.5：

- **类型 0**：零适配，附理由（命中速查表默认判定 / 注册表已覆盖的代码事实），无理由视同漏项；
- **类型 1-5**：给出落地方案——覆写点（`forward_oot` / `forward` / monkey patch / 扩展实现）、实现方式、落点（`ops/` / `patch/worker/` / `models/`）、代码模板编号（`reference/adapter-templates.md` 模板 A-F）、验收方式；
- **超出全集**（知识库未覆盖的全新 module 形态 / 加载形态）→ 显式声明「**新场景**」，描述特征、与最近章节的差异、建议机制；新场景的方案按「建抽象」而非「适配单模型」组织，并在 spec 中标注**需 Reviewer 重点关注**（这是知识库的扩展点，不允许静默略过）。**同时把新场景追加记录到 `$ASCENDBOT_FILE_PATH/self-evolving.md`**（不存在则新建），格式固定：

  ```markdown
  ## 新场景候选 — <场景名>（<模型>，<日期>）
  - 来源层：Worker 层（golden-worker-designer）
  - 特征描述：<模型的 module 形态 / 加载形态特征是什么>
  - 与最近场景的差异：<最接近的知识库章节及为何不覆盖>
  - 建议机制：<若落入知识库，应如何扩章>
  - 处置状态：待用户决策
  ```

  该文件是**知识库的演进候选池**：任务完成后由 SKILL.md 的自演进步骤向用户咨询是否把条目落入知识库（扩为新章节）。

### 步骤 4：加载期权重映射检查

对每个需要加载权重的 module，用必查命令三件套（知识库 §1.5.1）列出厂商权重名 vs vLLM 参数名的 **missing / unexpected 两组**，按四类差异（§1.5.2）定型，并入判定表「加载期差异」列。**不允许「名字看起来对、没实际验证」**——权重名以 safetensors index 为权威。

### 步骤 5：魔法数字复核

在 Phase 0 审计（raw_evidence.md §11）基础上，确认判定表涉及的维度 / 头数 / rope 参数 / expert 数在平台代码中无硬编码残留，产出参数化建议。

## 场景判定详表

| # | 场景（知识库章节） | 判定问题 | 零适配条件 | 需要适配时的方案要点 |
|---|---|---|---|---|
| 1 | 四级管线与执行流（§1.1） | 新注意力是否需要新 `MetadataBuilder`？五态状态机输出是否覆盖？ | 命中既有 backend（分派键已覆盖，MetadataBuilder 复用） | 新 backend 的 `MetadataBuilder` 必须正确输出五态（`PrefillNoCache` 等）以驱动图模式选择——落地时的显式验证项 |
| 2 | `_dummy_run` 三职责（§1.2） | （本阶段仅知悉：profile / 编译触发 / 图捕获三职责与计数可校验性） | 不适用——Stage 1 全程 eager，无图捕获 | 计数校验与图模式验收属 Stage 3（权威详述在 `.claude/agents/performance.md`） |
| 3 | 显存预算陷阱（§1.3） | （本阶段仅知悉：图捕获显存与 KV cache 预算竞争，v0.21.0rc1 起有预估） | 不适用——Stage 1 无图捕获显存 | 开图前确认预估逻辑覆盖本模型 capture 形态（如投机 1+k 查询长度），属 Stage 3 |
| 4 | 并行策略约束（§1.4） | 目标并行配置（DCP/PCP/EPLB）的整除与互斥约束逐条代入是否通过？是否命中已移除旧路径？ | 目标并行配置全部通过既有 fail-fast 校验 | 整除公式逐条代入验算（禁止只写结论）；EPLB 三件事（adaptor 注册 / expert map 校验 / 量化白名单）；不得依赖已移除开关（layer sharding / FlashComm2 / weight prefetch） |
| 5 | 权重加载适配（§1.5） | 每个需加载权重 module 的 missing/unexpected 两组是什么？四类差异命中哪些？量化 NZ 布局约束？ | 两组均为空且自检口径（§1.5.3）无阻断项 | `_stacked_params_mapping` / loader 重排 / `process_weights_after_loading` 确认 / `.view()` 兼容；量化模型 `packed_modules_model_mapping` 同步；NZ 转换点与 SFA 预处理层避免二次 NZ 转换 |
| 6 | 逐 module 判定流水线（§2.1） | 每个 module 的类型 0-5 判定（速查表 → 决策树 → 判定 A/B/C）；枚举两源交叉是否完整？ | 速查表全命中类型 0 + 枚举两源一致 + Q0 未命中 | 未命中 module 逐个走决策树；Q1' 不可 import → standalone 重写（`deepseek_v4.py` 模式）；规模分诊决定落地形态（纯补丁 / 补丁+局部新增 / `models/` 完整目录）；判定表不含 P 级列 |
| 7 | Attention backend 分派（§2.2） | 模型的 `(use_mla, use_sparse, use_compress)` 特征组合命中既有查表键吗？目标 backend 的图支持级别？ | 特征组合命中既有键 → backend 层零代码（GLM-5 先例） | 扩展查表键或新增 selector 探测函数（如 `model_uses_sfa_sparse`）+ 同步注册新 cache spec（联动调度层 E4）；310P 盲区显式排除 |
| 8 | 各注意力形态实现要点（§2.3） | 模型的注意力形态（MLA / SFA / DSA / GDN/KDA）对应的 NPU 实现路径要点是否齐备？ | 形态与已支持模型同构，要点全部已具备 | 按形态回查要点（MLA absorbed 路径 / SFA indexer cache / DSA compressor memoize / 线性注意力算子组）；NoPE、`qk_rope_head_dim==0` 等变体专项显式列出 |
| 9 | MoE 与缺算子处置（§2.4） | MoE 模型：`select_moe_comm_method` 按目标 SoC + EP size 落在哪个分支？路由形态是否为新（hash / swiglu_limit）？有无缺算子？ | 非 MoE 模型，或 EP 代入公式落在预期分支且无算子缺口 | 部署前代入公式验算并写明结论；新路由形态对齐精度先例（#14397）；缺算子按 L1-L3 技术下沉 + L4 拆分纪律处置；**算子兼容性扫描先于一切模型代码工作** |
| 10 | ACLGraph 图模式（§2.5） | ⚠️ **属 Stage 3，Stage 1 不加载、不判定** | — | 六类不可入图清单与 meta 实现要求仅作前置知悉——新自定义算子注册 meta 实现是 Stage 1 的实现纪律（Stage 3 开图前置） |

## 输出契约：Worker 层 design spec

落盘 `$ASCENDBOT_FILE_PATH/design/worker-design-spec.md`（同时作为 Worker 层章节汇入总设计文档），**必须包含**：

1. **Q0 结论**：分派是否覆盖 NPU；选哪个上游分支作基线及理由（判定表表头注明基线分支名）；
2. **module 枚举完整性结论**：config 来源 / modeling 来源两源交叉结果；非主流字段逐个确认了哪些、各自用途；只在一个来源出现的 module 的 ⚠️ 审查结论；
3. **逐 module 判定表**：`| module | 枚举来源 | 类型(0-5) | 加载期差异 | 工作量 | 备注 |`——覆盖全部枚举 module，**不含 P 级列**；
4. **逐 module 方案块**：类型 1-5 的 module 每个一个方案小节（选型与理由 / 实现方式 / 落点 / **实现依据：golden-worker-knowledge.md 章节号** / 代码模板编号 / 验收方式）——**「实现依据」是 Developer 选择性执行的索引，缺此字段视同方案不完整**；
5. **权重映射 missing/unexpected 清单**：逐 module 两组 + 四类差异定型 + loader 处理动作；
6. **魔法数字审计结论**：平台代码中对本模型维度/头数/rope/expert 数的硬编码扫描结果与参数化建议；
7. **dummy 减层方案**（供 Tester Phase 1 冒烟）：每种层类型的最小保留数与分类型裁剪键清单（如 `kda_layers` / `full_attn_layers` 须同步改且恰好划分层栈）、跨层机制的最小层数（如 `attn_res_block_size = N` → ≥ N+1，低于阈值的机制显式声明不覆盖）、TP 整除代入结果、按层类型分开的显存估算、`--hf-overrides` 嵌套穿透结论（不能穿透时给派生 config 目录方案）。**模型小到无需减层时显式声明「全层拉起」**。

**打回条件**：判定表缺 module 或带 P 级标注；「类型 0」无理由；枚举两源交叉未做；missing/unexpected 清单缺失或凭「名字看起来对」未实际验证；新场景未显式声明；缺 dummy 减层方案且未声明「全层拉起」——任一命中即由主控打回 Designer 补齐。

## 交接链

```
golden-designer.md 步骤 3（加载本文件）
  → 产出 design/worker-design-spec.md
  → golden_flow Phase 1 设计完整性检查（打回条件见上）
  → golden_flow Phase 2：Developer 按 golden-worker-adapter.md 落地流程执行——
      以 spec 逐 module 判定表为过滤条件，类型 0 不动，只实现类型 1-5；
      覆写点机制按方案块「实现依据」查阅 golden-worker-knowledge.md 对应章节，
      代码模板查 reference/adapter-templates.md
  → Phase 3 / G3：按知识库 §1.5.3 自检口径取加载证据
```
