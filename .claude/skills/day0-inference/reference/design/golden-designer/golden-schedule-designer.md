# Golden Schedule Designer — Stage 1 调度层设计规范

> 本文件由 `.claude/skills/day0-inference/reference/design/golden-designer/golden-designer.md` 的**步骤 2（调度层设计）**加载执行，不独立触发。
> 产出：**调度层 design spec**——它是调度层的唯一交接物，经 golden_flow Phase 1 设计完整性检查后串联传递至 Phase 2，Developer 按落地流程 `.claude/skills/day0-inference/reference/adapt/golden-adapter/golden-schedule-adapter.md` 依据本 spec 落地开发（代码写法参考知识库 `.claude/skills/day0-inference/reference/golden-schedule-knowledge.md`）。
> ⚠️ 行号防腐：引用代码以函数名/类名/注册项名为锚点，不写 `文件:行号`。

## 执行逻辑（三步）

### 步骤 1：通读调度相关代码

判定之前先建立事实基础，**三个来源都要读**，禁止仅凭 Phase 0 证据摘要下结论：

- **模型侧**（模型仓库内）：`config.json` 的注意力/cache 相关字段（注意力形态、滑窗参数、线性注意力 state 形状、MTP/draft 配置）、`modeling_*.py` 中各层类型的 cache 使用方式（indexer 独立缓存、recurrent state、多 cache group）；
- **上游 vLLM 侧**：`KVCacheSpec` 体系（FullAttention / MLA / Mamba / SlidingWindow 各 spec 的表达能力）、V1 `Scheduler`/`AsyncScheduler`、投机解码调度路径（`allocate_slots()` lookahead 预分配、`spec_token_ids` 回传、被拒 token 回滚）、connector 的 `SupportsHMA` 接口；
- **vllm-ascend 侧**：`vllm_ascend/core/kv_cache_interface.py::register_ascend_kv_cache_specs()` 已注册 spec 清单、`patch/platform/patch_kv_cache_{coordinator,utils}.py` 与 `patch_mamba_*` 系列、`check_and_update_config` 的调度相关收敛、专用调度器（`RecomputeScheduler` 等）与互斥校验、`patch_speculative_config.py`、`MooncakeHybridConnector`。

同时复核 Phase 0 五维扫描报告的 cache 语义维结论——它是初判，你是终判；结论不一致时以你通读代码后的判定为准，并在 spec 中显式修正。

### 步骤 2：场景判定——是否落在知识库既有章节内

调度层适配场景的全集 = **知识库 `golden-schedule-knowledge.md` 的全部章节**（每章一类场景，逐章判定对照见下「场景判定详表」；全集随知识库扩章而扩展，不写死数量）。逐章判定当前模型的调度特征是否命中：

- **命中**：进入步骤 3 逐场景输出；
- **不命中**：该场景输出「无需适配 + 理由（引用代码/证据）」；
- **超出全集**：模型的调度特征不被知识库任何一章覆盖 → 显式声明「**新场景**」，描述特征、与最近章节的差异、建议机制；新场景的方案按「建抽象」而非「适配单模型」组织，并在 spec 中标注**需 Reviewer 重点关注**（这是知识库的扩展点，不允许静默略过）。**同时把新场景追加记录到 `$ASCENDBOT_FILE_PATH/self-evolving.md`**（不存在则新建），格式固定：

  ```markdown
  ## 新场景候选 — <场景名>（<模型>，<日期>）
  - 来源层：调度层（golden-schedule-designer）
  - 特征描述：<模型的调度特征是什么>
  - 与最近场景的差异：<最接近的知识库章节及为何不覆盖>
  - 建议机制：<若落入知识库，应如何扩章>
  - 处置状态：待用户决策
  ```

  该文件是**知识库的演进候选池**：任务完成后由 SKILL.md 的自演进步骤向用户咨询是否把条目落入知识库（扩为新章节）。

### 步骤 3：逐场景输出 design spec

对每个场景产出一行判定 + 方案块：

- **无需适配**：给出理由（引用步骤 1 通读的代码事实或 Phase 0 证据章节），无理由视同漏项；
- **需要适配**：给出落地方案——选型与理由、实现方式（复用上游 spec / 复用 Ascend 既有 spec / 新增 spec + manager）、落点（注册点 / patch 文件 / 配置项）、验收方式（对应知识库 §二.6-8 的组合回归高发项）。

## 场景判定详表

| # | 场景（知识库章节） | 判定问题 | 零适配条件 | 需要适配时的方案要点 |
|---|---|---|---|---|
| 1 | KVCacheSpec 选型与 page size（§一.1-3） | 模型逐层注意力形态 × cache 构成是什么（对照 §一.2 全表）？既有 spec 清单能否承接？跨组 page size 是否一致、被撑大的副作用？ | 全部层命中既有 spec（上游 FullAttention / AscendMLA / SFA indexer / SlidingWindowMLA / 上游 Mamba）且 page size 自然一致 | 新 spec + manager 经 `register_ascend_kv_cache_specs()` 注册；混合模型评估归并策略（DSV4 逻辑 block 固定 256 native token 先例）；E6 对齐断言代入验算。**选型错误代价是显存量级的**（DSA 按 FullAttention 承接浪费 38.8% 显存先例） |
| 2 | 调度器装配（§一.4） | 部署形态（PD 分离 / DP 规模 / 吞吐策略）需要哪个调度器？混合线性注意力的 mamba 组 block 上限是否反噬 `max_num_seqs`？ | 上游 V1 `Scheduler`/`AsyncScheduler` 默认装配即可 | 专用调度器经 `scheduler_cls` 装配 + `check_and_update_config` 互斥校验 fail-fast；已知限制（如 recompute CPU offload 场景须关 async scheduling）显式声明 |
| 3 | EngineCore 配置中枢（§一.5） | E1-E12 逐项：本模型命中哪些项、归属哪层？ | 逐项判定后本层归口项（E4/E5/E6/E12）均无需改动 | E1-E12 标记清单逐项 `✅/❌/⚠️` + 改动点；归属其他层的项按 §一.5 指针分发到对应层 spec。**配置与调度适配必须放 platform 组**（engine-core 先于 worker 启动看到配置） |
| 4 | prefix cache 运行时语义（§二.6） | 模型是否混合线性注意力？开 prefix cache 的 align 前置、与 MTP 互斥项、三个已实证坑是否命中？ | 非线性注意力模型，或不开 prefix cache（附理由） | `mamba_cache_mode=align` + chunked prefill 强制依赖 + MTP 互斥显式检查；`prefix_cache_retention_interval` 默认值坑、mamba block 边界对齐（#51113 投毒先例）、block 撑大致短前缀失效——命中项转化为验收用例 |
| 5 | 投机解码调度（§二.7） | 模型带 MTP / draft model / DSpark 吗？draft 层 KV group 归属、回滚路径、block size 约束？ | 模型无 MTP/eagle/draft 权重 → 零适配（附理由） | `patch_speculative_config.py` 与 MTP 模型注册；线性注意力投机的 state 快照/回滚（每请求多 k 个 state 槽）；DSpark block size 约束（如 `num_speculative_tokens<5` 非法）与 `dynamic_spec_config`；组合回归高发三项（capture key 覆盖 1+k / chunked-prefill 边界 / DP padding）转化为验收用例 |
| 6 | PD 分离与混合注意力（§二.8） | 目标部署是否 PD 分离？模型是否多 KV group（混合注意力）？connector 是否支持？ | 无 PD 分离部署要求，或单 KV group 模型走既有 connector（附理由） | 多 group 须实现 `SupportsHMA` 接口语义（remote key 带 `kv_cache_group_id` / 跳 null block / LCM 对齐 / pin 线性 block）；layerwise KV Pool 不支持混合 group 的限制声明（混合模型只能走 bulk 路径） |

## 输出契约：调度层 design spec

落盘 `$ASCENDBOT_FILE_PATH/design/schedule-design-spec.md`（同时作为调度层章节汇入总设计文档），**必须包含**：

1. **场景判定总表**：`| 场景# | 是否命中 | 是否需要适配 | 理由（代码/证据引用） |`——知识库每章一行全覆盖，新场景追加行并标 ⚠️；
2. **逐场景方案块**：命中且需适配的场景，每个一个方案小节（选型与理由 / 实现方式 / 落点 / **实现依据：golden-schedule-knowledge.md 章节号** / 验收方式）——**「实现依据」是 Developer 选择性执行的索引，缺此字段视同方案不完整**；
3. **KVCacheSpec 定稿**：逐层注意力形态 × cache 构成 × spec 承接 × page size 的最终选型表，**显式标注「定稿前禁止进入性能工作」**——事后更改 spec 会级联推翻调度与图模式配置；
4. **E1-E12 标记清单**：逐项 `✅/❌/⚠️` + 改动点 + 归属层（归属其他层的项注明分发去向）。

**打回条件**：场景总表缺行；「无需适配」无理由；新场景未显式声明；缺 KVCacheSpec 定稿或定稿未标注「定稿前禁止进入性能工作」；E1-E12 标记缺项——任一命中即由主控打回 Designer 补齐。

## 交接链

```
golden-designer.md 步骤 2（加载本文件）
  → 产出 design/schedule-design-spec.md
  → golden_flow Phase 1 设计完整性检查（打回条件见上）
  → golden_flow Phase 2：Developer 按 golden-schedule-adapter.md 落地流程执行——
      以 spec 判定表为过滤条件，只实现「需要适配」的场景；
      代码写法按方案块「实现依据」查阅 golden-schedule-knowledge.md 对应章节
  → Stage 3 / G4：按 spec 的组合回归清单执行组合矩阵回归（量化 × 图 × 投机 × CP/PD）
```
