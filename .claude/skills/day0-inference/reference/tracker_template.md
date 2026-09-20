# Day0 流程跟踪单 — <model>

> 本文件是四阶段流程的**单一状态源**，由主控（`.claude/skills/day0-inference/SKILL.md`）在立项时用本模板实例化到 `$ASCENDBOT_FILE_PATH/tracker.md`（目录命名约束见 SKILL.md「执行步骤」步骤 1）。
>
> **读纪律**：所有子代理启动时先读本文件——确认当前阶段、当前阶段步骤表中自己是否被调用、本阶段产物目录。
>
> **写纪律**：子代理完成自己负责的步骤后（**无论成败**）立即回写本文件：成功 → 该步骤行状态置 **待签收**；失败 → 状态置 **打回**；两种结局都要在**备注列**填结果摘要 + 产物/证据路径，并在进度日志追加一行。**「待签收」翻转为「已完成」只能由主控在对应门禁/出口判据通过后执行**；整阶段所有步骤「已完成」且签收单落盘后，主控翻转阶段状态并前移「当前阶段」指针。门禁裁决权不下放。
>
> 状态取值：`未开始` / `进行中` / `待签收`（子代理完成，待主控过门禁）/ `已完成`（主控门禁通过）/ `打回`（失败，备注须注明路由目标）。

- 模型路径：<path>
- 创建时间：<date>
- **环境信息**（立项时主控填充——**所有子代理命令占位符与环境变量的唯一取值来源**，不在各文档中另行猜测）：
  - work-dir（serve 工作目录）：<path>
  - venv（虚拟环境根路径，含 `bin/`）：<path>
  - served-model-name：<name>
  - TP 大小：<TP>；硬件代次：<A2/A3/A5/310P>
  - max-model-len：<min(config.json 的 max_position_embeddings, 显存预算)>
  - $VLLM（上游 vLLM 源码路径，preflight §1 采集）：<path>
  - $VLLM_ASCEND（vllm-ascend 仓根）：<path>
- **当前阶段**：Stage 1
- **当前步骤**：S1.1

## Stage 1 Golden 基线（跑起来）— 状态：进行中

> 流程与门禁详述见 `flows/golden_flow.md`；每行步骤对应其中一个 Phase。

| 步骤 | 执行 agent | 产出 | 门禁 | 状态 | 产物路径 | 备注 |
|---|---|---|---|---|---|---|
| S1.1 依赖就绪与路径判定（Phase 0） | 主控（不调子代理） | raw_evidence.md + 依赖结论表 / 五维扫描报告 / 服务层初判 / 路径判定与排期 | G0 | 进行中 | `preflight/` | — |
| S1.2 适配设计（Phase 1） | designer | 设计文档（按层组织：服务/调度/Worker/跨层，每层含适配点判定表） | 设计完整性检查 | 未开始 | `design/` | — |
| S1.3 代码适配 + UT（Phase 2） | developer | 改动清单 + UT 结果 + OOT 自检证据 | G1 | 未开始 | `impl/` | — |
| S1.4 冒烟验证（Phase 3） | tester | dummy 冒烟证据 | G2 | 未开始 | `smoke/` | — |
| S1.5 真实权重精度（Phase 4） | tester（按 `accuracy.md` 的 G3 定义执行） | 权重加载证据 + 精度基线对比 | G3 | 未开始 | `accuracy/` | — |
| S1.6 评审 + 发布治理（Phase 5） | reviewer | 评审报告 + G4 检查结论 | G4 | 未开始 | `review/` | — |

签收单：`signoff.md`（全部步骤「已完成」后由主控产出）

## Stage 2 并行量化（跑得稳）— 状态：未开始

> ⚠️ `flows/parallel_flow.md` 为占位，以下步骤是**草案**；flow 实现后以 flow 为准重排。

| 步骤 | 执行 agent | 产出 | 状态 | 产物路径 | 备注 |
|---|---|---|---|---|---|
| S2.1 并行/量化方案设计 | designer | 并行策略与量化方案 | 未开始 | `parallel/` | — |
| S2.2 方案实现 | developer | 改动清单 + UT | 未开始 | `parallel/` | — |
| S2.3 部署运行验证 | tester | 目标并行配置拉起 + 冒烟 | 未开始 | `parallel/` | — |
| S2.4 量化精度对齐 golden 基线 | accuracy | 精度对比报告 | 未开始 | `parallel/` | — |
| S2.5 评审签收 | reviewer | 评审报告 | 未开始 | `parallel/` | — |

签收单：`parallel/signoff.md`

## Stage 3 特性叠加（跑得快）— 状态：未开始

> ⚠️ `flows/feature_flow.md` 为占位，以下步骤是**草案**；逐项叠加逐项回归，每特性一行在 flow 实现后展开。图模式（ACLGraph）验收素材见 `.claude/agents/performance.md`。

| 步骤 | 执行 agent | 产出 | 状态 | 产物路径 | 备注 |
|---|---|---|---|---|---|
| S3.1 特性清单与叠加顺序设计 | designer | 特性子集 + 叠加顺序 | 未开始 | `feature/` | — |
| S3.2 特性实现 | developer | 改动清单 + UT | 未开始 | `feature/` | — |
| S3.3 逐项叠加验证（含组合矩阵） | tester | 特性叠加矩阵 + benchmark | 未开始 | `feature/` | — |
| S3.4 叠加精度回归 | accuracy | 精度回归报告（对齐上一配置） | 未开始 | `feature/` | — |
| S3.5 评审签收 | reviewer | 评审报告 | 未开始 | `feature/` | — |

签收单：`feature/signoff.md`

## Stage 4 精度/性能验收（出口）— 状态：未开始

> ⚠️ `flows/performance_flow.md` 为占位，以下步骤是**草案**。本阶段不调用 designer / developer / tester / reviewer。

| 步骤 | 执行 agent | 产出 | 状态 | 产物路径 | 备注 |
|---|---|---|---|---|---|
| S4.1 瓶颈分析与定向调优 | performance | profiling 报告 + 调优前后对比 | 未开始 | `acceptance/` | — |
| S4.2 全量精度终验 + 服务矩阵 | accuracy | 精度验收报告（golden 基线 + 组合矩阵）+ 服务矩阵报告 | 未开始 | `acceptance/` | — |

签收单：`acceptance/signoff.md`

## 进度日志

| 时间 | 步骤 | 事件 | 产物路径 |
|---|---|---|---|
| <date> | S1.1 | 立项，跟踪单实例化 | 本文件 |
