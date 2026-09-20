# Parallel Adapter — Day0 Stage 2：并行量化方法论配套（占位）

> ⚠️ **占位声明**：本目录是 Stage 2（并行量化，跑得稳）的方法论配套。流程文件 `.claude/skills/day0-inference/flows/parallel_flow.md` 当前为占位，实现前本文档只提供**内容地图**——列出 Stage 2 设计/实现时应加载的方法论章节，不提供执行步骤；执行步骤以 parallel_flow.md 实现为准。

## 定位

Stage 2 在 Golden 基线（Stage 1 出口：eager+bf16 精度基线）上叠加**并行策略（TP/EP/DCP/PCP）与量化**，产出资源使用合理、精度对齐 golden 基线的版本。本阶段逐 module 判定方法论与 golden 阶段共用（`golden-adapter/` 文档族），此处只登记 Stage 2 特需的章节与差异化关注点。

## 内容地图（Stage 2 应加载的方法论章节）

| 主题 | 文档与章节 | Stage 2 关注点 |
|---|---|---|
| 并行策略约束 | `../golden-adapter/golden-worker-adapter.md` §1.4 | DCP/PCP 整除与互斥公式、PCP 仅 MRV2、EPLB 注册与量化白名单 |
| 权重加载与 NZ 布局 | `../golden-adapter/golden-worker-adapter.md` §1.5 | 量化模型的 `packed_modules_model_mapping` 同步、FRACTAL_NZ / A5 MX 量化强制转换 |
| KV 方案与 page 一致性 | `../golden-adapter/golden-schedule-adapter.md` §一.1-3 | KVCacheSpec 选型复核（并行形态改变 page/分片假设）、跨组 page size 一致性 |
| 量化方法识别 | `../../ascend-oot.md` §6 | 量化自动检测、`supported_quantization` 白名单、MXFP4 两道门 |
| MoE 通信方式 | `../golden-adapter/golden-worker-adapter.md` §2.4 | EP 规模与 `select_moe_comm_method` 分发（部署前代入公式验算） |

## 与流程的衔接

- 入口条件、出口判据草案、已知难点见 `../../../flows/parallel_flow.md`；
- Designer 在本阶段的设计规范见 `.claude/skills/day0-inference/reference/design/parallel-designer/parallel-designer.md`（由 designer 阶段路由器按当前阶段加载）；
- 精度对齐基准 = Stage 1 签收的 golden 基线（SKILL.md「跨阶段产物基线链」）。
