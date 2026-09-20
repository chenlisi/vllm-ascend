# Feature Adapter — Day0 Stage 3：特性叠加方法论配套（占位）

> ⚠️ **占位声明**：本目录是 Stage 3（特性叠加，跑得快）的方法论配套。流程文件 `.claude/skills/day0-inference/flows/feature_flow.md` 当前为占位，实现前本文档只提供**内容地图**——列出 Stage 3 设计/叠加时应加载的方法论章节，不提供执行步骤；执行步骤以 feature_flow.md 实现为准。

## 定位

Stage 3 在并行量化版本（Stage 2 出口）上**系统性集成 5+ 性能特性**（Prefix Caching / 投机解码 MTP/DSpark / ACLGraph 全图 / FlashComm / EP 等），逐项叠加逐项回归，确保叠加后精度无劣化。Stage 1 全程 eager 不验图——图模式验证从本阶段开始。

## 内容地图（Stage 3 应加载的方法论章节）

| 主题 | 文档与章节 | Stage 3 关注点 |
|---|---|---|
| ACLGraph 图模式 | `../golden-adapter/golden-worker-adapter.md` §2.5 + `.claude/agents/performance.md`（验收素材：逐级开图 / 捕获计数 / 六类不可入图） | 逐级开图顺序、meta 实现复核、UNIFORM_BATCH 预期管理 |
| 投机解码调度 | `../golden-adapter/golden-schedule-adapter.md` §二.7 | lookahead/draft 分组/回滚路径、verify 步 1+k 与 capture size 对齐 |
| prefix cache | `../golden-adapter/golden-schedule-adapter.md` §二.6 | `mamba_cache_mode=align`、chunked prefill mamba 边界、投毒先例 |
| PD 分离 | `../golden-adapter/golden-schedule-adapter.md` §二.8 | connector `SupportsHMA`、layerwise KV Pool 混合 group 限制 |
| 组合矩阵回归 | `../golden-adapter/golden-adapter.md` §5 | 量化 × 图 × 投机 × CP/PD 组合项（历史事故高发区）逐项覆盖 |

## 与流程的衔接

- 入口条件、出口判据草案、已知难点见 `../../../flows/feature_flow.md`；
- Designer 在本阶段的设计规范见 `.claude/skills/day0-inference/reference/design/feature-designer/feature-designer.md`（由 designer 阶段路由器按当前阶段加载）；
- 回归基准 = Stage 2 签收的并行量化基线（SKILL.md「跨阶段产物基线链」）；服务矩阵与性能达标验收属 Stage 4。
