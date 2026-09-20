# Feature Designer — Stage 3 特性叠加设计规范（占位）

> ⚠️ **占位声明**：本文件由 `.claude/agents/designer.md`（阶段路由器）在 **Stage 3** 时加载执行。`flows/feature_flow.md` 当前为占位，本文件是设计规范的**草案**——设计对象的清单是确定的，执行细节随 flow 实现后补充。
> 目标：产出特性清单、叠加顺序与回归计划的设计文档，供 Developer/Tester 逐项执行。

## 输入

- 读 `./.day0/<model>/tracker.md` 确认当前阶段 = Stage 3 与你的步骤行。
- Stage 2 签收单 + 并行量化精度基线——逐项叠加的回归基准是**上一配置**，不是 golden 基线。
- Stage 1 设计文档的**组合矩阵回归清单**（本阶段的覆盖依据）。

## 方法论文档（按内容地图加载）

`.claude/skills/day0-inference/reference/adapt/feature-adapter/feature-adapter.md` 的内容地图是本阶段的加载清单：

- `golden-adapter/golden-worker-adapter.md` §2.5 ACLGraph 图模式 + `.claude/agents/performance.md` 验收素材（逐级开图 / 捕获计数 / 六类不可入图）
- `golden-adapter/golden-schedule-adapter.md` §二.6-8 运行时调度语义（prefix cache / 投机解码 / PD 分离）
- `golden-adapter/golden-adapter.md` §5 组合矩阵回归清单（历史事故高发区）

## 设计产出（设计文档必须包含）

1. **特性清单与适用性**：本模型叠加的特性子集（Prefix Caching / 投机解码 / ACLGraph / FlashComm / EP/EPLB 等），≥5 项或逐项标注 not-applicable 及理由。
2. **叠加顺序与回归基准**：逐项开启的顺序（禁止一次性全开），每项的回归基准配置（= 上一项的最终配置）。
3. **组合矩阵裁剪结论**：量化 × 图 × 投机 × CP/PD 中本模型实际覆盖的组合项（对照 Stage 1 清单）。
4. **图模式目标级别声明**：eager / PIECEWISE / FULL_DECODE_ONLY 的目标与依据（稀疏/线性注意力只承诺 UNIFORM_BATCH）。
5. **给 Developer 的执行要点**：每个特性的改动点与验证方式，逐项列出。
