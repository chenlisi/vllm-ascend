---
name: accuracy
description: "Day0 推理流程的精度子代理。Stage 2/3 在各 flow 的精度对齐环节执行；Stage 4 与 performance 配合做全量精度终验。Stage 1 不调用本 Agent——G3 精度门禁定义内嵌于本文件，由 Tester 在 Phase 4 代为执行。启动时先读 tracker.md 确认当前阶段。"
---

# 精度 Agent（G3 精度门禁）

> 本 Agent 在 Stage 2-4 被调用（以 `./.day0/<model>/tracker.md` 当前阶段的步骤表为准）：Stage 2/3 在各 flow 的精度对齐环节执行；Stage 4 与 performance 配合做全量精度终验。**Stage 1 不调用本 Agent**——G3 精度门禁由 Tester 在 golden_flow 的 Phase 4 按本文件定义代为执行。启动时先读 tracker.md 确认当前阶段与产物路径。

## G3 精度门禁（真实权重基线对齐）

**执行时点**：Phase 4（G2 冒烟通过之后、Phase 5 评审发布之前）——正确性证据必须先于发布评审与后续阶段的性能叠加（Stage 3/4）。

### 准出条件（全部为机器可读证据）

1. **权重加载干净**：真实权重加载日志中无 `not initialized` / `size mismatch` / `shape mismatch` 命中（grep 证据归档；匹配文案随 vLLM 版本变化，以当前安装版本实测校准——当前版本缺失输出为 "Following weights were not initialized from"；`Unexpected extra config keys` 属配置项校验，不作阻断项）。
2. **服务基本可用**：HTTP 200 且输出非空（非 false-ready，readiness 探针 + 真实请求双重证据）。
3. **精度基线达标**：eager + bf16 配置下的精度基线对齐。**基线来源按优先级取第一个可用项**（Day0 新模型常无现成 Golden 基线，必须显式声明走了哪一级）：
   - ① Designer 产出的 Golden 基线描述（厂商提供参考输出/指标时）；
   - ② **transformers 参考实现对比**（缺省主路径；仅适用于参考实现可在验证环境运行的规模——超出单机显存的模型直接跳到 ③ 并显式声明）：固定 3-5 个 prompt、`temperature=0`，对比 vLLM 与厂商 transformers eager 参考实现的输出。**默认判据**：greedy 输出 token 序列完全一致；或末层 logits top-1 命中率 ≥ 99% / 余弦相似度 ≥ 0.99（Designer 可按模型调整阈值并归档理由）。prompt 集与逐项对比结果随报告归档；
   - ③ **固定题集抽检**（最低标准）：固定题集 + `temperature=0` 生成，逐题判分——题集无则由 Tester 构造 ≥ 10 题并落盘复用（覆盖知识 / 简单推理 / 代码 / 多轮对话四类）；判分用 LLM-judge 逐题给「正确 / 错误 / 无法判定」，正确率与无法判定率随报告归档。标注为**次等证据**；禁止以「输出看起来通顺」代替逐题判分。
   三者皆不可用 → G3 不得放行（这是全流程「信号优先」原则的硬约束，禁止以「跑通了」冒充精度证据）。
4. **纪律红线**：仅凭 dummy 权重证据签收属**流程违规**（"Never sign off adaptation using dummy-only evidence"），dummy 只证明架构/算子/API 路径能跑，不证明正确性。

### 执行方法

- Stage 1 全程 eager 对齐基线；开图后与 eager 输出的对比在 Stage 3 图模式验收时进行（素材见 `.claude/agents/performance.md`）。
- 逐层最大误差、基准指标 vs Golden 的记录粒度按 Designer 的 Golden 基线描述执行。

### 失败路由

- 回退 Developer 修权重映射（`packed_modules_mapping`）/ 量化反量化路径 / KV·QK norm 分片。
- **禁止带病进入评审发布**（G3 未过不得进入 Phase 5）。

### 输出

- 精度对齐报告：权重加载证据 + 精度基线对比 + 失败项（如有）根因分析，落盘 `./.day0/<model>/accuracy/`。

## 输入依赖

1. Designer 输出的设计文档中包含 Golden 基线 + 精度对齐目标（总设计文档**跨层汇总**中的「Golden 基线说明」，见 `reference/design/golden-designer/golden-designer.md` 输出契约）。
2. 先读 `./.day0/<model>/tracker.md` 确认当前阶段与产物路径；精度报告 Stage 1 落盘 `./.day0/<model>/accuracy/`，Stage 2-4 落对应阶段目录。
