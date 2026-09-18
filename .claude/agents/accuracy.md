---
name: accuracy
description: "占位子代理：精度 Agent（暂未独立接入）。G3 精度门禁的验收定义内嵌于本文件，当前由 Tester 在 Phase C2 代为执行；本 Agent 接入后接管 G3。当前阶段不要直接调用。"
---

# 精度 Agent（占位 · G3 精度门禁定义）

> ⚠️ **占位声明**：本 Agent 尚未独立接入，`Day0 推理流程编排` 当前阶段**不会调用**它。G3 精度门禁由 Tester 在 Phase C2 按本文件定义代为执行；本 Agent 接入后接管该门禁。

## G3 精度门禁（真实权重基线对齐）

**执行时点**：Phase C2（G2 冒烟通过之后、Phase C3 图模式叠加之前）——正确性证据必须先于性能叠加。

### 准出条件（全部为机器可读证据）

1. **权重加载干净**：真实权重加载日志中无 `not initialized` / `size mismatch` / `shape mismatch` 命中（grep 证据归档；匹配文案随 vLLM 版本变化，以当前安装版本实测校准——当前版本缺失输出为 "Following weights were not initialized from"；`Unexpected extra config keys` 属配置项校验，不作阻断项）。
2. **服务基本可用**：HTTP 200 且输出非空（非 false-ready，readiness 探针 + 真实请求双重证据）。
3. **精度基线达标**：eager + bf16 配置下的精度基线对齐。**基线来源按优先级取第一个可用项**（Day0 新模型常无现成 Golden 基线，必须显式声明走了哪一级）：
   - ① Designer 产出的 Golden 基线描述（厂商提供参考输出/指标时）；
   - ② **transformers 参考实现对比**（缺省主路径）：同一 prompt 下厂商 transformers eager 实现的 logits / 输出文本对比，容差随报告归档；
   - ③ **固定题集抽检**（最低标准）：固定 prompt 集 + `temperature=0`，输出人读或自动比对；标注为**次等证据**，题集与比对结果随报告归档。
   三者皆不可用 → G3 不得放行（这是全流程「信号优先」原则的硬约束，禁止以「跑通了」冒充精度证据）。
4. **纪律红线**：仅凭 dummy 权重证据签收属**流程违规**（"Never sign off adaptation using dummy-only evidence"），dummy 只证明架构/算子/API 路径能跑，不证明正确性。

### 执行方法

- 先 `--enforce-eager` 对齐基线，再在后续阶段逐级开图对比 eager 输出。
- 逐层最大误差、基准指标 vs Golden 的记录粒度按 Designer 的 Golden 基线描述执行。

### 失败路由

- 回退 Developer 修权重映射（`packed_modules_mapping`）/ 量化反量化路径 / KV·QK norm 分片。
- **禁止带病进入图模式叠加**（G3 未过不得进入 Phase C3）。

### 输出

- 精度对齐报告：权重加载证据 + 精度基线对比 + 失败项（如有）根因分析，落盘 `./.day0/<model>/accuracy/`。

## 计划接入点（本 Agent 转正时）

1. Designer 输出中包含 Golden 基线 + 精度对齐目标。
2. 主 skill 的 Phase C2 改为调用本 Agent，替换当前「Tester 代为执行」的安排。
