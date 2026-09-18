# Parallel Flow（占位）— Day0 Stage 2：并行量化版本（跑得稳）

> ⚠️ **占位声明**：本 flow 尚未实现。主控 `SKILL.md` 推进到 Stage 2 时，应**显式提示【并行量化 flow 尚未接入】**，输出本文件的框架定义供人工或后续版本接管，不得自行编造执行步骤。

## 阶段目标（对齐 AscendBot 四阶段定义）

基于 Golden 基线版本，按选定的**量化策略**与 **KV 缓存方案**，设计合理的**并行策略**（TP / EP / DCP / PCP），完成推理服务部署运行，确保推理精度正确——构建**资源使用合理的版本**。

## 入口条件（Stage 1 出口证据）

- Golden 阶段 G0-G5 全过：`./.day0/<model>/` 下 preflight/design/impl/smoke/accuracy/graph/review 产物齐全；
- eager + bf16 精度基线（G3 证据）归档——本阶段的精度对比基准。

## 规划出口判据（草案，实现时细化）

1. **并行策略合法性**：TP/EP/DCP/PCP 整除与互斥约束算术校验通过（MLA 模型 `TP % DCP == 0`；GQA `num_q_per_kv % DCP == 0`；PCP/DCP 互斥且 PCP 仅 MRV2）；
2. **量化路径正确**：量化格式自动检测命中、反量化路径无 Missing/Unexpected，EPLB 量化白名单校验（如需 EPLB）；
3. **服务部署运行**：目标并行配置下真实权重拉起成功，HTTP 200 且输出非空；
4. **精度对齐**：并行 + 量化配置下精度对齐 Golden eager 基线，量化损失在约定阈值内（阈值随模型量化格式在实现时定义）；
5. **组合回归**：量化 × 并行组合纳入 E2E 配置（`tests/e2e/models/configs/<Model>.yaml`）。

## 已知技术难点（实现时需覆盖）

- 并行切分策略需随量化权重分布动态重算；
- 量化策略 × 并行方案组合多，验证矩阵需显式裁剪；
- KV cache 方案（block size、C8、DCP 分片）与并行策略联动校验。

## 产物目录（规划）

`./.day0/<model>/parallel/`：并行策略报告、量化验证证据、精度对比报告。
