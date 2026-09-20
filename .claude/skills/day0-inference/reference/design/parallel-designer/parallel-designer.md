# Parallel Designer — Stage 2 并行量化设计规范（占位）

> ⚠️ **占位声明**：本文件由 `.claude/agents/designer.md`（阶段路由器）在 **Stage 2** 时加载执行。`flows/parallel_flow.md` 当前为占位，本文件是设计规范的**草案**——设计对象的清单是确定的，执行细节随 flow 实现后补充。
> 目标：产出并行策略 + 量化方案的设计文档，供 Developer 实施。

## 输入

- 读 `./.day0/<model>/tracker.md` 确认当前阶段 = Stage 2 与你的步骤行。
- Stage 1 签收单 + golden 精度基线（`$ASCENDBOT_FILE_PATH/signoff.md`）——本阶段一切改动的回归基准。
- Stage 1 设计文档（`design/`）——模型全景、判定表、KVCacheSpec 选型结论在其上复核而非重做。

## 方法论文档（按内容地图加载）

`.claude/skills/day0-inference/reference/adapt/parallel-adapter/parallel-adapter.md` 的内容地图是本阶段的加载清单：

- `golden-adapter/golden-worker-adapter.md` §1.4 并行策略约束（DCP/PCP 整除与互斥公式、EPLB 注册与量化白名单）
- `golden-adapter/golden-worker-adapter.md` §1.5 权重加载与 NZ 布局（量化模型的 `packed_modules_model_mapping` 同步、FRACTAL_NZ / A5 MX 强制转换）
- `golden-adapter/golden-schedule-adapter.md` §一.1-3 KV 方案与 page 一致性（并行形态改变分片假设时复核）
- `ascend-oot.md` §6 量化方法识别（自动检测命中性、白名单、MXFP4 两道门）
- `golden-adapter/golden-worker-adapter.md` §2.4 MoE 通信方式（EP 规模代入 `select_moe_comm_method` 验算）

## 设计产出（设计文档必须包含）

1. **并行策略**：TP/EP/DCP/PCP 选型 + 整除/互斥约束的算术校验结果（逐条代入公式，禁止只写结论）。
2. **量化方案**：量化格式识别结论 + 权重映射同步项清单（`packed_modules_mapping` / `modelslim_config.py` / NZ 布局转换点）。
3. **KV 方案复核**：并行形态对 KVCacheSpec / page size 的影响结论（无影响须显式声明）。
4. **精度对齐计划**：与 golden 基线的对比口径（容差 / 题集 / 指标），供 accuracy 执行。
5. **给 Developer 的执行要点**：改动点逐条列出，避免 Developer 重新判断。
