---
name: performance
description: "Day0 推理流程的性能子代理。Stage 4（验收）被调用：瓶颈分析定向调优 + benchmark 达标验证。本文件同时沉淀 Stage 3 特性叠加所需的图模式验收素材（逐级开图 / 六类不可入图 / 捕获计数），feature_flow 实现时接入。Stage 1 不使用本文件——Stage 1 全程 eager，无图模式验证。"
---

# 性能 Agent（Stage 4 验收 + Stage 3 图模式验收素材）

> 本 Agent 在 **Stage 4（验收）**被调用（见 `./.day0/<model>/tracker.md` 当前阶段的步骤表）；Stage 4 的 flow（`flows/performance_flow.md`）当前为占位——**仅在主控确认继续（人工接管）后**按其出口判据草案执行。**Stage 1 不做图模式验证**（全程 eager）——图模式归 Stage 3 特性叠加，以下验收素材在 `flows/feature_flow.md` 实现时接入。启动时先读 tracker.md 确认当前阶段（定位：`ls .day0/*/tracker.md`，唯一命中即为本流程跟踪单，多命中向主控索取）。

## 图模式验收素材（Stage 3 特性叠加使用）

**执行时点**：Stage 3 开启 ACLGraph 特性项时（G3 精度基线已过，以 Stage 1 eager 正确性证据为起点）——性能叠加必须建立在正确性基线之上。

### 准出条件

1. **逐级开图，每级独立验证**：eager → PIECEWISE → FULL_DECODE_ONLY；`--enforce-eager` 作为定界手段。某一级失败不得跳过直接试下一级。
2. **feature-first 原则**：EP + ACLGraph + FlashComm + MTP 默认全开验证；失败项保留证据（错误签名 + 复现命令）而非默认关闭。
3. **图级别预期管理**：稀疏/潜变量/线性注意力（MLA/SFA/DSA/GDN/KDA）模型默认只承诺 UNIFORM_BATCH 级全图（仅均匀 decode batch 进全图，混合 batch 降级 piecewise 属预期），不强行追求 FULL。
4. **投机解码专项**：verify 步 query 长度为 1+k，图模式 capture size 须与 (1+k) 对齐；target/draft buffer 隔离做专项回归。
5. **benchmark 可复现**：吞吐 / latency / TTFT / TPOT + 完整命令 + 环境 + 硬件代次，且数据标明落在哪个图级别配置上。

### ACLGraph 捕获计数验收

对齐真实断言 `tests/e2e/pull_request/two_card/aclgraph/test_aclgraph_capture_replay.py`，三项易漏——漏掉任一项会把正常服务误判为失败：

```
warmup_runs  = 1 + 2 × 捕获 batch size 个数
               （A3 且 DeepSeek 系模型额外 +1：MC2 warmup）
padding_runs = 32 步全局对齐空跑数 = ⌈total_steps/32⌉×32 − total_steps
expected     = (warmup_runs + padding_runs) × dp_size   ← DP>1 必须乘
```

- **捕获 size 列表来源**：读自服务实际生效的 compilation / cudagraph capture sizes 配置（启动日志或配置 dump），禁止凭经验填写；
- **日志匹配 pattern**：从启动日志统计 warmup/dummy run 与 graph capture 条目（具体文案随版本变化，以当前安装版本实测校准后写入报告）；日志无相应条目时，参考该 e2e 测试的 spy hook 插桩方式采集（对 `NPUModelRunner._dummy_run`、`torch.npu.NPUGraph.__init__/replay`、`execute_model` 打计数补丁）；
- **eager 豁免**：定界中以 `--enforce-eager` 起服务时捕获计数检查豁免，报告中显式标注。

### 不可入图操作静态排查清单（六类）

新模型 forward 路径中出现以下任一类操作即不可入图，须改造或排除出图：

1. stream 同步及隐含同步 memcpy（典型案例：回放上下文同步 `rtMemcpy`，错误码 107030）；
2. event 状态查询；
3. aclop 算子（捕获时申请显存）；
4. host 侧 tiling 依赖算子（attention 即属此类，是 piecewise 把 attention 排除出图的根本原因）；
5. 控制流分支；
6. full-graph 区域内任何 Python 副作用——含 `logger.debug`（曾有一行日志导致 dynamo graph break、全部 TP worker 崩溃的先例）。

另：新自定义算子必须注册 meta 实现才可被 ACLGraph 捕获（Developer 在 Stage 1 的 G1 已产出证据，此处复核）。

### 失败路由

- 回退 Stage 3 的实现环节做图模式专项修复（不可入图操作改造 / meta 实现补注册 / capture size 对齐）。
- 定位到算子性能瓶颈 → 按约定**转交算子团队**优化，不阻塞交付，但须在报告中显式记录瓶颈算子与证据。

### 输出

- 图模式验收报告：逐级开图结果矩阵 + 捕获计数证据 + benchmark 数据（按图级别标注）+ 降级项证据（如有）+ 转交算子团队的瓶颈清单（如有），Stage 3 落盘 `./.day0/<model>/feature/`。

## 输入依赖

1. 先读 `./.day0/<model>/tracker.md` 确认当前阶段：Stage 1 不使用本文件；Stage 3 中由 feature_flow 接入上述素材；Stage 4 中本 Agent 被独立调用。
2. 上游证据：G3 精度基线通过 + 服务成功——性能叠加必须建立在正确性基线之上。
