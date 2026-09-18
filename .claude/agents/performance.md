---
name: performance
description: "占位子代理：性能 Agent（暂未独立接入）。G4 图模式/性能门禁的验收定义内嵌于本文件，当前由 Tester 在 Phase C3 代为执行；本 Agent 接入后接管 G4。当前阶段不要直接调用。"
---

# 性能 Agent（占位 · G4 图模式/性能门禁定义）

> ⚠️ **占位声明**：本 Agent 尚未独立接入，`Day0 推理流程编排` 当前阶段**不会调用**它。G4 图模式/性能门禁由 Tester 在 Phase C3 按本文件定义代为执行；本 Agent 接入后接管该门禁。

## G4 图模式/性能门禁

**执行时点**：Phase C3（G3 精度基线通过之后）——性能叠加必须建立在正确性基线之上。

### 准出条件

1. **逐级开图，每级独立验证**：eager → PIECEWISE → FULL_DECODE_ONLY；`--enforce-eager` 作为定界手段。某一级失败不得跳过直接试下一级。
2. **feature-first 原则**：EP + ACLGraph + FlashComm + MTP 默认全开验证；失败项保留证据（错误签名 + 复现命令）而非默认关闭。
3. **图级别预期管理**：稀疏/潜变量/线性注意力（MLA/SFA/DSA/GDN/KDA）模型默认只承诺 UNIFORM_BATCH 级全图（仅均匀 decode batch 进全图，混合 batch 降级 piecewise 属预期），不强行追求 FULL。
4. **投机解码专项**：verify 步 query 长度为 1+k，图模式 capture size 须与 (1+k) 对齐；target/draft buffer 隔离做专项回归。
5. **benchmark 可复现**：吞吐 / latency / TTFT / TPOT + 完整命令 + 环境 + 硬件代次，且数据标明落在哪个图级别配置上。

### 不可入图操作静态排查清单（六类）

新模型 forward 路径中出现以下任一类操作即不可入图，须改造或排除出图：

1. stream 同步及隐含同步 memcpy（典型案例：回放上下文同步 `rtMemcpy`，错误码 107030）；
2. event 状态查询；
3. aclop 算子（捕获时申请显存）；
4. host 侧 tiling 依赖算子（attention 即属此类，是 piecewise 把 attention 排除出图的根本原因）；
5. 控制流分支；
6. full-graph 区域内任何 Python 副作用——含 `logger.debug`（曾有一行日志导致 dynamo graph break、全部 TP worker 崩溃的先例）。

另：新自定义算子必须注册 meta 实现才可被 ACLGraph 捕获（Developer 职责，此处复核证据）。

### 失败路由

- 回退 Developer 做图模式专项修复（不可入图操作改造 / meta 实现补注册 / capture size 对齐）。
- 定位到算子性能瓶颈 → 按约定**转交算子团队**优化，不阻塞 Day0 功能开箱，但须在报告中显式记录瓶颈算子与证据。

### 输出

- 性能报告：逐级开图结果矩阵 + benchmark 数据（吞吐 / latency / TTFT / TPOT + 命令与环境 + 硬件代次）+ 降级项证据（如有）+ 转交算子团队的瓶颈清单（如有），落盘 `./.day0/<model>/graph/`。

## 计划接入点（本 Agent 转正时）

1. Tester 产出 G3 通过证据 + 服务成功，供本 Agent 在其上做图模式叠加验证。
2. 主 skill 的 Phase C3 改为调用本 Agent，替换当前「Tester 代为执行」的安排。
