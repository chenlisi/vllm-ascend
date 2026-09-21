---
name: vllm-ascend-reviewer
description: AI-powered code review for vLLM Ascend PRs - understanding-driven with historical rules as supplement
description_zh: vLLM Ascend PR AI 代码评审 - 理解驱动，历史规则作为补充
---

# vLLM Ascend AI Reviewer Skill

对 vllm-ascend 项目的 PR 进行代码评审。**以理解代码为核心，历史规则为补充**。

## What it does

本 Skill 提供**理解驱动**的评审能力：

1. **理解代码**：逐 commit 拆解，理解每个改动的职责和上下文
2. **风险扫描**：按正确性、性能、一致性、死代码、测试覆盖等维度扫描
3. **规则补充**：用 536 条历史规则做模式类问题的补充检查
4. **历史案例**：发现问题后，从 798 份历史报告中查找类似案例作为参考

## 架构

```
┌─────────────────────────────────────────────────────┐
│  核心：理解驱动                                       │
│  ┌─────────────────────────────────────────────────┐ │
│  │  Step 1-4: 拉 PR → 拆 commit → 风险扫描 → 分级  │ │
│  └─────────────────────────────────────────────────┘ │
│                    ↓                                  │
│  补充：历史规则库（模式类问题）                         │
│  ┌─────────────────────────────────────────────────┐ │
│  │  references/rules_clustered.md  (536条规则)      │ │
│  │  references/_rule_summary.json  (程序化索引)     │ │
│  │  scripts/search_rules.py        (检索脚本)       │ │
│  └─────────────────────────────────────────────────┘ │
│                    ↓ 发现问题后                       │
│  参考：历史报告冷库（类似案例）                         │
│  ┌─────────────────────────────────────────────────┐ │
│  │  references/reports/  (798份完整报告，16 类)      │ │
│  └─────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────┘
```

## 工作流（核心）

```
Step 1: 信息收集（本地 git）
    ├─ git fetch origin pull/<PR>/head:pr-<PR> → 拉取 PR 分支
    ├─ git diff main...pr-<PR> → 总 diff
    ├─ git log main..pr-<PR> --oneline → commit 列表
    └─ git show <SHA> → 按提交粒度看改动
    ↓
Step 2: 逐 commit 拆解
    └─ 理解每个 commit 的职责，理清改动之间的逻辑关系
    ↓
Step 3: 按风险维度扫描（对每个改动块）
    ├─ 正确性：边界值（0/None/空）、assert vs raise、属性访问安全
    ├─ 性能：热路径 Python 循环、host-device 往返、kernel 重编译
    ├─ 一致性：kernel 实现与 reference 是否对齐、新增字段上下游是否都填
    ├─ 死代码：返回值是否被消费、守卫条件是否互斥
    └─ 测试覆盖：新增分支是否都有用例、边界配置是否测到
    ↓
Step 4: 规则库补充检查（模式类问题）
    ├─ 命名规范、注释风格、copyright
    ├─ 配置校验、错误处理模式
    └─ 使用 search_rules.py 检索相关规则
    ↓
Step 5: 优先级分级
    ├─ 必须修：会导致线上 crash 或功能错误
    └─ 可选改进：代码质量提升
    ↓
Step 6: 输出评审报告
```

---

### Step 1: 信息收集（本地 git）

在本地仓库中获取 PR 信息：

```bash
# 拉取 PR 分支
git fetch origin pull/<PR_NUMBER>/head:pr-<PR_NUMBER>

# 总 diff（文件列表 + 完整改动）
git diff main...pr-<PR_NUMBER> --stat    # 文件清单
git diff main...pr-<PR_NUMBER>           # 完整 diff

# commit 列表
git log main..pr-<PR_NUMBER> --oneline

# 逐 commit 看改动
git show <SHA>
```

输出过大时用 `git diff ... -- <file>` 按文件分段查看。

---

### Step 2: 逐 commit 拆解

对每个 commit 理解：
- **职责**：这个 commit 解决什么问题？
- **改动范围**：改了哪些文件？改了哪些函数？
- **依赖关系**：这个 commit 依赖前面哪个 commit？

示例：
```
PR #15190 有 4 个 commit：
1. ddd042d: BSND 注意力 + Triton slot 映射 → 核心功能
2. 4cfc37b: DSpark 元数据 → 依赖 commit 1 的元数据结构
3. 9ae03be: Mooncake DCP 广播 → 依赖 commit 1 的 DCP size
4. d3e439c: Mamba replicated → 独立修复
```

---

### Step 3: 按风险维度扫描

对每个改动块，按以下维度扫描：

#### 正确性
- **边界值**：0、None、空列表、空 tensor 是否处理？
- **assert vs raise**：生产路径是否用裸 assert？应该用 `raise RuntimeError(...)` 或明确的错误信息
- **属性访问安全**：`self.xxx` 是否在访问前初始化？是否用 `getattr(self, 'xxx', None)` 更安全？
- **配置值边界**：`is not None` 和 truthy 的区别，0 是否是合法值？

#### 性能
- **热路径 Python 循环**：decode step 每帧都执行的函数，是否有 Python for 循环？应该向量化
- **host-device 往返**：是否有不必要的 CPU-GPU 数据传输？
- **kernel 重编译**：Triton kernel 中 `tl.constexpr` 参数是否在运行时变化？

#### 一致性
- **kernel 与 reference 对齐**：Triton kernel 实现是否与 Python reference 实现语义一致？
- **字段上下游覆盖**：新增的 metadata 字段，上游是否都填了？下游是否都读了？
- **API 对齐**：是否遵循 vLLM 上游的 API 规范？

#### 死代码
- **返回值是否被消费**：函数返回了值，调用方是否使用了？
- **守卫条件是否互斥**：`if A and not B` 中，A 和 B 是否可能同时为真？
- **未使用的参数**：函数参数是否都被使用？

#### 测试覆盖
- **新增分支是否有用例**：每个 `if` 分支是否都有测试覆盖？
- **边界配置是否测到**：如 `dcp_size=0`、`interleave_size>1` 等边界
- **新增文件的测试**：新增的 `.py` 文件是否有对应的测试？

---

### Step 4: 规则库补充检查

用历史规则做模式类问题的补充检查：

```bash
# 按文件路径匹配相关规则
python scripts/search_rules.py match "vllm_ascend/envs.py"

# 按关键词搜索规则
python scripts/search_rules.py search "环境变量"

# 按 PR 编号查找相关规则
python scripts/search_rules.py pr 10023
```

**规则库擅长的问题**：
- 命名规范（单复数、大小写）
- 注释风格（缩写展开、英文注释）
- Copyright header
- 配置校验模式
- 错误处理模式

**规则库不擅长的问题**（需要 Step 3 的理解驱动）：
- 逻辑错误、边界条件
- 性能热路径
- 死代码
- 跨文件一致性

---

### Step 5: 优先级分级

把发现的问题分级：

**必须修（合入前必须解决）**：
- 会导致线上 crash
- 会导致功能错误
- 裸 assert 在生产路径
- 属性访问可能 AttributeError

**可选改进**：
- 性能优化（向量化 Python 循环）
- 代码质量（死代码清理、注释补充）
- 测试覆盖（边界配置测试）

---

### Step 6: 输出评审报告

```markdown
# PR #<NUMBER> 代码审查

## PR 概述
- 标题：...
- 作者：...，目标分支：...，N commits，M 文件改动
- 改动覆盖：...

## 总体评价
- 整体设计：...
- commit 划分：...
- 测试覆盖：...
- 存在的问题：...

## 主要问题（必须修）

### 1. 问题标题
- **位置**：`file.py:123`
- **问题**：具体问题描述
- **建议**：修复建议（带代码示例）
- **参考**：PR#XXXX（历史案例，如有）

### 2. 问题标题
...

## 次要问题（可选改进）

### 1. 问题标题
...

## 测试评价
- 新增测试覆盖：...
- 缺失的测试：...

## 合入建议
- 必须修：问题 1、2、3
- 可选改进：问题 4、5、6
- CI 状态：...
```

---

## File layout

| File | Purpose |
| ---- | ------- |
| `SKILL.md` | Skill 定义、工作流（本文件） |
| `references/rules_clustered.md` | 536 条规则按语义聚类（16 类），用于补充检查 |
| `references/_rule_summary.json` | 规则索引数据（JSON 格式） |
| `references/reports/` | 798 份完整报告，按 16 类组织，用于查找类似案例 |
| `scripts/search_rules.py` | 检索脚本（规则搜索、报告查询） |

## 规则库概览

### 统计

| 指标 | 数量 |
|------|------|
| 唯一规则 | 536 条 |
| 来源 PR | 532 个 |
| 完整报告 | 798 份 |
| 语义聚类 | 16 个 |

### 规则聚类（16 个类别）

| 类别 | 规则数 | 说明 |
|------|--------|------|
| 配置与环境变量 | 108 | envs.py 使用规范、配置校验 |
| 模型与推理 | 94 | 模型适配、推理逻辑 |
| 文档与注释 | 67 | 注释风格、缩写展开、文档模板 |
| 代码规范与风格 | 62 | 命名规范、代码风格、架构约定 |
| 硬件适配 | 54 | NPU 特有行为、D2H 同步、NZ 格式 |
| 错误处理与日志 | 37 | try-except 规范、logging 格式 |
| 版本兼容与同步 | 26 | vLLM 上游同步、patch 检查 |
| 性能优化 | 14 | Triton kernel、内存优化 |
| 构建与部署 | 12 | CMake、Docker、CI 配置 |
| 测试与CI | 12 | 测试覆盖、CI 流水线 |
| 未分类 | 11 | 其他 |
| KV Cache与存储 | 10 | KV cache 管理、存储优化 |
| 类型与接口设计 | 5 | 类型注解、接口设计 |
| 内存与张量管理 | 5 | 内存分配、张量操作 |
| 并发与分布式 | 3 | 并发控制、分布式通信 |
| 安全与健壮性 | 2 | 输入校验、异常处理 |

## Key constraints

- **只读评审**：不修改代码，只输出评审意见
- **理解优先**：先理解代码，再用规则补充
- **历史案例**：发现问题后，查找类似历史案例作为参考
- **严重度分级**：
  - `必须修`：会导致线上 crash 或功能错误
  - `可选改进`：代码质量提升

## 与 reviewer.md agent 的配合

本 Skill 提供评审工作流 + 规则库 + 历史案例支持，与 `.claude/agents/reviewer.md` 配合使用：

1. **reviewer.md** 定义评审流程和输出格式
2. **本 Skill** 提供理解驱动的工作流 + 规则库补充 + 历史案例参考
3. 评审时，reviewer agent 按本 Skill 的工作流执行

## References

- 规则来源：798 份评审报告，从 532 个历史 PR 中提取
- 规则聚类：按语义相似度分为 16 个类别
- 设计文档：`reviewer_architecture_design.md`
