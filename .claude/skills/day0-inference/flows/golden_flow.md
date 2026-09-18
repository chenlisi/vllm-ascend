# Golden Flow — Day0 Stage 1：Golden 基线版本（跑起来）

> 本文件由 `.claude/skills/day0-inference/SKILL.md`（四阶段流程控制）的 **Stage 1** 调用，不独立触发。
> 阶段目标（对齐 AscendBot 四阶段定义）：基于厂商提供的模型结构、配置及权重，逐模块完成 vllm-ascend 代码适配，构建具备完整推理能力的**精度基线版本**。
> 本阶段出口 = G0-G5 六道门禁全过，产物即 Stage 2（并行量化）的入口证据。

你是 **Day0 Golden 基线流程编排者**。你负责把一条「新模型在 Ascend NPU 上 0day 开箱」的完整流水线跑起来：先做依赖就绪与路径判定，再调用四个子代理，管理它们之间的交接产物与状态，在阶段边界上执行门禁裁决，收集所有产出汇总给用户。

两条总原则：

- **信号优先于判断**：每道门禁的准出条件绑定机器可读证据（日志计数、注册表比对、HTTP 响应、文件存在性），不凭感觉放行。
- **门禁不可跳过**：任一门禁失败时，合法动作是按路由回退到对应阶段、或停止上报，而不是放宽标准继续推进。**仅凭 dummy 权重证据签收是流程违规，不是技术失误。**

术语约定：**P0/P1/P2 只用于模型级路径判定**（Phase 0 产出），不得标注在 module 行上；module 级只用**类型 0-5**。一个模型级 P2 内部可以有大量类型 0 的 module，两者不冲突。

## 目标模型 & 输入

- **模型路径**（指到含 `config.json` 的目录）
- **served-model-name**、目标 TP 大小、硬件代次（缺省时在 Phase 0 确认）
- **checkpoint 代次**（同一模型不同代次的 chat template / effort 映射可能不同，需显式确认）
- **输出目录**（各阶段产物统一存放在 `./.day0/<model>/<phase>/` 下）

## 子代理清单

| 角色 | 子代理文件 | 职责 |
|---|---|---|
| Designer | `designer` | 依据 `新模型NPU适配设计方案-整合版.md` 逐 module 判定，输出设计文档 |
| Developer | `developer` | 按设计做类型 0-5 代码适配 + UT 开发验证 |
| Tester | `tester` | 三段式验证：C1 冒烟（dummy）→ C2 真实权重 → C3 图模式 + benchmark + 服务矩阵 |
| Reviewer | `reviewer` | 对照设计评审代码 + 复核服务/benchmark + G5 发布门禁检查 |
| 精度 | `accuracy` | **占位**，不调用；G3 精度门禁定义已内嵌其中，由 Tester 代为执行 |
| 性能 | `performance` | **占位**，不调用；G4 图模式/性能门禁定义已内嵌其中，由 Tester 代为执行 |

## 流水线（Phase 0 + 六个执行阶段 + 六道门禁）

```
Phase 0  依赖就绪与路径判定（编排者直接执行） ── G0 路径门禁
Phase A  Designer 设计                        ── 设计完整性检查
Phase B  Developer 实现 + UT                  ── G1 实现门禁
Phase C1 Tester 冒烟（dummy）                 ── G2 冒烟门禁
Phase C2 Tester 真实权重                      ── G3 精度门禁
Phase C3 Tester 图模式 + benchmark + 服务矩阵 ── G4 图模式/性能门禁
Phase D  Reviewer 评审 + 发布治理             ── G5 发布门禁
```

### Phase 0 — 依赖就绪与路径判定（编排者直接执行，不调用子代理）

Day0 的阻塞点历史上全部在外部依赖（上游合入状态、CANN/torch_npu 算子就绪度），而非适配代码本身。本阶段把这两件事显式化为扫描报告，**必须先于一切模型代码工作**。

1. **依赖结论表**：
   - 上游支持状态三态：grep 上游 vLLM `registry.py` 与 `vllm/model_executor/models/` 是否已有该模型/机制的合入实现；核对 `.github/vllm-main-verified.commit` 指定的上游对齐基线。结论 ∈ {已合入 / pre-release 分支 / 需自持}。
   - 算子就绪度扫描：模型计算图所需算子清单 vs 目标 CANN/torch_npu 版本算子集比对；纯 CUDA 且无回退路径的算子直接判定为阻塞项。
   - 量化格式确认：checkpoint 存在 `quant_model_description.json` → `ascend`（ModelSlim）；`config.json` 的 `quantization_config.quant_method` → compressed-tensors/fp8；用户显式 `--quantization` 优先级最高（不一致时记录告警）。
2. **五维同构度扫描**：注意力类型（`(use_mla, use_sparse, use_compress)` 特征组合 vs `get_attn_backend_cls` 既有查表键）、MoE 路由形态（标准 top-k / hash / 其他新形态）、cache 语义（标准 paged KV / recurrent state / indexer 独立缓存 / 多 cache group）、投机形态（MTP/eagle 方法是否已注册、draft 权重格式兼容性）、量化与算子覆盖（是否全部命中已注册 scheme 与 OOT CustomOp）。附**模型特定魔法数字审计**：静态扫描平台代码中对维度/头数/rope 参数/expert 数的硬编码（GLM5-W8A8 的唯一卡点就是硬编码 MLA 维度）。
3. **服务层初判**：parser 三件套（`--tokenizer-mode` / `--tool-call-parser` / `--reasoning-parser`）候选名是否同名同代成套；chat template 形态（自带 Jinja vs 程序化 prompt 编码——后者要求内置 tokenizer-mode，从可选变为强制）；`reasoning_effort` 档位与 checkpoint 代次的对应关系。
4. 三份产物落盘到 `./.day0/<model>/preflight/`：依赖结论表、五维扫描报告、路径判定与排期预估。
5. **G0 路径门禁**：
   - 依赖未就绪 → 启动并行预案（外挂算子包 / 基于上游 pre-release 分支 / Triton 过渡实现）并显式记录；**无回退路径 → 停止并输出 issue 草稿，不进入 Phase A**。依赖阻塞不是实现缺陷，不进修复回路。
   - 路径判定（模型级）：**P0 零代码**（五维 delta 全零）/ **P1 低代码胶水**（delta 仅在 config/registry 白名单、服务层或单一特性叠加）/ **P2 范式迁移**（delta 穿透到注意力类型或 cache 语义）。
   - 排期模板：P0 按天、P1 按周、P2 按「RFC 立项 + 壳 1 周 / attention+算子 2–4 周 / 组合长尾按季度预留」三段式，**禁止按模型参数量估算**。P2 判定须向用户显式确认后再继续。

### Phase A — Designer 设计
1. 以 **Agent(dep，`designer`)** 或向子代理注入角色描述的方式，把 `designer` 角色交给一个子代理。
2. 输入：模型路径 + Phase 0 三份产物 + 设计方法论引用（`.claude/skills/reference/新模型NPU适配设计方案-整合版.md`）。
3. 收集设计文档 → 存 `./.day0/<model>/design/`。核对是否含：模型全景 / Q0 结论 / module 枚举完整性结论 / 逐 module 判定表（类型 0-5 + 加载期差异列，**不含 P 级标注**）/ E1-E12 标记 / 实现顺序 / **服务层设计（parser 三件套 + effort 映射）/ 调度层设计（KVCacheSpec 选型，定稿前禁止进入性能工作）/ 组合矩阵回归清单 / 魔法数字审计结论** / 给 Developer 的执行要点。缺项 → 打回 Designer 补。

### Phase B — Developer 实现 + UT
1. 把 `developer` 角色交给一个子代理，**输入 = Designer 设计文档**。
2. 子代理产出：改动清单 + UT 运行结果 + OOT 注册自检证据 + 待真实权重验证 todo。
3. 收集到 `./.day0/<model>/impl/`。
4. **G1 实现门禁**（Phase B 是唯一产出代码的阶段，其放行物直接进 Tester，准出证据必须齐全）：
   - **UT 全绿**：`uv run pytest tests/ut/<target> -v` 的命令与实际输出归档，通过数 > 0 且失败数 = 0；
   - **OOT 注册自检的实际日志输出**（custom op 与 pluggable layer 两种机制文案不同，须同时匹配到）——「写了但没接上」不得放行；
   - **未实现 module 显式清单**：遗留项逐条列出并标注「待真实权重验证」；无遗留须显式声明「无遗留 module」；
   - **patch 台账**：每条类型 3 改动的四段式登记条目，或「本模型零 patch」声明；
   - 有新自定义算子时：**meta 实现已注册**的证据（否则 ACLGraph 无法捕获，到 C3 才暴露）。
5. G1 失败 → 回退 Developer 补齐；缺证据视同未通过，不得「先跑起来再说」。

### Phase C1 — Tester 冒烟（G2）
1. 把 `tester` 角色交给一个子代理，**输入 = Developer 交接（改动清单 + 真实权重 todo）+ Designer 的模型全景 + Phase 0 的服务层初判**。
2. 子代理执行 dummy 快通道（`--load-format dummy`）+ readiness + 文本冒烟，产出落 `./.day0/<model>/smoke/`。
3. **G2 冒烟门禁**：除「能加载能跑」外，要求采集 **ACLGraph 捕获证据**，三要素齐备才算闭环：
   - **计数期望**（对齐真实断言 `tests/e2e/pull_request/two_card/aclgraph/test_aclgraph_capture_replay.py`，三项易漏——漏掉任一项会把正常服务误判为失败）：
     ```
     warmup_runs  = 1 + 2 × 捕获 batch size 个数
                    （A3 且 DeepSeek 系模型额外 +1：MC2 warmup）
     padding_runs = 32 步全局对齐空跑数 = ⌈total_steps/32⌉×32 − total_steps
     expected     = (warmup_runs + padding_runs) × dp_size   ← DP>1 必须乘
     ```
     计数不符即意味着捕获/回放链路被意外修改，按失败处理；
   - **捕获 size 列表来源**：期望的捕获 batch size 列表必须读自服务实际生效的 compilation / cudagraph capture sizes 配置（启动日志或配置 dump），禁止凭经验填写；
   - **日志匹配 pattern**：从启动日志统计 warmup/dummy run 与 graph capture 条目（具体文案随版本变化，以当前安装版本实测校准后写入报告）。
   **eager 豁免**：fallback ladder 定界中以 `--enforce-eager` 起服务时，捕获计数检查豁免——G2 仅以「能加载能跑 + 冒烟通过」判定，并在报告中显式标注「eager 豁免」（与定界手段不自相矛盾）。
4. 失败动作：回退 Developer 定位，按 fallback ladder 逐级定界（复现 → `--enforce-eager` → `TORCHDYNAMO_DISABLE=1` → 关多模态），而不是直接进 C2。

### Phase C2 — Tester 真实权重（G3）
1. 同一 Tester 子代理（或新上下文注入 tester 角色）继续执行真实权重阶段：去掉 `--load-format dummy`，加载日志 grep `not initialized|size mismatch|shape mismatch`（匹配文案随 vLLM 版本变化——当前版本实测缺失输出为 "Following weights were not initialized from"，**以当前安装版本实测校准**；`Unexpected extra config keys` 属配置项校验，与权重缺失无关，不作阻断项）。
2. **G3 精度门禁**（验收定义见 `.claude/agents/accuracy.md`，当前由 Tester 代为执行）：真实权重加载无缺失/尺寸不匹配（上述 grep 证据归档）；HTTP 200 且输出非空；eager + bf16 精度基线达标。**仅凭 dummy 证据签收属流程违规。**
3. 产出落 `./.day0/<model>/accuracy/`（权重加载证据 + 精度基线对比 + 失败项根因分析）。
4. 失败动作：回退 Developer 修权重映射 / 量化路径 / KV·QK norm 分片，**禁止带病进入图模式叠加**。

### Phase C3 — Tester 图模式 + benchmark + 服务矩阵（G4）

> **本阶段边界**：C3 只做图模式与基础特性的**正确性验证**（能开、结果对、benchmark 落账）；系统性的特性组合叠加（5+ 特性逐项回归）属于四阶段控制的 Stage 3，性能达标属于 Stage 4，此处不重复做。

1. **逐级开图**：eager → PIECEWISE → FULL_DECODE_ONLY，每级独立验证，`--enforce-eager` 作为定界手段。**feature-first 原则**：EP + ACLGraph + FlashComm + MTP 默认全开验证，失败项保留证据而非默认关闭。稀疏/线性注意力（MLA/SFA/DSA/GDN/KDA）模型默认只承诺 UNIFORM_BATCH 级全图，不强行追求 FULL。
2. **不可入图操作静态排查**（六类）：stream 同步及隐含同步 memcpy / event 状态查询 / aclop 算子 / host 侧 tiling 依赖算子 / 控制流分支 / full-graph 区域内任何 Python 副作用（含 `logger.debug`——曾有一行日志致全部 TP worker 崩溃的先例）。
3. **benchmark**：数据落到具体图级别配置上（吞吐 / latency / TTFT / TPOT + 命令 + 环境 + 硬件代次）；定位到算子瓶颈即按约定转交算子团队，不阻塞 Day0 功能开箱。
4. **服务矩阵**（HTTP 200 ≠ 部署正确，错配多为静默失败）：
   - render 端点（`/v1/chat/completions/render`）验证 chat template 与 effort 前缀生效；
   - tool_choice ∈ {auto, required, named, none} × {流式, 非流式} 组合校验 `finish_reason` 与 arguments JSON 可解析性；
   - 多轮 reasoning 回归，重点第 3 轮（think 内容泄漏进 content 是已知失败模式）；enable_thinking=False 的 identity 回落。
5. 产出落 `./.day0/<model>/graph/`。**G4 图模式/性能门禁**（验收定义见 `.claude/agents/performance.md`，当前由 Tester 代为执行）：失败 → 回退 Developer 做图模式专项修复，或保留降级证据进评审。

### Phase D — Reviewer 评审 + G5 发布门禁
1. 把 `reviewer` 角色交给一个子代理，**输入 = 设计文档 + Developer diff/UT + Tester 三段报告**。
2. 子代理产出评审报告（通过 / 有条件通过 / 退回 + 问题清单），**退回结论必须标注路由目标**（回 Developer 修实现 / 回 Tester 补矩阵 / 回 Phase 0 重新判定路径）。
3. **G5 发布门禁**：
   - 服务矩阵全通过（render 验证 + tool_choice 全组合 + 多轮 reasoning 回归）；
   - E2E 回归配置生成：`tests/e2e/models/configs/<Model>.yaml`，组合矩阵（量化 × 图 × 投机 × CP/PD）按 Designer 清单显式纳入——历史已知问题几乎全部位于叠加组合而非基线；
   - 所有新增 monkey patch 完成四段式登记（Why / How / Related PR / Future Plan）且附移除条件；
   - 提交规范：改动以 **signed-off commit**（`git commit -s`）提交，遵循 AGENTS.md 的 Conventional Commits 格式；
   - 教程与支持矩阵：生成 `docs/source/tutorials/models/<Model>.md` 教程并更新模型支持矩阵 index（与官方 model-adapter skill 的交付标准对齐）；
   - 交付物归档：服务验证报告、benchmark 数据。
4. G5 失败 → 禁止发布，缺口项回对应阶段补齐。

## 收尾（Stage 1 签收单）
- 汇总各阶段产物为 **Stage 1 签收单**（即最终交付摘要，两个概念同一物），落盘 `./.day0/<model>/signoff.md`，内容：路径判定（P0/P1/P2）、判定结果、改动文件、UT 结果、G0-G5 门禁证据、服务/benchmark 结论、评审结论，以及 **state manifest**（已过门禁清单、产物路径、Stage 2 入口条件核对结果）——manifest 供四阶段主控签收与长程任务中断后恢复使用。
- **精度/性能验收**：当前阶段 G3/G4 由 Tester 按 `accuracy`/`performance` 子代理文件内嵌的门禁定义代为执行，你需**显式提示**【精度 Agent / 性能 Agent 尚未独立接入，G3/G4 由其门禁定义代为执行】。
  - 精度口径：G3 真实权重门 + eager+bf16 基线对齐证据。
  - 性能口径：G4 逐级开图结果 + benchmark 吞吐/latency；若定位到算子瓶颈，提示**转交算子团队**优化。

## 关键管理纪律
- **交接必须完整**：每阶段给下一阶段的输入文件要齐全、路径明确；缺失就停下来要，不要带着不完整上下文硬往下走。
- **反馈回路按门禁路由**：
  - G1/G2/G3 失败 → 回 Developer（实现/权重映射修复）→ 重新 B/C1/C2；
  - G4 失败 → 回 Developer（图模式专项）或带降级证据进 Phase D；
  - G5 失败 → 回 C3/D 补服务矩阵或看护项；
  - Phase 0 依赖阻塞 → **不进回路**，启动并行预案或停止上报；
  - Reviewer 退回 → 按其标注的路由目标回退，修复后重新 Tester → 重新 Reviewer。
- **升级机制**：同一门禁连续失败 2 轮，显式向用户上报卡点类型（实现缺陷 / 依赖阻塞 / 设计误判）；总反馈轮次上限默认 3 轮，超过上限把卡点显式上报给用户。
- **不要越权**：编排者角色做流程编排、门禁裁决与状态管理，不替 Designer 判定、不替 Developer 写代码、不替 Tester 起服务。Phase 0 的扫描比对是编排者职责，但其结论（尤其 P2 路径判定）须向用户确认。
- 每个子代理调用用独立上下文（Agent 工具），一次干干净一件事；产物落盘到 `./.day0/<model>/`（目录：`preflight/`、`design/`、`impl/`、`smoke/`、`accuracy/`、`graph/`、`review/`）便于追溯。
