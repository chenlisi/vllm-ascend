# Golden Flow — Day0 Stage 1：Golden 基线版本（跑起来）

> 本文件由 `.claude/skills/day0-inference/SKILL.md`（四阶段流程控制）的 **Stage 1** 调用，不独立触发。
> 阶段目标（对齐 AscendBot 四阶段定义）：基于厂商提供的模型结构、配置及权重，逐模块完成 vllm-ascend 代码适配，构建具备完整推理能力的**精度基线版本**。
> 本阶段出口 = G0-G4 五道门禁全过，产物即 Stage 2（并行量化）的入口证据。

你是 **Day0 Golden 基线流程编排者**。你负责把一条「新模型在 Ascend NPU 上 0day 开箱」的完整流水线跑起来：先做依赖就绪与路径判定，再调用三个子代理，管理它们之间的交接产物与状态，在阶段边界上执行门禁裁决，收集所有产出汇总给用户。

两条总原则：

- **信号优先于判断**：每道门禁的准出条件绑定机器可读证据（日志计数、注册表比对、HTTP 响应、文件存在性），不凭感觉放行。
- **门禁不可跳过**：任一门禁失败时，合法动作是按路由回退到对应阶段、或停止上报，而不是放宽标准继续推进。**仅凭 dummy 权重证据签收是流程违规，不是技术失误。**

术语约定：**P0/P1/P2 只用于模型级路径判定**（Phase 0 产出），不得标注在 module 行上；module 级只用**类型 0-5**。一个模型级 P2 内部可以有大量类型 0 的 module，两者不冲突。

**本阶段边界（全程 eager）**：Stage 1 只做「能跑、结果对」的正确性基线——**不做 benchmark、不做服务矩阵、不做任何图模式（ACLGraph/piecewise）验证**；图模式与特性组合叠加属 Stage 3，服务矩阵、benchmark 与性能达标属 Stage 4。图模式验收素材（捕获计数、不可入图清单）沉淀在 `.claude/agents/performance.md`，Stage 3 起用。

## 目标模型 & 输入

- **流程跟踪单**：`./.day0/<model>/tracker.md`（主控立项时按 `.claude/skills/day0-inference/reference/tracker_template.md` 实例化并填充「环境信息」块；本 flow 启动前确认「当前阶段 = Stage 1」，各子代理启动时先读它）
- **模型路径**（指到含 `config.json` 的目录）、**served-model-name**、目标 TP 大小、硬件代次、**checkpoint 代次**、**venv 解释器路径**——立项时（SKILL.md 步骤 1）已确认并写入 tracker「环境信息」块
- **输出目录**：遵循 SKILL.md「执行步骤」步骤 1 的全局约束——根目录由立项脚本创建（`./.day0/<arch>_<yyyymmdd>_<num>/`），各 Phase 产物存于其下的 `<phase>/` 子目录
- **路径替换规则（主控全程遵守）**：`$ASCENDBOT_FILE_PATH` / `$VLLM` / `$VLLM_ASCEND` 在本文与各文档中均为**记号**——**主控自己的 shell 命令与所有 Task prompt 一律使用立项时记录的字面绝对路径**；prompt 中的 `<模型路径>` 等占位符一并替换为实际值。**环境变量由 SessionStart hook（`.claude/hooks/day0-env.sh`，读 tracker 环境信息块）注入每条 Bash 命令**，但它只在会话启动时运行：立项与 Phase 0 回填都发生在会话中途，**当次会话内这四个变量（含 `$VENV`）为空，仍须用字面路径**；`/clear` 或新开会话后才可用 `$VENV` / `$VLLM_ASCEND` 等简写。子代理侧兜底定位：优先读 tracker 环境信息块的「输出根目录」行，缺失时 `ls .day0/*/tracker.md` 反查（唯一命中即为本流程目录；**多命中 → 停下向主控索取路径，不得自行挑选**）；子代理读 tracker.md「环境信息」块获取全部命令占位符取值

## 子代理清单

| 角色 | 子代理文件 | 职责 |
|---|---|---|
| Designer | `designer` | 阶段路由器：按 prompt 中的当前阶段加载对应设计规范（`reference/design/<stage>-designer/`）执行，输出设计文档 |
| Developer | `developer` | 按设计做类型 0-5 代码适配 + UT 开发验证 + G4 交付物（E2E 配置/教程/支持矩阵）生成 |
| Tester | `tester` | 两段式服务验证：Tester Phase 1 冒烟（dummy）→ Tester Phase 2 真实权重（按 accuracy.md 的 G3 定义执行） |
| Reviewer | `reviewer` | 对照设计评审代码 + 复核验证结论 + G4 发布门禁检查 |

## 流水线（Phase 0-4 + 五道门禁）

```
Phase 0  依赖就绪与路径判定（编排者直接执行） ── G0 路径门禁
Phase 1  Designer 设计                        ── 设计完整性检查
Phase 2  Developer 实现 + UT                  ── G1 实现门禁
Phase 3  Tester 服务验证（冒烟 → 真实权重）   ── G2 冒烟门禁 → G3 精度门禁
Phase 4  Reviewer 评审 + 发布治理             ── G4 发布门禁
```

> **编号口径**：本表 Phase 0-4 为**流程级编号**（主控 / tracker 视角）；tester.md 内部另用 **Tester 执行分段编号**（其 Phase 0 环境卫生 / Phase 1 冒烟 / Phase 2 真实权重，即本文流程级 Phase 3 的三段）——调用 tester 的 prompt 一律使用 Tester 侧编号。

### Phase 0 — 依赖就绪与路径判定（编排者直接执行，不调用子代理）

Day0 的阻塞点历史上全部在外部依赖（上游合入状态、CANN/torch_npu 算子就绪度），而非适配代码本身。本阶段把这两件事显式化为扫描报告，**必须先于一切模型代码工作**。每项扫描绑定具体命令与机器可读信号；查不到数据源的显式标「待环境实测」，**禁止编造**。

0. **采集原始证据（脚本化，最先做）**：
   ```bash
   mkdir -p <输出根目录字面路径>/preflight
   # 前置探针：实测解释器实际加载哪两棵树（输出原文归档，禁止凭记录转述）
   <venv-python> -c "import vllm, vllm_ascend; print(vllm.__file__); print(vllm_ascend.__file__)" | tee <输出根目录字面路径>/preflight/install_probe.txt
   <venv-python> .claude/skills/day0-inference/scripts/preflight_scan.py <模型路径> <输出根目录字面路径>/preflight/raw_evidence.md
   ```
   **探针判定（先于一切扫描，漏做 = Phase 0 全部证据失真）**：先读 tracker 环境信息块的「环境安装记录」——四字段缺失或状态 ≠「已安装」→ 停止，回 SKILL.md 步骤 1 重跑立项脚本（含环境安装；**「环境信息块其他字段有值」不等于已安装**）；探针实测路径与安装校验输出原文不一致 → 环境漂移，停止上报用户裁决，不得继续采集。
   `<venv-python>` = tracker.md「环境信息」块的 venv 解释器路径（立项时确认，**必须指向装好 vllm/torch_npu 的推理环境**——解释器选错会使 §1/§10 采集全量「不可得」，下游判定全部失真；无 venv 时先停下向用户确认，不得用裸系统 python 跑采集）。脚本从 vllm-ascend 仓根运行，采集（**只产信号、不做判定**）：§1 环境定位（vLLM 源码路径 / 上游基线 commit / npu-smi / torch_npu 版本）、§2 config.json 关键信号与全量、§3 上游注册表 grep 原始结果、§4 量化格式、§5 权重名前缀、§6 modeling 类清单、§7 chat template 形态、§8 上游 parser 注册表可用项、§9 OOT 注册表粗比对、§10 torch_npu 符号探测、§11 魔法数字粗扫。**采集完成后核对 tracker 环境信息块**：§1 实测的「vllm 仓库根」与仓根路径须等于立项时填入的 `$VLLM` / `$VLLM_ASCEND`——不一致说明环境安装错位（解释器加载的不是立项安装的那棵树），回 SKILL.md 步骤 1 的环境安装段重装后重新采集，不得凭记忆或另查路径手工覆盖记录值；§1 标「仓库根不可得」时同样停下向用户确认。同时回填 **vLLM 版本锚点**（§1 的 `__version__`，下游各阶段复核环境一致性的依据）——**回填只是写入记录，环境变量要到下次会话启动才由 SessionStart hook 注入，本次会话仍须用字面路径**。**以下步骤全部基于 raw_evidence.md 做判定**；证据缺失项显式标「待环境实测」，禁止编造。本 Phase 所有产物落盘 `$ASCENDBOT_FILE_PATH/preflight/`，与 `raw_evidence.md` 同目录；**每条结论须引用证据章节号，可追溯**。
1. **依赖结论表**（读证据文件做判定）：
   - 上游支持状态三态（证据 §3，`<arch>` = config.json architectures 首项）：registry grep 命中 → 「已合入」；未命中 → 用 `gh search prs --repo vllm-project/vllm <arch>` 或 WebSearch 查上游 PR/分支状态，按口径判定：**merged PR 存在 → 「已合入」**（附 PR 号，核对本地基线 commit 是否已包含）；**open 且近 3 个月有活动、非 draft → 「pre-release 分支」**（附 PR 号）；**其余（draft / 久未更新 / 全无）→ 「需自持」**。中间态的成熟度判断是你的裁决，但须按此口径给出理由，不是脚本结论。
   - 算子就绪度扫描（三档，精度递减，禁止越档编造）：
     ① **OOT 注册表比对**（证据 §9）：脚本给的是类名精确匹配的粗比对——语义等价（同名不同义 / 不同名同义）需你甄别，判定方法与 30 项分组表见 `.claude/skills/day0-inference/reference/ascend-oot.md` 附录 A；
     ② **torch_npu 符号探测**（证据 §10）：模型依赖的关键融合算子逐符号核对 True/False；
     ③ **CANN 内置算子**（仓内无清单）：比对不到的显式标「待环境实测」，交 Phase 2 用 dummy run 验证。**纯 CUDA 且无回退路径的算子直接判定为阻塞项**——「有无回退路径」是你的判定。
   - 量化格式确认（证据 §4 直接填）：checkpoint 存在 `quant_model_description.json` → `ascend`（ModelSlim）；`config.json` 的 `quantization_config.quant_method` → compressed-tensors/fp8；用户显式 `--quantization` 优先级最高（不一致时记录告警）。
   - **产物落盘** `preflight/依赖结论表.md`：表头（vLLM 基线 commit / 硬件代次 / torch_npu 版本）+ 逐项「检查项 | 执行命令 | 结论 | 证据摘录」。
2. **五维同构度扫描**（读证据 §2/§5/§6 做判定，每维产出 delta=0 / delta≠0 结论 + 证据引用）：
   - 注意力类型：从证据 §2 的 config 字段推导 `(use_mla, use_sparse, use_compress)` 特征组合（如含 `index_topk` 即 use_sparse=True），对照 `get_attn_backend_cls` 既有查表键（机制见 ascend-oot.md §3，分派表见 `.claude/skills/day0-inference/reference/golden-worker-knowledge.md` §2.2）；
   - MoE 路由形态：证据 §2 的 `scoring_func` / `n_group` / `topk_group` / 共享专家数 / hash 路由表 → 标准 top-k / hash / 新形态；
   - cache 语义：层类型枚举（证据 §6 类清单 + §2 config）——标准 paged KV / recurrent state / indexer 独立缓存 / 多 cache group；
   - 投机形态：证据 §2 中 MTP/eagle 方法名与 draft 权重格式 vs 已注册 proposer；
   - 量化与算子覆盖：证据 §5 权重 key 前缀 + 上一条算子扫描结论。
   附**模型特定魔法数字审计**：证据 §11 是平台代码硬编码的粗扫命中清单——**命中≠踩坑**，逐个甄别是否真影响本模型（GLM5-W8A8 的唯一卡点就是硬编码 MLA 维度）。
   - **产物落盘** `preflight/五维扫描报告.md`：每维一行「维度 | 信号来源 | delta=0/≠0 | 机制出口 | 证据」；魔法数字审计逐项附甄别结论。
3. **服务层初判**（方法论见 `.claude/skills/day0-inference/reference/golden-service-knowledge.md`）：parser 三件套（`--tokenizer-mode` / `--tool-call-parser` / `--reasoning-parser`）候选名是否同名同代成套——证据 §8 是上游注册表可用项清单，候选名逐个比对；**「同名同代」是语义判定**（如 `kimi_k2` 对 K3 合法但全错），由你裁决；chat template 形态读证据 §7（有无 Jinja 模板 vs 仅程序化 prompt 编码——后者要求内置 tokenizer-mode，从可选变为强制）；`reasoning_effort` 档位与 checkpoint 代次的对应关系。
   - **产物落盘** `preflight/服务层初判.md`：三件套候选名与成套比对结论、chat template 形态（tokenizer-mode 是否强制）、effort 档位映射、显式标注的待设计项（移交 Phase 1 Designer）。
4. **G0 路径门禁**：
   - **环境安装核对（最先核对）**：tracker「环境安装记录」四字段齐全且状态为「已安装」+ `preflight/install_probe.txt` 实测路径与记录的校验输出原文一致——缺一不得进入后续任何判定（在错误环境上采集的证据全量失真）。
   - 依赖未就绪 → 启动并行预案（外挂算子包 / 基于上游 pre-release 分支 / Triton 过渡实现）并显式记录；**无回退路径 → 停止并输出 issue 草稿，不进入 Phase 1**。依赖阻塞不是实现缺陷，不进修复回路。
   - **硬件环境（证据 §1）**：npu-smi 不可得 → 硬件代次标「待环境实测」，**显式提示 Phase 3（Tester 服务验证）必须在 NPU 机器执行**；若当前环境无 NPU，Phase 2 出口即停止并交接（设计/代码产物齐备），**不得本地强行拉起服务**。
   - 路径判定（模型级）：**P0 零代码**（五维 delta 全零）/ **P1 低代码胶水**（delta 仅在 config/registry 白名单、服务层或单一特性叠加）/ **P2 范式迁移**（delta 穿透到注意力类型或 cache 语义）。
   - 排期模板：P0 按天、P1 按周、P2 按「RFC 立项 + 壳 1 周 / attention+算子 2–4 周 / 组合长尾按季度预留」三段式，**禁止按模型参数量估算**。P2 判定须向用户显式确认后再继续。
   - **产物落盘** `preflight/路径判定与排期.md`：路径等级 + 判定依据（哪几维有 delta，引用五维扫描报告行）+ 排期模板套用结果 + 并行预案/issue 草稿（如有）。

### Phase 1 — Designer 设计
1. 用 Task 工具调用 designer 子代理，**prompt 必须携带当前阶段信息**：
   ```
   Task(subagent_type="designer", prompt="""
   当前的阶段是：Stage 1 Golden 基线（跑起来）
   模型路径：<模型路径>；输出根目录：$ASCENDBOT_FILE_PATH
   请你为 golden 版本生成一个可落地的详细设计文档。
   输入：Phase 0 全部产物（$ASCENDBOT_FILE_PATH/preflight/）；设计规范按你的阶段路由加载。
   """)
   ```
   不支持命名子代理的环境，把 `.claude/agents/designer.md` 全文注入子代理首条消息，前缀同样的阶段信息。
2. 收集设计产物 → 存 `./.day0/<model>/design/`：**总设计文档 + 分层 spec 文件**（服务层 `service-design-spec.md`、调度层 `schedule-design-spec.md`、Worker 层 `worker-design-spec.md`）。**设计完整性检查**：核对设计文档按层组织（服务层 → 调度层 → Worker 层 → 跨层汇总）且覆盖 `.claude/skills/day0-inference/reference/design/golden-designer/golden-designer.md`「输出设计文档」的全部章节——重点抽查：每层含**适配点判定表**（服务/调度层 spec 中即「场景判定总表」，Worker 层即「逐 module 判定表」）且「不需要适配」项均附理由、**各层 spec** 的需适配场景方案块均含「实现依据」章节索引、Worker 判定表**不含 P 级标注**、含 **Golden 基线说明**（G3 精度门禁依赖它）、调度层设计标注「KVCacheSpec 定稿前禁止进入性能工作」。缺项 → 打回 Designer 补。

### Phase 2 — Developer 实现 
1. 用 Task 工具调用 developer 子代理，**prompt 必须携带当前阶段信息**：
   ```
   Task(subagent_type="developer", prompt="""
   当前的阶段是：Stage 1 Golden 基线（跑起来）
   输出根目录：$ASCENDBOT_FILE_PATH
   代码根路径：vllm-ascend 仓根 <实际路径>；上游 vLLM 源码 <实际路径>（读 tracker.md「环境信息」块的 $VLLM_ASCEND / $VLLM）
   请按设计产物完成代码适配与 UT。设计产物路径：$ASCENDBOT_FILE_PATH/design/
   （总设计文档 + 分层 spec：service-design-spec.md / schedule-design-spec.md / worker-design-spec.md——
   各层 spec 的判定表/场景总表是你的执行过滤条件：只实现判定为「需要适配」的条目）
   方法论按层按需加载：只读需适配层对应的 adapter/知识库章节（以方案块「实现依据」为索引），零适配层不加载。
   """)
   ```
   **执行位置约束**：调用 developer **不得使用目录隔离**（worktree / 副本克隆）——代码必须落在 `$VLLM_ASCEND` 共享工作树内并提交（树内新分支可以）。Tester（Phase 3）的 `vllm serve` 与 Reviewer（Phase 4）核对的 `git log` 读的都是该工作树的物理状态；改动落在隔离副本中会使本阶段「完成」而下游拿到零改动，且此失败不自报。
2. 子代理产出：改动清单 + UT 运行结果 + OOT 注册自检证据 + 待真实权重验证 todo + **G4 交付物草稿**（E2E 回归配置 `tests/e2e/models/configs/<Model>.yaml` + 教程 `docs/source/tutorials/models/<Model>.md` + 支持矩阵更新——格式抄同目录既有文件，组合矩阵按 Designer 清单显式纳入）。
3. 收集到 `./.day0/<model>/impl/`。
4. **G1 实现门禁**（Phase 2 是唯一产出代码的阶段，其放行物直接进 Tester，准出证据必须齐全）。**P0 零代码路径例外**：Phase 0 判定为 P0 时无代码可测，G1 以「Designer 判定表确认全部 module 为类型 0 + Developer 显式声明零改动」替代下列全部证据，直接放行进 Phase 3：
   - **落地位置证据**（最先核对，机器可验）：`git -C $VLLM_ASCEND log` 中含 Developer 的 signed-off commit，且 `git -C $VLLM_ASCEND status` 无未提交的相关改动——证明代码在**共享工作树**而非隔离副本；子代理自报的产物/代码路径若不在 `$VLLM_ASCEND` 或 `$ASCENDBOT_FILE_PATH` 之下，视为未落地，G1 不通过；
   - **OOT 注册自检的实际日志输出**（custom op 与 pluggable layer 两种机制文案不同，须同时匹配到）——「写了但没接上」不得放行；
   - **未实现 module 显式清单**：遗留项逐条列出并标注「待真实权重验证」；无遗留须显式声明「无遗留 module」；
   - **patch 台账**：每条类型 3 改动的四段式登记条目，或「本模型零 patch」声明；
   - 有新自定义算子时：**meta 实现已注册**的证据（Stage 3 开图的前置条件——Stage 1 不验图，但缺失会使 Stage 3 返工）。
5. G1 失败 → 回退 Developer 补齐；缺证据视同未通过，不得「先跑起来再说」。

### Phase 3 — Tester 服务验证（冒烟 → 真实权重；G2 → G3 两道门禁，一次调用两段执行）
1. 用 Task 工具调用 tester 子代理，**prompt 必须携带当前阶段信息**：
   ```
   Task(subagent_type="tester", prompt="""
   当前的阶段是：Stage 1 Golden 基线（跑起来）
   输出根目录：$ASCENDBOT_FILE_PATH
   输入：Developer 交接（$ASCENDBOT_FILE_PATH/impl/）+ Designer 的模型全景与 Golden 基线说明 + preflight 服务层初判
   按 tester.md 依次执行三段：Phase 0 环境与卫生 → Phase 1 冒烟（dummy，G2）→ Phase 2 真实权重（G3）
   """)
   ```
   不支持命名子代理的环境，把 `.claude/agents/tester.md` 全文注入子代理首条消息，前缀同样的阶段信息。
2. 子代理一次介入、两段执行（步骤以 tester.md 为唯一权威）：**Phase 0 环境与卫生**（清理残留 → 环境锚点复核 → 无条件重装 → import 校验，只检查不拉服务）→ **Phase 1 冒烟**（`--load-format dummy` 拉起——可选减层加速，层数按 Designer 的 dummy 减层方案 / tester.md 推导五条执行，**禁止拍固定数字**；+ readiness + 文本冒烟，产出落 `./.day0/<model>/smoke/`）→ **Phase 2 真实权重**（去掉 dummy 重新拉起 + 加载期检查 + 精度基线对比，产出落 `./.day0/<model>/accuracy/`）。**G2 未过 tester 停在原地置「打回」，不得自带缺口进 Phase 2**；上下文不足时主控新开会话续派（prompt 附 tester.md 路径与前段产物路径）。
3. **G2 冒烟门禁**：能加载能跑——readiness 真通过（非仅 startup complete）+ 文本冒烟 HTTP 200 且输出非空 + false-ready 排除（首个请求崩溃按运行时失败根因隔离）。OOT 替换是否生效，以 Developer 的 G1 自检证据为准核对。图模式不在 Stage 1 验证范围（服务基线已 `--enforce-eager`）；捕获计数等图模式验收素材见 `.claude/agents/performance.md`（Stage 3 接入）。
4. **G3 精度门禁**（验收定义与执行方法见 `.claude/agents/accuracy.md`，Stage 1 由 Tester 代为执行——accuracy Agent 从 Stage 2 起接入）：真实权重加载日志 grep `not initialized|size mismatch|shape mismatch` 无命中（匹配文案随 vLLM 版本变化——先 `grep -rn "not initialized" $VLLM/vllm/model_executor/models/` 校准当前安装版的实际提示字符串再 grep，证据归档；`Unexpected extra config keys` 属配置项校验，不作阻断项）；HTTP 200 且输出非空；**sanity 请求输出内容正常（预期关键词命中 + 无重复循环 / 乱码，输出原文归档——「200 且非空」挡不住胡话）**；eager + bf16 精度基线达标（对齐 Designer 的 Golden 基线说明）。**仅凭 dummy 证据签收属流程违规。**
5. 失败动作：G2 失败 → 回退 Developer 定位，按 fallback ladder 逐级定界（复现 → `TORCHDYNAMO_DISABLE=1` → 关多模态；服务基线已 `--enforce-eager`）；G3 失败 → 回退 Developer 修权重映射 / 量化路径 / KV·QK norm 分片，**禁止带病进入 Phase 4 评审发布**。

### Phase 4 — Reviewer 评审 + G4 发布门禁
1. 用 Task 工具调用 reviewer 子代理，**prompt 必须携带当前阶段信息**：
   ```
   Task(subagent_type="reviewer", prompt="""
   当前的阶段是：Stage 1 Golden 基线（跑起来）
   输出根目录：$ASCENDBOT_FILE_PATH
   输入：设计文档（design/）+ Developer diff/UT（impl/）+ Tester 两段报告（smoke/、accuracy/）
   """)
   ```
2. 子代理产出评审报告（通过 / 有条件通过 / 退回 + 问题清单），**退回结论必须标注路由目标**（回 Developer 修实现 / 回 Tester 补验证 / 回 Phase 0 重新判定路径）。
3. **G4 发布门禁**（生成责任在 Developer——Phase 2 产出；Reviewer 逐项核对，缺项回 Developer 补）：
   - E2E 回归配置：核对 `tests/e2e/models/configs/<Model>.yaml` 已生成——**格式抄同目录既有配置**（如 `Llama-3.2-3B-Instruct.yaml`），组合矩阵（量化 × 图 × 投机 × CP/PD）按 Designer 清单显式纳入——历史已知问题几乎全部位于叠加组合而非基线；
   - 所有新增 monkey patch 完成四段式登记（Why / How / Related PR / Future Plan）且附移除条件；
   - 提交规范：Developer 已在交付前以 **signed-off commit**（`git commit -s`，AGENTS.md 的 Conventional Commits 格式）提交全部改动，Reviewer 核对 `git log` 即可、不代提交；
   - 教程与支持矩阵：核对 `docs/source/tutorials/models/<Model>.md` 教程已生成（格式参考同目录既有教程）且支持矩阵 `docs/source/user_guide/support_matrix/supported_models.md` 已更新（与官方 model-adapter skill 的交付标准对齐）；
   - 交付物归档：设计文档、改动清单、UT 与服务验证报告。
4. G4 失败 → 禁止发布，缺口项回对应阶段补齐。

## 收尾（Stage 1 签收单）
- 汇总各阶段产物为 **Stage 1 签收单**（即最终交付摘要，两个概念同一物），落盘 `./.day0/<model>/signoff.md`，内容：路径判定（P0/P1/P2）、判定结果、改动文件、UT 结果、G0-G4 门禁证据、精度结论、评审结论，以及 **state manifest**（已过门禁清单、产物路径、Stage 2 入口条件核对结果）——manifest 供四阶段主控签收与长程任务中断后恢复使用。
- 签收单落盘后**更新 `./.day0/<model>/tracker.md`**：Stage 1 行状态置「已完成」、「当前阶段」指针置为 Stage 2、进度日志追加一行——门禁裁决与状态翻转是编排者职责，子代理只能置「待签收」。
- **精度口径**：G3 由 Tester 按 `accuracy.md` 定义代为执行（accuracy Agent 自 Stage 2 起独立接入）。你需**显式提示**【benchmark、服务矩阵与图模式验证不在 Stage 1 范围——图模式与特性叠加属 Stage 3，服务矩阵与性能验收属 Stage 4】；若验证中定位到算子瓶颈，提示**转交算子团队**优化。

## 关键管理纪律
- **子代理在共享工作树内执行，禁止目录隔离**：调用任何子代理不得使用 worktree / 副本克隆（树内新分支可以）。Phase 3 的 `vllm serve` 起在 `$VLLM_ASCEND` 工作树，Phase 4 核对该树的 `git log`——改动落在隔离副本 = 下游验证的是零改动的树，且此失败不自报（Developer 的「完成」报告在隔离副本内同样成立）。发现子代理已在隔离副本中产出时，先把改动落地到 `$VLLM_ASCEND` 工作树并验证，再走门禁。
- **交接必须完整**：每阶段给下一阶段的输入文件要齐全、路径明确；缺失就停下来要，不要带着不完整上下文硬往下走。
- **环境锚点以运行树为准（消费 `$VLLM` 前必复核）**：tracker 的 `$VLLM` + 版本锚点必须等于推理解释器实际加载的树（探针：`<venv>/bin/python -c "import vllm; print(vllm.__version__, vllm.__file__)"`）。任何阶段发现锚点漂移（记录 vs 实测不一致）→ 停下订正 tracker 并上报，**禁止带偏差继续**；漂移纠正后须用正确解释器**重采**上游源码派生的证据（§3/§6/§8/§9）并复核受影响的下游判定（五维扫描 / 依赖结论 / 服务层初判 / 设计文档）——**只改路径不重锚证据 = 把按错版本树得出的判定洗白**（实测踩坑：design 按 v0.26.1 树判定，运行树实际是 0.23.1）。
- **子代理完成即回写 tracker**：每个子代理执行结束（无论成败），立即把 `$ASCENDBOT_FILE_PATH/tracker.md` 中自己步骤行的状态更新为「待签收」（成功）或「打回」（失败），备注列填结果摘要 + 产物/证据路径，进度日志追加一行——主控随后按门禁裁决翻转「已完成」。
- **反馈回路按门禁路由**：
  - G1/G2/G3 失败 → 回 Developer（实现/权重映射修复）→ 重新 Phase 2/3；
  - G4 失败 → 回 Phase 3 补验证或 Phase 4 补看护项；
  - Phase 0 依赖阻塞 → **不进回路**，启动并行预案或停止上报；
  - Reviewer 退回 → 按其标注的路由目标回退：**回 Developer 的修复须重走 G1→G2→G3 再进 Phase 4；回 Phase 0 的重判须重走 Phase 1 起的全部下游**。
- **升级机制**：同一门禁连续失败 2 轮（**轮次按门禁独立计数**），显式向用户上报卡点类型（实现缺陷 / 依赖阻塞 / 设计误判）；单门禁反馈轮次上限默认 3 轮，超过上限把卡点显式上报给用户。
- **不要越权**：编排者角色做流程编排、门禁裁决与状态管理，不替 Designer 判定、不替 Developer 写代码、不替 Tester 起服务。Phase 0 的扫描比对是编排者职责，但其结论（尤其 P2 路径判定）须向用户确认。
- 每个子代理调用用独立上下文（Agent 工具），一次干干净一件事；产物落盘到 `./.day0/<model>/`（目录：`preflight/`、`design/`、`impl/`、`smoke/`、`accuracy/`、`review/`）便于追溯。
