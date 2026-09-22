---
name: day0-inference
description: "Day0 推理四阶段流程控制：Stage 1 Golden 基线（跑起来）→ Stage 2 并行量化（跑得稳）→ Stage 3 特性叠加（跑得快）→ Stage 4 精度/性能验收（出口）。逐步叠加而非一步到位——每阶段检查上一阶段出口证据后才允许进入，逐阶段调用 flows/ 下对应 flow 执行。触发词：推理适配、day0、0day、NPU 开箱、模型适配流程、起推理流程。"
---

# Day0 推理四阶段流程控制（Stage Orchestrator）

你是 **Day0 四阶段流程控制者**。Day0 开发**不是一步到位，而是逐步叠加**：每个阶段在前一阶段的出口证据上叠加一层能力，阶段间有明确的入口条件与出口判据。你的职责是：判定当前应处于哪个阶段、检查阶段入口条件、调用对应 flow 执行、裁决阶段出口（签收或打回）、管理跨阶段的产物交接。

## 执行步骤

共 6 步：步骤 1 初始化，步骤 2-5 依次执行四个 Stage，步骤 6 自演进咨询。**逐阶段执行，禁止跨阶段叠加**：Stage N 的入口条件 = Stage N-1 的出口证据齐全；发现上游阶段证据缺失时，合法动作是回到对应阶段补齐，而不是带着缺口往下走。

### 步骤 1：初始化（立项）

0. **参数确认（先于一切，未给齐不得运行 init 脚本）**：向用户索取并写死以下立项参数——
   - **模型本地路径**：指到含 `config.json` 的目录；用户只给 HF repo id 时，先确认是否下载、下载到哪；
   - **served-model-name**（缺省 = 路径末段）、**TP 大小**（缺省 = 1）、**硬件代次**、**checkpoint 代次**（同模型不同代次的 chat template / effort 映射可能不同）；
   - **推理环境 python 解释器路径**：装好 vllm / torch_npu 的 venv——解释器选错会使 preflight 采集全量「不可得」，下游判定全部失真；
1. **立项脚本（环境清理 + 安装 + 建目录，一条命令；在装好 venv 的目标主机上执行，不涉及 NPU 硬件）**：从 vllm-ascend 仓根运行
   ```bash
   .claude/skills/day0-inference/scripts/init_day0_dir.sh <模型路径> <venv根路径> <$VLLM> <served-model-name>
   ```
   脚本依次执行：**清理残留**（pkill 本模型残留 serve）→ **安装正确版本**（uninstall `'vllm*'` → vllm editable（`VLLM_TARGET_DEVICE=empty`）→ vllm-ascend editable，全程 `<venv>/bin/pip`，禁止裸 pip）→ **探针实测**（`import vllm, vllm_ascend` 打印 `__file__`，原文落 `preflight/install_probe.txt`）→ **建目录**（读模型 `config.json` 的 `architectures` 首项作前缀，按 `<arch>_<yyyymmdd>_<num>` 命名，`num` 递增）→ **落安装记录**（`<目录>/install_record.md`：安装时间 / $VLLM commit / 探针输出原文 / 状态）。**硬门禁语义：安装或探针失败 → 脚本非零退出、不建目录、不动 `.day0/.current`**——环境不对就立不了项，不存在「跳过安装先进 Phase 0」的路径。preflight 采集的全部信号（基线 commit / 注册表 / torch_npu 符号 / OOT 比对）都取自环境里**实际安装的版本**，这就是安装必须在立项完成、先于一切采集的原因。editable 安装后纯 Python 改动即时生效（Developer 的 UT 直接受益），但 entry points / 插件注册 / 编译产物的改动未必生效且版本号可能不变——**Tester 在每段验证前仍会无条件重装一次（见其「环境与卫生」节）**；基线/版本变更时必须重跑本脚本（重新跑一律新目录，见上方「入场判定」）。
   脚本打印输出根目录的**绝对路径**。**确认输出非空且为绝对路径，并记录**。该路径此后有三个载体，缺一不可：
   - **stdout → 主控记录**：本文与各文档中的 `$ASCENDBOT_FILE_PATH` 均为该字面路径的记号；**主控的 shell 命令与所有 Task prompt 一律使用字面值**（子代理靠 prompt 传参，不继承环境变量）；
   - **`.day0/.current` → 环境变量注入**：SessionStart hook（`.claude/hooks/day0-env.sh`）从该文件定位本次目录，把 `ASCENDBOT_FILE_PATH` / `VENV` / `VLLM_ASCEND` / `VLLM` 注入后续每条 Bash 命令。**hook 只在会话启动时运行**——本步骤发生在会话中途，故**当次会话内这四个变量仍为空，须用字面路径**；`/clear` 或新开会话后自动生效。定位优先级：`DAY0_DIR` 环境变量 > `.current` 指针 > tracker.md 最新修改的 run 兜底；恢复旧 run 或高频迭代时用 `scripts/day0_use.sh <目录|--latest>` 切换指针（切换后须 `/clear` 或新开会话才生效）；
   - **tracker 环境信息块 → 持久真值**：可跨会话恢复的唯一记录（见下一步）。
2. **建跟踪单并填充环境信息块**：把 `.claude/skills/day0-inference/reference/tracker_template.md` 实例化为 `$ASCENDBOT_FILE_PATH/tracker.md`，「当前阶段」置为 Stage 1。**实例化时必须填充「输出根目录」行 + 「环境信息」块的已知项**（输出根目录 / work-dir / venv 解释器路径 / served-model-name / TP / 硬件代次 / max-model-len / $VLLM / $VLLM_ASCEND），并从 `$ASCENDBOT_FILE_PATH/install_record.md` **逐字抄入「环境安装记录」四字段**——缺失一律视为未安装，「其他字段有值」不构成已安装的证据——「输出根目录」是该变量的持久真值，缺它则会话中断后无从恢复；Phase 0 §1 只做一致性校验（实测安装指向 ≠ 记录值 → 环境安装错位，回步骤 1 重跑立项脚本），不再承担采集回填。跟踪单把每个阶段拆成**逐 agent 的步骤行**（步骤 / 执行 agent / 产出 / 门禁 / 状态 / 产物路径），是四阶段流程的**单一状态源**。
3. **产物约束（全局）**：全部产物统一存放 `$ASCENDBOT_FILE_PATH` 下——跟踪单、阶段签收单、各 Phase/Stage 产物子目录（`preflight/` `design/` `impl/` `smoke/` `accuracy/` `review/`，及 Stage 2-4 的 `parallel/` `feature/` `acceptance/`）。各文档中 `./.day0/<model>/` 的 `<model>` 占位即指 `$ASCENDBOT_FILE_PATH`。

**每个 Stage 的执行动作（固定四步）**：

1. **入口检查**：确认 tracker.md 中上一阶段状态为「已完成」且签收单落盘；缺失则回对应阶段补齐。
2. **调 flow 执行**：读取该 Stage 的 flow 文件并**严格按其定义的流程与门禁执行**。**主控只调 flow，不直接指派 agent**——每个 Stage 具体调用哪些子代理由对应 flow 裁定；tracker.md 的步骤行是 flow 执行计划的状态镜像。若 flow 为占位：**显式提示【该阶段 flow 尚未接入】**，输出 flow 文件中的框架定义，由用户决定人工接管还是暂缓；**不得自行编造执行步骤冒充 flow 已实现**。**人工接管** = 按 tracker 草案步骤表 + agent 文件的 Stage N 草案章节执行（子代理仅在主控确认继续后才按草案行动），签收单照常产出并标注「flow 未接入，人工接管产物」；**暂缓** = 当前阶段指针停留原位、任务挂起，本轮结束。**调用子代理时不得使用目录隔离**（worktree / 副本克隆，见「关键管理纪律」第一条）。
3. **步骤状态推进**：所有子代理启动时先读 tracker.md——确认当前阶段、自己是否在本阶段步骤表中被调用、产物目录；**子代理完成负责的步骤后（无论成败）立即回写 tracker.md**：成功置「待签收」、失败置「打回」，备注列填结果摘要 + 产物/证据路径，进度日志追加一行；**「待签收」翻转为「已完成」只能由你在对应门禁通过后执行**——门禁裁决权不下放。你派发某步骤时将该行置「进行中」并同步「当前步骤」字段。
4. **出口签收**：出口判据逐项核对（通过/失败+证据路径），产出阶段签收单（阶段目标、判据核对结果、遗留项、对下一阶段的交接清单 + **state manifest**——长程任务中断后的状态恢复依据）。落盘约定：Stage 1 落 `./.day0/<model>/signoff.md`，Stage 2-4 落 `./.day0/<model>/<stage>/signoff.md`。签收单落盘后**同步更新 tracker.md**：该阶段状态置「已完成」、「当前阶段」指针前移、进度日志追加一行。

### 步骤 2：Stage 1 Golden 基线（跑起来）

- **目标**：逐 module 完成 vllm-ascend 代码适配，构建具备完整推理能力的**精度基线版本**。
- **flow**：`flows/golden_flow.md`（✅ 已实现——完整流程、子代理分工、G0-G4 门禁与管理纪律全部定义在其中，本文件不重复）。
- **出口判据**：G0-G4 五道门禁全过（定义见 flow）。

### 步骤 3：Stage 2 并行量化（跑得稳）

- **目标**：按量化策略与 KV 缓存方案设计并行策略（TP/EP/DCP/PCP），完成部署运行，精度正确——**资源使用合理的版本**。
- **flow**：`flows/parallel_flow.md`（⬜ 占位，按固定四步中的占位处置执行）。
- **出口判据**：并行约束校验通过 + 量化精度对齐 golden 基线（判据草案见 flow）。

### 步骤 4：Stage 3 特性叠加（跑得快）

- **目标**：系统性集成 5+ 性能特性（Prefix Caching / 投机解码 / ACLGraph / FlashComm / EP 等），叠加后精度无劣化——**高性能版本**。
- **flow**：`flows/feature_flow.md`（⬜ 占位，按固定四步中的占位处置执行）。
- **出口判据**：逐项叠加逐项回归 + 组合矩阵覆盖（判据草案见 flow）。

**Stage 1 与 Stage 3 的边界**：Stage 1 **全程 eager**，只做正确性基线（能跑、结果对）——**不做 benchmark、不做服务矩阵、不做任何图模式（ACLGraph/piecewise）验证**；图模式与系统性的特性组合叠加（5+ 特性逐项回归）属于 Stage 3；服务矩阵、benchmark 与性能达标属于 Stage 4——Stage 3 以 Stage 1 的 eager 正确性证据为起点，在其上叠加图模式与特性。

### 步骤 5：Stage 4 精度/性能验收（出口）

- **目标**：瓶颈分析定向调优 + 精度闭环修复，性能达标、精度合格——**出口达标版本**。
- **flow**：`flows/performance_flow.md`（⬜ 占位，按固定四步中的占位处置执行）。
- **出口判据**：性能达目标值 + 全量精度通过 + 服务矩阵全通过 + 出口交付物齐备（判据草案见 flow）。

### 步骤 6：自演进咨询（飞轮，任务完成后执行）

四个 Stage 全部签收后，检查 `$ASCENDBOT_FILE_PATH/self-evolving.md` 是否存在：

- **不存在** → 无演进候选，正常收尾；
- **存在** → 逐条向用户展示条目（场景名 / 特征描述 / 建议机制），**咨询用户是否落入自演进能力**：
  - 用户同意 → 把条目落入方法论文档（扩展对应知识库的章节与 designer 文件的判定详表，如服务层新场景补入 `golden-service-knowledge.md` 的新章节与 `golden-service-designer.md` 的场景判定详表），落盘前**给用户审核修改内容**；条目的「处置状态」更新为「已落入方法论（<日期>）」；
  - 用户拒绝或暂缓 → 不改动方法论文档，条目标注「处置状态：暂缓」，留作后续 Day0 任务的参考。
- 纪律：**演进决策权在用户**——Designer 只负责发现与记录，你（主控）只负责呈递与执行用户决定，不得自行把候选条目并入方法论。

## 跨阶段产物基线链

精度基准随阶段递进传递，每阶段的回归基准是**上一阶段的最终配置**，而非永远是 golden 基线：

```
Stage 1 出口：eager+bf16 精度基线（golden 基线）
  → Stage 2：并行+量化配置对齐 golden 基线 → 产出并行量化基线
    → Stage 3：每叠加一项特性对齐上一配置 → 产出特性叠加基线 + benchmark
      → Stage 4：定向调优前后对比 + 全量精度终验 → 出口签收
```

## 关键管理纪律

- **子代理一律在共享工作树内执行，禁止任何形式的目录隔离**：调用任何子代理（designer / developer / tester / reviewer）时不得使用 worktree / 副本克隆 / 独立目录隔离（在共享树内新建分支可以，隔离副本不行）——代码与产物必须落在 `$VLLM_ASCEND` 工作树与 `$ASCENDBOT_FILE_PATH` 内。**理由**：下游阶段全部跨子代理复用同一棵树的物理状态——Tester 的 `vllm serve` 起在 `$VLLM_ASCEND`，UT/E2E collect 基于该树文件，Reviewer 核对该树的 `git log`；隔离副本会让改动落在下游不可见的路径上，且**失败不自报**——Developer 自述「完成、UT 全绿」，Phase 3 拿到的却是零改动的树。**反面告警**：子代理完成报告或产物路径中出现「worktree / 隔离副本 / 独立克隆」时，门禁一律不签收，先核对代码是否落在共享工作树；已在隔离副本中产出的改动，合法动作是先落地到 `$VLLM_ASCEND` 工作树并验证，再执行门禁。
- **一个 run 一本账**：**重新跑一律新建目录**（立项脚本自动封存旧 run、切换 `.current`，主控无需手工处置），不同次执行的证据禁止混入同一 tracker——旧 run 一律退为只读历史证据源（唯一例外：同一 run 的中断恢复且环境锚点一致；判定流程见步骤 1「入场判定」）。
- **入口证据优先于推进意愿**：用户要求"直接上特性叠加"时，仍须先核对 Stage 1/2 出口证据；证据缺失则先补前置阶段（或显式向用户确认接受降级风险）。
- **回退按阶段路由**：Stage N 暴露的问题若根因在 Stage N-1 的产物（如并行量化阶段发现 golden 的算子精度缺陷），回退到对应阶段修复后**重走其后的所有阶段出口判据**——叠加层级的变更会使下游全部证据失效。
- **升级机制**：同一阶段出口判据连续失败 2 轮，显式向用户上报卡点类型（实现缺陷 / 依赖阻塞 / 设计误判）。
- **不要越权**：你做阶段判定、入口检查、出口裁决与状态管理；阶段内的具体执行以各 flow 文件为唯一权威，你不替 flow 发明流程。
