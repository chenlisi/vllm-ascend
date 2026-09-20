# 服务层适配方法论（管理面 + 数据面）

> 本文是 Day0 分层适配文档族的服务层分册，索引见 `.claude/skills/day0-inference/reference/adapt/golden-adapter/golden-adapter.md`；同族文档：调度层 `.claude/skills/day0-inference/reference/adapt/golden-adapter/golden-schedule-adapter.md`、Worker 层 `.claude/skills/day0-inference/reference/adapt/golden-adapter/golden-worker-adapter.md`、OOT 机制背景 `.claude/skills/day0-inference/reference/ascend-oot.md`。
>
> **服务层适配的核心命题：错配多为静默失败——HTTP 200 ≠ 部署正确。** parser 名合法即可加载，错的是协议语义；只有矩阵化验证能暴露这类问题（见本文第二部分「服务矩阵验证」）。
>
> ⚠️ **行号防腐**：本文引用上游/插件代码时以函数名/类名/注册项名为锚点（如 `ToolParserManager`、`run_server_worker`），不写 `文件:行号`——行号随版本漂移，失效时按符号名 grep 重新定位。

服务层（API server / parser / tokenizer）决定**请求如何被编码与解析**。它分为两面：

- **管理面**：parser 往哪挂、三件套怎么配、chat template 是什么形态、effort 怎么管——回答「配置从哪来、错配为什么静」；
- **数据面**：请求在 parser 链上的实际流向、双模 parser 的写法、已知失败模式与验收矩阵——回答「请求怎么处理、怎么验证没配错」。

---

## 一、管理面（注册与配置管理）

### 1. parser 注册机制：两个惰性注册表

上游 vLLM 的解析器体系由两个惰性注册表承载：

- `ToolParserManager` 维护 `_TOOL_PARSERS_TO_REGISTER` 注册项；
- `ReasoningParserManager` 维护 30+ 项 reasoning parser 注册项。

两者都通过 `register_module` / `register_lazy_module` 挂载实现。需要自带 parser 时，启动参数 `--tool-parser-plugin` / `--reasoning-parser-plugin` 指向外部实现模块，服务启动时由 `run_server_worker` 导入并完成注册。

适配含义：**需要新 parser 时知道往哪挂**——优先经 plugin 参数注入外部实现，而不是改上游注册表；插件机制的总体背景见 `.claude/skills/day0-inference/reference/ascend-oot.md`。

### 2. 三件套配置纪律：同名同代成套

服务层适配的核心物是**解析器三件套**：`--tokenizer-mode` / `--tool-call-parser` / `--reasoning-parser`。三者必须**同名同代成套**配置。

错配的先例（Kimi K3 误配 `kimi_k2`）：parser 名合法、服务正常启动，但输出全错——思考文本落入 `content`、`reasoning_content` 为空、`tool_calls` 解析失败。**错代次的 parser 不报错，只静默产出语义错误的结果**（失败机理见第二部分「请求处理链」与「静默失败模式目录」）。

纪律：**三件套必须进入版本化的部署清单，并与 checkpoint 代次绑定校验**——同一代际模型内部仍存在 checkpoint 级分化（DeepSeek V4 的 0731 effort 映射补丁曾对 Preview checkpoint 造成回归，vllm-ascend #14951），服务层配置不是一次性工作，而是随 checkpoint 代次演化的持续维护面。

### 3. chat template 形态判定：自带 Jinja vs 程序化 prompt 编码

2026 代新模型的关键变化是**程序化 prompt 编码取代 Jinja 模板**，两种形态并存：

| 形态 | 代表 | 机制 |
|---|---|---|
| 自带 Jinja chat template | GLM、Qwen | tokenizer 配置自带模板，服务层直接渲染 |
| 程序化 prompt 编码 | DeepSeek V4、Kimi K3 | 模型仓库不提供 Jinja，只提供编码程序 |

- **DeepSeek V4**：仓库仅提供 `encoding/` Python helpers（`encode_messages`、`parse_message_from_completion_text`、`encode_arguments_to_dsml` 等），vLLM 以 `--tokenizer-mode deepseek_v4` 内置该编码；
- **Kimi K3**：chat template 是一个直接构建 prompt token 序列的 Python 渲染程序，vLLM 在 Python 与 Rust 前端同时实现了输入 renderer 与流式输出 parser，并与 XGrammar 集成以在解码期约束结构化区域。

由此 `--tokenizer-mode` 的性质发生升级：**从「tokenizer 实现选择」升级为「会话协议实现选择」**。输入 encoding 与输出 parsing 是同一协议的两半，必须同名同代成套配置，缺一不可。

### 4. reasoning_effort 管理

`reasoning_effort` 在 2026 代成为**协议层一等公民**：请求携带 `reasoning_effort` 时被自动写入 `chat_template_kwargs`（low/medium/high → `enable_thinking=true`，none → false），用户显式指定的 `enable_thinking` 优先。管理要点有四：

1. **档位语义按模型而异**。DeepSeek V4 为三档（low/high/max），其中 max 档在 prompt 首部注入固定英文 `REASONING_EFFORT_MAX` 前缀。
2. **档位命名分裂需网关归一化**。low/high/max（DSV4）与 low/medium/xhigh（Qwen3.5/3.8）并存，接入网关层时必须做归一化映射。
3. **强制思考模型的迁移改写**。GLM-5.3 的 `thinking.type` 仅支持 `enabled`，`disabled` 直接返回 400；从 GLM-5.2 迁移时必须把 disabled 请求改写为 enabled + effort=low。
4. **checkpoint 代次是显式配置维度**。effort 映射等前端行为须按 checkpoint 特征字段分支（DSV4 以 `dspark_*` 字段区分 Preview 与 0731/0813 checkpoint），对齐新代次时不得回归旧代次——DSV4 的 0731 effort 映射补丁曾回归 Preview checkpoint（vllm-ascend #14951）。

### 5. 多模态配套约束

多模态配置项须随三件套成套配置，不可漏项：Kimi K3 除 `--tokenizer-mode kimi_k3` 外还需 `--mm-encoder-tp-mode data` 等配置。另有本地处理器的输入模态限制：**vllm-ascend 当前本地处理器仅接受图像输入、不支持视频**。

### 6. NPU 侧已知修复面排查清单

vllm-ascend 在服务层的补丁集中在三类，适配新模型时按此清单逐项排查「该模型是否命中既有定制路径 / 是否触发已知回归」：

1. **前端行为对齐补丁族**：DeepSeek V4 的「reasoning-effort 对齐 + frontend 行为对齐」系列（#12262 / #13993 / #14624 / #14994）表明插件在 tokenizer/编码层维护了 **checkpoint 代次相关的定制路径**——新 checkpoint 到来时先查这组 patch 是否需要跟随更新。
2. **tool-call 流式状态机四类回归**（解析状态机在 NPU 发行版上的回归修复）：
   - GLM tool-call 流式终止的 final chunk 问题（#9787）；
   - GLM47 内联零参数流式工具调用（#9901）；
   - OpenAI 格式响应误发空 `tool_calls`（#9791）；
   - MiniMax-M2 流式参数畸形（#11505）。
3. **采样层上游对齐**：实验性的 reduce sampling（logits 保持分片、仅通信 top-k 候选）因无法产出全词表 logprobs、不兼容 lmhead-TP 与 PD 分离而将弃用，转向跟随上游 batch-sharded sampling（各 rank 本地计算词表分片后按 batch 维 all-to-all，峰值 logits 显存降为 1/P；RFC #15119）；同版本移除了自定义 top-k/top-p AscendC 实现，改用 CANN 算子。

### 代表模型三件套口径示例（原表 2-1）

> ⚠️ 本表是**实例参照而非判定依据**：新模型的三件套取值必须按其自身 chat template 形态与 checkpoint 代次重新判定（方法见本文第 2-4 节），禁止照抄下表。

| 模型 | tokenizer-mode | tool-call-parser | reasoning-parser | 特有约束 |
|---|---|---|---|---|
| DeepSeek V4 Flash/Pro | `deepseek_v4`（无 Jinja，内置 encoder） | `deepseek_v4`（DSML 格式） | `deepseek_v4`（委托式双模） | Preview 与 0731/0813 checkpoint 的 effort 映射不同，按 `dspark_*` 字段区分 |
| GLM-5.3 | 自带 Jinja | `glm47` | `glm45` | 强制思考，`thinking.disabled` 直接 400；effort 默认 max |
| Kimi K3 | `kimi_k3`（Python renderer） | `kimi_k3`（XTML） | `kimi_k3`（identity fallthrough） | 思考历史必须完整回传；误配 `kimi_k2` 名合法但静默错配 |
| Qwen3.5/3.8 | 自带 Jinja | `hermes` / `qwen3_coder` / `qwen3_xml` | `qwen3` | effort 命名分裂（low/medium/xhigh），需网关归一化 |

---

## 二、数据面（请求处理链与验证）

### 7. 请求处理链：preprocess → reason → tool

请求在 parser 链上的流向是固定的三段：

1. **tool parser 预处理**：经 `adjust_request` 改写请求，典型动作是设置 `skip_special_tokens=False`（让特殊 token 保留在输出中，供下游解析）；
2. **reasoning parser 后处理**：先从输出中拆出 `reasoning_content`；
3. **tool parser 解析**：剩余的 `content` 交 tool parser 产出 `tool_calls`。

关键顺序约束：**工具调用只在 `reasoning_end` 之后解析**，以保证 interleaved thinking 的顺序。

这条链解释了错配为何静默：**parser 名合法即可加载，加载即跑通三段流程，错的是协议语义**——reasoning 边界、特殊 token 约定、tool 参数格式任意一环代次不符，产出就是「结构合法、内容全错」的响应，HTTP 层无任何异常。

### 8. 委托式 parser 标准写法

「思考/非思考双模」模型的 reasoning parser 标准写法是**委托**：按 `chat_template_kwargs.thinking` 在两个 parser 间分发。先例：

- `DeepSeekV3ReasoningParser`：按 `thinking` 在 `DeepSeekR1ReasoningParser`（思考模）与 `IdentityReasoningParser`（非思考模，原样透传）之间委托；
- `KimiK3ReasoningParser`：thinking 关闭时同样回落 identity。

新双模模型写 parser 时沿用此模式，不要另起单模 parser 再在网关侧分流。

### 9. 静默失败模式目录

服务层的失败以静默为主，已知三类模式须在排查与验收时逐项对照：

| 模式 | 症状 | 先例 |
|---|---|---|
| parser 代次错配 | 思考文本落入 `content`，`reasoning_content` 为空，tool_calls 解析失败 | 对 K3 误配 `kimi_k2`（名合法、输出全错） |
| 多轮 + 工具调用的 think 泄漏 | 第 3 轮起 `<think>` 内容泄漏进 content | GLM-4.5（vllm#27703） |
| 长上下文省略特殊 token | 特殊 token 被省略导致原文泄漏 | DeepSeek V4 的 DSML START token 先例 |

共同根因都可回溯到第 7 节的请求处理链：代次错配错在解析边界，think 泄漏错在多轮状态下 reasoning 段的识别，特殊 token 泄漏错在预处理/解码对 `skip_special_tokens` 类开关的处理。

### 10. 服务矩阵验证（验收用）

服务层验证需**矩阵化而非冒烟**——再次强调：HTTP 200 ≠ 部署正确。验收矩阵共 4 项：

1. **render 验证**：用 `/v1/chat/completions/render` 端点比对渲染 token_ids 数，验证模板与 effort 前缀生效（DSV4 三档应得不同前缀长度，公开报告值为 5/84/97 tokens——单一来源数据，仅作方法示范，不构成其他模型的期望值）；
2. **tool_choice 全组合**：tool_choice ∈ {auto, required, named, none} × {流式, 非流式} × {有无 reasoning_effort}，校验 `finish_reason` 与 arguments JSON 可解析性；
3. **多轮 reasoning 回归**：重点第 3 轮（think 泄漏进 content 是已知失败模式，vllm#27703）；并覆盖 `enable_thinking=False` 的 identity 回落——此时 `reasoning_content` 应为空、全文进 `content`；
4. **长上下文泄漏回归**：针对「长上下文省略特殊 token 致原文泄漏」模式的专项回归（DSV4 DSML START token 先例），须覆盖。

---

## 三、与流程的衔接

本层结论在 Day0 流程中的落点（Stage 1 见 `.claude/skills/day0-inference/flows/golden_flow.md`，G0-G4 门禁）：

| 流程阶段 | 本文对应内容 | 落点 |
|---|---|---|
| Phase 0 服务层初判 | 三件套候选名同名同代比对（第 2 节）、chat template 形态判定（第 3 节）、effort 档位与 checkpoint 代次对应关系（第 4 节） | 随依赖结论表与五维扫描报告落盘 `./.day0/<model>/preflight/` |
| Phase 1 设计文档 | 服务层设计章节：parser 三件套 + effort 映射（含 checkpoint 代次分支策略） | 设计完整性检查的必查项 |
| Stage 4 服务矩阵验收 | 本文第 10 节的 4 项验收矩阵 | 产出落 `./.day0/<model>/acceptance/`，报告不齐备不得签收 |

术语约定与文档族一致：**P0/P1/P2 只用于模型级路径判定**（Phase 0 产出），module 级只用**类型 0-5**，本文不涉及 module 级判定，不出现类型编号。
