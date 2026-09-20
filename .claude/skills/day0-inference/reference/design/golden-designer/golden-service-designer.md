# Golden Service Designer — Stage 1 服务层设计规范

> 本文件由 `.claude/skills/day0-inference/reference/design/golden-designer/golden-designer.md` 的**步骤 1（服务层设计）**加载执行，不独立触发。
> 产出：**服务层 design spec**——它是服务层的唯一交接物，经 golden_flow Phase 1 设计完整性检查后串联传递至 Phase 2，Developer 按落地流程 `.claude/skills/day0-inference/reference/adapt/golden-adapter/golden-service-adapter.md` 依据本 spec 落地开发（代码写法参考知识库 `.claude/skills/day0-inference/reference/golden-service-knowledge.md`）。
> ⚠️ 行号防腐：引用代码以函数名/类名/注册项名为锚点，不写 `文件:行号`。

## 执行逻辑（三步）

### 步骤 1：通读服务化相关代码

判定之前先建立事实基础，**三个来源都要读**，禁止仅凭 Phase 0 证据摘要下结论：

- **模型侧**（模型仓库内）：`tokenizer_config.json` / `chat_template*.jinja` / `encoding_*.py`（程序化 prompt 编码实现）/ `preprocessor_config.json` / `tokenization_*.py`；
- **上游 vLLM 侧**：`ToolParserManager` / `ReasoningParserManager` 注册表实现（Phase 0 证据 §8 给出的可用项清单，逐个读实现而非只看名字）、tokenizer-mode 注册点、chat template 渲染路径、`run_server_worker` 的 plugin 导入逻辑；
- **vllm-ascend 侧**：服务层既有 patch 族（`vllm_ascend/patch/` 中前端行为对齐、tool-call 流式状态机、采样层三组，对照知识库第 6 章的清单）。

同时复核 Phase 0 的 `preflight/服务层初判.md`——它是初判，你是终判；结论不一致时以你通读代码后的判定为准，并在 spec 中显式修正。

### 步骤 2：场景判定——是否落在知识库既有章节内

服务层适配场景的全集 = **知识库 `golden-service-knowledge.md` 的全部章节**（每章一类场景，逐章判定对照见下「场景判定详表」；全集随知识库扩章而扩展，不写死数量）。逐章判定当前模型的服务化特征是否命中：

- **命中**：进入步骤 3 逐场景输出；
- **不命中**：该场景输出「无需适配 + 理由（引用代码/证据）」；
- **超出全集**：模型的服务化特征不被知识库任何一章覆盖 → 显式声明「**新场景**」，描述特征、与最近章节的差异、建议机制；新场景的方案按「建抽象」而非「适配单模型」组织，并在 spec 中标注**需 Reviewer 重点关注**（这是知识库的扩展点，不允许静默略过）。**同时把新场景追加记录到 `$ASCENDBOT_FILE_PATH/self-evolving.md`**（不存在则新建），格式固定：

  ```markdown
  ## 新场景候选 — <场景名>（<模型>，<日期>）
  - 来源层：服务层（golden-service-designer）
  - 特征描述：<模型的服务化特征是什么>
  - 与最近场景的差异：<最接近的知识库章节及为何不覆盖>
  - 建议机制：<若落入知识库，应如何扩章>
  - 处置状态：待用户决策
  ```

  该文件是**知识库的演进候选池**：任务完成后由 SKILL.md 的自演进步骤向用户咨询是否把条目落入知识库（扩为新章节）。

### 步骤 3：逐场景输出 design spec

对每个场景产出一行判定 + 方案块：

- **无需适配**：给出理由（引用步骤 1 通读的代码事实或 Phase 0 证据章节），无理由视同漏项；
- **需要适配**：给出落地方案——选型与理由、实现方式（复用上游 / 委托式 / 新写 plugin）、落点（配置项 / 文件 / 注册点）、验收方式（对应知识库末章服务矩阵的验收项）。

## 场景判定详表

| # | 场景（知识库章节） | 判定问题 | 零适配条件 | 需要适配时的方案要点 |
|---|---|---|---|---|
| 1 | parser 注册机制（§一.1） | 本模型是否需要新 parser？ | 三件套全部命中上游注册表 | 经 `--tool-parser-plugin` / `--reasoning-parser-plugin` 注入外部实现，不改上游注册表 |
| 2 | 三件套配置纪律（§一.2） | `--tokenizer-mode` / `--tool-call-parser` / `--reasoning-parser` 的同名同代候选是什么？ | 三个候选同名同代且均已在注册表 | 给出一套绑定 checkpoint 代次的成套配置；错代次先例：K3 误配 `kimi_k2` 名合法但输出全错 |
| 3 | chat template 形态（§一.3） | 自带 Jinja 还是程序化 prompt 编码？（证据 §7） | 自带 Jinja 模板 | 程序化编码 → `--tokenizer-mode` 从可选变强制；确认上游是否已内置对应实现，未内置则方案含新增编码/renderer |
| 4 | reasoning_effort 管理（§一.4） | 模型是否有 effort 概念？档位语义与 checkpoint 代次分支？ | 模型无 effort 概念 | 档位映射表（含代次分支，对齐新代次不得回归旧代次）；命名分裂需网关归一化的显式说明 |
| 5 | 多模态配套约束（§一.5） | 是否多模态？（证据 §2 vision tower / media placeholder） | 纯文本模型 | `--mm-encoder-tp-mode` 等成套配置 + 本地处理器仅图像、不支持视频的限制声明 |
| 6 | NPU 已知修复面排查（§一.6） | 模型族/checkpoint 是否命中既有定制路径或已知回归？ | 不命中任何一组 | 命中项列明 patch 编号（#12262/#13993/#14624/#14994/#9787/#9901/#9791/#11505），给出跟随更新结论 |
| 7 | 请求处理链（§二.7） | 本模型在 preprocess → reason → tool 链上有无特殊顺序/开关需求？ | 标准三段链即可，无特殊 token 约定 | `adjust_request` 改写需求（如 `skip_special_tokens=False`）、reasoning_end 后才解析工具调用的顺序约束 |
| 8 | 委托式 parser 写法（§二.8） | 是否思考/非思考双模？ | 单模（纯思考或纯非思考） | 按 `chat_template_kwargs.thinking` 在两个 parser 间委托；非思考模 identity 回落必须显式设计 |
| 9 | 静默失败模式排查（§二.9） | 三类已知静默模式（parser 代次错配 / 多轮 think 泄漏 / 长上下文特殊 token 省略）本模型是否有对应风险面？ | 逐项对照后无风险面（逐条写理由） | 命中的模式转化为服务矩阵的专项测试用例 |
| 10 | 服务矩阵验证设计（§二.10） | 4 项验收矩阵（render 端点 / tool_choice 全组合 / 多轮 reasoning 第 3 轮 / 长上下文泄漏回归）的用例参数是什么？ | 不适用——矩阵是验收义务，非可选适配点 | 给出本模型的矩阵用例参数（各 effort 档位的预期前缀形态、tool_choice 组合、多轮脚本）；Stage 1 只设计不执行，执行属 Stage 4 |

## 输出契约：服务层 design spec

落盘 `$ASCENDBOT_FILE_PATH/design/service-design-spec.md`（同时作为设计文档的服务层章节汇入总设计文档），**必须包含**：

1. **场景判定总表**：`| 场景# | 是否命中 | 是否需要适配 | 理由（代码/证据引用） |`——知识库每章一行全覆盖，新场景追加行并标 ⚠️；
2. **逐场景方案块**：命中且需适配的场景，每个一个方案小节（选型与理由 / 实现方式 / 落点 / **实现依据：golden-service-knowledge.md 章节号** / 验收方式）——**「实现依据」是 Developer 选择性执行的索引，缺此字段视同方案不完整**；
3. **三件套最终选型**：`--tokenizer-mode` / `--tool-call-parser` / `--reasoning-parser` + 多模态配套项，附成套性自检（同名同代、与 checkpoint 代次绑定）；
4. **服务矩阵用例参数**（知识库末章对应场景产出）：Stage 4 验收的直接输入。

**打回条件**：场景总表缺行；「无需适配」无理由；新场景未显式声明；三件套缺成套性自检——任一命中即由主控打回 Designer 补齐。

## 交接链

```
golden-designer.md 步骤 1（加载本文件）
  → 产出 design/service-design-spec.md
  → golden_flow Phase 1 设计完整性检查（打回条件见上）
  → golden_flow Phase 2：Developer 按 golden-service-adapter.md 落地流程执行——
      以 spec 判定表为过滤条件，只实现「需要适配」的场景；
      代码写法按方案块「实现依据」查阅 golden-service-knowledge.md 对应章节
  → Stage 4：按 spec 的服务矩阵用例参数执行服务矩阵验收
```
