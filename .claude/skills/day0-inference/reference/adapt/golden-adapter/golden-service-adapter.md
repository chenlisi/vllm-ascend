# Golden Service Adapter — Stage 1 服务层落地流程

> 本文是**流程步骤**，由 `.claude/agents/developer.md`（Stage 1 实现步骤）在服务层落地时加载执行。
> 输入是设计侧产出的服务层 design spec；**代码怎么写，参考知识库 `.claude/skills/day0-inference/reference/golden-service-knowledge.md`**（机制细节：注册点、委托式写法、失败模式避坑）。

## 输入

- `$ASCENDBOT_FILE_PATH/design/service-design-spec.md`：场景判定表 + 逐场景方案块（选型 / 实现方式 / 落点 / 实现依据章节号）+ 三件套最终选型 + 服务矩阵用例参数。

## 执行步骤

1. **圈定执行范围**：读 spec 的场景判定表，圈出判定为「需要适配」的条目（含 ⚠️ 标注的新场景）；「无需适配」的条目不落地、不加载对应知识章节。
2. **逐条目落地**：每条按其方案块执行——
   - 方案块有「实现依据」章节号 → 查阅 `golden-service-knowledge.md` 对应章节，按其机制实现（parser 注册走 `--tool-parser-plugin` / `--reasoning-parser-plugin` 注入，不改上游注册表；双模 parser 用委托式写法）；
   - ⚠️ 新场景 → 知识库无对应章节，直接按方案块的「建抽象」设计实现，并在交付物中保留 Reviewer 重点关注标注；
   - 典型落地物：parser plugin 实现、tokenizer-mode 配置、三件套与多模态配套配置项。
3. **成套性自检**：逐项核对 spec 的三件套最终选型——同名同代成套、与 checkpoint 代次绑定（自检口径见知识库第 2 节）。
4. **验证与证据归集**：实现级验证随 UT（汇入 G1 门禁证据）；**服务矩阵验证不在本阶段执行**——按 spec 的矩阵用例参数，属 Stage 4 验收（矩阵定义见知识库第 10 节）。

## 输出

- 服务层改动清单（parser / plugin / 配置文件）+ UT 证据 + 成套性自检结论 → 汇入 `./.day0/<model>/impl/` 的 G1 门禁证据。

## 纪律

- spec 是唯一权威输入：判定疑问回 Designer 澄清，不自行改判定；实现中发现 spec 与代码事实冲突时，以代码事实为准并回 Designer 修正 spec。
- 判定纪律（与知识库一致）：三件套「同名同代」是语义判定（`kimi_k2` 对 K3 合法但全错），「parser 名能加载」只证明名字合法，不证明协议语义正确。
