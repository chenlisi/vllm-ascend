# Golden Worker Adapter — Stage 1 Worker 层落地流程

> 本文是**流程步骤**，由 `.claude/agents/developer.md`（Stage 1 实现步骤）在 Worker 层落地时加载执行。
> 输入是设计侧产出的 Worker 层 design spec；**代码怎么写：覆写点机制参考知识库 `.claude/skills/day0-inference/reference/golden-worker-knowledge.md`**（§2.1 判定流水线、§2.1.5 判定 B 覆写点、§1.5 权重映射、§2.4 MoE 通信），**代码模板查 `.claude/skills/day0-inference/reference/adapter-templates.md`**（类型 0-5 模板 A-F）。

## 输入

- `$ASCENDBOT_FILE_PATH/design/worker-design-spec.md`：Q0 结论 + module 枚举完整性结论 + **逐 module 判定表**（`| module | 枚举来源 | 类型(0-5) | 加载期差异 | 工作量 | 备注 |`）+ 逐 module 方案块（含实现依据章节号）+ 权重映射 missing/unexpected 清单 + 魔法数字审计结论。

## 执行步骤

1. **圈定执行范围**：读 spec 的逐 module 判定表，圈出判定类型 1-5 的 module（含 ⚠️ 标注的新场景）；**类型 0 的 module 不动**（OOT 自动替换，不写代码），也不加载对应知识章节。
2. **逐 module 落地**：每个 module 按其判定类型执行——
   - 方案块有「实现依据」章节号 → 查阅 `golden-worker-knowledge.md` 对应章节确认机制，代码骨架从 `reference/adapter-templates.md` 对应模板起步，仓内真实参考实现按模板索引阅读：
     - **类型 1**：新增 CustomOp OOT 实现，覆写 `forward_oot`；**类型 2**：新增 PluggableLayer OOT 实现，覆写 `forward`——**覆写点写错不报错、只静默不生效**，机制查知识库 §2.1.5 判定 B；
     - **类型 3**：monkey patch（无法走注册表）——走 patch 决策树纪律（见 `.claude/agents/developer.md` 通用约束），产出四段式登记条目；
     - **类型 4**：扩展现有 Ascend 实现（补参数丢弃 / 分支缺失）；
     - **类型 5**：全新结构（算子 + 可能的 attention backend + KV cache spec）——组织方式查知识库 §2.1.1（Q1' standalone 重写）与 §2.1.4 决策树特殊分支；
   - ⚠️ 新场景 → 知识库无对应章节，直接按方案块的「建抽象」设计实现，并在交付物中保留 Reviewer 重点关注标注。
3. **权重映射 loader 处理**：按 spec 的 missing/unexpected 清单逐条处理——融合打包写 `_stacked_params_mapping` 或 loader 重排、子模块并入确认 `process_weights_after_loading`、旧版兼容 loader 做 `.view()`、命名拼写对照同族层 loader（四类差异机制查知识库 §1.5.2）；量化模型同步 `packed_modules_model_mapping`、落实 NZ 布局转换点（§1.5.4）。
4. **自检**：两项——
   - **UT 全绿**：UT 落在 `tests/ut/` 下，每个改动 module 覆盖构造级（能构造、能 import、OOT 注册链路生效）+ 精度级（eager 下对齐 torch 等价实现或 Golden 值），`uv run pytest tests/ut/<target> -v` 失败数 = 0；
   - **OOT 注册自检**：确认注册实际生效——custom op 与 pluggable layer 两种机制日志文案不同，须同时匹配到（口径见 `.claude/skills/day0-inference/reference/ascend-oot.md` 附录 C）。
5. **证据归集**：改动文件清单 + UT 运行结果 + OOT 自检日志摘录 + 未实现 module 显式清单（无遗留须显式声明）+ patch 台账（类型 3 四段式登记或「本模型零 patch」声明）+ 新算子 meta 实现注册证据。

## 输出

- Worker 层改动清单（`models/` / `patch/worker/` / `ops/` / loader 映射等）+ UT 证据 + OOT 自检证据 + 上述归集项 → 汇入 `./.day0/<model>/impl/` 的 G1 门禁证据。

## 纪律

- spec 是唯一权威输入：判定疑问回 Designer 澄清，不自行改判定；实现中发现 spec 与代码事实冲突时，以代码事实为准并回 Designer 修正 spec。
- 术语纪律：module 级只用类型 0-5；P0/P1/P2 只属于模型级路径判定，不出现在实现标注中。
- 图模式（知识库 §2.5 ACLGraph）属 Stage 3，本阶段不加载、不验证；但新自定义算子的 meta 实现注册是本阶段的实现纪律（Stage 3 开图前置）。
