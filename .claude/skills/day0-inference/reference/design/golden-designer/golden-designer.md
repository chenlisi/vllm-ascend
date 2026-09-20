# Golden Designer — Stage 1 Golden 基线设计规范

> 本文件由 `.claude/agents/designer.md`（阶段路由器）在 **Stage 1** 时加载执行，不独立触发。
> 目标：产出一份可供 Developer 直接执行的 golden 版本详细设计文档。

## 方法论文档（Stage 1 全量加载）

- **总纲**：`.claude/skills/day0-inference/reference/adapt/golden-adapter/golden-adapter.md` → 分层框架、Q0 模型级前置检查（`is_rocm()` 二分陷阱 → 注册覆盖实现）、组合矩阵回归清单、术语纪律与实现顺序铁律
- **Worker 层**：`.claude/skills/day0-inference/reference/adapt/golden-adapter/golden-worker-adapter.md` → 数据面 §2.1 判定流水线（速查表快通道 → 决策树 Q1/Q1'/Q2-Q4 → 判定表模板 → 枚举完整性交叉验证）；控制面 §1.5 加载期权重映射检查（厂商权重名 ↔ vLLM 参数名的 missing/unexpected 判定）。（§2.5 ACLGraph 属 Stage 3，本阶段不加载）
- **调度层**：`.claude/skills/day0-inference/reference/adapt/golden-adapter/golden-schedule-adapter.md` → KVCacheSpec 选型、E1-E12 配置中枢清单、prefix cache / 投机调度 / PD 运行时语义
- **服务层**：`.claude/skills/day0-inference/reference/adapt/golden-adapter/golden-service-adapter.md` → parser 三件套、effort 映射、服务矩阵验证
- **机制参考**：`.claude/skills/day0-inference/reference/ascend-oot.md` → OOT 注册机制、六类适配类型（类型 0-5 覆写点）、附录 B 常见陷阱 / 附录 C 适配自检

配套参考：`.claude/skills/day0-inference/reference/adapter-templates.md`（类型 0-5 的代码模板）。

## 执行步骤

0. **输入确认**
   - 读 `./.day0/<model>/tracker.md` 确认你在 Stage 1 步骤表中的步骤行与产物目录。
   - 通读 Phase 0 三份产物（`$ASCENDBOT_FILE_PATH/preflight/`：raw_evidence.md + 依赖结论表 / 五维同构度扫描报告 / 服务层初判）。缺失时向主流程索取，不要自行重扫。
   - 模型级路径判定（P0/P1/P2，Phase 0 产出）是你的设计基调：P0 以验证清单为主，P2 必须按「建抽象」而非「适配单模型」组织设计。

1. **模型侧枚举**（上游支持状态、硬件代次、算子就绪度已由 Phase 0 覆盖，不重扫）
   - 读待适配模型所在路径的 `config.json` + `modeling_*.py`，列出全部 `nn.Module`。
   - **枚举完整性（golden-worker-adapter.md §2.1.7）**：从 `config.json` 模块清单 + `modeling_*.py` 类清单两个来源交叉枚举，两源不一致的标 `⚠️` 单独审查；对**非主流 config 字段**（`attn_res_block_size`/`e_score_correction_bias`/`routed_expert_hidden_size`/`mla_use_output_gate`/`use_full_rank_gate` 等）逐个确认其在 forward 里的使用分支。
   - grep `$VLLM_ASCEND/vllm_ascend/ops/triton/` 是否有同族算子（kernel 存在≠已接线，存在即大幅降低工作量评估）。

2. **逐 module 判定**
   - 先走速查表快通道；未命中 module 走决策树（golden-worker-adapter.md §2.1）。
   - 产出**判定表**：每个 module 一行 = 类型 0-5 + 工作量（**不标 P 级**）。
   - 标记阻塞项 vs 非阻塞项。

3. **加载期权重映射检查（golden-worker-adapter.md §1.5）**
   - 对每个需加载权重的 module（重点新注意力类），列出「厂商权重名集合」vs「vLLM 层参数名集合」的 missing/unexpected 两组。
   - 重点排查四类差异：融合打包（三套 q/k/v → packed）、子模块并入（kv_b_proj → W_UK_T/W_UV）、旧版兼容（A_log 4D→1D）、命名拼写（conv1d unsqueeze、dt_bias 初始化）。
   - 结果并入判定表（每个 module 加一行「加载期差异」列）。

4. **EngineCore 核对**
   - 对判定表整体跑 E1-E12，标记需改项（重点：E1 模型注册 / E2 量化识别 / E3 attention backend / E4 KV cache spec / E6 block-page 对齐 / E9 并行约束）。

5. **分层设计（服务层 / 调度层）**——细节方法论全部在 adapter 文档族，本节只规定产出内容：
   - **服务层设计**（依据 golden-service-adapter.md）：parser 三件套选型、chat template / effort 映射策略（含 checkpoint 代次分支）、多模态输入边界。
   - **调度层设计**（依据 golden-schedule-adapter.md）：KVCacheSpec 选型与 page size 推导、投机调度结论、prefix cache / PD 兼容性结论。**本节定稿前禁止进入任何性能工作**——事后更改 spec 会级联推翻调度与图模式配置。
   - **组合矩阵回归清单**（依据 golden-adapter.md §5）：明确该模型需覆盖的 量化 × 图模式 × 投机 × CP/PD 组合项，作为 E2E 配置与 Stage 3 特性叠加的输入（Stage 1 不执行图相关组合项）。
   - **魔法数字审计复核**：在 Phase 0 审计（raw_evidence.md §11）基础上，确认判定表涉及的维度/头数/rope 参数/expert 数在平台代码中无硬编码残留（GLM5-W8A8 的唯一卡点就是硬编码 MLA 维度）。

6. **设计决策**
   - 给出落地形态，明确模型级路径（P0/P1/P2）走到哪一步。
   - 若存在类型 5 全新结构，明确是 standalone 重写还是复用平台中立基类。

## 输出设计文档（必须交付）

在指定输出路径写入 Markdown 设计文档，**必须包含**：

1. **模型全景**：架构类、量化类型、多模态能力、max-seq-len 目标。
2. **Q0 前置检查结论**：分派是否覆盖 NPU；选哪个上游分支作基线及理由。
3. **module 枚举完整性结论（golden-worker-adapter.md §2.1.7）**：config 来源 / modeling 来源两源交叉结果；非主流字段逐个确认了哪些、各自用途。
4. **逐 module 判定表**：`| module | 枚举来源 | 类型(0-5) | 加载期差异 | 工作量 | 备注 |`（覆盖全部 module；**不含 P 级列**）。
5. **EngineCore E1-E12 标记清单**：逐项 `✅/❌/⚠️`+ 改动点。
6. **实现顺序建议**：按 golden-adapter.md 的实现顺序铁律（①算子 ②backend → ③KV spec → ④组装 → ⑤配置），标注可并行的步骤。
7. **Golden 基线说明**：精度对比基线的来源与对齐目标（供后续精度验收）。Day0 新模型常无现成 Golden 基线——须显式声明走了哪级缺省路径：厂商参考输出（有则给）→ transformers 参考实现 logits/输出对比（缺省主路径）→ 固定题集抽检（最低标准，给出题集与证据等级标注），详见 `.claude/agents/accuracy.md` 的基线来源优先级。
8. **服务层设计**：parser 三件套选型与理由、chat template / effort 映射策略（含 checkpoint 代次分支）、多模态输入边界。
9. **调度层设计**：KVCacheSpec 选型与 page size 推导、投机调度（draft 分组归属、回滚路径）、prefix cache / PD 兼容性结论；标注「本节定稿前禁止进入性能工作」。
10. **组合矩阵回归清单**：量化 × 图 × 投机 × CP/PD 需覆盖的组合项（供 E2E 配置与 Stage 3 特性叠加使用）。
11. **魔法数字审计结论**：平台代码中对本模型维度/头数/rope/expert 数的硬编码扫描结果与参数化建议。
12. **给 Developer 的执行要点**：明确哪些 module 走何种覆写点（forward_oot / forward / monkey patch 等），避免 Developer 重新判断；**加载期差异（golden-worker-adapter.md §1.5）里 missing/unexpected 需要哪些 loader 处理，逐条列出**。
