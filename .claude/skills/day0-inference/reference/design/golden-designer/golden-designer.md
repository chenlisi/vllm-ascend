# Golden Designer — Stage 1 Golden 基线设计规范

> 本文件由 `.claude/agents/designer.md`（阶段路由器）在 **Stage 1** 时加载执行，不独立触发。
> 目标：产出可供 Developer 直接执行的设计产物——**三份分层 design spec + 一份跨层汇总**。

## 组织原则：三个步骤用三个子文件完成设计

**严格按 服务层 → 调度层 → Worker 层 的顺序执行**（顺序依据 `.claude/skills/day0-inference/reference/adapt/golden-adapter/golden-adapter.md` §2：服务层静默失败最先排除、KVCacheSpec 先于一切性能工作定稿、Worker 层逐 module 收口）。**设计的主流程就是依次加载执行三个 designer 子文件**，每个子文件是其所产 spec 的执行逻辑、判定详表、输出契约与打回条件的**唯一权威**——本文件不复述细节，只做路由与跨层汇总。

每层统一三步判定模式（细则在各子文件）：**适配点清单（= 对应知识库既有章节）→ 逐点判定（不需要必须附理由；超出全集显式声明新场景并记录 `$ASCENDBOT_FILE_PATH/self-evolving.md`）→ 逐场景方案块（必含「实现依据：知识库章节号」）**。

## 执行步骤

### 0. 输入确认

- 读 `./.day0/<model>/tracker.md` 确认你在 Stage 1 步骤表中的步骤行与产物目录。
- 通读 Phase 0 全部产物（`$ASCENDBOT_FILE_PATH/preflight/`：raw_evidence.md + 依赖结论表 / 五维扫描报告 / 服务层初判 / 路径判定与排期）。缺失时向主流程索取，不要自行重扫。
- 模型级路径判定（P0/P1/P2，Phase 0 产出）是你的设计基调：P0 以验证清单为主，P2 必须按「建抽象」而非「适配单模型」组织设计。

### 1. 服务层设计

加载 `.claude/skills/day0-inference/reference/design/golden-designer/golden-service-designer.md` 执行。
产出：**服务层 design spec**，落盘 `$ASCENDBOT_FILE_PATH/design/service-design-spec.md`（场景判定总表 + 逐场景方案块 + 三件套最终选型 + 服务矩阵用例参数）。

### 2. 调度层设计

加载 `.claude/skills/day0-inference/reference/design/golden-designer/golden-schedule-designer.md` 执行。
产出：**调度层 design spec**，落盘 `$ASCENDBOT_FILE_PATH/design/schedule-design-spec.md`（场景判定总表 + 逐场景方案块 + **KVCacheSpec 定稿——标注「定稿前禁止进入性能工作」** + E1-E12 标记清单）。

### 3. Worker 层设计

加载 `.claude/skills/day0-inference/reference/design/golden-designer/golden-worker-designer.md` 执行。
产出：**Worker 层 design spec**，落盘 `$ASCENDBOT_FILE_PATH/design/worker-design-spec.md`（Q0 结论 + module 枚举完整性结论 + **逐 module 判定表——类型 0-5、不含 P 级列** + 逐 module 方案块 + missing/unexpected 清单 + 魔法数字审计结论）。

### 4. 跨层汇总（本文件的自有职责）

- **组合矩阵回归清单**（golden-adapter.md §5）：量化 × 图模式 × 投机 × CP/PD 需覆盖的组合项，作为 E2E 配置与 Stage 3 特性叠加的输入（Stage 1 不执行图相关组合项）；
- **实现顺序建议**：按实现顺序铁律（golden-adapter.md §7：①算子 ②backend → ③KV spec → ④组装 → ⑤配置），标注可并行的步骤；
- **Golden 基线说明**：精度对比基线的来源与对齐目标（供 G3 精度验收）。Day0 新模型常无现成基线——须显式声明走了哪级缺省路径：厂商参考输出（有则给）→ transformers 参考实现 logits/输出对比（缺省主路径）→ 固定题集抽检（最低标准，给出题集与证据等级标注），详见 `.claude/agents/accuracy.md`；
- **设计决策**：落地形态（P0/P1/P2 走到哪一步）；类型 5 全新结构是 standalone 重写还是复用平台中立基类；
- **给 Developer 的执行要点**：哪些 module 走何种覆写点（forward_oot / forward / monkey patch 等），逐条明确，避免 Developer 重新判断；加载期差异的 missing/unexpected 需要哪些 loader 处理，逐条列出。

## 输出设计文档（必须交付，按层组织）

在 `$ASCENDBOT_FILE_PATH/design/` 写入以下产物，**必须包含**：

**全局**
1. **模型全景**：架构类、量化类型、多模态能力、max-seq-len 目标（总设计文档首章）。

**三份分层 spec**（各自的完整契约与打回条件以对应 designer 子文件为唯一权威）
2. **服务层**：`service-design-spec.md`——场景判定总表（知识库每章一行全覆盖、零适配附理由、新场景显式声明）+ 逐场景方案块 + 三件套最终选型（含成套性自检）+ 服务矩阵用例参数；
3. **调度层**：`schedule-design-spec.md`——场景判定总表 + KVCacheSpec 定稿（标注禁止性能工作）+ E1-E12 标记清单 + 逐场景方案块；
4. **Worker 层**：`worker-design-spec.md`——Q0 结论 + 枚举完整性结论 + 逐 module 判定表（不含 P 级列）+ 逐 module 方案块 + missing/unexpected 清单 + 魔法数字审计结论。

**跨层汇总**（总设计文档末章）
5. **Golden 基线说明 + 组合矩阵回归清单 + 实现顺序建议 + 给 Developer 的执行要点**。

**打回条件**（主控设计完整性检查依据）：三份 spec 任一缺失或路径不符；总设计文档缺模型全景或跨层汇总任一项；Worker 判定表带 P 级标注；调度层 spec 未标注「KVCacheSpec 定稿前禁止性能工作」——以上任一命中即打回 Designer 补齐。**各 spec 内部完整性（判定表缺行、零适配无理由、方案块缺「实现依据」等）按对应 designer 子文件的打回条件检查。**

配套参考（全阶段可查）：`.claude/skills/day0-inference/reference/adapt/golden-adapter/golden-adapter.md`（分层框架 / Q0 / 组合矩阵 / 术语纪律与铁律）、`.claude/skills/day0-inference/reference/ascend-oot.md`（OOT 机制与类型 0-5）、`.claude/skills/day0-inference/reference/adapter-templates.md`（代码模板 A-F）。
