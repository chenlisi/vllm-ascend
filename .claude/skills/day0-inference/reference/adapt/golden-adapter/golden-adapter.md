# Golden 阶段分层适配总纲

> 本文件是 `.claude/skills/day0-inference/flows/golden_flow.md` 的方法论配套文档，服务 **Phase 1（Designer 设计）** 与 **Phase 2（Developer 实现）**。它是文档族的索引与总纲：每层细节一律在子文档详述，本文只给框架、判据与指向。
>
> 同族文档：
> - `.claude/skills/day0-inference/reference/ascend-oot.md` —— OOT 机制与六类适配类型（判定树与模板索引）
> - `.claude/skills/day0-inference/reference/adapter-templates.md` —— 类型 0-5 代码模板 A-F
> - 每层「知识库 + 落地流程 + designer 规范」三件套：
>   - 服务层：`.claude/skills/day0-inference/reference/golden-service-knowledge.md` + `.claude/skills/day0-inference/reference/adapt/golden-adapter/golden-service-adapter.md` + `.claude/skills/day0-inference/reference/design/golden-designer/golden-service-designer.md`
>   - 调度层：`.claude/skills/day0-inference/reference/golden-schedule-knowledge.md` + `.claude/skills/day0-inference/reference/adapt/golden-adapter/golden-schedule-adapter.md` + `.claude/skills/day0-inference/reference/design/golden-designer/golden-schedule-designer.md`
>   - Worker 层：`.claude/skills/day0-inference/reference/golden-worker-knowledge.md` + `.claude/skills/day0-inference/reference/adapt/golden-adapter/golden-worker-adapter.md` + `.claude/skills/day0-inference/reference/design/golden-designer/golden-worker-designer.md`
>
> ⚠️ **行号防腐**：全文 `文件:行号` 引用以函数名/类名为锚点，行号随版本漂移，失效时按函数名 grep 重新定位。

## 1. 文档定位与主流程

- **设计（Phase 1）**：以 `.claude/skills/day0-inference/reference/design/golden-designer/golden-designer.md` 为唯一权威——它依次调用三个 designer 子文件产出三份 design spec（`design/service-design-spec.md` / `schedule-design-spec.md` / `worker-design-spec.md`）+ 跨层汇总，交付物契约与打回条件不在本文重复。
- **开发（Phase 2）**：按 §3 的落地三步执行——**三个步骤就是用三个 adapter 子文件完成开发**，各 adapter 以对应 design spec 为输入、以对应知识库为写法参考，产出统一汇入 `./.day0/<model>/impl/` 的 G1 门禁证据。
- **本文的自有内容**：分层框架（§2）、落地三步（§3）、Q0 模型级前置检查（§4）、组合矩阵回归清单（§5）、术语纪律与实现顺序铁律（§7）——这些是跨层公共物，不属任何单一子文件。

> 依赖阻塞（上游未合入 / CANN 算子缺口）不是实现缺陷，不进修复回路——处置规则（并行预案 / 停止提 issue）见 golden_flow Phase 0 门禁。

## 2. 分层框架：两套视角的融合

**视角一（工作量在哪里）——L1/L2/L3 三层**：

- **L1 模型定义层（vLLM 侧）**：上游是否已有该模型实现。有 → 零工作量；无 → 需写 vLLM 风格模型定义。**关键认知：L1 往往已经完成**——上游有 `vllm/models/<model>/{nvidia,amd}/` 厂商分支约定，此时 vllm-ascend 的工作是**补一个 ascend 覆盖实现，而非从零实现**（基线怎么选见 §4）。
- **L2 算子层（vllm-ascend 侧）**：模型用到的每个 module 在 NPU 上是否已有实现（已注册 → 零适配；未注册 → 新增 OOT；上游硬编码 CUDA 分支 → monkey patch）。
- **L3 框架配置层（EngineCore / Platform / Worker）**：KV cache 形态、注意力后端、调度、并行、图模式、量化识别。**这一层不改，模型能构造出来但跑不起来或跑错。**

**视角二（挂什么）——运行链路四层**：服务层（API server / parser / tokenizer，决定请求如何被编码与解析）→ 调度层（Scheduler / KV Cache / 投机解码 / PD 分离，决定显存如何被切块与复用）→ Worker 层（NPUWorker / ModelRunner / 并行策略 / 权重加载，决定模型如何被实例化与驱动）→ 算子层（Worker 层之下的执行细节：attention backend 与自定义算子，决定每个算子落在哪条 NPU 实现路径上）。两套视角不冲突：L1 对应 Worker 层的模型实现面，L2 即算子层，L3 横切调度层与 Worker 层控制面。

**分层不是人为切分**：四层各自的注册点（parser 注册表、`KVCacheSpecRegistry`、`ModelRegistry`、`get_attn_backend_cls`）与失败模式（静默错配、显存错估、图捕获崩溃、算子缺失）**互不重叠**，因此适配顺序基本固定：

1. **服务层协议先行验证**——参数与 parser 错配是静默失败，最先排除；
2. **KVCacheSpec 定稿决定调度可行性**——block size、分组、prefix 命中语义全部由 spec 导出，事后更改级联推翻调度与图模式配置，必须先于一切性能工作定稿；
3. **Worker 层实例化**——管线、并行约束、权重加载；
4. **算子层收口**——性能与正确性长尾。

## 3. 落地三步（Phase 2 开发主流程）

按 §2 的顺序，依次执行三个 adapter 子文件完成开发。三者输入均为对应 design spec（判定表是执行过滤条件：只落地「需要适配」的条目），写法参考均为对应知识库，产出统一汇入 G1 门禁证据：

| 步骤 | 执行文件（落地流程） | 输入 spec | 写法参考（知识库） | 落地物 |
|---|---|---|---|---|
| ① 服务层 | `.claude/skills/day0-inference/reference/adapt/golden-adapter/golden-service-adapter.md` | `design/service-design-spec.md` | `.claude/skills/day0-inference/reference/golden-service-knowledge.md` | parser plugin / tokenizer-mode / 三件套与多模态配置 + 成套性自检 |
| ② 调度层 | `.claude/skills/day0-inference/reference/adapt/golden-adapter/golden-schedule-adapter.md` | `design/schedule-design-spec.md` | `.claude/skills/day0-inference/reference/golden-schedule-knowledge.md` | KVCacheSpec 注册与选型落实 + E1-E12 配置改动 |
| ③ Worker 层 | `.claude/skills/day0-inference/reference/adapt/golden-adapter/golden-worker-adapter.md` | `design/worker-design-spec.md` | `.claude/skills/day0-inference/reference/golden-worker-knowledge.md`（代码模板：`reference/adapter-templates.md`） | 逐 module 类型 0-5 实现 + 权重映射 loader + UT/OOT 自检 |

## 4. Q0：模型级平台分派前置检查

进入逐 module 判定**之前**，先做这个**模型级**检查——同一模型只问一次，不是 per-module 的。

**检查动作**：读上游 `vllm/models/<model>/__init__.py` 的分派逻辑。**陷阱**：若分派只按 `is_rocm()` 二分（`if not current_platform.is_rocm(): from .nvidia... else: from .amd...`），NPU 会**静默落入 nvidia 分支**，import CUDA 专属代码直接崩。

**处置（Q0 命中）**：必须在 `vllm_ascend/models/__init__.py::register_model()` 注册覆盖实现（即 EngineCore 清单的 E1 项），且确保 ascend 分支**不 import 上游 nvidia 模块**。Q0 命中直接影响模型级路径判定（覆盖注册本身计入工作量），路径分级的完整规则见 golden_flow Phase 0。

**基线分支选择**：上游有多个厂商分支时，**不要默认选 nvidia**——选错会让工作量翻倍。检查命令：

```bash
# ① 查各分支的模块级厂商 import（NPU 上 import 即崩的那种）
grep -nE "^(from|import).*(cute|cutlass|aiter|hip|rocm|nvshmem)" <branch>/*.py

# ② 对比各分支用的是「可注册层」还是「私有类」
grep -rn "PluggableLayer.register\|CustomOp.register" <该分支引用的层>
```

判据（优先级从高到低）：

| 观察 | 含义 |
|---|---|
| 模块级 import 厂商专属 kernel（如 `cute_dsl`） | ❌ 该分支不可用，import 即崩 |
| 用带 `@PluggableLayer.register` 的**共享层** | ✅ 可走类型 2（注册替换） |
| 用**私有类**（无注册装饰器） | ⚠️ 只能走类型 3（monkey patch），成本更高 |
| 行数更少 | ✅ 通常意味着厂商特化更少 |

> 实践中曾出现 nvidia 分支不可用而 amd 分支干净的情况，务必逐个查。判定表表头须注明选定的基线分支名——基线选错，整张判定表跟着错。

通过 Q0 后，才进入逐 module 判定（决策树与类型 0-5 定义见 `.claude/skills/day0-inference/reference/ascend-oot.md`，逐 module 落地机制见 `.claude/skills/day0-inference/reference/golden-worker-knowledge.md`）。

## 5. 跨层公共物：组合矩阵回归清单

历史已知问题几乎全部位于**叠加组合**而非基线。Day0 验收须按 Designer 产出的清单覆盖以下笛卡尔积的相关子集，**每个启用组合至少一条 E2E 用例，配置落在 `tests/e2e/models/configs/<Model>.yaml`（汇入 G4 发布门禁）**：

- **量化**：BF16 / W8A8 / W8A8C8 / W4A8MXFP / MXFP4 等该模型实际发布的权重族；
- **图模式**：eager / PIECEWISE / FULL_DECODE_ONLY（稀疏与线性注意力只承诺 UNIFORM_BATCH）；
- **投机解码**：无 / MTP / eagle / DSpark，verify 步 1+k 与 capture size 对齐；
- **CP/PD**：DCP / PCP / PD 分离（含 connector 形态）。

**三条历史交叉项事故先例**（新读者理解"为什么必须覆盖组合"的依据）：MTP × prefix cache 静默输出损坏；DSpark × 混合 KV 分组 padding 膨胀；量化 draft × BF16 target page-size 统一失败。组合维度的机制细节分别见 golden-schedule-knowledge.md 与 golden-worker-knowledge.md（路径同在 `.claude/skills/day0-inference/reference/` 下）。

## 7. 术语纪律与实现顺序铁律

**术语纪律**（两套体系不得混用、不得出现在同一张表）：**P0/P1/P2** 只用于**模型级路径判定**——回答"走哪条路"，是 golden_flow Phase 0 的产出与 G0 门禁对象；**类型 0-5** 只用于 **module 级适配判定**——回答"怎么适配"，是逐 module 判定表的唯一合法标注。一个模型级 P2 内部可以有大量类型 0 的 module，两者不冲突。

**实现顺序四条铁律**：

1. **先正确后性能**：所有新算子先用 torch 等价实现跑通精度，再换融合算子；
2. **先 eager 后图**：`enforce_eager=True` 跑通再开 ACL Graph；
3. **先小规模后并行**：单卡跑通再上 TP/EP/DP；
4. **静默错误显式化**：所有兜底 `else` 分支加 `raise NotImplementedError`，把静默错误变成早期失败。
