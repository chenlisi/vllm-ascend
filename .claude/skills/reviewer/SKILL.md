---
name: reviewer
description: "代码评审知识库与只读工具（Reviewer 子 agent 的资产）。承载评审思路（提问参考，14 组 50 条）、评审规则表（复查清单，14 组 302 条，按文件 glob 匹配）与三个确定性只读脚本（选文件+覆盖记账、影响面+风险排序、行号回填）。当需要按仓内真实评审沉淀的标准评审一段 diff、或复用这些脚本做覆盖记账与行号定位时使用。触发词：代码评审、review、规则表、评审思路、review_scope、impact_scan、locate_comment。"
---

# Reviewer 知识库与只读工具

本 skill 承载 **Reviewer 子 agent**（定义在 `.claude/agents/reviewer.md`）的**知识库与确定性工具**——工具与规则自包含，可被 Day0 之外的流程复用。

## 评审顺序（先读源码，再用规则表复查）

参照 `code-review-graph` 的原则——**工具只缩小范围，判断交给读过源码的评审者；冲突时以源码为准**：

| 阶段 | 谁做 | 用什么 |
|---|---|---|
| ① 跑脚本 A / B | 脚本 | 收窄范围：选文件 + 覆盖记账；影响面 + 缺失测试 + 风险排序 |
| ② 读源码出意见 | 评审者 | 提问参考 `reference/review-heuristics.md`——**此阶段不展开规则表** |
| ③ 用规则表复查 | 评审者 | `reference/review-rules.md`——补漏 + 剔除与历史**未被采纳**意见同型的噪音 |
| ④ 定位 | 脚本 C | 行号回填（唯一命中才认定），行号不由模型手写 |

## 目录结构

| 路径 | 内容 | 用在 |
|---|---|---|
| `reference/review-heuristics.md` | 评审思路：14 组 50 条跨组推理思路，四段式（思路 / 触发 / 典型问句 / 支撑来源） | ② |
| `reference/review-rules.md` | 评审规则表：14 组 302 条，按文件路径 glob 分组，每条附真实来源 PR | ③（复查） |
| `scripts/review_scope.py` | **脚本 A**：确定性选文件 + glob 规则匹配 + 覆盖记账 → `scope.json` | ① |
| `scripts/impact_scan.py` | **脚本 B**：改动符号 → N 跳调用者/测试反查 → 风险排序 + 缺失测试 → `impact.json` | ① |
| `scripts/locate_comment.py` | **脚本 C**：分级行号定位（hunk 新增行 → 全文精确 → 滑窗模糊）+ findings 批量回填 | ④ |

## 使用方式

```bash
# ① 脚本 A：选文件 + 覆盖记账（--preview 与正式评审共用同一实现；规则匹配结果落盘，③ 才用）
python .claude/skills/reviewer/scripts/review_scope.py --preview
python .claude/skills/reviewer/scripts/review_scope.py --base <base> --head <head> --out <输出根目录>

# ① 脚本 B：影响面 + 缺失测试 + 风险排序（限深 ≤2、限量）
python .claude/skills/reviewer/scripts/impact_scan.py --base <base> --head <head> --out <输出根目录>

# ④ 脚本 C：findings 批量回填行号（同时给 --base 与 --ref 才启用 hunk 侧别匹配）
python .claude/skills/reviewer/scripts/locate_comment.py \
  --findings <输出根目录>/review/findings.json --ref <head> --base <base> \
  --out <输出根目录>/review/findings.json
```

## 四条纪律

1. **只读**：三个脚本不写源码，只写 `<输出根目录>/review/`。禁止改 tracked 源码、代提交、自动应用修复。
2. **脚本产证据、判定归评审者**：脚本给「证据与线索」，结论判定归 Reviewer 与主控；脚本与源码冲突时**以源码为准**。**必须读过源码再下结论**，不得凭脚本输出或规则命中直接判定。
3. **空结果 ≠ 不存在**（同 crg：「empty graph result can mean *not indexed* or *not statically visible*, not *does not exist*」）：脚本 B 报 0 条缺失测试、脚本 A 报无规则命中，都只代表「静态未发现」——不得据此写「此处无问题」。
4. **证据先行**：每条问题先给证据（`文件:行` 或机器可读信号）再给结论；**建议级无证据即剔除**。行号一律由脚本回填，不由模型手写。

## 与 day0-inference 的关系

Day0 的 Stage 1 Phase 4 会**调用** Reviewer 与这些脚本（见 `.claude/skills/day0-inference/flows/golden_flow.md`）。边界要说准，不要把"自包含"说过头：

- **工具与规则自包含**（本 skill 内）——规则表与评审思路是全仓通用的开发约定，可独立用于任何一段 diff 的评审；
- **领域判定依据外挂** day0-inference 的知识库：A 项用 `reference/golden-worker-knowledge.md`（§2.1.7 枚举完整性、§1.5 加载期映射），B 项用 `reference/ascend-oot.md`（附录 B/C）。这是**合理耦合**（评审对象就是 Day0 的适配改动），不在本 skill 内复制副本。

规则的全量出处与设计取舍见仓根 `DESIGN.md`（§6.2、§7、§10）。
