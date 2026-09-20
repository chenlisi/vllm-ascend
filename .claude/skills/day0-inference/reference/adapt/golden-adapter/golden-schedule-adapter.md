# Golden Schedule Adapter — Stage 1 调度层落地流程

> 本文是**流程步骤**，由 `.claude/agents/developer.md`（Stage 1 实现步骤）在调度层落地时加载执行。
> 输入是设计侧产出的调度层 design spec；**代码怎么写，参考知识库 `.claude/skills/day0-inference/reference/golden-schedule-knowledge.md`**（机制细节：KVCacheSpec 注册与选型判据、page size 对齐、调度器装配、E1-E12 配置中枢、prefix cache / 投机调度 / PD 运行时语义）。

## 输入

- `$ASCENDBOT_FILE_PATH/design/schedule-design-spec.md`：场景判定总表 + 逐场景方案块（选型与理由 / 实现方式 / 落点 / 实现依据章节号）+ **KVCacheSpec 定稿（标注「定稿前禁止进入性能工作」）** + E1-E12 标记清单（逐项 `✅/❌/⚠️` + 改动点 + 归属层）。

## 执行步骤

1. **圈定执行范围**：读 spec 的场景判定总表，圈出判定为「需要适配」的条目（含 ⚠️ 标注的新场景）；「无需适配」的条目不落地、不加载对应知识章节。
2. **逐条目落地**：每条按其方案块执行——
   - 方案块有「实现依据」章节号 → 查阅 `golden-schedule-knowledge.md` 对应章节，按其机制实现：
     - KVCacheSpec 新增/复用 → `vllm_ascend/core/kv_cache_interface.py::register_ascend_kv_cache_specs()` 注册（机制 §一.1-2）；混合 KV cache 与 page 对齐 → `patch/platform/patch_kv_cache_{coordinator,utils}.py`、`patch_mamba_config.py` 对齐断言、`utils.py::refresh_block_size`（§一.3）；
     - 调度器装配 → `scheduler_cls` 派生与 `check_and_update_config` 互斥校验（§一.4）；
     - EngineCore 本层归口项（E4/E5/E6/E12）→ `check_and_update_config` / `patch/platform/` / `patch_speculative_config.py`（§一.5、§二.7）；
     - prefix cache / PD 运行时项 → §二.6、§二.8 的机制与前置检查；
   - ⚠️ 新场景 → 知识库无对应章节，直接按方案块的「建抽象」设计实现，并在交付物中保留 Reviewer 重点关注标注；
   - 归属其他层的 E1-E12 项（E1/E2/E3/E7/E8/E9/E10/E11）不在本层落地——按 spec 标注的归属层分发到对应层的落地流程。
3. **自检**：两项——
   - **KVCacheSpec 定稿确认**：实现与 spec 的 KVCacheSpec 定稿逐字段一致（各层 spec 承接、page_size、分组、merge/prefix 命中语义）；任何偏离即推翻定稿，回 Designer 重新定稿——**禁止带着未定稿的 spec 进入任何性能工作**；
   - **E1-E12 标记落实**：spec 标记为本层的项逐项核对落实证据（注册点 / assert / 配置收敛），标记为其他层的项确认已在对应层 spec 中有着落。
4. **验证与证据归集**：实现级验证随 UT（汇入 G1 门禁证据）；组合回归（量化 × 图 × 投机 × CP/PD）不在本阶段执行——按 spec 的组合矩阵清单，属 Stage 3 / G4（组合高发项定义见知识库 §二.6-8）。

## 输出

- 调度层改动清单（KVCacheSpec 注册 / platform 配置 patch / 调度器装配与互斥配置 / 投机调度配置）+ UT 证据 + KVCacheSpec 定稿确认与 E1-E12 落实自检结论 → 汇入 `./.day0/<model>/impl/` 的 G1 门禁证据。

## 纪律

- spec 是唯一权威输入：判定疑问回 Designer 澄清，不自行改判定；实现中发现 spec 与代码事实冲突时，以代码事实为准并回 Designer 修正 spec。
- **配置与调度适配必须放 platform 组**（`pre_register_and_update` / `check_and_update_config`），不放 worker 组补丁——engine-core 进程在 worker 启动前就要看到（机制见知识库 §一.5 关键原则）。
- 术语纪律：本层场景以知识库章节为单位，不出现 P 级标注（P0/P1/P2 只属于模型级路径判定）与类型 0-5 标注（只属于 Worker 层 module 级判定）。
