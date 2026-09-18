# MySR 前端仓库开发日志

记录 Python 前端、用户接口、桥接、导出和前端测试的重大变更。

## 记录格式

后续重大变更至少记录：日期、变更类型、影响范围、原因、修改路径、结果/验证证据、遗留风险和后续行动。

## 2026-09-04 - 建立文件夹级说明与日志约定

- 变更类型：结构与工作流规范化。
- 影响范围：本开发单元的说明文件、局部约束和日志入口。
- 结果：建立本目录的持续记录入口；未来必要的大幅修改应追加到本文件。
- 验证：目录职责已与 `MySR_Dev/AGENTS.md` 和 `memory/DEVELOPMENT_MEMORY.md` 的总规则对齐。

## 2026-09-05 - 暴露 mutation affinity Python 接口

- 变更类型：跨仓库公共接口桥接。
- 影响范围：`mysr/sr.py`、`mysr/test/test_dimensional_formula_type.py`。
- Decision：MySRRegressor 新增 `mutation_affinity`、强度、探索比例、operator/feature affinity 参数，并在 `_run` 中转换为 Julia `Options`；保留前端参数校验与后端矩阵校验的双层契约。
- Confirmed：构造器/get_params 验证通过；使用本地 MySRCore checkout 的 env_mysr fit smoke 成功（niterations=0，输出方程表可生成）。
- 验证：`python -m pytest -q mysr/test/test_dimensional_formula_type.py`：16 passed；Julia 后端正式 checkout 直接测试与 `Pkg.test()` 全部通过。
- 遗留风险：当前 `juliapkg.json` 仍指向发布版 MySRCore v1.1.0；使用新参数前需在开发环境启用本地 MySRCore 或等待包含该接口的后端发布版本。普通 env_mysr 已通过 Pkg.develop 指向本地 checkout，冻结 Benchmark 环境未改动。

- 补充验证：在本地 MySRCore checkout、`niterations=1`、单 population/单 cycle 的小型搜索中，带 operator/feature affinity 参数的 Python `fit` 成功，`equations_` 返回 1 行 × 6 列；运行产物已放入父级 `outputs/MySR/mutation-affinity-python-smoke/`。
- 兼容性修正：默认参数不再向旧版 MySRCore 发送未知 affinity 关键字；仅当用户改变 affinity、强度/探索比例或提供矩阵时才转发。额外 Python 参数测试与本地后端 `niterations=1` fit smoke 均通过。

## 2026-09-06 - 修复 AFE operator alias 命名

- 变更类型：AI Feynman-like feature provenance 可追溯性修复。
- 影响范围：`mysr/feature_engineering.py`、`mysr/test/test_feature_engineering.py`。
- **Confirmed**：旧实现对 signature 做全局 `replace("x", "")`，会把 `exp(x0)`/`exp_ratio(x0,x1)`
  错命名为 `afe_ep_0`/`afe_ep_ratio_0_1`；现改为仅移除 `x<index>` 变量 token，保留 operator
  名称，重复 basename 仍由 ensemble 的 `__2` 规则消歧。
- **Verification**：新增 `exp` 与 `exp_ratio` 命名回归断言；在 `env_mysr`、临时可写 Julia
  depot（叠加环境原 depot）下 `mysr/test/test_feature_engineering.py` 为 `61 passed`。
- **Residual**：本修复发生在已完成的 v0.2 formal run 之后；该 run 的原始 alias 与 expanded
  provenance 保持不变，后续若需要能力结论应在新版本 benchmark 中重新部署。

## 2026-09-06 - 增加 AFE 结构候选档案与 provenance

- 变更类型：AI Feynman-like AFE 结构能力增强。
- 影响范围：`mysr/feature_engineering.py`、`mysr/test/test_feature_engineering.py`。
- **Confirmed**：新增可选的 deterministic structural basis，支持 radial square、symmetric
  sum/product、二元及三元小规模组合，并支持 antisymmetric difference；新增
  `relation_kind`、`structural_score` 和 `feature_graph_`/`get_feature_graph()`，记录原始节点、
  派生节点、输入映射、复杂度和生成证据。
- **Decision**：新候选保持显式开关 `enable_structural_basis=False`、
  `enable_anti_invariance=False`，避免改变已有配置的增广列数量；新的高能力 benchmark profile
  必须显式开启两个开关。
- **Verification**：`python -m compileall -q mysr`、Ruff 检查通过；在 `env_mysr` 下完整
  `mysr/test/test_feature_engineering.py` 为 62 passed；独立结构测试成功生成并接受
  `((x0)^2 + (x1)^2)` 与 `(x0 - x1)`，远程 `env_1_mysr` 编译/import 检查通过。
- **Unknown**：结构候选尚未实现递归 MySRCore 子问题和 HOF 回代；新候选对匹配 benchmark 的
  recovery、evaluations、耗时和内存收益必须在新协议中验证。

## 2026-09-06 - AFE 梯度探针与扩展结构基

- **Confirmed**：增加有限差分 sum/difference 结构证据、低阶多项式、elementary-symmetric 与
  dimensionless ratio 候选，并持久化 `reduction_plan`；新分支默认关闭以保持旧 API 行为。
- **Verification**：Ruff/compileall 通过；`mysr/test/test_feature_engineering.py` 为 63 passed；
  梯度结构、结构图与 reduction plan 回归测试已加入。
- **Unknown**：尚未在 RS 题库上测量质量和资源收益。

## 2026-09-06 - AFE 递归入口与反馈语料修正

- **Confirmed**：fit 流程先归档 structural candidates，再把接受的结构中间量送入 composition beam；梯度探针增加活动度门控。
- **Confirmed**：MySRCore 第一次有效反馈可清理 structural bootstrap，只保留当前真实 loss 语料。
- **Verification**：AFE/RNN 前端 106 passed，MySRCore `Pkg.test()` passed。

## 2026-09-06 - RNN-GPSR ranking/diversity decoding

- **Confirmed**：`TorchRNNGenerator` 使用 pairwise ranking loss 学习整个反馈批次的质量顺序，
  以 marginal entropy 抑制 token 模式坍缩，并支持长度惩罚与 temperature/top-k/top-p 采样。
- **Confirmed**：`MySRRegressor` 暴露并校验对应参数，默认配置不改变已有调用。
- **Verification**：RNN-GPSR seeding suite 为 38 passed；compileall/Ruff 通过。
- **Unknown**：真实 GPSR feedback round 的 proposal quality 和额外训练成本待 RS benchmark 验证。

## 2026-09-06 - Grammar-aware RNN teacher forcing

- **Confirmed**：RNN-GPSR training logits now receive per-prefix legality masks derived from token arities and the length ceiling, matching the batched sampler's grammar constraints. This focuses learning capacity on syntactically completable expressions; an overlong training sequence is rejected rather than silently truncated.
- **Verification**：`mysr/test/test_rnn_gpsr_seeding.py` 为 `42 passed`；compileall 和 Ruff 通过。
- **Unknown**：当前 RS array 30052 使用的是此前同步的 snapshot，不包含本轮改动。

## 2026-09-06 - RNN PQT supervision and AFE permutation structure

- **Confirmed**：`TorchRNNGenerator` now adds a configurable normalized elite-likelihood term (`rnn_elite_supervision_weight`, high profile `0.2`) while preserving the old positional `_policy_loss` call contract. Grammar masks apply to this term as well.
- **Confirmed**：`SurrogateEngineConfig.enable_permutation_basis` archives the three-variable alternating Vandermonde factor when explicitly enabled; the high-capability profile enables it.
- **Verification**：combined AFE/RNN suite `111 passed`; no backend or scheduler files changed.
- **Unknown**：the current RS job does not contain this snapshot and must not be interpreted as an evaluation of these changes.

## 2026-09-06 - Data-aware RNN bootstrap and elitist proposal replay

- **Confirmed**：RNN-GPSR bootstrap proposals are ordered by finite dataset loss plus a tiny complexity tie-break, rather than complexity-only pseudo-cost; the resulting evaluation count is returned to MySRCore. Each accepted proposal batch replays the best finite training sequences according to `rnn_replay_fraction`, while retaining sampled diversity.
- **Confirmed**：the direct-Slurm high-capability profile enables the data-aware path with 256 candidates, four feedback rounds, tied ranking, constrained sampling, and 25% replay.
- **Verification**：AFE/RNN focused suite `107 passed`; benchmark suite validation and repository tests passed. Real-task recovery and runtime impact remain Unknown until RS job 30052 completes.

## 2026-09-07 - 1.1.1 RNN-GPSR length alignment release

- **Confirmed**：`rnn_gpsr.py` now sizes the teacher-forcing grammar mask to the actual
  language-batch time dimension while retaining `max_length` for completion legality.
- **Verification**：eight mixed sequences produced matching `(8, 5, 3)` tensors and finite
  log probabilities; `py_compile` and `git diff --check` passed.
- **Decision**：released as MySR `v1.1.1` on `codex-rnn-align-v1.1.1-20260907`, commit
  `adc93bb1d3c6dd66324d2a16bb63f1b3e2174dc6`, and pushed with tag `v1.1.1`.

## 2026-09-08 - 1.1.2 benchmark release record

- **Confirmed**：MySR package metadata and juliapkg pin are at `1.1.2`/MySRCore `v1.1.2`;
  release commit `ea6f30b` and tag `v1.1.2` are present.
- **Decision**：the next remote comparison invokes the same frontend through three declared
  capability profiles (AFE, RNN-GPSR, and empty) so the ablation isolates feature paths without
  changing the solver environment or backend.
- **Verification**：frontend regression checks and compile validation passed before packaging;
  final resource/result evidence is deferred to the direct Slurm run.
- **Unknown**：ablation recovery and frontier quality are not yet measured.

## 2026-09-09 - Keep raw feature names available when AFE is disabled

- **变更类型**：前端能力元数据契约修复。
- **Confirmed**：禁用 `auto_feature_engineering` 时，拟合初始化曾将
  `augmented_feature_names_` 留为 `None`；这与该属性的公开类型及 benchmark capability
  evidence 需求不一致，并触发了 RS 四组消融中 M_RNN/M_EMPTY 的报告阶段崩溃。
- **Decision**：在预变换流程结束后，禁用 AFE 的模型将 `augmented_feature_names_` 设置为
  当前（含可选 feature selection 后）的原始搜索特征名；启用 AFE 的增广列行为保持不变。
- **修改路径**：`mysr/sr.py`、`mysr/test/test_feature_engineering.py`。
- **Verification**：AFE/RNN 与 feature-engineering focused suite 在临时可写 Julia depot 下
  `113 passed`；新增 disabled-AFE raw-name regression 通过。
- **Residual/Unknown**：该修复尚未发布新的 MySR tag，也尚未在 RS 上重跑四组 benchmark。

## 2026-09-11 - Final 1.1.3 enhancement pass

- **变更类型**：AFE 与 RNN-GPSR 能力极限补强（本地可控范围内）与版本对齐。
- **Confirmed**：在前端扩展了 AFE 结构候选能力（`normalized_sub`、`reciprocal`、`exp`）并放宽了
  FEAT-like 种群探索上限与 beam 规模；增强 RNN-GPSR proposals 的 replay 顺序与回填策略；
  并将 RNN-GPSR 默认候选/提案数与 feedback 轮次上调（160/160/3）。
- **确认路径**：`mysr/feature_engineering.py`、`mysr/feat_engine.py`、`mysr/rnn_gpsr.py`、
  `mysr/sr.py`、`mysr/juliapkg.json`、`mysr/test/test_juliapkg_config.py`、`pyproject.toml`。
- **Validation**：`python -m compileall` 覆盖了改动文件；`test_juliapkg_config` 在当前
  `env_mysr` 下受 Julia depot 只读约束而不可完成（`read-only file system`）。
- **Decision**：将本次变更归档为 `MySR 1.1.3`，与 MySRCore 1.1.3 同步发布候选。

## 2026-09-11 - RNN fallback and AFE naming alignment stabilization (commit a5d4c2a)

- **变更类型**：防止 AFE 与 RNN-GPSR 在边界条件下失去输出（空提案）或命名错配。
- **Confirmed**：
  - [mysr/rnn_gpsr.py](mysr/rnn_gpsr.py): 当 quality gate 未通过时，`TorchRNNGenerator` 改为退化到训练序列回放；新增回放去重、`fallback_generated_count` 与 `sampling_attempts` 诊断字段。
  - [mysr/feat_engine.py](mysr/feat_engine.py): 在 `FeatureEngineeringEnsemble` 中按 proposal signature 做重名回放对齐，避免 bundle names/downstream_columns 被原 name 映射误配。
  - [mysr/test/test_rnn_gpsr_seeding.py](mysr/test/test_rnn_gpsr_seeding.py)：新增 fallback 行为回归测试。
  - [mysr/test/test_feature_engineering.py](mysr/test/test_feature_engineering.py)：增加 bundle 重名映射与顺序健壮性覆盖。
- **Verification**：`python -m py_compile mysr/feat_engine.py mysr/rnn_gpsr.py mysr/test/test_feature_engineering.py mysr/test/test_rnn_gpsr_seeding.py` 通过。`pytest` 仍受 `env_mysr` 中 Julia depot 只读导致 `mysr` 包导入错误（`read-only file system`）阻断。
- **Residual/Unknown**：未跟踪输出目录 `MySR/outputs/` 仍在本地保留；本次提交未清理。

## 2026-09-12 - MySR 前端 AFE 质量审查与修复

- **Confirmed**：前端审查发现两个问题：`FeatureEngineeringEnsemble` 对 bundle 中未登记的原始/占位 node signature 直接查 proposal 表，重复名称场景触发 `KeyError`；FEAT-like 初始化预算会在深度组合 beam 前耗尽，导致配置注释承诺的组合候选无法进入搜索。
- **Decision**：bundle 排序只使用已登记 proposal signature，并为 bundle 使用独立的 improvement/validation/complexity rank；名称无法唯一映射时安全跳过 bundle 记录。FEAT-like seed 阶段优先保留变量/一元节点和受控深度组合，并为 residual beam 预留 evaluation budget。
- **修改路径**：`mysr/feat_engine.py`；提交 `9d0cf13`，修复前备份 `backup/pre-mysr-feature-ensemble-fix-20260912`。
- **Verification**：FEAT 完整测试 `68 passed`；量纲/RNN bridge `62 passed`；release JuliaPkg 配置测试 `2 passed`；`ruff check mysr/feat_engine.py`、`py_compile` 和 `git diff --check` 通过。
- **Residual/Unknown**：全仓库 Ruff 仍有历史遗留的 146 条风格/类型提示，本次未批量修改；未进行大规模 benchmark。

## 2026-09-12 - TypeSpec backend namespace and nonnumeric compatibility audit

- **Confirmed**：本地 `MySRCore` 公开边界是 `MySRCore.SymbolicRegression`，而 TypeSpec 动态生成代码仍硬编码顶层 `SymbolicRegression`；这会使本地 backend 的 TypeSpec 安装失败。
- **Decision**：`mysr/type_specs.py` 的生成模块、runtime 子模块和内部导入统一使用 `MySRCore.SymbolicRegression`；提交 `e2eaeb2`，备份分支 `backup/pre-type-spec-namespace-fix-20260912`。
- **Confirmed**：非数值 TypeSpec 在半理论包装路径中会触发 backend `one(::Type{T})`；已在 MySRCore 隔离分支加入单位元能力检查，并修正 mutation/crossover 的 eager fallback。
- **Verification**：该修改前端 TypeSpec 单用例通过；MySRCore `Pkg.test()` 全部通过。TypeSpec 全量在前 41 个测试通过后暴露另一个既有 TemplateExpression/custom combiner 特征映射越界问题，尚未修复。
- **Residual/Unknown**：TemplateExpression 自定义 combiner 的多特征随机树约束仍需单独设计和回归；未进行大规模性能 benchmark。

## 2026-09-13 - Python↔Julia bridge predicate cache

- **Confirmed**：`mysr/julia_helpers.py` 对 `jl_is_function` 使用模块级 Julia 函数谓词缓存，避免重复 `jl.seval`；语义与原实现一致。
- **Verification**：针对缓存行为的 pytest 单测通过（1 passed）；完整 backend 回归在对应 MySRCore 集成分支通过。
- **Residual/Unknown**：未缓存 `julia_state_`/`julia_options_` 的反序列化，因为对象流可能携带可变状态，需独立失效策略后再评估。

## 2026-09-13 - Focused bridge integration verification

- **Confirmed**：集成分支 `feature/integrate-bridge-cache-20260913` 保持 canonical `main` 未修改，工作树仅保留既有未跟踪 `outputs/` 与 `worktrees/`。
- **Verification**：`env_mysr` + 可写临时 depot 下 bridge cache 单测 `1 passed`，量纲/RNN 聚焦套件 `62 passed`；改动文件 `compileall` 通过。全文件 Ruff 仍报告历史遗留问题，本轮未批量改动。

## 2026-09-13 - Local-primary population migration bridge merge

- **Decision**：以 canonical MySR 分支 `feature/local-mysr-merge-20260913` 为主线，在隔离 worktree `feature/local-primary-population-migration-python-merge-20260913` 合入 population profile、migration topology/policy 参数及参数分组；保留本地 AFE、RNN-GPSR 和 bridge cache 改动。
- **Confirmed**：合并提交 `3d3cf2c`；新增 population migration focused test，未留下冲突标记。
- **Verification**：临时 Julia project 实际加载后端合并 worktree，`pytest -q mysr/test/test_population_migration.py` 为 `3 passed`；量纲/RNN 聚焦套件 `test_dimensional_formula_type.py` + `test_rnn_gpsr_seeding.py` 为 `62 passed`（1 个既有 sklearn 收敛警告）；`python -m compileall -q mysr` 通过。当前环境没有 `ruff` 可执行文件，因此未声称 Ruff 通过。
- **Unknown**：全仓库历史 lint 提示和新迁移策略的匹配预算性能仍需独立工作。

## 2026-09-13 - Surrogate convergence retry and Ruff verification

- **Decision**：`SurrogateFeatureEngineer` 对 MLP surrogate 捕获 `ConvergenceWarning` 后，使用相同 random seed 将 `max_iter` 有界翻倍重试一次（上限 4000），避免小数据集在首次预算内欠训练并向用户泄漏非阻断警告。
- **Verification**：`ruff 0.16.3` 对修改的 `feature_engineering.py` 和 migration test 通过；population migration `3 passed`；量纲/RNN 聚焦套件 `62 passed` 且不再出现 sklearn 收敛警告；`compileall` 通过。
- **Unknown**：更长重试预算的端到端 AFE 成本与搜索质量收益尚未独立 benchmark；全仓库仍有既有 lint 提示。

## 2026-09-13 - Merge bridge cache into local MySR branch

- **Decision**：将已验证的 bridge cache 集成分支合并到本地工作分支 `feature/local-mysr-merge-20260913`，不直接修改 `main`；备份分支为 `backup/pre-local-mysr-merge-20260913`。
- **Confirmed**：合并提交 `ac36a0b` 仅包含 `jl_is_function` predicate cache、对应回归测试和日志记录。
- **Verification**：`env_mysr` + 可写临时 depot 下 bridge 单测 `1 passed`，量纲/RNN 聚焦套件 `62 passed`，`compileall` 与 `git diff --check` 通过；仅保留既有线程/sklearn 警告。
## 2026-09-12 - Merge canonical local frontend into crossover worktree

- **Decision**：以 canonical MySR `main` 为前端代码主线完成合并；worktree 中没有需要独立保留的
  代码分支，最终 `feat_engine.py` 与 `type_specs.py` 与 canonical 主线一致。
- **Confirmed**：合并提交为 `7d76fc8`，未留下源码冲突或相对 canonical 的代码差异。
- **Verification**：使用 env_mysr、临时 Julia bridge 指向合并后的 MySRCore worktree，量纲/RNN
  聚焦测试 `62 passed`；`compileall` 与 Ruff 检查通过。
- **Unknown**：未运行大规模搜索或 benchmark；既有线程配置和 sklearn 收敛警告仍存在。

## 2026-09-12 - Three basic test rounds completed

- **Confirmed**：第三轮临时 Julia bridge 明确加载
  `/home/taomingyu/MySR_Dev/worktrees/crossover-optimization` backend，量纲/RNN
  聚焦测试 `62 passed`（94.39s）。
- **Verification**：`python -m compileall -q mysr` 与目标文件 Ruff 均通过；未发现跨仓库接口 BUG。
- **Unknown**：线程配置与 sklearn 收敛警告仍为既有环境/训练提示，未归因于本次合并。

## 2026-09-12 - Full static quality scan

- **Confirmed**：全量 `ruff check mysr` 报告 145 条历史问题，主要集中在旧导出器、测试辅助代码和兼容层；本轮目标文件 `feat_engine.py`、`type_specs.py` 仍保持 Ruff 通过。
- **Decision**：不对 145 条跨模块历史提示进行自动批量修复，避免改变既有 API 或测试语义；继续采用按模块、按回归覆盖逐项治理。
- **Unknown**：其余历史 lint 项需要独立的分模块清理计划。

## 2026-09-13 - Canonical backend integration bridge verification

- **Confirmed**：临时 Julia project 指向 canonical MySRCore 集成分支，前端量纲/RNN-GPSR
  测试 `62 passed`（33.65s）；`compileall` 与目标文件 Ruff 通过。
- **Unknown**：本轮未修改 Python runtime；Python 全量 145 条历史 lint 提示仍按模块治理。

## 2026-09-13 - Cache Julia function predicate in Python bridge

- **Confirmed**：`mysr/julia_helpers.py` 现在在模块初始化期间只通过一次 `jl.seval` 创建
  `_JL_IS_FUNCTION`，后续 `jl_is_function` 调用复用该 Julia closure；公共函数签名和返回
  语义保持不变。
- **Decision**：本轮仅优化无状态函数谓词的重复桥接开销；`julia_state_`/
  `julia_options_` 反序列化缓存仍作为 Proposal，待设计显式失效契约后单独处理。
- **修改路径**：`mysr/julia_helpers.py`、`mysr/test/test_main.py`；提交 `336f85b`。
- **Verification**：在 `env_mysr`（临时可写 depot 加环境 depot）下新增回归
  `pytest -q mysr/test/test_main.py -k jl_is_function_uses_cached_predicate`，结果
  `1 passed, 175 deselected`；`compileall` 与 `git diff --check` 通过。
- **Residual/Unknown**：目标测试文件仍含既有 Ruff 历史提示；未进行大规模 benchmark，
  因此尚无端到端吞吐提升量化证据。
## 2026-09-13 - Multi-agent Python quality audit

- **Confirmed**：RNN-GPSR 配置和请求边界、prefix token 合法性、replay 去重均已加固；cluster manager 仅接受受支持名称；pytest 限定 canonical 测试路径，开发 Docker/README 与 MySRCore 1.1.3 对齐。
- **Verification**：RNN-GPSR `54 passed`；目标源文件 Ruff、compileall、`git diff --check` 通过。
- **Unknown**：全仓 Ruff 历史 debt 尚未清理；完整跨语言测试需可写且依赖齐全的 Julia depot。

## 2026-09-14 - Cluster-manager validation before optional package loading

- 变更类型：Julia bridge 配置校验与测试隔离。
- **Confirmed**：此前 `load_required_packages` 会在 `_load_cluster_manager` 校验前尝试安装
  `ClusterManagers`；错误的 manager 名称因此可能先触发无关 registry 操作并遮蔽配置错误。
- **Decision**：新增 `_validate_cluster_manager`，在所有 package 操作前复用白名单校验；Slurm
  manager 的显式 package 请求不会隐式加载 `LoopVectorization`/turbo 扩展。
- 修改路径：`mysr/julia_helpers.py`、`mysr/julia_extensions.py`、`mysr/test/test_main.py`。
- **Verification**：目标 bridge 文件 Ruff、compileall、`git diff --check` 通过；新增测试覆盖无效
  manager 的零 package 调用和 Slurm package 请求集合。完整 Julia 集成测试受当前 env_mysr
  depot 的只读编译缓存/registry 状态限制，未伪造为通过。
- **Residual/Unknown**：真实 Slurm allocation 仍需在已安装 `SlurmClusterManager` 的 Julia
  project 中验证；缺失 optional package 属于环境准备问题，不由该校验修复。

## 2026-09-14 - Python Ruff implementation and test debt cleanup

- **Confirmed**：对 `mysr/` 执行 Ruff 0.16.3 全仓扫描，基线 141 条诊断；生产源码中的
  低风险规则已按模块修复，包括动态导出器初始化缓存、可变默认参数、桥接导入、约束
  类型注解、checkpoint 临时文件清理和回归器中的安全等价重写。修改提交为 `40ee006`
  和 `86ac5fd`。
- **Decision**：保留 `ValueError` 等既有公开异常类型，无法安全替换的动态 `exec` 与
  checkpoint broad exception 采用有理由的局部 noqa；不使用 unsafe Ruff 自动修复，避免
  改变测试或用户接口语义。
- **Confirmed**：测试目录仅应用字面量、导入排序、字符串列选择和 `readlines()` 等安全
  等价清理，提交 `ac08ead`；全仓 Ruff 降至 67 条，剩余全部位于测试/Notebook，不再
  包含生产 `mysr/` 源码诊断。
- **Verification**：生产目标文件和 `sr.py` Ruff 通过；`python -m py_compile mysr/sr.py`、
  `python -m compileall -q mysr/test`、`git diff --check` 通过。pytest 仍被
  `env_mysr` Julia depot 的只读编译缓存（EROFS）阻断，未将其记录为源码测试通过。
- **Residual/Unknown**：测试/Notebook 中剩余 C408、SIM117、subprocess check/capture、
  BLE001/S102 等 67 条提示暂不批量修改；需要后续按测试模块逐项审查并在可写 Julia
  depot 中运行回归。

## 2026-09-14 - Preserve abstract no-op semantics during lint cleanup

- **Decision**：抽象方法继续使用原有 `pass`（仅局部抑制 PIE790），不以 `...` 替换，
  从而保持潜在 `super()` 调用返回 `None` 的既有语义；提交 `e288a1e`。
- **Verification**：相关四个配置模块 Ruff、`py_compile` 和 `git diff --check` 通过；
  全仓 Ruff 仍为 67 条，新增提交未引入诊断。
## 2026-09-14 - Ruff and cluster-manager hardening

- **Confirmed**：生产 `mysr/` 源码 Ruff 诊断从 141 降为 0；剩余 67 条位于测试/Notebook，避免以 unsafe fix 改变测试语义。
- **Confirmed**：cluster manager 在任何 Julia optional package 操作前执行白名单校验；Slurm 路径不会隐式加载 turbo/LoopVectorization。非法名称和 package-call 顺序回归测试通过。
- **Verification**：Ruff production-only 通过；compileall 和 diff-check 通过；cluster manager focused tests `3 passed`。
- **Environment limitation**：真实 Slurm/Docker 测试仍需要缺失的 Docker fixture、可写 Julia depot 和完整 optional registry，未将环境阻断报告为源码失败。

## 2026-09-14 - Full test and notebook Ruff cleanup

- **Confirmed**：`ruff check mysr` 的 67 条剩余诊断全部位于测试/Notebook；采用字面量、
  `capture_output`/显式 `check=False`、多上下文 `with`、集合字面量和导入清理等语义等价
  修改，故意的 broad-exception/`exec` 测试保留局部、有原因的 noqa。
- **Decision**：不触碰生产运行语义，也不删除用户未跟踪的 `outputs/` 或 `worktrees/`。
  修改提交为 `643731f`。
- **Verification**：Ruff 全仓报告 `All checks passed`；`python -m compileall -q mysr/test`
  和 `git diff --check` 通过。目标 pytest 收集阶段受 `env_mysr` Julia 编译缓存目录只读
  （EROFS）阻断，未伪造为测试通过。
- **Residual/Unknown**：待在可写、依赖完整的 Julia depot 中重跑受影响的 Python 测试。

## 2026-09-14 - Full Ruff cleanup completed

- **Confirmed**：剩余测试/Notebook Ruff 诊断已按语义等价方式清理，`ruff check mysr` 现在 0 diagnostics；未使用 unsafe fix 改变运行逻辑。
- **Verification**：`python -m compileall -q mysr`、`git diff --check` 通过；提交 `643731f`。
- **Unknown**：pytest 仍需可写 Julia depot 才能完成完整 collection，环境 EROFS 不构成源码断言失败。

## 2026-09-18 - Expose parent-selection and survival policies

- **Decision**：为 `MySRRegressor` 增加显式 opt-in 的 `parent_selection` 与
  `survival_strategy` 参数，默认分别为 `"tournament"` 和
  `"regularized_evolution"`，并转发为 Julia `Symbol`。
- **Confirmed**：参数出现在 `get_params()`，前端校验拒绝未知策略；参数分组文档已同步。
- **Verification**：`env_mysr` Python constructor/get-params smoke 通过；对应后端
  worktree 的 epsilon-lexicase + AFP 搜索 smoke 通过。
- **Unknown**：当前 Python smoke 使用的是环境中已注册的 backend；完整前端 bridge
  运行需在后端 worktree 被注册到该环境后再验证，尚未作为 benchmark 证据记录。

## 2026-09-18 - Submit parent-selection benchmark

- **Confirmed**：前端 worktree 与对应 MySRCore snapshot 一起部署到新的 Carbon run root；
  profile 明确转发四个 parent/survival arm，且关闭 AFE/RNN seeding 以隔离后端选择策略。
- **Verification**：`sr.py` compile/ruff 和 Python constructor/get-params smoke 已通过；
  Carbon doctor/matched-environment gate 通过，四组 array/reducer 已提交。
- **Unknown**：远程搜索尚未完成；前端参数对最终 HOF、测试误差和资源的影响待完整 reducer
  产物验证，不能把异步提交视作性能证据。

## 2026-09-18 - Move parent-selection benchmark to node2

- **Decision**：取消旧 Carbon-pinned benchmark jobs，保留旧结果目录；新的 frontend snapshot
  固定在 node2 run root，不影响 canonical checkout。
- **Verification**：新四组 array/reducer `32696/32697`、`32702/32703`、`32708/32709`、
  `32714/32715` 已提交，首批 array elements 均在 node2 运行。
- **Unknown**：搜索尚未完成，前端参数对 HOF、测试误差和资源的影响待 reducer 产物验证。
