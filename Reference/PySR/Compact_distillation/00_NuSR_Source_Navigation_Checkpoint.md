# NuSR Source Navigation Checkpoint

> 生成时间：2026-05-06
> 基于会话：2026-05-05 至 2026-05-06，多轮 PySR / SymbolicRegression.jl 源码分析
> 状态：**上下文 compact 前的知识蒸馏稿，不可替代源码，仅供导航**

---

## 0. 当前 checkpoint 的目的

本文件是 PySR（Python Symbolic Regression）和 SymbolicRegression.jl（Julia 后端）源码分析的**知识蒸馏产物**。

**核心目标**：

1. 让陶明宇（NuSR 开发者）理解 PySR 与 SymbolicRegression.jl 的整体结构，不需要每次从头读源码
2. 让 compact 之后的 Claude Code 重新读取本文件后，可以在 ~5 分钟内恢复当前源码理解水平
3. 让未来的 Claude Code 能根据 NuSR 目标，精准定位应阅读/修改/跳过的文件，避免通读全仓库
4. 避免未来再次浪费数十轮对话通读 PySR / SymbolicRegression.jl 全部源码

**使用方式**：

- 新 session 开始时，CC 应**先读本文件**，再根据用户的具体问题查第 9 节路由表
- 每次深入阅读新源码后，应**更新本文件**（追加新条目到对应表格，或修正待确认项）
- 如果某个"待确认"项被确认，应将其移到第 11 节"已确认结论"

**重要声明**：本文件所有内容均基于 2026-05-05 ~ 2026-05-06 的源码阅读。源码版本为本地解压的 PySR-master 和 SymbolicRegression.jl-master。后续 PySR/SR.jl 上游更新可能导致信息过时。

---

## 1. 当前已经读取或接触过的源码范围

### 1.1 Python 前端（PySR）

| 路径 | 已接触程度 | 主要作用 | 与 NuSR 的关系 | 是否需要后续深入 |
|---|---|---|---|---|
| `pysr/__init__.py` | 已详细阅读 | 包入口，导入全部公开 API | 理解 PySR 公开接口 | 否（已读完） |
| `pysr/sr.py` | 已详细阅读（关键方法） | `PySRRegressor` 类（~2900行），fit/predict/export 全部用户接口 | **最核心**：NuSRRegressor wrapper 的直接目标 | 是（部分方法如 `_read_equation_file` 细节待补） |
| `pysr/julia_import.py` | 已详细阅读 | juliacall 初始化，加载 SymbolicRegression.jl | 理解 Python-Julia 桥接机制 | 否（已读完，75行） |
| `pysr/julia_helpers.py` | 已详细阅读 | `jl_array`、`jl_serialize`、`jl_deserialize` 等工具函数 | NuSR wrapper 的数据转换参考 | 否（已读完，75行） |
| `pysr/julia_extensions.py` | 已详细阅读 | 按需加载 Julia 扩展包（Zygote/Mooncake/Enzyme 等） | 低（NuSR 初期不需要自动微分） | 否 |
| `pysr/julia_registry_helpers.py` | 已详细阅读 | Julia 包注册表 eager fallback 机制 | 低（部署相关） | 否（已读完，47行） |
| `pysr/expression_specs.py` | 已详细阅读 | `AbstractExpressionSpec` + `ExpressionSpec` + `TemplateExpressionSpec` + `CallableJuliaExpression` | **极重要**：NuSR 核物理模板的核心机制 | 否（已读完，435行） |
| `pysr/export.py` | 部分（~50行） | `add_export_formats`：对每个方程调用 pysr2sympy → sympy2numpy/jax/torch | NuSR 导出功能参考 | 是（需补全阅读） |
| `pysr/export_sympy.py` | 部分（~50行，映射表） | Julia 运算符 → SymPy 函数映射表 | NuSR 公式导出参考 | 是（需补全阅读） |
| `pysr/export_numpy.py` | 未读 | 待确认 | NuSR 可调用表达式导出 | 按需 |
| `pysr/export_jax.py` | 未读 | 待确认 | 低（NuSR 初期不用 JAX） | 按需 |
| `pysr/export_torch.py` | 未读 | 待确认 | 低（NuSR 初期不用 PyTorch） | 按需 |
| `pysr/export_latex.py` | 未读 | 待确认 | NuSR LaTeX 导出 | 按需 |
| `pysr/feature_selection.py` | 未读 | 待确认 | 低（核物理数据通常已预处理） | 按需 |
| `pysr/denoising.py` | 未读 | 待确认 | 低 | 按需 |
| `pysr/utils.py` | 未读 | `_preprocess_julia_floats` 等工具函数 | 低 | 按需 |
| `pysr/deprecated.py` | 未读 | 废弃 API 兼容层 | 否（NuSR 不需要向后兼容） | 否 |
| `pysr/logger_specs.py` | 未读 | 待确认 | 低 | 按需 |

### 1.2 Julia 后端（SymbolicRegression.jl）

| 路径 | 已接触程度 | 主要作用 | 与 NuSR 的关系 | 是否需要后续深入 |
|---|---|---|---|---|
| `src/SymbolicRegression.jl` | 已详细阅读 | 主模块，`equation_search` 函数（Python 入口），6 步搜索流水线 | **核心**：理解搜索全流程 | 否（已读完关键部分） |
| `src/OptionsStruct.jl` | 已详细阅读 | `ComplexityMapping` 结构体 | NuSR 物理复杂度加权 | 否（已读完） |
| `src/RegularizedEvolution.jl` | 已详细阅读 | `reg_evol_cycle`：tournament selection + 变异/交叉 | 理解进化算法核心 | 否（已读完，161行） |
| `src/SingleIteration.jl` | 已详细阅读 | `s_r_cycle`：模拟退火 + `optimize_and_simplify_population` | 理解单次迭代 | 否（已读完，142行） |
| `src/HallOfFame.jl` | 已详细阅读 | `HallOfFame` 结构体（按复杂度索引），Pareto 前沿筛选 | **重要**：NuSR 结果筛选 | 否（已读完，302行） |
| `src/LossFunctions.jl` | 已详细阅读 | `eval_loss`（3条路径），`loss_to_cost`，`update_baseline_loss!` | **极重要**：自定义物理损失 | 否（已读完，248行） |
| `src/Complexity.jl` | 已详细阅读 | `compute_complexity`：默认/加权/自定义 | NuSR 物理复杂度 | 否（已读完，65行） |
| `src/AdaptiveParsimony.jl` | 已详细阅读 | `RunningSearchStatistics`：频率跟踪用于探索-利用平衡 | 理解搜索行为调优 | 否（已读完，96行） |
| `src/Configure.jl` | 已详细阅读 | 搜索前配置测试 + `move_functions_to_workers` | 理解自定义函数如何分发到 workers | 否（已读完） |
| `src/TemplateExpression.jl` | 已详细阅读 | `TemplateExpression` 结构体，`eval_tree_array`，变异覆写，ValidVector 约定 | **极重要**：核物理模板机制 | 否（已读完，1093行） |
| `src/TemplateExpressionMacro.jl` | 已详细阅读 | `@template_spec` 宏，将 do-block 转为 `TemplateStructure` | 理解 Python→Julia 模板传递 | 否（已读完，154行） |
| `src/SearchUtils.jl` | 部分（`save_to_file`、`update_hall_of_fame!`、`construct_datasets`） | 文件持久化 + HoF 更新 | 理解 Python 如何读取结果 | 是（`@sr_spawner` 宏等未读） |
| `src/MutationFunctions.jl` | 部分（前 100 行） | 基础变异操作（swap_operands、mutate_operator、random_node） | 理解变异机制 | 是（`get_contents_for_mutation` 抽象未完整追踪） |
| `src/Mutate.jl` | 未读 | `next_generation`、`crossover_generation` 主函数 | **重要**：理解变异/交叉分发逻辑 | 是 |
| `src/Population.jl` | 未读 | `Population` 结构体，`best_of_sample`（tournament selection） | 理解种群管理 | 是 |
| `src/PopMember.jl` | 未读 | `PopMember` 结构体（tree, loss, cost, birth, parent, ref） | 理解个体表示 | 是 |
| `src/Core.jl` | 未读（但知其 export 内容） | `Dataset`、`Options`、`AbstractExpressionSpec`、`AbstractExpression` 等核心类型定义 | **重要**：理解 Julia 侧类型系统 | 是 |
| `src/ComposableExpression.jl` | 未读 | ValidVector 定义 + 子表达式包装器 | **重要**：TemplateExpression 的基础组件 | 是 |
| `src/ConstantOptimization.jl` | 未读 | BFGS/NelderMead 常数优化 | 理解参数优化机制 | 是 |
| `src/ParametricExpression.jl` | 未读 | 废弃的旧参数化表达式（`ParametricExpressionSpec`） | 低（已被 TemplateExpression 替代） | 否 |
| `src/Migration.jl` | 未读 | 种群间迁移逻辑 | 低（NuSR 初期单种群为主） | 按需 |
| `src/DimensionalAnalysis.jl` | 未读 | 物理量纲约束 | **可能重要**：核物理量纲检查 | 待确认 |
| `src/CheckConstraints.jl` | 未读 | 约束检查 | NuSR 物理约束 | 按需 |
| `src/Dataset.jl` | 未读 | `Dataset` 结构体完整定义 | 理解数据流 | 是 |
| `src/Operators.jl` | 未读 | `OperatorEnum` 结构体定义 | 理解算子枚举 | 按需 |

### 1.3 已创建的文件

| 路径 | 作用 |
|---|---|
| `/home/taomingyu/Experiment/2026/05/05/test_template_expression_spec.ipynb` | TemplateExpressionSpec 测试 notebook（21 cells），验证 `(p[1] * x1 * f(x2)) / (g(x3) - 1)` 模板 |

---

## 2. PySR 的整体结构地图

### 2.1 包入口：`pysr/__init__.py`

- **作用**：PySR 包的公开 API 入口
- **关键导入**：
  - `from juliacall import Main as jl` — Julia 运行时句柄
  - `jl.SymbolicRegression` — Julia 后端模块
  - `from .sr import PySRRegressor` — 核心类
  - `from .expression_specs import TemplateExpressionSpec, ExpressionSpec, AbstractExpressionSpec`
  - 版本信息、deprecation warning 等
- **与 NuSR 的关系**：NuSR 可以通过 `from pysr import ...` 复用，或直接 fork 并替换 `__init__.py`
- **是否建议第一版修改**：否（wrapper 层不需要改上游包入口）

### 2.2 核心类：`pysr/sr.py` — `PySRRegressor`

- **作用**：scikit-learn 兼容的符号回归估计器（~2900 行），所有用户操作的入口
- **关键类**：`PySRRegressor(BaseEstimator, MultiOutputMixin, RegressorMixin)`
- **关键方法**（按调用链顺序）：

| 方法 | 行号（约） | 作用 | NuSR 相关性 |
|---|---|---|---|
| `__init__` | 862-1136 | 存储 ~70 个超参数，不调用 Julia | 高：NuSR 可覆写默认参数 |
| `fit` | 2305-2473 | 参数校验 → `_setup_equation_file` → `_run` | **极高**：NuSR wrapper 的核心包装目标 |
| `_run` | 1905-2303 | Python-Julia 桥接核心：构建 Options → 转换数据 → 调用 `equation_search` | **极高**：理解参数如何传递给 Julia |
| `predict` | 2495-2593 | 使用 `best_equation["lambda_format"]` 可调用对象预测 | 高：NuSR 预测接口 |
| `get_best` | 1439-1485 | 按 accuracy/score/best 选择最佳方程 | 高 |
| `get_hof` | 2786-2829 | 从 CSV 读取 HoF，添加导出格式 | 高 |
| `_read_equation_file` | 2744-2773 | 读取 Julia 写入的 CSV | 中 |
| `sympy` | 2595-2623 | 返回 `best_equation["sympy_format"]` | 中 |
| `latex` | 2625-2661 | sympy → LaTeX 渲染 | 中 |
| `_setup_equation_file` | 待确认行号 | 创建输出目录和 equation CSV | 低 |

- **`_run` 方法的关键代码段**（从阅读记忆重建，非逐字复制）：

```
1. 构建 OperatorEnum：
   jl_operator_enum = SymbolicRegression.OperatorEnum(
       binary_operators=..., unary_operators=...
   )

2. 构建 Options：
   options = SymbolicRegression.Options(
       operators=jl_operator_enum,
       expression_spec=self.expression_spec_.julia_expression_spec(),
       ...
   )

3. 转换数据：
   jl_X = jl_array(X.T)  # numpy → Julia [features, rows]
   jl_y = jl_array(y)

4. 调用搜索：
   out = SymbolicRegression.equation_search(
       jl_X, jl_y, niterations=..., options=options, ...
   )

5. 保存状态：
   self.julia_state_ = out["state"]
   self.julia_options_stream_ = jl_serialize(options)
```

- **可能修改点**：NuSR 可以通过继承 `PySRRegressor` 并覆写 `_run` 或 `__init__` 来注入核物理默认参数
- **是否建议第一版修改**：否，建议用 wrapper 而非继承

### 2.3 Python-Julia 连接：`pysr/julia_import.py`

- **作用**：通过 juliacall（PythonCall.jl）建立同进程 Julia 连接
- **关键操作**：
  1. 设置环境变量：`PYTHON_JULIACALL_HANDLE_SIGNALS=yes`、`PYTHON_JULIACALL_THREADS=auto`、`PYTHON_JULIACALL_OPTLEVEL=3`
  2. `from juliacall import Main as jl`
  3. `jl.seval("using SymbolicRegression")`
  4. `SymbolicRegression = jl.SymbolicRegression`
- **与 NuSR 的关系**：NuSR 可以直接复用此机制，无需重新实现
- **是否建议第一版修改**：否

### 2.4 Julia 工具函数：`pysr/julia_helpers.py`

- **作用**：Python ↔ Julia 数据转换工具
- **关键函数**：
  - `jl_array(x, dtype)` — numpy → Julia Array（低开销）
  - `jl_serialize(obj)` — Julia 对象 → numpy uint8 数组
  - `jl_deserialize(s)` — 反序列化
  - `jl_named_tuple(d)` — Python dict → Julia NamedTuple
  - `jl_is_function(x)` — 检查是否为 Julia 函数
- **与 NuSR 的关系**：NuSR wrapper 可能需要调用这些工具函数
- **是否建议第一版修改**：否

### 2.5 Expression Spec 系统：`pysr/expression_specs.py`

- **作用**：定义表达式规范的抽象层，控制搜索空间的结构
- **关键类**：

| 类 | 行号 | 作用 | NuSR 相关性 |
|---|---|---|---|
| `AbstractExpressionSpec` | 32-85 | ABC，定义 `julia_expression_spec()` 和 `create_exports()` 抽象方法 | 理解扩展点 |
| `ExpressionSpec` | 87-126 | 默认 spec（无结构约束），返回 `SymbolicRegression.ExpressionSpec()` | 低（NuSR 主要用模板） |
| `TemplateExpressionSpec` | 128-322 | **模板 spec**：固定外层结构，只搜索子表达式 | **极高** |
| `ParametricExpressionSpec` | 352-403 | 已废弃的旧参数化表达式 | 否 |
| `CallableJuliaExpression` | 406-412 | 包装 Julia 表达式为 Python 可调用对象 | 高（预测接口） |

- **`TemplateExpressionSpec.__init__` 两种输入方式**：
  - 新格式（推荐）：`combine="..."` + `expressions=["f","g"]` + `variable_names=["x1","x2","x3"]` + `parameters={"p": 1}`
  - 旧格式（待废弃）：`function_symbols` + `combine`
- **`_template_macro_str()`**：将 Python 参数拼接为 Julia 代码字符串，例如：
  ```
  @template_spec(expressions=(f, g,), parameters=(p=1,)) do x1, x2, x3
      (p[1] * x1 * f(x2)) / (g(x3) - 1)
  end
  ```
- **`julia_expression_spec()`**：缓存调用 `jl.seval(self._template_macro_str())`，返回 Julia 端的 `TemplateExpressionSpec{TemplateStructure{...}}`
- **局限性**：
  - `evaluates_in_julia == True` — 不支持 sympy/latex 导出
  - `supports_torch == False`、`supports_jax == False`
  - `create_exports()` 返回 `CallableJuliaExpression` 而非 sympy 格式
- **是否建议第一版修改**：否，但 NuSR 应深度使用此机制

### 2.6 导出系统

- **`export.py`**：`add_export_formats` 函数
  - 调用链：`pysr2sympy(julia_equation)` → `sympy2numpy`/`sympy2jax`/`sympy2torch` → `sympy2latex`
  - 为每个方程生成 `sympy_format`、`lambda_format`、`jax_format`、`torch_format`、`latex_format`
  - **待确认**：如何处理 TemplateExpression 的导出（已确认不支持 sympy/latex）
- **`export_sympy.py`**：
  - `sympy_mappings` 字典：Julia 运算符名 → SymPy 函数
    - `"div" → lambda x,y: x/y`、`"mult" → lambda x,y: x*y`
    - `"sqrt" → sympy.sqrt`、`"sin" → sympy.sin`
    - `"square" → lambda x: x**2` 等
  - **待确认**：完整映射表、`pysr2sympy` 主函数的实现细节
- **`export_numpy.py`、`export_jax.py`、`export_torch.py`、`export_latex.py`**：均未读，待确认

### 2.7 未读但与 NuSR 潜在相关的 Python 文件

| 文件 | 预计作用 | NuSR 相关性 | 优先级 |
|---|---|---|---|
| `feature_selection.py` | 特征选择 | 低（核物理数据通常预筛选） | 低 |
| `denoising.py` | 降噪 | 低 | 低 |
| `utils.py` | 通用工具 | 中（CSV 读取兼容） | 低 |
| `deprecated.py` | 废弃 API | 否 | 否 |
| `logger_specs.py` | 日志配置 | 低 | 低 |

---

## 3. SymbolicRegression.jl 的整体结构地图

### 3.1 主模块：`src/SymbolicRegression.jl`

- **作用**：Julia 包主入口（~1291 行），include 全部子模块，export 全部公开类型/函数
- **关键函数**：

| 函数 | 行号（约） | 作用 |
|---|---|---|
| `equation_search` | 469-580 | **Python 的入口点**。接收 X/y 矩阵，构造 Dataset，调用 `_equation_search` |
| `_equation_search` | 582-597 | 6 步流水线调度器 |
| `_validate_options` | （在 Configure.jl 中） | 验证参数配置 |
| `_create_workers` | （在 Configure.jl 中） | 创建分布式 workers |
| `_initialize_search!` | 待确认 | 初始化种群、HoF |
| `_warmup_search!` | 待确认 | 渐进式 maxsize 预热 |
| `_main_search_loop!` | 883-1132 | **核心调度循环**：轮询 populations → 更新 HoF → Pareto 前沿 → 保存 → 迁移 → 分派新周期 → 早停 |
| `_tear_down!` | 待确认 | 清理 workers |
| `_info_dump` | 待确认 | 输出搜索信息 |
| `_format_output` | 待确认 | 格式化返回给 Python 的结果 |
| `_dispatch_s_r_cycle` | 1169-1212 | 包装 `s_r_cycle` + `optimize_and_simplify_population` |

- **搜索流水线（6 步）**：
  1. `_validate_options` → 参数校验
  2. `_create_workers` → 创建分布式 workers
  3. `_initialize_search!` → 初始化种群和 HoF
  4. `_warmup_search!` → 预热（渐进式 maxsize）
  5. `_main_search_loop!` → 主搜索循环
  6. `_tear_down!` → 清理 + `_info_dump` → `_format_output`

- **与 PySR 的关系**：`equation_search` 是 PySR `_run()` 直接调用的 Julia 函数
- **与 NuSR 的关系**：NuSR 如果要深度定制搜索策略，可能需要修改此文件
- **是否建议第一版修改**：否（wrapper 层不需要改 Julia 核心）

### 3.2 参数系统：`src/OptionsStruct.jl`

- **作用**：定义 `ComplexityMapping` 结构体
- **关键结构**：
  - `ComplexityMapping{T, VC, D}`：
    - `use::Bool` — 是否启用
    - `op_complexities::NTuple{D, Vector{T}}` — 每个算子的复杂度权重
    - `variable_complexity::VC` — 变量复杂度
    - `constant_complexity::T` — 常数复杂度
- **与 NuSR 的关系**：可以为不同物理算子设置不同复杂度权重
- **待确认**：`Options` 结构体的完整定义在哪个文件（Core.jl？还是 OptionsStruct.jl？）

### 3.3 进化算法核心

#### `src/RegularizedEvolution.jl` — `reg_evol_cycle` (161 行)

- **作用**：进化算法的原子单位
- **流程**：
  1. 计算本周期迭代次数 = `ceil(pop.n / tournament_selection_n)`
  2. 每轮迭代：
     - ~97.4% 概率 → `best_of_sample` 选亲本 → `next_generation` 变异 → 子代替换最老个体
     - ~2.6% 概率 → 选两个亲本 → `crossover_generation` 交叉 → 两个子代替换两个最老个体
  3. 返回 (pop, num_evals)
- **与 NuSR 的关系**：理解进化压力如何控制，但第一版不需要修改

#### `src/SingleIteration.jl` — `s_r_cycle` (142 行)

- **作用**：一个搜索迭代 = 多次 `reg_evol_cycle`（模拟退火温度递减）+ `optimize_and_simplify_population`
- **流程**：
  1. 温度从 `alpha * T` 递减到 `T`
  2. 每次 `reg_evol_cycle` 后降低温度
  3. 跟踪 `best_examples_seen`（局部 HoF）
  4. 迭代结束后 `optimize_and_simplify_population`
- **`optimize_and_simplify_population`**：可选择性地简化表达式树 + BFGS/NelderMead 常数优化

### 3.4 Hall of Fame：`src/HallOfFame.jl` (302 行)

- **核心结构**：`HallOfFame{T, L, N, PM}`
  - `members::Array{PM, 1}` — 按复杂度索引的最佳个体
  - `exists::Array{Bool, 1}` — 该复杂度位置是否已填充
- **关键函数**：
  - `HallOfFame(options, dataset)` — 构造函数，创建原型 member，分配 `options.maxsize` 大小的数组
  - `calculate_pareto_frontier` — 筛选 Pareto 前沿：每个复杂度的方程必须优于所有更低复杂度方程
  - `format_hall_of_fame` — 计算 score（基于 loss 比率和复杂度差异）
- **与 NuSR 的关系**：Pareto 前沿选择对核物理经验公式发现至关重要

### 3.5 损失函数系统：`src/LossFunctions.jl` (248 行)

- **`eval_loss`**（3 条路径）：
  1. 自定义 `loss_function(tree, dataset, options)` — 完全自定义
  2. 自定义 `loss_function_expression(expression, dataset, options)` — 表达式级自定义
  3. 默认 `_eval_loss` — 调用 `eval_tree_array` → `_loss(prediction, y, elementwise_loss)`
- **`eval_cost`**：`cost = loss_to_cost(result_loss, dataset.use_baseline, dataset.baseline_loss, member, options)`
- **`loss_to_cost`**：`loss_val = loss / normalization + size * parsimony + frequency_adaptation`
  - `normalization`：baseline_loss（常数预测均值的损失）
  - `parsimony`：复杂度惩罚系数
  - `frequency_adaptation`：来自 `RunningSearchStatistics` 的频率调整
- **`update_baseline_loss!`**：通过评估常数树来计算 baseline
- **与 NuSR 的关系**：**这是 NuSR 自定义物理损失的核心扩展点**

### 3.6 复杂度：`src/Complexity.jl` (65 行)

- **`compute_complexity`**：
  - 若 `complexity_mapping isa Function` → 调用自定义函数
  - 若 `ComplexityMapping.use == true` → 加权求和
  - 否则 → `count_nodes`（每节点 = 1）
- **与 NuSR 的关系**：可以为不同物理算子设置不同复杂度

### 3.7 自适应简约：`src/AdaptiveParsimony.jl` (96 行)

- **核心结构**：`RunningSearchStatistics`
  - `window_size::Int` — 窗口大小
  - `frequencies::Vector{Float64}` — 各复杂度的方程出现次数
  - `normalized_frequencies::Vector{Float64}` — 归一化频率
- **关键函数**：
  - `update_frequencies!` — 对某复杂度的出现频率 +1
  - `move_window!` — 当频率总和超过 window_size 时等比缩放
  - `normalize_frequencies!` — 更新归一化频率
- **与 NuSR 的关系**：理解探索-利用平衡，但第一版不需要修改

### 3.8 配置测试：`src/Configure.jl`

- **关键函数**：
  - `assert_operators_well_defined` — 用随机输入测试每个算子
  - `test_option_configuration` — 验证选项组合的合法性
  - `test_dataset_configuration` — 验证数据集维度
  - `move_functions_to_workers` — **将用户自定义函数（算子、损失函数等）复制到分布式 workers**
  - `configure_workers` — 完整的 worker 配置流程
- **`move_functions_to_workers` 覆盖的函数集**：
  - `unaops`、`binops`、`elementwise_loss`、`early_stop_condition`
  - `expression_type`、`loss_function`、`loss_function_expression`、`complexity_mapping`
- **与 NuSR 的关系**：理解自定义函数如何在多进程环境下工作

### 3.9 TemplateExpression 系统（已详细阅读）

#### `src/TemplateExpression.jl` (1093 行)

- **核心结构**：

| 结构体 | 作用 |
|---|---|
| `TemplateStructure{K, Kp, E, NF, NP}` | 模板结构定义：`combine::E`（组合函数）、`num_features::NF`（各子表达式特征数）、`num_parameters::NP`（参数长度） |
| `TemplateExpression{T, F, N, E, TS, D}` | 模板表达式实例：`trees::TS`（NamedTuple of ComposableExpression）、`metadata::Metadata{D}` |
| `ArgumentRecorder` | 代理对象，用于自动推断子表达式的参数个数 |

- **关键函数**：

| 函数 | 行号（约） | 作用 |
|---|---|---|
| `infer_variable_constraints` | 213-241 | 通过 `ArgumentRecorder` 代理调用 `combine`，自动判断每个子表达式允许的特征数 |
| `eval_tree_array` (for TemplateExpression) | 684-723 | 调用 `combine(trees, params..., ValidVector.(eachrow(cX)))`，返回 `(result.x, result.valid)` |
| `get_contents_for_mutation` | 797-803 | 随机选一个子表达式进行变异 |
| `with_contents_for_mutation` | 806-821 | 将变异后的子表达式替换回原位置 |
| `mutate_constant` | 869-900 | 两条路径：变异子表达式内部常数 或 变异 `ParamVector` 中的参数值 |
| `compute_complexity` | 552-561 | 各子表达式复杂度之和（外层结构贡献 0） |
| `has_invalid_variables` | 942-950 | 检查子表达式的特征索引是否越界 |

- **ValidVector 约定**（已确认）：
  - `combine` 函数中 `x1`、`x2`、`x3` 不是标量而是 `ValidVector`（向量 + `valid::Bool` flag）
  - 所有算术运算（`+`、`-`、`*`、`/`、`sin` 等）是向量化的
  - `combine` 必须返回 `ValidVector`，否则抛 `TemplateReturnError`
  - 零除和 NaN 通过 `valid=false` 传播

- **变异机制**：TemplateExpression 覆写了 `get_contents_for_mutation` 和 `with_contents_for_mutation`。每次变异随机选一个子表达式，对该子表达式执行标准变异操作（由 `MutationFunctions.jl` 提供），然后替换回原位置。

#### `src/TemplateExpressionMacro.jl` (154 行)

- **`@template_spec` 宏**（line 34-151）：
  - 输入：`expressions=(f,g,...)`、`parameters=(p1=size1,...)`、`num_features=(...)` 关键字 + do-block
  - 处理：将 do-block 重新包装为带 `NamedTuple` 参数的匿名函数
    - 子表达式参数：`(; f, g)` NamedTuple
    - 数据参数：`(x1, x2, x3)` tuple
  - 输出：`TemplateExpressionSpec{TemplateStructure{(:f,:g),...}}`
  - 内部生成 hash-named 函数（避免命名冲突）

### 3.10 未读但与 NuSR 潜在高度相关的 Julia 文件

| 文件 | 预计作用 | NuSR 相关性 | 优先级 |
|---|---|---|---|
| `Core.jl` | `Dataset`、`Options`、`AbstractExpression`、`AbstractExpressionSpec` 等核心类型定义 | **极高**（理解类型系统） | Phase 1 |
| `ComposableExpression.jl` | ValidVector 定义 + 子表达式包装器 | **极高**（TemplateExpression 基础） | Phase 1 |
| `Mutate.jl` | `next_generation`、`crossover_generation` 主函数 | 高（理解变异分发） | Phase 2 |
| `Population.jl` | `Population` 结构体，`best_of_sample` | 高（理解种群和选择） | Phase 2 |
| `PopMember.jl` | `PopMember` 结构体 | 中 | Phase 2 |
| `Dataset.jl` | `Dataset` 结构体完整定义 | 中（理解数据流） | Phase 2 |
| `ConstantOptimization.jl` | BFGS/NelderMead 常数优化 | 中（理解参数优化） | Phase 3 |
| `DimensionalAnalysis.jl` | 物理量纲约束 | **可能重要** | Phase 4 |
| `CheckConstraints.jl` | 约束检查 | 中 | Phase 4 |
| `Operators.jl` | `OperatorEnum` 定义 | 中 | Phase 2 |
| `Migration.jl` | 种群间迁移 | 低（NuSR 初期单种群） | 按需 |

---

## 4. PySR 与 SymbolicRegression.jl 的连接关系

### 4.1 整体调用链

```
用户代码
  ↓
NuSRRegressor.fit(X, y)                     ← NuSR wrapper（计划中）
  ↓
PySRRegressor.fit(X, y)                     ← pysr/sr.py:2305
  ↓
PySRRegressor._run(X, y)                    ← pysr/sr.py:1905
  ├─ 构建 Julia OperatorEnum                ← jl.SymbolicRegression.OperatorEnum(...)
  ├─ 构建 Julia Options                     ← jl.SymbolicRegression.Options(...)
  │    └─ expression_spec.julia_expression_spec()  ← jl.seval(@template_spec ...)
  ├─ 转换数据：numpy → Julia Array           ← jl_array(X.T), jl_array(y)
  ├─ 调用搜索                               ← SymbolicRegression.equation_search(...)
  │    ↓                                    ← src/SymbolicRegression.jl:469
  │    _equation_search(...)                ← src/SymbolicRegression.jl:582
  │    ├─ _validate_options
  │    ├─ _create_workers
  │    ├─ _initialize_search!
  │    ├─ _warmup_search!
  │    ├─ _main_search_loop!                ← 核心调度（src/SymbolicRegression.jl:883）
  │    │    └─ _dispatch_s_r_cycle          ← 每个 worker 的搜索循环
  │    │         └─ s_r_cycle               ← 模拟退火 + 多次 reg_evol_cycle
  │    │              └─ reg_evol_cycle      ← tournament selection + 变异/交叉
  │    ├─ _tear_down!
  │    ├─ _info_dump
  │    └─ _format_output
  ├─ 保存状态：self.julia_state_ = out["state"]
  └─ 返回
  ↓
PySRRegressor.get_hof()                     ← 从 CSV 读取，添加导出格式
  ↓
用户获取 equations / predict / sympy / latex
```

### 4.2 Python → Julia 参数转换路径

| Python 参数 | 转换方式 | Julia 目标 |
|---|---|---|
| `binary_operators` (list of str) | `jl.seval("safe_log(x) = ...")` 为每个算子在 Julia 中定义函数 → `jl_convert(jl.Vector, ...)` | `OperatorEnum.binops` |
| `unary_operators` (list of str) | 同上 | `OperatorEnum.unaops` |
| `constraints` (dict) | `jl_constraints_dict = jl.Dict(...)` | `Options.constraints` |
| `expression_spec` (`TemplateExpressionSpec`) | `self.expression_spec_.julia_expression_spec()` → `jl.seval(@template_spec ...)` | `Options.expression_spec` |
| `elementwise_loss` (callable) | 包装为 Julia 函数 | `Options.elementwise_loss` |
| `loss_function` (callable) | 包装为 Julia 函数 | `Options.loss_function` |
| `complexity_mapping` (callable) | 包装为 Julia 函数 | `Options.complexity_mapping` |
| `niterations` (int) | 直接传递 | `equation_search(niterations=...)` |
| `maxsize` (int) | 直接传递 | `Options(maxsize=...)` |
| `parsimony` (float) | 直接传递 | `Options(parsimony=...)` |
| `X` (numpy array) | `jl_array(X.T)` — [features, rows] 格式 | `equation_search(X, ...)` |
| `y` (numpy array) | `jl_array(y)` | `equation_search(y, ...)` |
| `variable_names` (list of str) | `jl_array([str(v) for v in ...])` | `equation_search(variable_names=...)` |

### 4.3 Julia → Python 结果返回路径

1. `equation_search` 返回 dict 包含 `"hall_of_fame"`、`"state"` 等
2. Python `_run()` 保存 `self.julia_state_` 和 `self.julia_options_stream_`
3. HoF 同时写入 CSV 文件（`hall_of_fame.csv`）
4. Python `get_hof()` 从 CSV 读取，调用 `add_export_formats()` 生成 `sympy_format`、`lambda_format` 等
5. 对于 TemplateExpressionSpec，`create_exports()` 返回 `CallableJuliaExpression`（因为表达式求值完全在 Julia 中）

### 4.4 Python-Julia 桥接关键文件

| 文件 | 作用 | 是否建议修改 |
|---|---|---|
| `pysr/julia_import.py` | juliacall 初始化 | 否 |
| `pysr/julia_helpers.py` | 数据转换工具 | 否（可复用） |
| `pysr/sr.py:_run()` | 参数转换 + 搜索调用 | **NuSR wrapper 的主要参考** |
| `pysr/expression_specs.py:TemplateExpressionSpec` | 模板 spec 的 Python → Julia 传递 | **NuSR 模板系统的核心** |

### 4.5 NuSR 最可能常用的扩展入口

1. **`PySRRegressor.__init__`** — 覆写默认参数（核物理默认算子、默认 parsimony 等）
2. **`PySRRegressor._run`** — 注入自定义 Options、自定义数据预处理
3. **`TemplateExpressionSpec`** — 定义核物理公式模板
4. **`Options.elementwise_loss` / `Options.loss_function`** — 自定义物理感知损失
5. **`Options.constraints`** — 施加物理约束
6. **`Options.complexity_mapping`** — 核物理算子复杂度加权
7. **`Options.operators`** — 注册核物理专用算子

---

## 5. 与 NuSR 最相关的机制清单

### 5.1 TemplateExpressionSpec（模板表达式）

- **解决的问题**：固定公式的整体结构，只搜索子表达式和参数，大幅缩小搜索空间
- **对 NuSR 的作用**：**极重要**。核物理经验公式通常有已知的函数形式（如液滴模型质量公式的 5 项结构），可以用模板固定外层，让 SR 搜索子项的具体形式
- **相关文件**：
  - Python: `pysr/expression_specs.py:128-322`
  - Julia: `src/TemplateExpression.jl`（全文件，1093行）、`src/TemplateExpressionMacro.jl`（全文件，154行）
- **关键类/函数**：
  - Python: `TemplateExpressionSpec.__init__`、`_template_macro_str`、`julia_expression_spec`
  - Julia: `TemplateStructure`、`TemplateExpression`、`@template_spec`、`eval_tree_array`（TemplateExpression 特化）、`get_contents_for_mutation`、`with_contents_for_mutation`
- **已确认结论**：
  - 可以表达 `(p[1] * x1 * f(x2)) / (g(x3) - 1)` 这种结构
  - 裸变量（如 `x1`）可以直接出现在 `combine` 表达式中的任何位置
  - `f(x2)` 只搜索涉及 `x2` 的子表达式（由 `infer_variable_constraints` 自动推断）
  - 子表达式复杂度之和 = 总复杂度（外层结构不计入）
  - 变异时随机选一个子表达式进行标准变异
- **待确认问题**：
  - `ComposableExpression` 的具体接口（如何包装子表达式、如何传递特征映射）
  - `ParamVector` 的优化机制（BFGS 的具体集成方式）
  - 多参数支持（`parameters={"p": 3}` 时 `p[1]`、`p[2]`、`p[3]` 的索引方式）
- **最小学习路径**：`expression_specs.py:TemplateExpressionSpec` → 测试 notebook → `TemplateExpression.jl:eval_tree_array` → `TemplateExpressionMacro.jl:@template_spec`
- **是否适合 NuSR 第一版使用**：是，这是 NuSR 最核心的机制
- **是否需要修改核心**：不需要，通过 PySR 公开 API 即可使用

### 5.2 自定义损失函数

- **解决的问题**：用户自定义损失计算逻辑，可以引入物理先验（如残差惩罚、边界约束、渐近行为检查）
- **对 NuSR 的作用**：核物理中可能需要加权 MSE（按实验误差加权）、物理合理性惩罚（如越界惩罚）、量纲一致性检查等
- **相关文件**：
  - Julia: `src/LossFunctions.jl:139-159`（`eval_loss` 的 3 条路径）
  - Python: `pysr/sr.py:_run()`（`elementwise_loss` / `loss_function` 参数传递）
- **关键函数**：
  - `eval_loss(tree, dataset, options)` — 完全自定义
  - `eval_loss(expression, dataset, options)` — 表达式级自定义
  - `loss_to_cost(loss, normalization, size, parsimony)` — loss → cost 转化
- **已确认结论**：
  - 支持 `elementwise_loss`（逐元素损失，如自定义 MSE 变体）
  - 支持 `loss_function`（完全接管损失计算）
  - `loss_to_cost` 中 `normalization` = baseline_loss（常数预测均值的损失）
- **待确认问题**：
  - 自定义 `loss_function` 的签名约定（tree, dataset, options → scalar）
  - 在 TemplateExpression 下自定义 loss 的行为
  - 权重（weights）在自定义 loss 中的传递方式
- **最小学习路径**：`LossFunctions.jl:eval_loss` → `LossFunctions.jl:_eval_loss` → `LossFunctions.jl:loss_to_cost`
- **是否适合 NuSR 第一版使用**：是，可以通过 PySR 公开 API 使用
- **是否需要修改核心**：不需要

### 5.3 自定义算子（Custom Operators）

- **解决的问题**：注册领域专用数学运算符（如核物理中的 `shell_correction`、`pairing_term`）
- **对 NuSR 的作用**：核物理中有大量专有函数形式（壳修正、对力项、库仑势等），需要作为算子注册
- **相关文件**：
  - Python: `pysr/sr.py:_run()`（算子字符串 → Julia 函数定义 → `OperatorEnum`）
  - Julia: `Configure.jl:assert_operators_well_defined`（算子合法性检查）
- **关键机制**：PySR 通过 `jl.seval("my_op(x) = ...")` 在 Julia 侧定义函数，然后放入 `OperatorEnum`
- **已确认结论**：
  - 支持 Python lambda/函数作为自定义算子
  - 匿名函数**不能**作为算子（`Configure.jl` 中有显式检查）
  - 算子必须返回与输入相同类型的输出
  - 二元算子和一元算子不能有交集
- **待确认问题**：
  - 自定义算子的 `bumper`（无感缓冲区）行为
  - 算子对 `ValidVector` 的支持（在 TemplateExpression 中）
  - `OperatorEnum` 结构体的完整字段
- **最小学习路径**：`PySRRegressor.__init__` 中 `binary_operators`/`unary_operators` 参数文档 → `Configure.jl:assert_operators_well_defined`
- **是否适合 NuSR 第一版使用**：是
- **是否需要修改核心**：不需要

### 5.4 约束（Constraints）

- **解决的问题**：限制表达式中算子的嵌套关系（如 `sin` 里面不能嵌套 `sin`）
- **对 NuSR 的作用**：核物理中可能需要约束某些算子的嵌套（如指数不能嵌套指数、物理项与纯数学项分离）
- **相关文件**：Python `pysr/sr.py:_run()`（`constraints` dict → `jl.Dict`）；Julia 端 `CheckConstraints.jl`（未读）
- **待确认问题**：
  - 约束 dict 的确切格式
  - 嵌套约束 vs 类型约束的区别
  - 在 TemplateExpression 下约束的行为
- **最小学习路径**：PySR 文档 `constraints` 参数 → `CheckConstraints.jl`
- **是否适合 NuSR 第一版使用**：是（如果 PySR API 足够）
- **是否需要修改核心**：待确认

### 5.5 复杂度控制（Complexity Control）

- **解决的问题**：控制表达式的复杂度惩罚，使搜索结果倾向于简洁公式
- **对 NuSR 的作用**：核物理中不同算子的物理复杂度不同（如壳修正比简单多项式复杂得多），需要差异化加权
- **相关文件**：
  - Julia: `src/OptionsStruct.jl`（`ComplexityMapping` 结构体）、`src/Complexity.jl`（`compute_complexity`）、`src/AdaptiveParsimony.jl`
- **关键机制**：
  - `ComplexityMapping`：`op_complexities`（各算子权重）、`variable_complexity`（变量权重）、`constant_complexity`（常数权重）
  - `compute_complexity`：3 条路径（自定义函数 / `ComplexityMapping` / `count_nodes`）
  - TemplateExpression 的复杂度 = 各子表达式复杂度之和
- **已确认结论**：`ComplexityMapping` 支持算子级/变量级/常数级的差异化复杂度
- **待确认问题**：
  - `ComplexityMapping` 的完整构造函数签名
  - 与 TemplateExpression 的结合行为
- **最小学习路径**：`OptionsStruct.jl` → `Complexity.jl`
- **是否适合 NuSR 第一版使用**：是
- **是否需要修改核心**：不需要

### 5.6 模型选择（Model Selection）

- **解决的问题**：从 Pareto 前沿中选择"最好"的方程
- **对 NuSR 的作用**：核物理经验公式需要在精度和简洁性之间权衡
- **相关文件**：
  - Python: `pysr/sr.py:get_best`（line 1439-1485）
  - Julia: `src/HallOfFame.jl:calculate_pareto_frontier`、`format_hall_of_fame`
- **关键机制**：
  - `model_selection="best"`：选 score 最高的
  - `model_selection="accuracy"`：选 loss 最低的
  - score 计算考虑了 loss 比率和复杂度差异
- **已确认结论**：Pareto 前沿筛选条件是 loss 必须低于所有更低复杂度方程
- **待确认问题**：score 计算的精确公式
- **是否适合 NuSR 第一版使用**：是，不需要修改

### 5.7 公式导出（Equation Export）

- **解决的问题**：将 Julia 表达式树转换为 Python 可用的公式（SymPy、NumPy、JAX、Torch、LaTeX）
- **对 NuSR 的作用**：NuSR 需要输出可读的物理公式（LaTeX）和可计算的形式（NumPy/SymPy）
- **相关文件**：
  - Python: `pysr/export.py`、`pysr/export_sympy.py`、`pysr/export_numpy.py`、`pysr/export_latex.py`
  - `pysr/expression_specs.py:CallableJuliaExpression`
- **已确认结论**：
  - TemplateExpressionSpec **不支持** `sympy()` 和 `latex()` 导出
  - 原因是表达式求值完全在 Julia 中进行（`evaluates_in_julia=True`）
  - TemplateExpressionSpec 的 `create_exports()` 返回 `CallableJuliaExpression`（只能用 Julia 后端预测）
- **待确认问题**：
  - 是否有 workaround 让 TemplateExpression 也能导出 LaTeX
  - `export_sympy.py` 中 `pysr2sympy` 的完整实现
  - `export_latex.py` 的实现
- **最小学习路径**：`export.py:add_export_formats` → `export_sympy.py:pysr2sympy`
- **是否适合 NuSR 第一版使用**：TemplateExpression 导出是 NuSR 的**关键痛点和待解决问题**
- **是否需要修改核心**：可能需要扩展导出系统以支持 TemplateExpression

### 5.8 并行与批处理

- **解决的问题**：多 populations 并行搜索、大数据批处理
- **对 NuSR 的作用**：核物理数据集通常不大（<10,000 点），并行主要用于多 populations 探索
- **相关文件**：Python `pysr/sr.py:_run()`（`parallelism`、`procs`、`numprocs`）；Julia `Configure.jl:configure_workers`
- **已确认结论**：
  - 支持 `:multithreading`（单机多线程）和 `:multiprocessing`（分布式）
  - `deterministic=True` 时只能用 `:serial` 模式
  - `@sr_spawner` 宏处理并行分派（未读）
- **是否适合 NuSR 第一版使用**：是，学校服务器上使用 `:multiprocessing`

### 5.9 物理量纲分析（Dimensional Analysis）

- **状态**：`src/DimensionalAnalysis.jl` **未读**，但已知存在
- **对 NuSR 的作用**：**可能极其重要**。核物理公式有明确的量纲约束（能量、截面、质量等）
- **待确认问题**：全部待确认
- **最小学习路径**：`DimensionalAnalysis.jl`（未读）
- **是否适合 NuSR 第一版使用**：待确认（取决于该模块的成熟度）

### 5.10 自定义复杂度映射（ComplexityMapping）

- **参考 5.5**，此处补充 Python 侧使用方式
- 通过 PySR 的 `complexity_mapping` 参数传入 Python callable
- Python callable 被 `move_functions_to_workers` 复制到所有 worker

---

## 6. TemplateExpressionSpec 专题总结

### 6.1 定义位置

- **Python 定义**：`pysr/expression_specs.py:128-322`（`TemplateExpressionSpec` 类）
- **Julia 定义**：
  - 结构体：`src/TemplateExpression.jl:106-149`（`TemplateStructure`）、`:293-321`（`TemplateExpression`）
  - 宏：`src/TemplateExpressionMacro.jl:34-151`（`@template_spec`）

### 6.2 主要参数

| 参数 | 类型 | 含义 | 示例 |
|---|---|---|---|
| `combine` | str | 外层组合表达式，支持 `+`、`-`、`*`、`/`、`()` 和子表达式调用 | `"(p[1] * x1 * f(x2)) / (g(x3) - 1)"` |
| `expressions` | list of str | 需要搜索的子表达式名 | `["f", "g"]` |
| `variable_names` | list of str | 数据列名 | `["x1", "x2", "x3"]` |
| `parameters` | dict (str→int) 或 None | 可优化参数及其长度 | `{"p": 1}` |

### 6.3 `combine` 的意义

- 固定公式的**外层结构**（整体运算关系）
- 其中可以出现：
  - 子表达式调用：`f(x2)`、`g(x3)`（这些会被搜索）
  - 裸数据列：`x1`（直接参与外层运算，不进入子表达式搜索）
  - 可优化参数：`p[1]`（在搜索中被 BFGS/NelderMead 优化）
  - 算术运算符：`+`、`-`、`*`、`/`
  - 常数：`1`、`2` 等
- `combine` 中的子表达式调用参数决定了该子表达式可以使用哪些特征（由 `infer_variable_constraints` 自动推断）
  - 例如：`f(x2)` → f 只能使用 x2（即第 2 列）；`f(x1, x2)` → f 可以使用 x1 和 x2

### 6.4 `expressions` 的意义

- 声明需要被搜索的子表达式的符号名
- 必须与 `combine` 中出现的函数名一致
- 每个 expression 在搜索中独立进化（独立的表达式树）
- 搜索到的子表达式会被插回 `combine` 中参与整体求值

### 6.5 `variable_names` 的意义

- 数据列的名称列表，与 `X` 矩阵的列一一对应
- 这些名称在 `combine` 字符串中作为变量名使用
- 也传递给 Julia 侧用于 CSV 输出中的变量可读性

### 6.6 `parameters` 的意义

- 声明模板中的可优化参数
- `{"p": 1}` 表示一个名为 `p` 的 1 维参数向量，通过 `p[1]` 访问
- 参数值在搜索过程中被 Julia 侧的常数优化器（BFGS/NelderMead）优化
- `mutate_constant` 有两种变异方式：变异子表达式内部常数 或 变异 `ParamVector` 中的参数值

### 6.7 如何被 PySR 接收

1. 用户创建 `TemplateExpressionSpec(combine=..., expressions=..., variable_names=..., parameters=...)`
2. `PySRRegressor.__init__` 将 spec 存储在 `self.expression_spec_`
3. `PySRRegressor._run()` 调用 `self.expression_spec_.julia_expression_spec()`
4. `julia_expression_spec()` 调用 `_template_macro_str()` 生成 Julia 代码字符串
5. 通过 `jl.seval(code_string)` 在 Julia 中执行 `@template_spec` 宏
6. 返回 Julia 端的 `TemplateExpressionSpec{TemplateStructure{(:f,:g),...}}`
7. 将该对象传入 `SymbolicRegression.Options(expression_spec=...)`

### 6.8 如何传递给 SymbolicRegression.jl

- 通过 `Options.expression_spec` 字段
- Julia 侧在 `_main_search_loop!` 中创建个体时使用该 spec 生成表达式树
- 搜索过程中，`eval_tree_array` 根据 expression_spec 的类型分发到 TemplateExpression 专用实现

### 6.9 Julia 侧如何处理

1. `@template_spec` 宏将 do-block 包装为 `TemplateStructure`
2. `infer_variable_constraints` 通过 `ArgumentRecorder` 代理推断子表达式特征数
3. 搜索初始化时，每个子表达式创建独立的 `ComposableExpression`（标准表达式树）
4. `eval_tree_array` 调用 `combine`：
   - 先求值各子表达式得到 ValidVector
   - 再用 `map(x -> ValidVector(copy(x), true), eachrow(cX))` 构造裸数据的 ValidVector
   - 将子表达式结果、参数、裸数据传入 `combine` 函数
   - `combine` 返回的 ValidVector 的 `.x` 和 `.valid` 即为最终预测值和有效性标记
5. 变异时随机选一个子表达式，对其做标准变异操作，替换回去

### 6.10 能限制表达式结构到什么程度

- **完全固定**外层运算结构（如 `(a*X)/(Y-1)` 的形式不可改变）
- 子表达式的**参数范围**被限定（`f(x2)` 只能使用第 2 列数据）
- 子表达式的**内部结构**完全自由（任意算子组合）
- 参数化常数可被优化但**不参与结构搜索**

### 6.11 能否表示 `(p[1] * x1 * f(x2)) / (g(x3) - 1)`

**已确认可以**。测试 notebook 验证了该模板的有效性。

对应的 Python 代码：
```python
TemplateExpressionSpec(
    combine="(p[1] * x1 * f(x2)) / (g(x3) - 1)",
    expressions=["f", "g"],
    variable_names=["x1", "x2", "x3"],
    parameters={"p": 1},
)
```
测试结果（使用 `x1, x2, x3` 为 `sin(x2)+x2^2` 和 `x3^2+2` 生成的人工数据）：
- RMSE: 2.73e-07, MAE: 1.45e-07, R²: 1.000000
- 搜索到的 f 接近 `-(square(x2) + sin(x2))`（负号来自整体分式结构）
- 搜索到的 g 应接近 `x3^2 + 2`

### 6.12 使用它构建 NuSR 核物理公式模板的潜力

**潜力极高**。具体应用场景：

1. **液滴模型质量公式**（Bethe-Weizsäcker formula）：
   ```python
   TemplateExpressionSpec(
       combine="p[1] + p[2]*A + p[3]*A^(2/3) + f(N,Z) + g(N,Z)/A^(1/2) + h(N,Z)/A^(3/4)",
       expressions=["f", "g", "h"],
       variable_names=["N", "Z", "A"],
       parameters={"p": 3},
   )
   ```
   固定体积项+表面项+库仑项，让 SR 搜索壳修正和对力项。

2. **衰变能公式**：
   ```python
   TemplateExpressionSpec(
       combine="f(Z,N) * g(Q_value) + h(shell_correction)",
       expressions=["f", "g", "h"],
       ...
   )
   ```

3. **截面公式**：
   ```python
   TemplateExpressionSpec(
       combine="p[1] * (E^p[2]) * f(Z_target) * g(A_projectile) / (E - h(Z_target, A_target))",
       expressions=["f", "g", "h"],
       parameters={"p": 2},
       ...
   )
   ```

### 6.13 目前仍需要验证的点

1. **TemplateExpression 的 LaTeX/SymPy 导出**：目前不支持，NuSR 需要找到 workaround
2. **多参数 `parameters={"p": 3}` 时参数在 `combine` 中的访问方式**：`p[1]`、`p[2]`、`p[3]`？还是 `p1`、`p2`、`p3`？
3. **TemplateExpression 与自定义 loss 的交互**：`eval_loss` 在 TemplateExpression 下的行为
4. **TemplateExpression 与 constraints 的交互**：子表达式内部是否受全局 constraints 约束
5. **`ComposableExpression` 的具体机制**：子表达式如何封装特征映射
6. **TemplateExpression 的 `node_type`**：是否支持 `Float32`/`Float64` 切换
7. **多个 datasets 时的 TemplateExpression**：多输出回归是否支持模板

### 6.14 下一步应该阅读的具体文件和关键词

- **文件**：`src/ComposableExpression.jl`（ValidVector 定义）、`src/Core.jl`（`AbstractExpressionSpec` Julia 端定义）
- **关键词**：`ComposableExpression`、`ValidVector`、`ParamVector`、`get_contents`、`set_contents`、`ArgumentRecorder`、`infer_variable_constraints`、`evaluates_in_julia`

---

## 7. NuSR 第一版开发建议

### 7.1 强烈建议先用 wrapper 层实现的功能

| 功能 | 目标 | 推荐入口 | 是否第一版做 | 风险 |
|---|---|---|---|---|
| **NuSRRegressor** | 包装 `PySRRegressor`，提供核物理专用默认参数和简化 API | 新建 `nusr/regressor.py` | **是** | 低（纯 Python wrapper） |
| **核物理数据加载器** | 从 AME/ENSDF/EXFOR 格式加载数据，统一转换为 numpy | 新建 `nusr/data/` | **是** | 低（独立模块） |
| **核物理公式模板库** | 预置液滴模型、壳修正、对力项等常用模板 | 新建 `nusr/templates/`，使用 `TemplateExpressionSpec` | **是** | 低（依赖 PySR API） |
| **常用核物理算子库** | `shell_correction`、`pairing_term`、`coulomb_energy` 等 | 新建 `nusr/operators.py` | **是** | 中（算子需在 Julia 侧定义） |
| **公式解释器** | 将搜索到的公式翻译为物理语言（识别已知物理项） | 新建 `nusr/interpreter.py` | 是（但复杂度可控） | 中 |
| **结果导出** | 统一管理 sympy/latex/callable 导出 | 包装 PySR 导出功能 | **是** | 低 |
| **实验 Notebooks** | 每个核物理数据集的探索性分析 notebook | `Experiment/` 目录下 | **是** | 无 |

### 7.2 可以通过 PySR 公开 API 实现的功能

| 功能 | PySR API | 是否第一版做 | 备注 |
|---|---|---|---|
| 模板表达式 | `TemplateExpressionSpec` | **是** | NuSR 最核心功能 |
| 自定义损失 | `elementwise_loss` / `loss_function` | **是** | 物理感知损失 |
| 算子约束 | `constraints` | 是（如果需要） | 限制算子嵌套 |
| 自定义算子 | `binary_operators` / `unary_operators` | **是** | 核物理专用算子 |
| 复杂度控制 | `complexity_mapping` / `parsimony` | 是 | 物理项复杂度加权 |
| 变量命名 | `variable_names` | **是** | 核物理意义命名 |
| 模型选择 | `model_selection` | **是** | "best" 通常适合经验公式 |
| 并行控制 | `parallelism` / `procs` | **是** | 学校服务器配置 |

### 7.3 可能需要修改 PySR / SymbolicRegression.jl 核心的功能

| 功能 | 目标 | 推荐入口文件 | 是否第一版做 | 风险 | 替代方案 |
|---|---|---|---|---|---|
| TemplateExpression 的 LaTeX/SymPy 导出 | 让模板表达式也能导出为可读公式 | `pysr/export.py`、`pysr/export_sympy.py` | **否（但需要 plan）** | 中 | 手动解析 `julia_expression` 字符串，构建 SymPy 表达式 |
| 物理维度强约束搜索 | 在搜索过程中过滤违反物理量纲的表达式 | `src/DimensionalAnalysis.jl`（未读） | **否** | 高（需理解 Julia 侧搜索循环） | 用 TemplateExpression 固定外层结构，在 loss 中加量纲惩罚 |
| 物理项级别结构约束 | 约束表达式中必须包含某些物理项 | `src/TemplateExpression.jl` + `src/Mutate.jl` | **否** | 高 | 用 TemplateExpression 固定外层，子表达式内部不限制 |
| 特殊种群初始化 | 用已知物理公式初始化种群 | `PySRRegressor.__init__` 的 `guesses` 参数？ | **待确认** | 待确认 | 查看 PySR 是否支持 `warm_start` |
| 核物理专用变异算子 | 插入/删除已知物理结构的变异 | `src/Mutate.jl` | **否** | 极高 | 先用 TemplateExpression + 自定义算子替代 |

---

## 8. 面向未来 CC 的操作规则

以下规则面向 compact 后重新加载本文件的 Claude Code（以及未来的 AI 助手）：

1. **先读取本文件**：在任何 PySR / SymbolicRegression.jl 相关操作之前，完整阅读本文件（特别是第 9 节路由表）
2. **不要通读整个仓库**：PySR + SR.jl 合计数万行代码，通读浪费 context。根据用户的具体问题，用第 9 节路由表定位到 2-3 个关键文件
3. **根据用户目标选择相关文件**：NuSR 开发的不同阶段需要不同的源码理解深度。参考第 10 节的阶段阅读计划
4. **优先使用 grep/rg 定位**：对于代码中的具体符号/函数/类，用 `rg` 搜索而非打开文件浏览
5. **每次只读相关片段**：用 `Read` 工具的 offset/limit 参数只读需要的行，不要一次读整个大文件
6. **读完要更新本文件或新建专题 md**：每次深入阅读新源码后，应更新本文件（追加新条目到对应表格、修正待确认、补充新发现）。如果某个主题足够复杂，新建专题 md（如 `01_TemplateExpression_Deep_Dive.md`）
7. **修改代码前必须先写设计说明**：在修改 PySR / SR.jl 源码前（NuSR 需要 patch 或 fork 时），先写设计文档说明修改理由、影响范围、回退方案
8. **优先 wrapper，不优先改核心**：NuSR 的开发哲学是"能用包装器解决的就不改上游"。只有确认 PySR 公开 API 无法满足需求时才考虑修改核心
9. **不确定就标注待确认**：永远不要假装知道没看过的文件内容。不确定的地方标注"待确认"并给出验证路径（搜索关键词 + 建议文件）
10. **永远把结论写入 docs/**：每个 session 的源码理解结论应持久化到 `/home/taomingyu/Reference/PySR/` 或 `/home/taomingyu/Development/NuSR/docs/` 中

---

## 9. 未来问题到源码文件的路由表

| 如果我的问题是... | 优先阅读文件 | 次要阅读文件 | 可能修改文件 | 不建议一开始看的文件 | 备注 |
|---|---|---|---|---|---|
| 我想理解 `PySRRegressor.fit` 调用链 | `pysr/sr.py:2305-2473`（fit）、`pysr/sr.py:1905-2303`（_run） | `pysr/julia_import.py`、`pysr/julia_helpers.py` | NuSR 新建 `nusr/regressor.py` | `pysr/export*.py`、Julia 全部 | 先读 fit → 再读 _run → 理解参数转换 |
| 我想加核物理公式模板 | `pysr/expression_specs.py:128-322`（TemplateExpressionSpec） | `src/TemplateExpression.jl:684-723`（eval_tree_array） | NuSR 新建 `nusr/templates/` | `src/Mutate.jl`、`src/Population.jl` | 先运行测试 notebook 验证理解 |
| 我想加自定义 loss | `src/LossFunctions.jl:139-208`（eval_loss + eval_cost） | `pysr/sr.py:_run()`（loss 参数传递部分） | `pysr/sr.py:_run()`（如果需要新的 loss 入口） | `src/Mutate.jl`、`src/RegularizedEvolution.jl` | 3 条路径选最适合的 |
| 我想加自定义 operator | `src/Configure.jl:5-58`（assert_operators_well_defined） | `pysr/sr.py:_run()`（算子定义 + OperatorEnum 构造） | 新建 `nusr/operators.py` | `src/Mutate.jl` | 算子必须是命名函数，返回同类型 |
| 我想限制公式复杂度 | `src/Complexity.jl`（全文件） | `src/OptionsStruct.jl`（ComplexityMapping） | `pysr/sr.py:_run()`（传入 complexity_mapping） | `src/TemplateExpression.jl` | 复杂度控制有 3 条路径 |
| 我想限制公式结构 | `pysr/expression_specs.py:128-322`（TemplateExpressionSpec） | `src/TemplateExpression.jl`（全文件） | 新建 `nusr/templates/` | `src/RegularizedEvolution.jl` | TemplateExpression 是最佳方案 |
| 我想让公式满足物理维度 | `src/DimensionalAnalysis.jl`（**未读**） | `src/CheckConstraints.jl`（未读） | 待确认 | 待确认 | 此模块状态未知 |
| 我想让搜索偏向某些物理项 | `src/Complexity.jl` + `src/LossFunctions.jl` | `src/AdaptiveParsimony.jl` | `pysr/sr.py:_run()`（自定义 loss 入口） | `src/Mutate.jl` | 通过 loss 加权 + 复杂度差异化实现 |
| 我想导出 LaTeX / SymPy | `pysr/export.py`、`pysr/export_sympy.py` | `pysr/export_latex.py`（未读） | 同上文件（TemplateExpression 导出是痛点） | Julia 全部 | 注意：TemplateExpression 不支持导出 |
| 我想改搜索算法 | `src/RegularizedEvolution.jl` | `src/SingleIteration.jl`、`src/SymbolicRegression.jl:_main_search_loop!` | 同上文件 | Python 全部 | **高风险**，建议先用 PySR API |
| 我想改 mutation | `src/MutationFunctions.jl` | `src/Mutate.jl`（未读）、`src/TemplateExpression.jl:797-821`（模板变异） | 同上文件 | Python 全部 | **高风险** |
| 我想理解 Julia Options | `src/Core.jl`（未读，Options 定义可能在此） | `src/OptionsStruct.jl` | 不需要修改 | `src/Mutate.jl` | Options 是搜索的核心配置 |
| 我想调试 Python-Julia 接口 | `pysr/julia_import.py` | `pysr/julia_helpers.py`、`pysr/sr.py:_run()` | 同上（如果发现 bug） | Julia 全部 | 通常问题在参数转换 |
| 我想写 NuSRRegressor wrapper | `pysr/sr.py:__init__` + `pysr/sr.py:fit` + `pysr/sr.py:_run` | `pysr/expression_specs.py` | 新建 `nusr/regressor.py` | Julia 全部 | 纯 Python wrapper，不改上游 |
| 我想写测试 | PySR 现有 tests（未读，路径待确认） | 新建 `nusr/tests/` | 新建测试文件 | Julia 全部 | 参考 PySR 的测试结构 |

---

## 10. 推荐后续阅读顺序

### Phase 1：只读结构地图（当前阶段，已完成 80%）

- **目标**：理解 PySR + SR.jl 的宏观结构，能回答"功能 X 在哪个文件"
- **要看的文件**：本 checkpoint（不要读新源码）
- **不看的文件**：所有其他文件
- **要回答的问题**：
  - PySR 有哪些模块？每个模块的职责是什么？
  - SR.jl 有哪些模块？每个模块的职责是什么？
  - Python-Julia 的交互机制是什么？
- **要生成的 md 文档**：本文件（已生成）

### Phase 2：PySRRegressor.fit 完整调用链（必要时）

- **目标**：精读 `fit()` → `_run()` 的每一行，理解所有参数如何转换和传递
- **要看的文件**：
  - `pysr/sr.py:2305-2473`（fit，约 170 行）
  - `pysr/sr.py:1905-2303`（_run，约 400 行）
  - `pysr/julia_helpers.py`（75 行）
- **不看的文件**：Julia 全部、export 全部、feature_selection、denoising
- **要回答的问题**：
  - `_run` 中每个代码块的作用
  - 自定义 loss / operator / complexity_mapping 的精确传递路径
  - 如何在不修改 `_run` 的情况下注入 NuSR 自定义行为
- **要生成的 md 文档**：`01_PySRRegressor_Run_Deep_Dive.md`

### Phase 3：TemplateExpressionSpec 完整机制（用于 NuSR 模板）

- **目标**：完全理解 TemplateExpressionSpec 从 Python 到 Julia 的完整生命周期
- **要看的文件**：
  - Python: `pysr/expression_specs.py:128-322`（已读）
  - Julia: `src/TemplateExpression.jl`（已读）
  - Julia: `src/TemplateExpressionMacro.jl`（已读）
  - Julia: `src/ComposableExpression.jl`（**未读，需补**）
- **不看的文件**：export 全部、其他 Julia 模块
- **要回答的问题**：
  - `ComposableExpression` 的接口（特征映射、ValidVector 传播）
  - TemplateExpression 下自定义 loss 的行为
  - TemplateExpression 下 constraints 的行为
  - 如何导出 TemplateExpression 为 LaTeX
- **要生成的 md 文档**：`02_TemplateExpression_Complete_Guide.md`

### Phase 4：Custom Loss / Operators / Constraints（用于 NuSR 物理注入）

- **目标**：理解如何通过 PySR API 注入核物理先验
- **要看的文件**：
  - `src/LossFunctions.jl`（已读）
  - `src/CheckConstraints.jl`（未读）
  - `src/Configure.jl`（已读，move_functions_to_workers 部分）
- **不看的文件**：`Mutate.jl`、`Population.jl`、`PopMember.jl`
- **要回答的问题**：
  - 自定义 loss 的完整签名和支持的操作
  - constraints dict 的确切格式
  - 自定义算子在 TemplateExpression 中的行为
- **要生成的 md 文档**：`03_Custom_Loss_Operators_Constraints.md`

### Phase 5：NuSR Wrapper 实现（开始写代码）

- **目标**：实现 `NuSRRegressor` 的第一版
- **要看的文件**：Phase 2 的全部 + `pysr/export.py`
- **不看的文件**：Julia 全部（除非 debug）
- **要回答的问题**：
  - NuSRRegressor 的 API 设计
  - 默认参数的核物理最优值
  - 模板库的组织方式
- **要生成的 md 文档**：`04_NuSR_Wrapper_Design.md` + 代码

### Phase 6：评估是否需要修改 SR.jl 核心（远期）

- **目标**：判断 NuSR 是否真的需要 fork SR.jl
- **要看的文件**：按需（取决于前 5 个 Phase 的发现）
- **要回答的问题**：
  - PySR 公开 API 的边界在哪里
  - 哪些核物理需求无法通过公开 API 满足
  - fork 的维护成本评估
- **要生成的 md 文档**：`05_Fork_vs_Wrapper_Decision.md`

---

## 11. 已确认结论

以下结论均来自已读取的源码（Python 端或 Julia 端），有明确的文件/行号依据（但此处从略，详细见各节）：

1. **PySR 是 juliacall（PythonCall.jl）的同进程嵌入架构**，不是子进程或 HTTP RPC。Python 直接持有 Julia 对象引用。来源：`pysr/julia_import.py`。

2. **`.fit()` 的核心链路**：`fit()` → `_run()` → 构建 Julia `Options` → 转换数据 → `SymbolicRegression.equation_search()` → 保存状态 → `get_hof()` 从 CSV 读取。来源：`pysr/sr.py`。

3. **进化搜索的层次结构**：`niterations` × `populations` 个 workers 并行 → 每个 worker 执行 `s_r_cycle` → `s_r_cycle` 包含多次 `reg_evol_cycle`（模拟退火）+ 简化优化。来源：`src/SymbolicRegression.jl`、`src/SingleIteration.jl`、`src/RegularizedEvolution.jl`。

4. **Tournament selection + 变异/交叉**：`reg_evol_cycle` 用 `best_of_sample` 选亲本，~97.4% 概率变异（`next_generation`），~2.6% 概率交叉（`crossover_generation`）。子代替换最老个体。来源：`src/RegularizedEvolution.jl`。

5. **Hall of Fame 按复杂度索引**：`members[c]` 存储复杂度为 c 的最佳个体。Pareto 前沿筛选条件：loss 必须低于所有更低复杂度方程的 loss。来源：`src/HallOfFame.jl`。

6. **Loss → Cost 转化**：`cost = loss / baseline + size * parsimony + frequency_adaptation`。baseline 通过常数预测均值得到。来源：`src/LossFunctions.jl`。

7. **`eval_loss` 有 3 条路径**：自定义 `loss_function` → 自定义 `loss_function_expression` → 默认 `_eval_loss`。这是 NuSR 自定义物理损失的核心扩展点。来源：`src/LossFunctions.jl`。

8. **TemplateExpressionSpec 可以表达 `(p[1] * x1 * f(x2)) / (g(x3) - 1)` 这种结构**。裸变量可以直接出现在 `combine` 中参与外层运算。来源：`pysr/expression_specs.py` + 测试 notebook 验证。

9. **Python → Julia 的模板传递路径**：`TemplateExpressionSpec` Python 对象 → `_template_macro_str()` 生成 Julia 代码 → `jl.seval()` 执行 `@template_spec` 宏 → 生成 Julia 端 `TemplateExpressionSpec{TemplateStructure{...}}`。来源：`pysr/expression_specs.py` + `src/TemplateExpressionMacro.jl`。

10. **特征约束自动推断**：`infer_variable_constraints` 通过 `ArgumentRecorder` 代理调用 `combine`，自动判断每个子表达式允许的特征数。来源：`src/TemplateExpression.jl:213-241`。

11. **变异机制（TemplateExpression）**：覆写 `get_contents_for_mutation` 和 `with_contents_for_mutation`。每次变异随机选一个子表达式，执行标准变异操作后替换回去。来源：`src/TemplateExpression.jl:797-821`。

12. **常数/参数变异**：`mutate_constant` 有两条路径 — 变异子表达式内部常数节点 或 变异 `ParamVector` 中的参数值。来源：`src/TemplateExpression.jl:869-900`。

13. **复杂度计算**：`compute_complexity` 有 3 条路径 — 自定义函数 / `ComplexityMapping` 加权 / 默认 `count_nodes`。TemplateExpression 的复杂度 = 各子表达式复杂度之和。来源：`src/Complexity.jl` + `src/TemplateExpression.jl:552-561`。

14. **TemplateExpression 的局限性**：`evaluates_in_julia=True`，不支持 `sympy()` / `latex()` / `torch` / `jax` 导出。`create_exports()` 返回 `CallableJuliaExpression`。来源：`pysr/expression_specs.py:309`。

15. **自定义函数必须在多进程间复制**：`move_functions_to_workers` 将用户自定义的算子、损失函数、复杂度映射等分发到所有 workers。来源：`src/Configure.jl:128-214`。

16. **匿名函数不能作为算子**：`Configure.jl` 中有显式检查，算子必须是命名函数。来源：`src/Configure.jl:76-83`。

17. **算子必须返回与输入同类型的输出**：如果算子返回了不同类型，会抛出错误。来源：`src/Configure.jl:20-29`。

18. **`ComplexityMapping` 支持算子级差异化权重**：可以为不同算子设置不同复杂度。来源：`src/OptionsStruct.jl`。

19. **数据格式**：X 是 `[features, rows]` 格式的 Julia 矩阵，y 是向量。Python 侧通过 `jl_array(X.T)` 转换（numpy 默认是 `[rows, features]`）。来源：`pysr/sr.py:_run()` + `pysr/julia_helpers.py`。

20. **搜索结果通过 CSV 持久化**：`save_to_file` 写入 `hall_of_fame.csv`（含 `Complexity,Loss,Equation` 列），同时写 `.bak` 备份。Python 侧 `get_hof()` 从 CSV 读取。来源：`src/SearchUtils.jl:621-665` + `pysr/sr.py:2786-2829`。

---

## 12. 待确认问题

以下是当前已识别但未确认的问题。按重要性和验证难度排序：

| # | 问题 | 为什么重要 | 建议搜索关键词 | 建议阅读文件 |
|---|---|---|---|---|
| 1 | `Options` 结构体完整字段定义在哪个文件？ | 理解搜索的全部可配置参数 | `struct Options` | `src/Core.jl`（最可能的定义位置） |
| 2 | `ComposableExpression` 的完整接口 | TemplateExpression 的基础组件 | `ComposableExpression`、`ValidVector` | `src/ComposableExpression.jl` |
| 3 | TemplateExpression 能否导出 LaTeX？ | NuSR 的核心需求 | `latex`、`TemplateExpression` | `pysr/export_latex.py`、`pysr/export.py` |
| 4 | TemplateExpression + 自定义 loss 的交互 | 物理感知损失需要与模板协同 | `eval_loss`、`TemplateExpression` | `src/LossFunctions.jl` + `src/TemplateExpression.jl` |
| 5 | constraints dict 的确切格式 | 嵌套约束对核物理有意义 | `constraints`、`nested` | `src/CheckConstraints.jl`、PySR 文档 |
| 6 | `DimensionalAnalysis.jl` 的内容和可用性 | 核物理量纲约束可能极其重要 | `dimensional`、`units` | `src/DimensionalAnalysis.jl` |
| 7 | `ParamVector` 的优化机制（BFGS 集成方式） | 参数优化影响模板中 p[1] 的精度 | `ParamVector`、`BFGS`、`NelderMead` | `src/ConstantOptimization.jl` |
| 8 | `@sr_spawner` 宏的并行调度机制 | 理解 multiprocessing vs multithreading | `sr_spawner` | `src/SearchUtils.jl` |
| 9 | `next_generation` / `crossover_generation` 的具体实现 | 理解变异/交叉的分发逻辑 | `next_generation`、`crossover_generation` | `src/Mutate.jl` |
| 10 | `Core.jl` 中 `AbstractExpression` 的定义 | Julia 端类型系统的根 | `AbstractExpression`、`AbstractExpressionSpec` | `src/Core.jl` |
| 11 | 多输出回归（multitarget）的实现 | NuSR 远期可能需要多目标 | `multi`、`v_dim_out` | `src/SymbolicRegression.jl` |
| 12 | PySR 是否支持 warm start / initial guesses | 用已知物理公式初始化种群 | `guesses`、`warm_start`、`initial_population` | `pysr/sr.py:__init__`、`src/SearchUtils.jl:parse_guesses` |
| 13 | `export_sympy.py` 中 `pysr2sympy` 的完整实现 | 理解非模板表达式的导出机制 | `pysr2sympy` | `pysr/export_sympy.py`（需补全阅读） |
| 14 | `export_latex.py` 的实现 | NuSR 的 LaTeX 导出 | `sympy2latex` | `pysr/export_latex.py` |
| 15 | `ValidVector` 支持的完整运算集合 | TemplateExpression 中 `combine` 的表达能力 | `ValidVector` | `src/ComposableExpression.jl` |
| 16 | `Population` 结构体 + `best_of_sample` 实现 | 理解 tournament selection | `best_of_sample`、`Population` | `src/Population.jl` |
| 17 | `PopMember` 结构体字段 | 理解个体表示 | `PopMember` | `src/PopMember.jl` |
| 18 | TemplateExpression 是否支持不同子表达式用不同算子集 | 核物理中不同项可能需要不同算子 | `operators`、`TemplateExpression` | `src/TemplateExpression.jl`、`src/ComposableExpression.jl` |

---

## 13. 下一次 compact 后的启动 prompt

以下是一段可以直接复制给 compact 后 Claude Code 的启动 prompt。复制后在新 session 中粘贴即可：

---

```
请先完整阅读 /home/taomingyu/Reference/PySR/00_NuSR_Source_Navigation_Checkpoint.md。

这份文件是我（陶明宇）在 2026-05-06 花了两天时间通读 PySR 和 SymbolicRegression.jl 源码后写的知识蒸馏文档。你的任务是基于这份文档（而非重新通读源码）来帮助我开发 NuSR——一个面向核物理经验公式发现的符号回归 Python 库。

请遵守以下规则：

1. **不要通读整个 PySR / SymbolicRegression.jl 仓库。** 文档已经标明了每个文件的作用、我已读程度、与 NuSR 的关系。如果你需要看某个具体文件，先看文档第 9 节的路由表，然后用 grep 定位到具体行，用 Read 的 offset/limit 只读需要的片段。

2. **不确定就标注"待确认"。** 文档中有大量"待确认"标记。如果你基于已有知识能判断，告诉我；如果必须读源码才能回答，读完之后更新文档。

3. **读完新源码要更新文档。** 每当你深入阅读了一个之前标注"未读"或"待确认"的文件，请在文档对应位置追加你的发现，并修正过时信息。

4. **优先用 wrapper 模式。** NuSR 第一版的开发哲学是：能用 PySR 公开 API 实现的就不改 PySR 代码；能用 Python wrapper 解决的不改 Julia 代码。只有确认公开 API 无法满足时才考虑 fork/修改核心。

5. **修改代码前先写设计说明。** 在改任何文件之前，先告诉我你要改什么、为什么、影响范围、回退方案。

6. **写代码优先考虑物理正确性。** 我是物理博士，代码的物理含义比代码的工程优雅更重要。如果某个实现方案在物理上有问题，直接指出。

7. **使用中文交流，专业名词后括号附英文原名。**

我的当前目标是：[在此描述你的具体目标，例如："实现 NuSRRegressor 的第一版 wrapper，包含核物理默认参数和液滴模型模板"]。
```

---

## 附录：文件路径索引

### PySR 源码
```
/home/taomingyu/Development/PySR/src/PySR-master/pysr/
├── __init__.py
├── sr.py                  ← 核心类 PySRRegressor
├── julia_import.py        ← juliacall 初始化
├── julia_helpers.py       ← Python-Julia 数据转换
├── julia_extensions.py    ← Julia 扩展包加载
├── julia_registry_helpers.py ← Julia 注册表 fallback
├── expression_specs.py    ← TemplateExpressionSpec 等
├── export.py              ← 导出流水线
├── export_sympy.py        ← SymPy 导出
├── export_numpy.py        ← NumPy 导出
├── export_jax.py          ← JAX 导出
├── export_torch.py        ← PyTorch 导出
├── export_latex.py        ← LaTeX 导出
├── feature_selection.py   ← 特征选择
├── denoising.py           ← 降噪
├── utils.py               ← 通用工具
├── deprecated.py          ← 废弃 API
└── logger_specs.py        ← 日志配置
```

### SymbolicRegression.jl 源码
```
/home/taomingyu/Development/SymbolicRegression/src/SymbolicRegression.jl-master/src/
├── SymbolicRegression.jl      ← 主模块 + equation_search
├── OptionsStruct.jl           ← ComplexityMapping
├── Core.jl                    ← Dataset, Options, AbstractExpression 等（未读）
├── RegularizedEvolution.jl    ← reg_evol_cycle
├── SingleIteration.jl         ← s_r_cycle + optimize_and_simplify_population
├── HallOfFame.jl              ← HallOfFame + Pareto 前沿
├── LossFunctions.jl           ← eval_loss + eval_cost
├── Complexity.jl              ← compute_complexity
├── AdaptiveParsimony.jl       ← RunningSearchStatistics
├── Configure.jl               ← 配置测试 + move_functions_to_workers
├── MutationFunctions.jl       ← 基础变异操作
├── Mutate.jl                  ← next_generation + crossover_generation（未读）
├── Population.jl              ← Population + best_of_sample（未读）
├── PopMember.jl               ← PopMember 结构体（未读）
├── SearchUtils.jl             ← save_to_file + construct_datasets + @sr_spawner
├── TemplateExpression.jl      ← TemplateExpression + ValidVector
├── TemplateExpressionMacro.jl ← @template_spec 宏
├── ComposableExpression.jl    ← ComposableExpression + ValidVector（未读）
├── ParametricExpression.jl    ← 废弃的旧参数化表达式（未读）
├── ConstantOptimization.jl    ← BFGS/NelderMead（未读）
├── DimensionalAnalysis.jl     ← 物理量纲约束（未读）
├── CheckConstraints.jl        ← 约束检查（未读）
├── Dataset.jl                 ← Dataset 结构体（未读）
├── Operators.jl               ← OperatorEnum（未读）
└── Migration.jl               ← 种群迁移（未读）
```

### NuSR 开发目录
```
/home/taomingyu/Development/NuSR/
├── NuSR_Development_Plan.md   ← 12 个月开发规划
├── CLAUDE.md                  ← NuSR 项目指令
├── task_queue.md              ← 任务队列
├── session_log.md             ← 会话日志
├── nusr/                      ← NuSR 包本体（待创建）
│   ├── __init__.py
│   ├── regressor.py
│   ├── templates/
│   ├── operators.py
│   ├── interpreter.py
│   └── data/
├── docs/                      ← 设计文档
└── tests/                     ← 测试
```

### 参考文档
```
/home/taomingyu/Reference/PySR/
├── 00_NuSR_Source_Navigation_Checkpoint.md  ← 本文件
└── general_pysr_expression_discovery_guide.md ← 已有的 PySR 使用指南
```

---

> **文档维护规则**：
> - 每次深入阅读新源码后，更新第 1 节表格中的"已接触程度"
> - 每次确认一个"待确认"项后，将其移到第 11 节
> - 每次发现新问题后，追加到第 12 节
> - 每次完成一个 Phase 后，更新第 10 节的进度
> - 如果某个主题内容超过 200 行，拆分为独立专题 md
