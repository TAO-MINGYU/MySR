# 10 — 源文件索引

## PySR 前端 (Python)

| 文件 | 行数 | 内容 | 依赖 |
|------|------|------|------|
| `pysr/__init__.py` | 67 | 包初始化，设置日志，导入 Julia，定义公共 API | julia_import, sr |
| `pysr/sr.py` | 3122 | `PySRRegressor` 类 + 所有 fit/predict/export 逻辑 | 所有其他 pysr 模块 |
| `pysr/julia_import.py` | 75 | Julia 桥接初始化，加载 SymbolicRegression.jl | julia_registry_helpers |
| `pysr/julia_helpers.py` | 75 | 数据转换 (jl_array, jl_serialize, jl_deserialize) | julia_import |
| `pysr/julia_extensions.py` | 68 | 可选 Julia 包的安装和加载 | julia_import, julia_registry_helpers |
| `pysr/julia_registry_helpers.py` | 47 | Julia 注册表容错机制 | (standalone) |
| `pysr/expression_specs.py` | 435 | ExpressionSpec 类层次 (默认/模版/参数化) | julia_import, julia_helpers, export |
| `pysr/export.py` | ~150 | 导出编排器，调用各格式导出器 | export_sympy/numpy/jax/torch |
| `pysr/export_sympy.py` | ~100 | PySR 字符串→SymPy，操作符映射表 | sympy |
| `pysr/export_numpy.py` | ~60 | CallableEquation 类，lambdify 封装 | sympy, numpy |
| `pysr/export_jax.py` | ~200 | SymPy→JAX 源码生成，参数可训练 | sympy, jax |
| `pysr/export_torch.py` | ~250 | SymPy→PyTorch nn.Module 递归构建 | sympy, torch |
| `pysr/export_latex.py` | ~200 | LaTeX 方程和表格生成 | sympy |
| `pysr/feature_selection.py` | ~40 | RandomForest 特征选择 | sklearn |
| `pysr/denoising.py` | ~60 | GaussianProcess 去噪 | sklearn |
| `pysr/utils.py` | ~80 | 工具函数 (float 转换, 下标, 关键词建议) | numpy |
| `pysr/logger_specs.py` | ~100 | TensorBoardLoggerSpec | julia_import |
| `pysr/deprecated.py` | ~80 | 旧 API 兼容层 | sr |
| `pysr/param_groupings.yml` | ~100 | 超参数分组文档 | (文档) |
| `pyproject.toml` | 80 | 构建配置和依赖声明 | (构建) |

## SymbolicRegression.jl 后端 (Julia)

### 核心模块
| 文件 | 内容 |
|------|------|
| `src/SymbolicRegression.jl` | 主模块: `equation_search`, `_main_search_loop!`, `_dispatch_s_r_cycle` |
| `src/Core.jl` | 核心聚合模块: 汇集所有子模块，声明 `create_expression` 接口 |

### 算法
| 文件 | 内容 |
|------|------|
| `src/RegularizedEvolution.jl` | `reg_evol_cycle`: 锦标赛+变异/交叉+年龄替换主循环 |
| `src/SingleIteration.jl` | `s_r_cycle`: ncycles 次进化; `optimize_and_simplify_population` |
| `src/MutationFunctions.jl` | 树变异原语 (swap_operands 到 gen_random_tree 到 crossover_trees) |
| `src/Mutate.jl` | 变异调度: `next_generation`, `crossover_generation`, `_dispatch_mutations!` |
| `src/MutationWeights.jl` | `MutationWeights` 结构体, `sample_mutation`, 编译时优化 |

### 数据结构
| 文件 | 内容 |
|------|------|
| `src/PopMember.jl` | `PopMember{T,L,N}`: 个体 (树+成本+谱系) |
| `src/Population.jl` | `Population{T,L,N,PM}`: 群体容器, 锦标赛选择, 复制 |
| `src/HallOfFame.jl` | `HallOfFame{T,L,N,PM}`: 按复杂度的最佳方程, Pareto 前沿, 分数 |
| `src/ComposableExpression.jl` | `ComposableExpression`, `ValidVector` 系统 |
| `src/ParametricExpression.jl` | `ParametricExpression`, `ParametricExpressionSpec` [已弃用] |
| `src/TemplateExpression.jl` | `TemplateExpression`, `TemplateStructure`, 评估/显示/突变 |
| `src/TemplateExpressionMacro.jl` | `@template_spec` 宏, 确定性函数名散列 |
| `src/ExpressionBuilder.jl` | `create_expression`, `embed_metadata`, `strip_metadata` |
| `src/ExpressionSpec.jl` | `AbstractExpressionSpec`, `ExpressionSpec` |

### 评估与优化
| 文件 | 内容 |
|------|------|
| `src/LossFunctions.jl` | `eval_loss`, `eval_cost`, `loss_to_cost`, baseline_loss |
| `src/ConstantOptimization.jl` | `optimize_constants`, `Evaluator`, `GradEvaluator` |
| `src/Complexity.jl` | `compute_complexity`, `ComplexityMapping` 加权计算 |
| `src/AdaptiveParsimony.jl` | `RunningSearchStatistics`, 自适应频率跟踪 |
| `src/CheckConstraints.jl` | `check_constraints`, `flag_illegal_nests` |

### 配置与数据
| 文件 | 内容 |
|------|------|
| `src/Options.jl` | `Options` 构造函数: 参数验证, 默认值, 操作符映射 |
| `src/OptionsStruct.jl` | `Options`, `ComplexityMapping` 结构体定义 |
| `src/Operators.jl` | safe_* 操作符定义, `get_safe_op`, 操作符别名 |
| `src/Dataset.jl` | `Dataset`, `BasicDataset`, `SubDataset` |
| `src/Configure.jl` | 配置验证, 操作符测试, worker 设置 |
| `src/DimensionalAnalysis.jl` | `WildcardQuantity`, 维度违规检测 |

### 工具与并行
| 文件 | 内容 |
|------|------|
| `src/SearchUtils.jl` | `SearchState`, `RuntimeOptions`, `@sr_spawner`, `save_to_file`, `parse_guesses` |
| `src/Utils.jl` | `MutableTuple`, `bottomk_fast`, `PerTaskCache`, `@save_kwargs` |
| `src/Migration.jl` | `migrate!`: 岛屿模型迁移逻辑 |
| `src/ProgressBars.jl` | `WrappedProgressBar`: ProgressMeter 封装 |
| `src/Logging.jl` | `SRLogger`, Pareto 体积日志, 凸包计算 |
| `src/Recorder.jl` | `@recorder` 宏, 条件状态快照 |
| `src/ProgramConstants.jl` | 全局类型别名 (`DATA_TYPE`, `LOSS_TYPE`, `RecordType`) |
| `src/precompile.jl` | 预编译工作负载 |
| `src/deprecates.jl` | 已弃用函数签名映射 |
| `src/MLJInterface.jl` | MLJ 框架集成: `SRRegressor`, `MultitargetSRRegressor` |

### 扩展
| 文件 | 内容 |
|------|------|
| `ext/SymbolicRegressionEnzymeExt.jl` | Enzyme.jl 自动微分后端 (32MB 栈) |
| `ext/SymbolicRegressionMooncakeExt.jl` | Mooncake.jl 自动微分 (支持 TemplateExpression) |
| `ext/SymbolicRegressionJSON3Ext.jl` | JSON3 序列化 (record 文件) |
| `ext/SymbolicRegressionSymbolicUtilsExt.jl` | SymbolicUtils.jl 符号代数转换 |
