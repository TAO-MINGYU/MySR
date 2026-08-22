# 03 — Python↔Julia 桥接机制

## 1. 桥接层架构

### 依赖关系
```
pysr/__init__.py
  └── from .julia_import import jl, SymbolicRegression
        │
        ├── juliacall (Python 包) → PythonCall.jl (Julia 包)
        │   │
        │   ├── jl.seval(code)              # 执行 Julia 代码字符串
        │   ├── jl.ModuleName               # 访问 Julia 模块
        │   ├── AnyValue                     # 任意 Julia 对象的 Python 包装
        │   └── VectorValue                  # Julia Vector 的 Python 包装
        │
        └── SymbolicRegression (Julia 模块引用)
```

### 导入顺序至关重要
`__init__.py` 必须在导入 numpy/pytorch 之前导入 `juliacall`：
- juliacall 若在 numpy 之后导入，可能与 OpenBLAS/MKL 产生库链接冲突
- 因此 `from pysr.julia_import import jl, SymbolicRegression` 是所有 `__init__.py` 导入中的第一个

## 2. julia_import.py — Julia 初始化

```
关键步骤:
1. 检测 juliacall 是否已加载
2. 设置环境变量:
   - PYTHON_JULIACALL_HANDLE_SIGNALS=yes  # 防止多线程 segfault
   - PYTHON_JULIACALL_THREADS=auto        # 最大 CPU 利用
   - PYTHON_JULIACALL_OPTLEVEL=3          # 最大优化
3. try_with_registry_fallback(_import_juliacall)
4. 从 juliacall 导入: AnyValue, VectorValue, Main (as jl)
5. 记录 Julia 版本: jl.VERSION
6. jl.seval("using SymbolicRegression")
7. 暴露: SymbolicRegression, D, less, greater_equal, less_equal, Pkg
```

## 3. julia_helpers.py — 数据转换工具

### 关键函数

```python
jl_array(x, dtype=None) → Julia Array
    # Python array/list → julia.Array[dtype]

jl_serialize(obj: Any) → NDArray[np.uint8]
    # Julia 对象 → IOBuffer → Serialization.serialize → numpy uint8 数组
    # 用于: pickle PySRRegressor 时保存 Julia 状态

jl_deserialize(s: NDArray[np.uint8]) → AnyValue
    # numpy uint8 数组 → IOBuffer → Serialization.deserialize → Julia 对象
    # 用于: 加载 pickle 时恢复 Julia 状态

jl_named_tuple(d) → Julia NamedTuple
    # Python dict → Julia NamedTuple

jl_is_function(f) → bool
    # 检查 Julia 值是否是 Function 类型
```

### 模块加载时副作用
```python
# 在 julia_helpers.py 模块加载时自动执行:
jl.seval("using Serialization: Serialization")
jl.seval("using PythonCall: PythonCall")
jl.seval("using SymbolicRegression: plus, sub, mult, div, pow")
```

## 4. julia_extensions.py — 扩展包管理

### 可选扩展
| 扩展 | 用途 |
|------|------|
| LoopVectorization.jl (turbo) | 加速表达式评估 |
| Bumper.jl (bumper) | 自定义分配器，减少GC |
| Zygote.jl / Enzyme.jl / Mooncake.jl | 自动微分后端 |
| ClusterManagers.jl | Slurm/SGE 集群管理 |
| TensorBoardLogger.jl | TensorBoard 日志 |

### 加载流程
```python
load_package(package_name, uuid):
    1. isinstalled(uuid) → 检查 Pkg.dependencies()
    2. 如果未安装: Pkg.add(name=..., uuid=...), Pkg.resolve()
    3. wrap in try_with_registry_fallback
    4. jl.seval(f"using {package_name}: {package_name}")
```

## 5. julia_registry_helpers.py — Registry 容错

### 问题
Julia 包注册表有时会报告 "Unsatisfiable requirements"，即使包是可安装的

### 解决方案: try_with_registry_fallback
```python
try_with_registry_fallback(f, *args, **kwargs):
    1. 保存当前 JULIA_PKG_SERVER_REGISTRY_PREFERENCE
    2. 执行 f(*args, **kwargs)
    3. 如果失败且错误匹配 "Unsatisfiable requirements":
       - 设置 JULIA_PKG_SERVER_REGISTRY_PREFERENCE="eager"
       - 重试 f(*args, **kwargs)
    4. 恢复原始设置
```

## 6. 状态持久化

### PySRRegressor 的 warm_start 机制
```
fit() 开始时:
  if warm_start:
    options = jl_deserialize(julia_options_stream_)  # 恢复 Julia Options
    state = jl_deserialize(julia_state_stream_)      # 恢复搜索状态
  else:
    options = 新构建
    state = nothing

_run() 结束时:
  julia_options_stream_ = jl_serialize(options)
  julia_state_stream_ = jl_serialize(state)
  _checkpoint()  # pickle 整个 PySRRegressor 到 checkpoint.pkl
```

### checkpoint 格式
```
output_directory_/run_id_/checkpoint.pkl:
  ├── Python 属性 (equations_, feature_names_in_, 等)
  ├── julia_options_stream_  (numpy uint8 数组)
  └── julia_state_stream_    (numpy uint8 数组)
```

## 7. 内联操作符

PySR 支持在 `operators` 字典中直接写 Julia 代码定义新操作符:
```python
operators = {
    2: ["my_op(x, y) = x^2 + y^2"]
}
```
- `_maybe_create_inline_operators()` 检测含 `(` 的字符串
- 通过 `jl.seval()` 在 Julia 中注册函数
- 提取函数名，替换 `extra_sympy_mappings` 检查
