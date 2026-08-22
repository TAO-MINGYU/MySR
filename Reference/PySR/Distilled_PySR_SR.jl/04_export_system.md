# 04 — 表达式导出系统

## 1. 导出架构

```
PySR 方程字符串 (如 "plus(mult(x1, 3.5), sin(x2))")
    │
    ├── export_sympy.py: pysr2sympy() → sympy.Expr
    │       │
    │       ├── export_numpy.py: sympy2numpy() → CallableEquation
    │       ├── export_jax.py:   sympy2jax()   → (callable, parameters)
    │       ├── export_torch.py: sympy2torch() → nn.Module
    │       └── export_latex.py: sympy2latex() → LaTeX string
    │
    └── export.py: add_export_formats() → DataFrame with all formats
```

## 2. pysr2sympy — 字符串→SymPy

**文件**: `export_sympy.py`

### sympy_mappings 字典
将 PySR 内部操作符名映射到 SymPy 函数:
```python
sympy_mappings = {
    "div": lambda x, y: x / y,
    "inv": lambda x: 1 / x,
    "mult": lambda x, y: x * y,
    "plus": lambda x, y: x + y,
    "sub": lambda x, y: x - y,
    "neg": lambda x: -x,
    "pow": lambda x, y: x ** y,
    "sqrt": sqrt,
    "cos": cos,
    "sin": sin,
    "exp": exp,
    "log": log,
    "cond": lambda x, y: Piecewise((y, x > 0), (0, True)),
    "greater": lambda x, y: Piecewise((x, x > y), (0, True)),
    "relu": lambda x: Max(0, x),
    # ... 50+ 操作符
}
```

### 核心转换
```python
pysr2sympy(equation, feature_names_in, extra_sympy_mappings=None):
    symbols = create_sympy_symbols_map(feature_names_in)
    locals = {**symbols, **sympy_mappings, **(extra_sympy_mappings or {})}
    return sympy.sympify(equation, locals=locals, evaluate=False)
```

### 关键设计
- `evaluate=False`: 保留原始表达式结构，不自动展开
- `create_sympy_symbols_map()`: 为每个特征名创建 `sympy.Symbol`

## 3. NumPy 导出

**文件**: `export_numpy.py`

```python
class CallableEquation:
    def __call__(self, X):
        # 1. DataFrame → 按列名提取
        # 2. ndarray → 转置并解包列
        # 3. 通过 lambdify 评估:
        #    self._lambda = lambdify(self._sympy_symbols, self._sympy)
        # 返回 shape (n_samples,) 的结果
        
    def _lambda(self):
        # 懒加载: 只在首次调用时编译
```

## 4. JAX 导出

**文件**: `export_jax.py`

### 流程
```
SymPy 表达式
  → sympy2jaxtext() - 递归生成 JAX Python 源代码字符串
    - sympy.Float → append to parameters list, use parameters[idx]
    - sympy.Integer → literal
    - sympy.Symbol → X[:, col_idx]
    - function → jnp.funcname(subexprs...)
  → 用 exec() 定义函数
  → 返回 (callable, jnp.array(parameters))
```

### 参数化
- 所有浮点常数自动提取为可训练参数
- `Rational` 和 `NumberSymbol` 保持为固定常数
- 可选 `selection` 遮罩用于特征选择

### 支持的 JAX 函数
```python
_jnp_func_lookup = {
    sympy.sqrt: "jnp.sqrt",
    sympy.sin: "jnp.sin",
    sympy.exp: "jnp.exp",
    sympy.erf: "jsp.erf",
    sympy.log: "jnp.log",
    # ... etc
}
# Mul → 用 * 连接
# Add → 用 + 连接
```

## 5. PyTorch 导出

**文件**: `export_torch.py`

### 递归模块构建
```
sympy.Expr → _Node (nn.Module 树)
  ├── sympy.Float → nn.Parameter (可训练)
  ├── sympy.Rational → buffer (固定)
  ├── sympy.Symbol → 输入占位符
  └── sympy.Function → nn.ModuleList([子节点...])
      forward: torch.func(子节点输出...)
```

### SingleSymPyModule
```python
class SingleSymPyModule(nn.Module):
    def forward(self, X):
        symbols = {name: X[:, i] for i, name in enumerate(symbol_names)}
        return self._node.forward(symbols)
```

## 6. LaTeX 导出

**文件**: `export_latex.py`

### PreciseLatexPrinter
```python
class PreciseLatexPrinter(LatexPrinter):
    def _print_Float(self, expr):
        # 截断到指定精度
        expr = sympy.Float(expr, self.prec)
        return super()._print_Float(expr)
```

### 表格生成
```python
latex_table(indices=None, precision=3, columns=None):
    # 使用 booktabs 格式
    # 长方程 (>50 字符) 用 minipage + dmath* 环境换行
    # 支持自定义列: equation, complexity, loss, score
```

## 7. TemplateExpression 导出

对于 `TemplateExpressionSpec`:
- `create_exports()` 不调用 `add_export_formats()`
- 而是调用 `_search_output_to_callable_expressions()`
- 从 Julia 的 hall-of-fame 中提取 `AbstractExpression` 对象
- 包装为 `CallableJuliaExpression` (在 Julia 中评估)
- `evaluates_in_julia = True`

### CallableJuliaExpression
```python
class CallableJuliaExpression:
    def __call__(self, X, *args):
        # X → 转置 → Julia 评估 → 转置回 → numpy array
```

## 8. 特征选择 + 导出

当 `select_k_features` 启用时:
- 导出函数包含特征选择遮罩
- `CallableEquation._selection` 存储 bool mask
- JAX: `X = X[:, list(selection)]`
- PyTorch: `self.selection = selection`
