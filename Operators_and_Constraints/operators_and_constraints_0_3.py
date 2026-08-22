# region libraries
# ======================================================
# libraries
# ======================================================
import numpy as np
import pandas as pd
import sympy as sp
# endregion


# region oerators
# ==============================================================
# Binary Operators
# ==============================================================
my_binary_operators = [
    # --- 基础算子 ---
    "+", 
    "-", 
    "*", 
    "/",
]

# ==============================================================
# Unary Operators
# ==============================================================
my_unary_operators = [

    # --- 2.1 PySR/Julia 内置算子---
    "square",    # 平方: x^2
    "cube",      # 立方: x^3
    "cbrt",      # 立方根: x^(1/3)
    "sqrt",      # 平方根: √x (注意：如果数据可能为负，建议不用或改为 sqrt_abs)
    "abs",       # 绝对值: |x|
    "exp",       # 自然指数: e^x
    "log",       # 自然对数: ln(x)
    "sin",       # 正弦: sin(x)
    "cos",       # 余弦: cos(x)
    "tan",       # 正切: tan(x)
    "asin",      # 反正弦: arcsin(x)
    "acos",      # 反余弦: arccos(x)
    "atan",      # 反正切: arctan(x)
    "sinh",      # 双曲正弦: sinh(x)
    "cosh",      # 双曲余弦: cosh(x)
    "tanh",      # 双曲正切: tanh(x)
    "asinh",     # 反双曲正弦: arcsinh(x)
    "acosh",     # 反双曲余弦: arccosh(x)
    "atanh",     # 反双曲正切: arctanh(x)
    
    # # --- 物理与几何衍生算子 (新增自定义) ---
    "inv(x) = 1 / x",                       # 倒数
    "inv2(x) = 1 / (x^2)",                  # 平方反比
    "inv3(x) = 1 / (x^3)",                  # 立方反比
    "pow23(x) = cbrt(x)^2",                   # 2/3次方 (改写为cbrt的平方，完美避开负数底数报错)
]

# ==============================================================
# SymPy Mappings
# ==============================================================
my_extra_sympy_mappings = {

    # # ---一元算子翻译---
    "inv": lambda x: 1 / x,
    "inv2": lambda x: 1 / (x**2),
    "inv3": lambda x: 1 / (x**3),
    "pow23": lambda x: x**sp.Rational(2, 3),
}
# endregion

# region constraints
# ==============================================================
# Helper: extract active operator names from PySR operator strings
# ==============================================================
# 目的：
# - 如果以后你临时注释掉 primitive set 中的某些算子；
# - complexity_of_operators / constraints / nested_constraints 会自动过滤；
# - 避免 constraints 中出现当前 primitive set 没启用的 operator，从而导致 PySR 报错。

def _get_pysr_operator_name(operator_string):
    operator_string = operator_string.strip()

    # 自定义算子，例如：
    # "inv(x) = 1 / x"
    # 提取为：
    # "inv"
    if "(" in operator_string and "=" in operator_string:
        return operator_string.split("(", 1)[0].strip()

    # 内置算子，例如：
    # "+", "-", "*", "/", "square", "exp"
    return operator_string


_active_operator_names = {
    _get_pysr_operator_name(op)
    for op in (my_binary_operators + my_unary_operators)
}


def _keep_active_dict(raw_dict):
    return {
        key: value
        for key, value in raw_dict.items()
        if key in _active_operator_names
    }


def _active_ops(op_list):
    return [
        op
        for op in op_list
        if op in _active_operator_names
    ]


def _forbid_nested(nested_dict, outer_ops, inner_ops):
    for outer in _active_ops(outer_ops):
        nested_dict.setdefault(outer, {})
        for inner in _active_ops(inner_ops):
            nested_dict[outer][inner] = 0


def _limit_nested(nested_dict, outer_ops, inner_ops, limit):
    for outer in _active_ops(outer_ops):
        nested_dict.setdefault(outer, {})
        for inner in _active_ops(inner_ops):
            nested_dict[outer][inner] = limit


# ==============================================================
# 1. Operator Complexities
# ==============================================================
# 基本原则：
# - +, -, * 最便宜；
# - / 比普通乘加略贵，因为会引入奇异性；
# - square/cube/cbrt/abs 是简单代数结构；
# - exp/log/trig/hyperbolic 是复杂函数；
# - tan, asin, acos, sinh, cosh, acosh, atanh 等更容易引入不稳定结构，因此略贵；
# - inv/inv2/inv3 是有奇异性的反比结构，因此比普通代数算子略贵。

my_complexity_of_operators_raw = {
    # ==========================================================
    # Binary Operators
    # ==========================================================
    "+": 1,
    "-": 1,
    "*": 1,
    "/": 2,

    # ==========================================================
    # Built-in unary algebraic operators
    # ==========================================================
    "square": 1,
    "cube": 1,
    "cbrt": 1,
    "sqrt": 2,
    "abs": 1,

    # ==========================================================
    # Exponential / logarithmic functions
    # ==========================================================
    "exp": 3,
    "log": 3,

    # ==========================================================
    # Trigonometric functions
    # ==========================================================
    "sin": 3,
    "cos": 3,
    "tan": 4,
    "asin": 4,
    "acos": 4,
    "atan": 3,

    # ==========================================================
    # Hyperbolic functions
    # ==========================================================
    "sinh": 4,
    "cosh": 4,
    "tanh": 3,
    "asinh": 4,
    "acosh": 5,
    "atanh": 5,

    # ==========================================================
    # Custom inverse / power operators
    # ==========================================================
    "inv": 2,
    "inv2": 3,
    "inv3": 3,
    "pow23": 2,
}

my_complexity_of_operators = _keep_active_dict(my_complexity_of_operators_raw)


# ==============================================================
# 2. Basic Constraints
# ==============================================================
# 格式：
#   一元算子: "op": max_complexity_inside
#   二元算子: "op": (left_max_complexity, right_max_complexity)
#
# -1 表示不限制。
#
# 核心原则：
# 1. +, -, * 基本不限制；
# 2. / 的分母复杂度适度限制，防止出现过深、过怪的分母；
# 3. 简单代数算子内部可以稍微复杂；
# 4. exp/log/trig/hyperbolic 内部表达式要更浅；
# 5. inv/inv2/inv3 允许承担物理反比结构，但也不应无限套娃。

my_constraints_raw = {
    # ==========================================================
    # Binary Operators
    # ==========================================================
    "+": (-1, -1),
    "-": (-1, -1),
    "*": (-1, -1),

    # 分母复杂度限制为 8：
    # 允许 x / (a + b*y)、x / sqrt(y) 这类结构；
    # 但不鼓励 x / very_complex_expression。
    "/": (-1, 8),

    # ==========================================================
    # Built-in unary algebraic operators
    # ==========================================================
    "square": 8,
    "cube": 8,
    "cbrt": 8,
    "sqrt": 6,
    "abs": 8,

    # ==========================================================
    # Exponential / logarithmic functions
    # ==========================================================
    # 允许 exp(a*x+b), exp(inv(x)), log(x+y) 等；
    # 不鼓励 exp(very_complex_expression)。
    "exp": 5,
    "log": 5,

    # ==========================================================
    # Trigonometric functions
    # ==========================================================
    "sin": 5,
    "cos": 5,
    "tan": 4,
    "asin": 4,
    "acos": 4,
    "atan": 5,

    # ==========================================================
    # Hyperbolic functions
    # ==========================================================
    "sinh": 4,
    "cosh": 4,
    "tanh": 5,
    "asinh": 5,
    "acosh": 4,
    "atanh": 4,

    # ==========================================================
    # Custom inverse / power operators
    # ==========================================================
    "inv": 8,
    "inv2": 7,
    "inv3": 7,
    "pow23": 8,
}

my_constraints = _keep_active_dict(my_constraints_raw)


# ==============================================================
# 3. Nested Constraints
# ==============================================================
# nested_constraints[outer][inner] = n
#
# 含义：
# - outer 是外层算子；
# - inner 是被包在里面的算子；
# - n = 0 表示禁止 inner 出现在 outer 内部；
# - n = 1 表示最多允许出现一次。
#
# 核心原则：
# 1. +, -, *, / 不做 nested 限制，只靠 complexity 和 maxsize 控制；
# 2. exp/log/trig/hyperbolic 视为复杂函数，复杂函数之间不互相嵌套；
# 3. inv/inv2/inv3 不互相嵌套，避免 inv(inv(x))、inv2(inv(x))；
# 4. 幂次/根式结构不要乱套娃；
# 5. 允许简单代数结构进入复杂函数，例如 exp(inv(x)), sin(square(x))；
# 6. 禁止复杂函数被简单代数外壳包装，例如 square(exp(x))，避免隐藏复杂度。

# --------------------------------------------------------------
# 3.1 Operator groups
# --------------------------------------------------------------

basic_binary_ops = [
    "+",
    "-",
    "*",
    "/",
]


simple_algebra_unary_ops = [
    "square",
    "cube",
    "cbrt",
    "sqrt",
    "abs",
    "pow23",
]


inverse_unary_ops = [
    "inv",
    "inv2",
    "inv3",
]


exp_log_unary_ops = [
    "exp",
    "log",
]


trig_unary_ops = [
    "sin",
    "cos",
    "tan",
    "asin",
    "acos",
    "atan",
]


hyperbolic_unary_ops = [
    "sinh",
    "cosh",
    "tanh",
    "asinh",
    "acosh",
    "atanh",
]


complex_unary_ops = (
    exp_log_unary_ops
    + trig_unary_ops
    + hyperbolic_unary_ops
)


root_power_unary_ops = [
    "square",
    "cube",
    "cbrt",
    "sqrt",
    "pow23",
]


my_nested_constraints = {}


# --------------------------------------------------------------
# 3.2 复杂函数之间禁止互相嵌套
# --------------------------------------------------------------
# 禁止例子：
# - exp(exp(x))
# - exp(log(x))
# - log(exp(x))
# - sin(cos(x))
# - tanh(exp(x))
# - sinh(sin(x))

_forbid_nested(
    my_nested_constraints,
    outer_ops=complex_unary_ops,
    inner_ops=complex_unary_ops,
)


# --------------------------------------------------------------
# 3.3 简单代数外壳禁止包裹复杂函数
# --------------------------------------------------------------
# 禁止例子：
# - square(exp(x))
# - cube(log(x))
# - sqrt(sin(x))
# - pow23(tanh(x))
#
# 原因：
# - square(exp(x)) 等价于 exp(2x)，会隐藏复杂度；
# - cube(sin(x))、sqrt(tanh(x)) 这类结构通常缺乏明确物理含义。

_forbid_nested(
    my_nested_constraints,
    outer_ops=simple_algebra_unary_ops,
    inner_ops=complex_unary_ops,
)


# --------------------------------------------------------------
# 3.4 inverse 类算子禁止互相嵌套
# --------------------------------------------------------------
# 禁止例子：
# - inv(inv(x))
# - inv2(inv(x))
# - inv3(inv2(x))
#
# 原因：
# - 这些结构多数可以化简；
# - 容易制造无意义奇异点。

_forbid_nested(
    my_nested_constraints,
    outer_ops=inverse_unary_ops,
    inner_ops=inverse_unary_ops,
)


# --------------------------------------------------------------
# 3.5 inverse 类算子禁止包裹复杂函数
# --------------------------------------------------------------
# 禁止例子：
# - inv(exp(x))
# - inv(log(x))
# - inv2(sin(x))
# - inv3(tanh(x))
#
# 如果需要 1/exp(x)，更推荐让模型用 exp(-x) 形式表达。

_forbid_nested(
    my_nested_constraints,
    outer_ops=inverse_unary_ops,
    inner_ops=complex_unary_ops,
)


# --------------------------------------------------------------
# 3.6 根式 / 幂次结构之间限制嵌套
# --------------------------------------------------------------
# 默认禁止：
# - square(cube(x))
# - cube(square(x))
# - cbrt(sqrt(x))
# - pow23(pow23(x))
# - sqrt(cbrt(x))
#
# 例外：
# - 允许 square(square(x)) 一次，用来表达 x^4。

_forbid_nested(
    my_nested_constraints,
    outer_ops=root_power_unary_ops,
    inner_ops=root_power_unary_ops,
)

# 例外：允许 square(square(x)) 最多出现一次
if "square" in _active_operator_names:
    my_nested_constraints.setdefault("square", {})
    my_nested_constraints["square"]["square"] = 1


# --------------------------------------------------------------
# 3.7 exp/log 之间再次明确禁止
# --------------------------------------------------------------
# 这一步与 3.2 有重叠，是为了让规则更清晰。
# 禁止：
# - exp(exp(x))
# - exp(log(x))
# - log(exp(x))
# - log(log(x))

_forbid_nested(
    my_nested_constraints,
    outer_ops=exp_log_unary_ops,
    inner_ops=exp_log_unary_ops,
)


# --------------------------------------------------------------
# 3.8 三角函数之间禁止互相嵌套
# --------------------------------------------------------------
# 禁止：
# - sin(cos(x))
# - tan(sin(x))
# - asin(cos(x))

_forbid_nested(
    my_nested_constraints,
    outer_ops=trig_unary_ops,
    inner_ops=trig_unary_ops,
)


# --------------------------------------------------------------
# 3.9 双曲函数之间禁止互相嵌套
# --------------------------------------------------------------
# 禁止：
# - sinh(cosh(x))
# - tanh(sinh(x))
# - atanh(tanh(x))

_forbid_nested(
    my_nested_constraints,
    outer_ops=hyperbolic_unary_ops,
    inner_ops=hyperbolic_unary_ops,
)


# --------------------------------------------------------------
# 3.10 三角函数与双曲函数之间禁止互相嵌套
# --------------------------------------------------------------
# 禁止：
# - sin(tanh(x))
# - cosh(atan(x))
# - atanh(sin(x))

_forbid_nested(
    my_nested_constraints,
    outer_ops=trig_unary_ops,
    inner_ops=hyperbolic_unary_ops,
)

_forbid_nested(
    my_nested_constraints,
    outer_ops=hyperbolic_unary_ops,
    inner_ops=trig_unary_ops,
)


# --------------------------------------------------------------
# 3.11 Final cleanup
# --------------------------------------------------------------
# 双保险：
# - 删除未启用的 outer operator；
# - 删除未启用的 inner operator；
# - 删除空字典。

my_nested_constraints = {
    outer: {
        inner: limit
        for inner, limit in inner_dict.items()
        if inner in _active_operator_names
    }
    for outer, inner_dict in my_nested_constraints.items()
    if outer in _active_operator_names
}

my_nested_constraints = {
    outer: inner_dict
    for outer, inner_dict in my_nested_constraints.items()
    if len(inner_dict) > 0
}

# endregion

