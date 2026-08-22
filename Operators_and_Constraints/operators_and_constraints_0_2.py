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


# ==============================================================
# 1. Operator Complexities
# ==============================================================
# 设计原则：
# - +, -, * 最便宜；
# - / 略贵，因为它可能引入奇异性；
# - square/cube/cbrt/sqrt/abs/pow23 作为代数结构，整体较便宜；
# - exp/log 物理上有意义，但容易引入指数爆炸、定义域限制和复杂嵌套，
#   因此复杂度稍高；
# - inv/inv2/inv3 是反比结构，物理上常见，但有奇异性，复杂度略高。
#
# 注意：
# - 这里不通过 complexity 严格禁止代数嵌套；
# - 只是让更危险的结构在 Pareto ranking 里付出更高代价。

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
    "sqrt": 1,
    "abs": 1,

    # ==========================================================
    # Exponential / logarithmic operators
    # ==========================================================
    "exp": 3,
    "log": 3,

    # ==========================================================
    # Custom inverse / power operators
    # ==========================================================
    "inv": 2,
    "inv2": 2,
    "inv3": 2,
    "pow23": 1,
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
# 你的当前原则：
# - 除 exp/log 外，其余算子尽量不限制嵌套；
# - 这样可以保留构造物理形式的自由度；
# - 比如：
#     sqrt(cbrt(x))       -> x^(1/6)
#     square(cbrt(x))     -> x^(2/3)
#     inv(square(x))      -> 1/x^2
#     square(inv(x))      -> 1/x^2
#     inv(cbrt(x))        -> x^(-1/3)
#
# 因此：
# - +, -, * 完全开放；
# - / 的分母仍然给一个较宽松限制，防止 very_complex_expression 出现在分母；
# - 代数算子、根式算子、反比算子都放宽；
# - exp/log 内部适度限制，防止 exp(very_complex_expression)。

my_constraints_raw = {
    # ==========================================================
    # Binary Operators
    # ==========================================================
    "+": (-1, -1),
    "-": (-1, -1),
    "*": (-1, -1),

    # 分母复杂度限制为 10：
    # 比之前更宽松，允许更多物理反比结构；
    # 但仍然避免 x / very_complex_expression。
    "/": (-1, 10),

    # ==========================================================
    # Built-in unary algebraic operators
    # ==========================================================
    "square": 12,
    "cube": 12,
    "cbrt": 12,
    "sqrt": 12,
    "abs": 12,

    # ==========================================================
    # Exponential / logarithmic operators
    # ==========================================================
    # exp/log 是唯一重点限制对象：
    # - 允许 exp(a*x+b), exp(inv(x)), exp(sqrt(x))；
    # - 允许 log(x+y), log(square(x)), log(abs(x)+1)；
    # - 不鼓励 exp(very_complex_expression)。
    "exp": 6,
    "log": 6,

    # ==========================================================
    # Custom inverse / power operators
    # ==========================================================
    "inv": 12,
    "inv2": 12,
    "inv3": 12,
    "pow23": 12,
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
# - n = 0 表示禁止 inner 出现在 outer 内部。
#
# 你的当前物理原则：
# - 只不能接受 exp 和 log 的彼此嵌套；
# - 其他嵌套都暂时允许，让 PySR 保留构造物理表达式的能力。
#
# 因此这里只禁止：
# - exp(exp(x))
# - exp(log(x))
# - log(exp(x))
# - log(log(x))
#
# 其他例如：
# - sqrt(cbrt(x))
# - cbrt(sqrt(x))
# - square(inv(x))
# - inv(square(x))
# - inv(cbrt(x))
# - pow23(inv(x))
# - square(square(x))
# - cube(cbrt(x))
# 全部允许。

exp_log_unary_ops = [
    "exp",
    "log",
]

my_nested_constraints = {}

_forbid_nested(
    my_nested_constraints,
    outer_ops=exp_log_unary_ops,
    inner_ops=exp_log_unary_ops,
)


# --------------------------------------------------------------
# Final cleanup
# --------------------------------------------------------------
# 双保险：
# - 删除未启用的 outer operator；
# - 删除未启用的 inner operator；
# - 删除空字典。
#
# 这样以后你在 primitive set 中注释掉任何 operator，
# 后面的 constraints 会自动适配，不会把未启用算子传给 PySR。

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




