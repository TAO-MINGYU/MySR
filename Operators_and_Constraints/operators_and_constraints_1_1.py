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

    # --- 核质量 / 结合能 ---
    "symmetry_kernel(x, y) = (x - y)^2 / (abs(x + y) + 1e-12)",      # (N-Z)^2/A
    "isospin_asym(x, y) = (x - y) / (abs(x + y) + 1e-12)",          # I=(N-Z)/A
    "coulomb_kernel(x, y) = x * (x - 1) / (abs(y)^(1/3) + 1e-12)",  # Z(Z-1)/A^(1/3)
    "fissility_kernel(x, y) = x^2 / (abs(y) + 1e-12)",             # Z^2/A

    # --- 半径 / 几何截面 / 反应截面 ---
    "radius_sum13(x, y) = abs(x)^(1/3) + abs(y)^(1/3)",             # Ap^(1/3)+At^(1/3)
    "geom_cross_kernel(x, y) = (abs(x)^(1/3) + abs(y)^(1/3))^2",    # 几何截面核心

    # --- α 衰变 ---
    "alpha_gn_kernel(x, y) = x / (sqrt(abs(y)) + 1e-12)",           # Z/sqrt(Q_alpha)
    "alpha_brown_kernel(x, y) = abs(x)^0.6 / (sqrt(abs(y)) + 1e-12)", # Z^0.6/sqrt(Q_alpha)

    # --- 级密度 / 统计模型 ---
    "sqrt_prod(x, y) = sqrt(abs(x * y))",                          # sqrt(aU)
    "exp_sqrt_prod(x, y) = exp(min(80.0, 2.0 * sqrt(abs(x * y))))", # exp(2sqrt(aU))
    "backshift(x, y) = x - y",                                     # U-Delta

    # --- Breit-Wigner / Lorentzian ---
    "lorentz_denom(x, y) = x^2 + y^2 / 4",                         # (E-Er)^2 + Gamma^2/4
    "lorentz_kernel(x, y) = 1 / (x^2 + y^2 / 4 + 1e-12)",           # Lorentzian core

    # --- 比值型物理结构 ---
    "safe_ratio(x, y) = x / (abs(y) + 1e-12)",
    "inv_sum(x, y) = 1 / (abs(x + y) + 1e-12)",
    "product_over_sum(x, y) = x * y / (abs(x + y) + 1e-12)",
]
# ==============================================================
# Unary Operators
# ==============================================================
my_unary_operators = [
    # --- PySR / Julia 内置算子 ---
    "square",    # x^2
    "cube",      # x^3
    "cbrt",      # x^(1/3)
    "sqrt",      # sqrt(x)
    "abs",       # |x|
    "exp",       # exp(x)
    "log",       # log(x)

    # --- 三角函数 ---
    "sin",
    "cos",
    "tan",
    "asin",
    "acos",
    "atan",

    # --- 双曲函数 ---
    "sinh",
    "cosh",
    "tanh",
    "asinh",
    "acosh",
    "atanh",

    # --- 保护型基础算子 ---
    "inv(x) = 1 / (x + 1e-12)",                       # 保护倒数
    "inv_abs(x) = 1 / (abs(x) + 1e-12)",              # 绝对值保护倒数
    "inv2(x) = 1 / (x^2 + 1e-12)",                    # 平方反比
    "inv3(x) = 1 / (x^3 + 1e-12)",                    # 立方反比
    "inv_square_abs(x) = 1 / (x^2 + 1e-12)",          # 平方保护倒数
    "sqrt_abs(x) = sqrt(abs(x))",                     # sqrt(|x|)
    "log_abs(x) = log(abs(x) + 1e-12)",               # log(|x|)
    "log1p_abs(x) = log(abs(1 + x) + 1e-12)",         # log(|1+x|)

    # --- 幂次结构：质量公式、半径、表面项 ---
    "pow13_abs(x) = abs(x)^(1/3)",                    # A^(1/3)
    "pow23_abs(x) = abs(x)^(2/3)",                    # A^(2/3)
    "pow23(x) = cbrt(x)^2",                           # x^(2/3)，允许负数输入

    # --- Woods-Saxon / Fermi 分布 ---
    "woods_saxon(x) = 1 / (1 + exp(clamp(x, -80.0, 80.0)))",
    "dws(x) = exp(clamp(x, -80.0, 80.0)) / (1 + exp(clamp(x, -80.0, 80.0)))^2",

    # --- 高斯 / 指数阻尼 / 势函数结构 ---
    "gauss_sq(x) = exp(-min(80.0, x^2))",              # exp(-x^2)
    "gauss_std(x) = exp(-min(80.0, x^2 / 2))",         # exp(-x^2/2)
    "exp_neg_abs(x) = exp(-abs(x))",                   # exp(-|x|)
    "yukawa(x) = exp(-abs(x)) / (abs(x) + 1e-12)",     # Yukawa-like: exp(-r)/r
    "morse_attr(x) = 1 - exp(-abs(x))",                # Morse 吸引项

    # --- 简单阻尼 / 形状结构 ---
    "inv_one_plus(x) = 1 / (1 + abs(x))",              # 1/(1+|x|)
    "one_minus_sq(x) = 1 - x^2",                       # 1-x^2
    "sqrt_sq_minus_one(x) = sqrt(abs(x^2 - 1))",       # sqrt(|x^2-1|)

    # --- 集体模型 / 转动谱 ---
    "rotor(x) = x * (x + 1)",                          # J(J+1)
    "rotor_sq(x) = (x * (x + 1))^2",                   # [J(J+1)]^2
]

# ==============================================================
# SymPy Mappings
# ==============================================================
EPS = sp.Float("1e-12")
my_extra_sympy_mappings = {
    # ==========================================================
    # Unary Operators
    # ==========================================================

    # --- 保护型基础算子 ---
    "inv": lambda x: 1 / (x + EPS),
    "inv_abs": lambda x: 1 / (sp.Abs(x) + EPS),
    "inv2": lambda x: 1 / (x**2 + EPS),
    "inv3": lambda x: 1 / (x**3 + EPS),
    "inv_square_abs": lambda x: 1 / (x**2 + EPS),
    "sqrt_abs": lambda x: sp.sqrt(sp.Abs(x)),
    "log_abs": lambda x: sp.log(sp.Abs(x) + EPS),
    "log1p_abs": lambda x: sp.log(sp.Abs(1 + x) + EPS),

    # --- 幂次结构：质量公式、半径、表面项 ---
    "pow13_abs": lambda x: sp.Abs(x) ** sp.Rational(1, 3),
    "pow23_abs": lambda x: sp.Abs(x) ** sp.Rational(2, 3),
    "pow23": lambda x: sp.real_root(x, 3) ** 2,

    # --- Woods-Saxon / Fermi 分布 ---
    "woods_saxon": lambda x: 1 / (1 + sp.exp(x)),
    "dws": lambda x: sp.exp(x) / (1 + sp.exp(x))**2,

    # --- 高斯 / 指数阻尼 / 势函数结构 ---
    "gauss_sq": lambda x: sp.exp(-(x**2)),
    "gauss_std": lambda x: sp.exp(-(x**2) / 2),
    "exp_neg_abs": lambda x: sp.exp(-sp.Abs(x)),
    "yukawa": lambda x: sp.exp(-sp.Abs(x)) / (sp.Abs(x) + EPS),
    "morse_attr": lambda x: 1 - sp.exp(-sp.Abs(x)),

    # --- 简单阻尼 / 形状结构 ---
    "inv_one_plus": lambda x: 1 / (1 + sp.Abs(x)),
    "one_minus_sq": lambda x: 1 - x**2,
    "sqrt_sq_minus_one": lambda x: sp.sqrt(sp.Abs(x**2 - 1)),

    # --- 集体模型 / 转动谱 ---
    "rotor": lambda x: x * (x + 1),
    "rotor_sq": lambda x: (x * (x + 1))**2,

    # ==========================================================
    # Binary Operators
    # ==========================================================

    # --- 核质量 / 结合能 ---
    "symmetry_kernel": lambda x, y: (x - y)**2 / (sp.Abs(x + y) + EPS),
    "isospin_asym": lambda x, y: (x - y) / (sp.Abs(x + y) + EPS),
    "coulomb_kernel": lambda x, y: x * (x - 1) / (
        sp.Abs(y) ** sp.Rational(1, 3) + EPS
    ),
    "fissility_kernel": lambda x, y: x**2 / (sp.Abs(y) + EPS),

    # --- 半径 / 几何截面 / 反应截面 ---
    "radius_sum13": lambda x, y: (
        sp.Abs(x) ** sp.Rational(1, 3)
        + sp.Abs(y) ** sp.Rational(1, 3)
    ),
    "geom_cross_kernel": lambda x, y: (
        sp.Abs(x) ** sp.Rational(1, 3)
        + sp.Abs(y) ** sp.Rational(1, 3)
    )**2,

    # --- α 衰变 ---
    "alpha_gn_kernel": lambda x, y: x / (sp.sqrt(sp.Abs(y)) + EPS),
    "alpha_brown_kernel": lambda x, y: (
        sp.Abs(x) ** sp.Float("0.6")
    ) / (sp.sqrt(sp.Abs(y)) + EPS),

    # --- 级密度 / 统计模型 ---
    "sqrt_prod": lambda x, y: sp.sqrt(sp.Abs(x * y)),
    "exp_sqrt_prod": lambda x, y: sp.exp(2 * sp.sqrt(sp.Abs(x * y))),
    "backshift": lambda x, y: x - y,

    # --- Breit-Wigner / Lorentzian ---
    "lorentz_denom": lambda x, y: x**2 + y**2 / 4,
    "lorentz_kernel": lambda x, y: 1 / (x**2 + y**2 / 4 + EPS),

    # --- 比值型物理结构 ---
    "safe_ratio": lambda x, y: x / (sp.Abs(y) + EPS),
    "inv_sum": lambda x, y: 1 / (sp.Abs(x + y) + EPS),
    "product_over_sum": lambda x, y: x * y / (sp.Abs(x + y) + EPS),
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
    # "woods_saxon(x) = 1 / (1 + exp(x))"
    # 提取为：
    # "woods_saxon"
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


# ==============================================================
# Operator Complexities
# ==============================================================
# 注意：
# - 这里保留完整版本，命名为 my_complexity_of_operators_raw；
# - 真正传给 PySR 的是最后过滤后的 my_complexity_of_operators。

my_complexity_of_operators_raw = {
    # ==========================================================
    # Unary Operators
    # ==========================================================

    # --- 1. PySR / Julia 内置标准一元算子 ---
    "square": 1,
    "cube": 1,
    "cbrt": 1,
    "sqrt": 1,
    "abs": 1,
    "exp": 1,
    "log": 1,

    # --- 2. 三角函数 ---
    "sin": 1,
    "cos": 1,
    "tan": 2,
    "asin": 2,
    "acos": 2,
    "atan": 2,

    # --- 3. 双曲函数 ---
    "sinh": 2,
    "cosh": 2,
    "tanh": 2,
    "asinh": 2,
    "acosh": 3,
    "atanh": 3,

    # --- 4. 保护型基础算子 ---
    "inv": 1,
    "inv_abs": 2,
    "inv2": 2,
    "inv3": 2,
    "inv_square_abs": 2,
    "sqrt_abs": 2,
    "log_abs": 2,
    "log1p_abs": 3,

    # --- 5. 幂次结构：质量公式、半径、表面项 ---
    "pow13_abs": 2,
    "pow23_abs": 2,
    "pow23": 2,

    # --- 6. Woods-Saxon / Fermi 分布 ---
    "woods_saxon": 3,
    "dws": 4,

    # --- 7. 高斯 / 指数阻尼 / 势函数结构 ---
    "gauss_sq": 2,
    "gauss_std": 3,
    "exp_neg_abs": 2,
    "yukawa": 3,
    "morse_attr": 2,

    # --- 8. 简单阻尼 / 形状结构 ---
    "inv_one_plus": 2,
    "one_minus_sq": 2,
    "sqrt_sq_minus_one": 3,

    # --- 9. 集体模型 / 转动谱 ---
    "rotor": 2,
    "rotor_sq": 3,

    # ==========================================================
    # Binary Operators
    # ==========================================================

    # --- 1. 基础二元算子 ---
    "+": 1,
    "-": 1,
    "*": 1,
    "/": 1,

    # 如果你在 primitive set 中启用了 "^"，这里会自动保留；
    # 如果没启用，则会自动过滤掉。
    "^": 2,

    # --- 2. 核质量 / 结合能 ---
    "symmetry_kernel": 3,
    "isospin_asym": 2,
    "coulomb_kernel": 3,
    "fissility_kernel": 2,

    # --- 3. 半径 / 几何截面 / 反应截面 ---
    "radius_sum13": 2,
    "geom_cross_kernel": 3,

    # --- 4. alpha 衰变 ---
    "alpha_gn_kernel": 2,
    "alpha_brown_kernel": 3,

    # --- 5. 级密度 / 统计模型 ---
    "sqrt_prod": 2,
    "exp_sqrt_prod": 4,
    "backshift": 1,

    # --- 6. Breit-Wigner / Lorentzian ---
    "lorentz_denom": 2,
    "lorentz_kernel": 3,

    # --- 7. 比值型物理结构 ---
    "safe_ratio": 1,
    "inv_sum": 2,
    "product_over_sum": 3,
}

my_complexity_of_operators = _keep_active_dict(my_complexity_of_operators_raw)


# ==============================================================
# my_constraints
# ==============================================================
# 格式：
#   一元算子: "op": max_complexity_inside
#   二元算子: "op": (left_max_complexity, right_max_complexity)
#
# -1 表示不限制。
#
# 注意：
# - 这里保留完整版本，命名为 my_constraints_raw；
# - 真正传给 PySR 的是最后过滤后的 my_constraints。

my_constraints_raw = {
    # ==========================================================
    # Binary Operators
    # ==========================================================

    # --- 基础二元算子 ---
    "+": (-1, -1),
    "-": (-1, -1),
    "*": (-1, -1),
    "/": (-1, -1),

    # --- 幂算子 ---
    # 如果 primitive set 没启用 "^"，这里会自动被过滤掉。
    # 右侧指数 complexity <= 1：
    # 尽量允许 A^0.333, Q^-0.5, Z^0.6；
    # 尽量禁止 A^(N-Z), A^(x+y), A^exp(x)。
    "^": (-1, 1),

    # --- 核质量 / 结合能 ---
    "symmetry_kernel": (6, 6),
    "isospin_asym": (6, 6),
    "coulomb_kernel": (6, 6),
    "fissility_kernel": (6, 6),

    # --- 半径 / 几何截面 / 反应截面 ---
    "radius_sum13": (6, 6),
    "geom_cross_kernel": (6, 6),

    # --- alpha 衰变 ---
    "alpha_gn_kernel": (6, 6),
    "alpha_brown_kernel": (6, 6),

    # --- 级密度 / 统计模型 ---
    "sqrt_prod": (6, 6),
    "exp_sqrt_prod": (5, 5),
    "backshift": (6, 6),

    # --- Breit-Wigner / Lorentzian ---
    "lorentz_denom": (6, 6),
    "lorentz_kernel": (6, 6),

    # --- 比值型物理结构 ---
    "safe_ratio": (6, 6),
    "inv_sum": (6, 6),
    "product_over_sum": (6, 6),

    # ==========================================================
    # Unary Operators
    # ==========================================================

    # --- 简单内置数学算子 ---
    "square": 8,
    "cube": 8,
    "cbrt": 8,
    "sqrt": 8,
    "abs": 8,

    # --- 复杂内置数学函数 ---
    "exp": 5,
    "log": 5,

    # --- 三角函数：视为复杂函数 ---
    "sin": 4,
    "cos": 4,
    "tan": 3,
    "asin": 3,
    "acos": 3,
    "atan": 4,

    # --- 双曲函数：视为复杂函数 ---
    "sinh": 3,
    "cosh": 3,
    "tanh": 3,
    "asinh": 4,
    "acosh": 3,
    "atanh": 3,

    # --- 保护型简单算子 ---
    "inv": 8,
    "inv_abs": 8,
    "inv2": 8,
    "inv3": 8,
    "inv_square_abs": 8,
    "sqrt_abs": 8,

    # --- log 型保护算子：略收紧 ---
    "log_abs": 5,
    "log1p_abs": 5,

    # --- 幂次结构 ---
    "pow13_abs": 8,
    "pow23_abs": 8,
    "pow23": 8,

    # --- Woods-Saxon / Fermi 分布 ---
    "woods_saxon": 5,
    "dws": 5,

    # --- 高斯 / 指数阻尼 / 势函数结构 ---
    "gauss_sq": 5,
    "gauss_std": 5,
    "exp_neg_abs": 5,
    "yukawa": 5,
    "morse_attr": 5,

    # --- 简单阻尼 / 形状结构 ---
    "inv_one_plus": 6,
    "one_minus_sq": 8,
    "sqrt_sq_minus_one": 6,

    # --- 集体模型 / 转动谱 ---
    "rotor": 8,
    "rotor_sq": 6,
}

my_constraints = _keep_active_dict(my_constraints_raw)


# ==============================================================
# my_nested_constraints
# ==============================================================
# 核心原则：
# 1. + - * / 不加 nested 限制；
# 2. ^ 不在 nested_constraints 中禁止，只靠 my_constraints["^"] = (-1, 1) 限制指数；
# 3. 简单数学函数可以进入复杂函数内部；
# 4. exp/log/三角函数/双曲函数/强物理模板都视为复杂函数；
# 5. 复杂函数内部禁止再出现复杂函数；
# 6. 如果某个 operator 没有在 primitive set 中启用，会自动从 nested_constraints 中过滤掉。

# --------------------------------------------------------------
# 1. 简单二元组合器
# --------------------------------------------------------------
# 不写入 nested_constraints，表示默认不限制。

simple_binary_ops = [
    "+",
    "-",
    "*",
    "/",
    "^",
]


# --------------------------------------------------------------
# 2. 简单数学一元算子
# --------------------------------------------------------------
# 这些可以出现在复杂函数内部。
# 即便某些当前没启用，也没关系，后面会自动过滤。

simple_unary_ops = [
    "square",
    "cube",
    "cbrt",
    "sqrt",
    "abs",

    "inv",
    "inv_abs",
    "inv2",
    "inv3",
    "inv_square_abs",
    "sqrt_abs",

    "pow13_abs",
    "pow23_abs",
    "pow23",

    "one_minus_sq",
    "sqrt_sq_minus_one",
    "rotor",
    "rotor_sq",
]


# --------------------------------------------------------------
# 3. 复杂数学一元函数
# --------------------------------------------------------------
# 这些不允许互相嵌套，也不允许直接套强物理模板。

complex_math_unary_ops = [
    "exp",
    "log",
    "log_abs",
    "log1p_abs",

    "sin",
    "cos",
    "tan",
    "asin",
    "acos",
    "atan",

    "sinh",
    "cosh",
    "tanh",
    "asinh",
    "acosh",
    "atanh",
]


# --------------------------------------------------------------
# 4. 强物理模板一元算子
# --------------------------------------------------------------

strong_physics_unary_ops = [
    "woods_saxon",
    "dws",

    "gauss_sq",
    "gauss_std",
    "exp_neg_abs",
    "yukawa",
    "morse_attr",
    "inv_one_plus",
]


# --------------------------------------------------------------
# 5. 强物理模板二元算子
# --------------------------------------------------------------

strong_physics_binary_ops = [
    # --- 核质量 / 结合能 ---
    "symmetry_kernel",
    "isospin_asym",
    "coulomb_kernel",
    "fissility_kernel",

    # --- 半径 / 几何截面 / 反应截面 ---
    "radius_sum13",
    "geom_cross_kernel",

    # --- alpha 衰变 ---
    "alpha_gn_kernel",
    "alpha_brown_kernel",

    # --- 级密度 / 统计模型 ---
    "sqrt_prod",
    "exp_sqrt_prod",
    "backshift",

    # --- Breit-Wigner / Lorentzian ---
    "lorentz_denom",
    "lorentz_kernel",

    # --- 比值型物理结构 ---
    "safe_ratio",
    "inv_sum",
    "product_over_sum",
]


# --------------------------------------------------------------
# 6. 复杂函数总表
# --------------------------------------------------------------
# 注意：
# - 这里先写完整列表；
# - 后面构造 nested_constraints 时会自动只保留 active operator。

complex_ops = (
    complex_math_unary_ops
    + strong_physics_unary_ops
    + strong_physics_binary_ops
)


# --------------------------------------------------------------
# 7. 构造 nested_constraints
# --------------------------------------------------------------

my_nested_constraints = {}

for op_outer in _active_ops(complex_ops):
    my_nested_constraints[op_outer] = {}

    for op_inner in _active_ops(complex_ops):
        my_nested_constraints[op_outer][op_inner] = 0


# --------------------------------------------------------------
# 8. 最终清理 nested_constraints
# --------------------------------------------------------------
# 这一步是双保险：
# - 删除没有启用的 outer operator；
# - 删除没有启用的 inner operator；
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

