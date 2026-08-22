# region libraries
# ======================================================
# libraries
# ======================================================
import numpy as np
import pandas as pd
import sympy as sp
# endregion

# region operators

# ==============================================================
# Binary Operators
# ==============================================================
my_binary_operators = [
    # ==========================================================
    # 基础算子
    # ==========================================================
    "+",
    "-",
    "*",
    "/",

    # ==========================================================
    # 核质量 / 结合能
    # ==========================================================
    "symmetry_kernel(x, y) = (x - y)^2 / (abs(x + y) + 1e-12)",          # (N-Z)^2/A
    "isospin_asym(x, y) = (x - y) / (abs(x + y) + 1e-12)",              # I=(N-Z)/A
    "coulomb_kernel(x, y) = x * (x - 1) / (abs(y)^(1/3) + 1e-12)",      # Z(Z-1)/A^(1/3)
    "fissility_kernel(x, y) = x^2 / (abs(y) + 1e-12)",                 # Z^2/A

    # ==========================================================
    # 半径 / 几何截面 / 反应截面
    # ==========================================================
    "radius_sum13(x, y) = abs(x)^(1/3) + abs(y)^(1/3)",                 # Ap^(1/3)+At^(1/3)
    "geom_cross_kernel(x, y) = (abs(x)^(1/3) + abs(y)^(1/3))^2",        # 几何截面核心

    # ==========================================================
    # α 衰变
    # ==========================================================
    "alpha_gn_kernel(x, y) = x / (sqrt(abs(y)) + 1e-12)",               # Z/sqrt(Q_alpha)
    "alpha_brown_kernel(x, y) = abs(x)^0.6 / (sqrt(abs(y)) + 1e-12)",   # Z^0.6/sqrt(Q_alpha)

    # ==========================================================
    # 级密度 / 统计模型
    # ==========================================================
    "sqrt_prod(x, y) = sqrt(abs(x * y))",                               # sqrt(aU)
    "exp_sqrt_prod(x, y) = exp(min(80.0, 2.0 * sqrt(abs(x * y))))",     # exp(2sqrt(aU))
    "backshift(x, y) = x - y",                                          # U-Delta

    # ==========================================================
    # Breit-Wigner / Lorentzian
    # ==========================================================
    "lorentz_denom(x, y) = x^2 + y^2 / 4",                              # (E-Er)^2 + Gamma^2/4
    "lorentz_kernel(x, y) = 1 / (x^2 + y^2 / 4 + 1e-12)",               # Lorentzian core

    # ==========================================================
    # 比值型物理结构
    # ==========================================================
    "safe_ratio(x, y) = x / (abs(y) + 1e-12)",
    "inv_sum(x, y) = 1 / (abs(x + y) + 1e-12)",
    "product_over_sum(x, y) = x * y / (abs(x + y) + 1e-12)",

    # ==========================================================
    # 相对论 / 几何 / 统计因子结构
    # 来自第二套算子，即使原来被注释，也纳入最终 primitive set
    # ==========================================================
    "beta_sq(x, y) = x^2 / (y^2 + 1e-12)",                                      # beta^2
    "lorentz_gamma(x, y) = 1 / (sqrt(abs(1 - x^2 / (y^2 + 1e-12))) + 1e-12)",   # gamma
    "diff_inv(x, y) = 1 / (y + 1e-12) - 1 / (x + 1e-12)",                       # 1/y - 1/x
    "hypot_2d(x, y) = sqrt(abs(x^2 + y^2))",                                    # sqrt(x^2+y^2)
    "inv_lorentz(x, y) = sqrt(abs(1 - x^2 / (y^2 + 1e-12)))",                   # sqrt(|1-beta^2|)
    "sum_sq(x, y) = x^2 + y^2",                                                 # x^2+y^2
    "cos_diff(x, y) = cos(x - y)",                                              # cos(x-y)

    # ==========================================================
    # 比例 / 平均 / 分布函数结构
    # ==========================================================
    "inv_one_minus_ratio(x, y) = 1 / (1 - x / (y + 1e-12) + 1e-12)",             # 1/(1-x/y)
    "geom_mean(x, y) = sqrt(abs(x * y))",                                       # sqrt(|xy|)
    "exp_neg_ratio(x, y) = exp(clamp(-x / (y + 1e-12), -80.0, 80.0))",          # exp(-x/y)
    "log_ratio(x, y) = log(abs(x / (y + 1e-12)) + 1e-12)",                      # log(|x/y|)
    "parallel_sum(x, y) = x * y / (x + y + 1e-12)",                             # xy/(x+y)

    # ==========================================================
    # Bose-Einstein / Fermi-Dirac / Logistic-like 统计因子
    # ==========================================================
    "bose_einstein(x, y) = 1 / (exp(clamp(x / (y + 1e-12), -80.0, 80.0)) - 1 + 1e-12)",
    "logistic_prod(x, y) = 1 / (exp(clamp(-x * y, -80.0, 80.0)) + 1)",
    "fermi_dirac(x, y) = 1 / (exp(clamp(x / (y + 1e-12), -80.0, 80.0)) + 1)",
    "bose_einstein_neg(x, y) = 1 / (exp(clamp(-x * y, -80.0, 80.0)) - 1 + 1e-12)",
]


# ==============================================================
# Unary Operators
# ==============================================================
my_unary_operators = [
    # ==========================================================
    # PySR / Julia 内置算子
    # ==========================================================
    "square",    # x^2
    "cube",      # x^3
    "cbrt",      # x^(1/3)
    "sqrt",      # sqrt(x)
    "abs",       # |x|
    "exp",       # exp(x)
    "log",       # log(x)

    # ==========================================================
    # 三角函数
    # ==========================================================
    "sin",
    "cos",
    "tan",
    "asin",
    "acos",
    "atan",

    # ==========================================================
    # 双曲函数
    # ==========================================================
    "sinh",
    "cosh",
    "tanh",
    "asinh",
    "acosh",
    "atanh",

    # ==========================================================
    # 保护型基础算子
    # ==========================================================
    "inv(x) = 1 / (x + 1e-12)",                         # 保护倒数
    "inv_abs(x) = 1 / (abs(x) + 1e-12)",                 # 绝对值保护倒数
    "inv2(x) = 1 / (x^2 + 1e-12)",                       # 平方反比
    "inv3(x) = 1 / (x^3 + 1e-12)",                       # 立方反比
    "inv_square_abs(x) = 1 / (x^2 + 1e-12)",             # 平方保护倒数
    "sqrt_abs(x) = sqrt(abs(x))",                        # sqrt(|x|)
    "log_abs(x) = log(abs(x) + 1e-12)",                  # log(|x|)
    "log1p_abs(x) = log(abs(1 + x) + 1e-12)",            # log(|1+x|)

    # ==========================================================
    # 幂次结构：质量公式、半径、表面项
    # ==========================================================
    "pow13_abs(x) = abs(x)^(1/3)",                       # |x|^(1/3)
    "pow23_abs(x) = abs(x)^(2/3)",                       # |x|^(2/3)
    "pow23(x) = cbrt(x)^2",                              # x^(2/3)，允许负数输入

    # ==========================================================
    # Woods-Saxon / Fermi 分布
    # ==========================================================
    "woods_saxon(x) = 1 / (1 + exp(clamp(x, -80.0, 80.0)))",
    "dws(x) = exp(clamp(x, -80.0, 80.0)) / (1 + exp(clamp(x, -80.0, 80.0)))^2",

    # ==========================================================
    # 高斯 / 指数阻尼 / 势函数结构
    # ==========================================================
    "gauss_sq(x) = exp(-min(80.0, x^2))",                 # exp(-x^2)
    "gauss_std(x) = exp(-min(80.0, x^2 / 2))",            # exp(-x^2/2)
    "exp_neg_abs(x) = exp(-abs(x))",                      # exp(-|x|)
    "yukawa(x) = exp(-abs(x)) / (abs(x) + 1e-12)",        # exp(-|x|)/|x|
    "morse_attr(x) = 1 - exp(-abs(x))",                   # 1-exp(-|x|)

    # ==========================================================
    # 第二套中与第一套定义不同的 signed/raw 版本
    # 为了完整继承第二套物理含义，保留为新名字
    # ==========================================================
    "yukawa_signed(x) = exp(clamp(-x, -80.0, 80.0)) / (abs(x) + 1e-12)",   # exp(-x)/|x|
    "morse_attr_signed(x) = 1 - exp(clamp(-x, -80.0, 80.0))",             # 1-exp(-x)
    "inv_one_plus_signed(x) = 1 / (1 + x + 1e-12)",                       # 1/(1+x)

    # ==========================================================
    # 简单阻尼 / 形状结构
    # ==========================================================
    "inv_one_plus(x) = 1 / (1 + abs(x))",                 # 1/(1+|x|)
    "one_minus_sq(x) = 1 - x^2",                          # 1-x^2
    "sqrt_sq_minus_one(x) = sqrt(abs(x^2 - 1))",          # sqrt(|x^2-1|)

    # ==========================================================
    # 集体模型 / 转动谱
    # ==========================================================
    "rotor(x) = x * (x + 1)",                             # J(J+1)
    "rotor_sq(x) = (x * (x + 1))^2",                      # [J(J+1)]^2
]

# ==============================================================
# SymPy Mappings
# ==============================================================
my_extra_sympy_mappings = {
    # ==========================================================
    # 一元算子翻译
    # ==========================================================
    "inv": lambda x: 1 / (x + sp.Float("1e-12")),
    "inv_abs": lambda x: 1 / (sp.Abs(x) + sp.Float("1e-12")),
    "inv2": lambda x: 1 / (x**2 + sp.Float("1e-12")),
    "inv3": lambda x: 1 / (x**3 + sp.Float("1e-12")),
    "inv_square_abs": lambda x: 1 / (x**2 + sp.Float("1e-12")),

    "sqrt_abs": lambda x: sp.sqrt(sp.Abs(x)),
    "log_abs": lambda x: sp.log(sp.Abs(x) + sp.Float("1e-12")),
    "log1p_abs": lambda x: sp.log(sp.Abs(1 + x) + sp.Float("1e-12")),

    "pow13_abs": lambda x: sp.Abs(x)**sp.Rational(1, 3),
    "pow23_abs": lambda x: sp.Abs(x)**sp.Rational(2, 3),

    # 注意：这里不用 x**(2/3)，避免 SymPy 对负数走复数分支
    "pow23": lambda x: sp.real_root(x, 3)**2,

    "woods_saxon": lambda x: 1 / (1 + sp.exp(sp.Min(sp.Float(80), sp.Max(sp.Float(-80), x)))),
    "dws": lambda x: (
        sp.exp(sp.Min(sp.Float(80), sp.Max(sp.Float(-80), x)))
        / (1 + sp.exp(sp.Min(sp.Float(80), sp.Max(sp.Float(-80), x))))**2
    ),

    "gauss_sq": lambda x: sp.exp(-sp.Min(sp.Float(80), x**2)),
    "gauss_std": lambda x: sp.exp(-sp.Min(sp.Float(80), x**2 / 2)),
    "exp_neg_abs": lambda x: sp.exp(-sp.Abs(x)),
    "yukawa": lambda x: sp.exp(-sp.Abs(x)) / (sp.Abs(x) + sp.Float("1e-12")),
    "morse_attr": lambda x: 1 - sp.exp(-sp.Abs(x)),

    "yukawa_signed": lambda x: (
        sp.exp(sp.Min(sp.Float(80), sp.Max(sp.Float(-80), -x)))
        / (sp.Abs(x) + sp.Float("1e-12"))
    ),
    "morse_attr_signed": lambda x: 1 - sp.exp(sp.Min(sp.Float(80), sp.Max(sp.Float(-80), -x))),
    "inv_one_plus_signed": lambda x: 1 / (1 + x + sp.Float("1e-12")),

    "inv_one_plus": lambda x: 1 / (1 + sp.Abs(x)),
    "one_minus_sq": lambda x: 1 - x**2,
    "sqrt_sq_minus_one": lambda x: sp.sqrt(sp.Abs(x**2 - 1)),

    "rotor": lambda x: x * (x + 1),
    "rotor_sq": lambda x: (x * (x + 1))**2,

    # ==========================================================
    # 二元算子翻译：核质量 / 结合能
    # ==========================================================
    "symmetry_kernel": lambda x, y: (x - y)**2 / (sp.Abs(x + y) + sp.Float("1e-12")),
    "isospin_asym": lambda x, y: (x - y) / (sp.Abs(x + y) + sp.Float("1e-12")),
    "coulomb_kernel": lambda x, y: x * (x - 1) / (sp.Abs(y)**sp.Rational(1, 3) + sp.Float("1e-12")),
    "fissility_kernel": lambda x, y: x**2 / (sp.Abs(y) + sp.Float("1e-12")),

    # ==========================================================
    # 二元算子翻译：半径 / 几何截面
    # ==========================================================
    "radius_sum13": lambda x, y: sp.Abs(x)**sp.Rational(1, 3) + sp.Abs(y)**sp.Rational(1, 3),
    "geom_cross_kernel": lambda x, y: (
        sp.Abs(x)**sp.Rational(1, 3) + sp.Abs(y)**sp.Rational(1, 3)
    )**2,

    # ==========================================================
    # 二元算子翻译：α 衰变
    # ==========================================================
    "alpha_gn_kernel": lambda x, y: x / (sp.sqrt(sp.Abs(y)) + sp.Float("1e-12")),
    "alpha_brown_kernel": lambda x, y: sp.Abs(x)**sp.Float("0.6") / (
        sp.sqrt(sp.Abs(y)) + sp.Float("1e-12")
    ),

    # ==========================================================
    # 二元算子翻译：级密度 / 统计模型
    # ==========================================================
    "sqrt_prod": lambda x, y: sp.sqrt(sp.Abs(x * y)),
    "exp_sqrt_prod": lambda x, y: sp.exp(sp.Min(sp.Float(80), 2 * sp.sqrt(sp.Abs(x * y)))),
    "backshift": lambda x, y: x - y,

    # ==========================================================
    # 二元算子翻译：Breit-Wigner / Lorentzian
    # ==========================================================
    "lorentz_denom": lambda x, y: x**2 + y**2 / 4,
    "lorentz_kernel": lambda x, y: 1 / (x**2 + y**2 / 4 + sp.Float("1e-12")),

    # ==========================================================
    # 二元算子翻译：比值型物理结构
    # ==========================================================
    "safe_ratio": lambda x, y: x / (sp.Abs(y) + sp.Float("1e-12")),
    "inv_sum": lambda x, y: 1 / (sp.Abs(x + y) + sp.Float("1e-12")),
    "product_over_sum": lambda x, y: x * y / (sp.Abs(x + y) + sp.Float("1e-12")),

    # ==========================================================
    # 二元算子翻译：相对论 / 几何结构
    # ==========================================================
    "beta_sq": lambda x, y: x**2 / (y**2 + sp.Float("1e-12")),
    "lorentz_gamma": lambda x, y: 1 / (
        sp.sqrt(sp.Abs(1 - x**2 / (y**2 + sp.Float("1e-12")))) + sp.Float("1e-12")
    ),
    "diff_inv": lambda x, y: 1 / (y + sp.Float("1e-12")) - 1 / (x + sp.Float("1e-12")),
    "hypot_2d": lambda x, y: sp.sqrt(sp.Abs(x**2 + y**2)),
    "inv_lorentz": lambda x, y: sp.sqrt(sp.Abs(1 - x**2 / (y**2 + sp.Float("1e-12")))),
    "sum_sq": lambda x, y: x**2 + y**2,
    "cos_diff": lambda x, y: sp.cos(x - y),

    # ==========================================================
    # 二元算子翻译：比例 / 平均 / 分布函数
    # ==========================================================
    "inv_one_minus_ratio": lambda x, y: 1 / (
        1 - x / (y + sp.Float("1e-12")) + sp.Float("1e-12")
    ),
    "geom_mean": lambda x, y: sp.sqrt(sp.Abs(x * y)),
    "exp_neg_ratio": lambda x, y: sp.exp(
        sp.Min(sp.Float(80), sp.Max(sp.Float(-80), -x / (y + sp.Float("1e-12"))))
    ),
    "log_ratio": lambda x, y: sp.log(sp.Abs(x / (y + sp.Float("1e-12"))) + sp.Float("1e-12")),
    "parallel_sum": lambda x, y: x * y / (x + y + sp.Float("1e-12")),

    "bose_einstein": lambda x, y: 1 / (
        sp.exp(sp.Min(sp.Float(80), sp.Max(sp.Float(-80), x / (y + sp.Float("1e-12")))))
        - 1
        + sp.Float("1e-12")
    ),
    "logistic_prod": lambda x, y: 1 / (
        sp.exp(sp.Min(sp.Float(80), sp.Max(sp.Float(-80), -x * y))) + 1
    ),
    "fermi_dirac": lambda x, y: 1 / (
        sp.exp(sp.Min(sp.Float(80), sp.Max(sp.Float(-80), x / (y + sp.Float("1e-12")))))
        + 1
    ),
    "bose_einstein_neg": lambda x, y: 1 / (
        sp.exp(sp.Min(sp.Float(80), sp.Max(sp.Float(-80), -x * y)))
        - 1
        + sp.Float("1e-12")
    ),
}
# endregion

# region constraints
# ==============================================================
# Helper: extract active operator names from PySR operator strings
# ==============================================================
# 目的：
# - 如果以后你临时注释掉某些算子，这段代码会自动过滤掉不存在的算子；
# - 避免 constraints / nested_constraints 中出现当前 primitive set 没启用的算子名。

def _get_pysr_operator_name(operator_string):
    operator_string = operator_string.strip()
    if "(" in operator_string and "=" in operator_string:
        return operator_string.split("(", 1)[0].strip()
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


# ==============================================================
# 1. Operator Complexities
# ==============================================================
# 基本思想：
# - 基础四则运算复杂度低；
# - 简单代数算子复杂度低；
# - 物理宏算子复杂度中等；
# - exp/log/trig/hyperbolic/统计分布/Woods-Saxon/Gaussian/Yukawa 复杂度较高；
# - 含 exp/log/ratio/singularity 的算子要更贵，防止搜索空间爆炸。

my_complexity_of_operators_raw = {
    # ==========================================================
    # Basic binary operators
    # ==========================================================
    "+": 1,
    "-": 1,
    "*": 1,
    "/": 2,

    # ==========================================================
    # Nuclear mass / binding-energy kernels
    # ==========================================================
    "symmetry_kernel": 4,
    "isospin_asym": 3,
    "coulomb_kernel": 4,
    "fissility_kernel": 3,

    # ==========================================================
    # Radius / geometry / reaction-cross-section kernels
    # ==========================================================
    "radius_sum13": 3,
    "geom_cross_kernel": 4,

    # ==========================================================
    # Alpha-decay kernels
    # ==========================================================
    "alpha_gn_kernel": 4,
    "alpha_brown_kernel": 5,

    # ==========================================================
    # Level-density / statistical-model kernels
    # ==========================================================
    "sqrt_prod": 3,
    "exp_sqrt_prod": 6,
    "backshift": 1,

    # ==========================================================
    # Breit-Wigner / Lorentzian kernels
    # ==========================================================
    "lorentz_denom": 3,
    "lorentz_kernel": 5,

    # ==========================================================
    # Ratio-type physical structures
    # ==========================================================
    "safe_ratio": 3,
    "inv_sum": 3,
    "product_over_sum": 4,

    # ==========================================================
    # Relativity / geometry / phase structures
    # ==========================================================
    "beta_sq": 3,
    "lorentz_gamma": 5,
    "diff_inv": 4,
    "hypot_2d": 3,
    "inv_lorentz": 4,
    "sum_sq": 2,
    "cos_diff": 4,

    # ==========================================================
    # Ratio / average / distribution-like binary structures
    # ==========================================================
    "inv_one_minus_ratio": 5,
    "geom_mean": 3,
    "exp_neg_ratio": 5,
    "log_ratio": 5,
    "parallel_sum": 4,

    # ==========================================================
    # Bose-Einstein / Fermi-Dirac / Logistic-like structures
    # ==========================================================
    "bose_einstein": 7,
    "logistic_prod": 6,
    "fermi_dirac": 6,
    "bose_einstein_neg": 7,

    # ==========================================================
    # Built-in unary algebraic operators
    # ==========================================================
    "square": 1,
    "cube": 1,
    "cbrt": 1,
    "sqrt": 2,
    "abs": 1,

    # ==========================================================
    # Elementary complex unary functions
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
    # Protected inverse / log / sqrt operators
    # ==========================================================
    "inv": 2,
    "inv_abs": 2,
    "inv2": 3,
    "inv3": 3,
    "inv_square_abs": 3,
    "sqrt_abs": 2,
    "log_abs": 3,
    "log1p_abs": 3,

    # ==========================================================
    # Power-law structures
    # ==========================================================
    "pow13_abs": 2,
    "pow23_abs": 2,
    "pow23": 2,

    # ==========================================================
    # Woods-Saxon / Fermi-distribution structures
    # ==========================================================
    "woods_saxon": 5,
    "dws": 6,

    # ==========================================================
    # Gaussian / damping / potential structures
    # ==========================================================
    "gauss_sq": 5,
    "gauss_std": 5,
    "exp_neg_abs": 4,
    "yukawa": 6,
    "morse_attr": 4,

    # ==========================================================
    # Signed/raw versions inherited from the second operator set
    # ==========================================================
    "yukawa_signed": 6,
    "morse_attr_signed": 4,
    "inv_one_plus_signed": 3,

    # ==========================================================
    # Shape / damping / collective-model structures
    # ==========================================================
    "inv_one_plus": 3,
    "one_minus_sq": 2,
    "sqrt_sq_minus_one": 3,
    "rotor": 2,
    "rotor_sq": 3,
}

my_complexity_of_operators = _keep_active_dict(my_complexity_of_operators_raw)


# ==============================================================
# 2. Basic Constraints
# ==============================================================
# 格式：
# - unary operator: "op": max_complexity_inside
# - binary operator: "op": (left_max_complexity, right_max_complexity)
#
# -1 表示不限制。
#
# 总原则：
# 1. +, -, * 基本不限制；
# 2. / 的分母复杂度适当限制，防止出现特别复杂的奇异结构；
# 3. 物理宏算子的两个输入都限制为中等复杂度；
# 4. exp/log/trig/hyperbolic 的内部表达式限制更严格；
# 5. 统计分布类、Woods-Saxon、Gaussian、Yukawa 内部限制更严格。

my_constraints_raw = {
    # ==========================================================
    # Basic binary operators
    # ==========================================================
    "+": (-1, -1),
    "-": (-1, -1),
    "*": (-1, -1),
    "/": (-1, 8),

    # ==========================================================
    # Nuclear mass / binding-energy kernels
    # ==========================================================
    "symmetry_kernel": (6, 6),
    "isospin_asym": (6, 6),
    "coulomb_kernel": (6, 6),
    "fissility_kernel": (6, 6),

    # ==========================================================
    # Radius / geometry / reaction-cross-section kernels
    # ==========================================================
    "radius_sum13": (6, 6),
    "geom_cross_kernel": (6, 6),

    # ==========================================================
    # Alpha-decay kernels
    # ==========================================================
    "alpha_gn_kernel": (6, 6),
    "alpha_brown_kernel": (6, 6),

    # ==========================================================
    # Level-density / statistical-model kernels
    # ==========================================================
    "sqrt_prod": (6, 6),
    "exp_sqrt_prod": (4, 4),
    "backshift": (8, 8),

    # ==========================================================
    # Breit-Wigner / Lorentzian kernels
    # ==========================================================
    "lorentz_denom": (6, 6),
    "lorentz_kernel": (5, 5),

    # ==========================================================
    # Ratio-type physical structures
    # ==========================================================
    "safe_ratio": (8, 6),
    "inv_sum": (6, 6),
    "product_over_sum": (6, 6),

    # ==========================================================
    # Relativity / geometry / phase structures
    # ==========================================================
    "beta_sq": (6, 6),
    "lorentz_gamma": (4, 4),
    "diff_inv": (5, 5),
    "hypot_2d": (6, 6),
    "inv_lorentz": (4, 4),
    "sum_sq": (8, 8),
    "cos_diff": (5, 5),

    # ==========================================================
    # Ratio / average / distribution-like binary structures
    # ==========================================================
    "inv_one_minus_ratio": (5, 5),
    "geom_mean": (6, 6),
    "exp_neg_ratio": (5, 5),
    "log_ratio": (5, 5),
    "parallel_sum": (6, 6),

    # ==========================================================
    # Bose-Einstein / Fermi-Dirac / Logistic-like structures
    # ==========================================================
    "bose_einstein": (4, 4),
    "logistic_prod": (4, 4),
    "fermi_dirac": (4, 4),
    "bose_einstein_neg": (4, 4),

    # ==========================================================
    # Built-in unary algebraic operators
    # ==========================================================
    "square": 8,
    "cube": 8,
    "cbrt": 8,
    "sqrt": 6,
    "abs": 8,

    # ==========================================================
    # Elementary complex unary functions
    # ==========================================================
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
    # Protected inverse / log / sqrt operators
    # ==========================================================
    "inv": 8,
    "inv_abs": 8,
    "inv2": 7,
    "inv3": 7,
    "inv_square_abs": 7,
    "sqrt_abs": 8,
    "log_abs": 6,
    "log1p_abs": 6,

    # ==========================================================
    # Power-law structures
    # ==========================================================
    "pow13_abs": 8,
    "pow23_abs": 8,
    "pow23": 8,

    # ==========================================================
    # Woods-Saxon / Fermi-distribution structures
    # ==========================================================
    "woods_saxon": 5,
    "dws": 5,

    # ==========================================================
    # Gaussian / damping / potential structures
    # ==========================================================
    "gauss_sq": 5,
    "gauss_std": 5,
    "exp_neg_abs": 5,
    "yukawa": 5,
    "morse_attr": 5,

    # ==========================================================
    # Signed/raw versions inherited from the second operator set
    # ==========================================================
    "yukawa_signed": 5,
    "morse_attr_signed": 5,
    "inv_one_plus_signed": 6,

    # ==========================================================
    # Shape / damping / collective-model structures
    # ==========================================================
    "inv_one_plus": 6,
    "one_minus_sq": 8,
    "sqrt_sq_minus_one": 6,
    "rotor": 8,
    "rotor_sq": 8,
}

my_constraints = _keep_active_dict(my_constraints_raw)


# ==============================================================
# 3. Nested Constraints
# ==============================================================
# 含义：
# nested_constraints[outer][inner] = n
#
# 表示：
# - 当 outer 是外层算子时，inner 最多可以在它的子树中出现 n 次；
# - n = 0 表示完全禁止 inner 出现在 outer 里面；
# - n = 1 表示最多出现一次。
#
# 核心原则：
# 1. 基础四则运算不做 nested 限制；
# 2. 简单代数算子可以进入复杂函数内部；
# 3. 复杂函数不能互相嵌套；
# 4. 物理宏算子不能互相作为输入；
# 5. exp/log/trig/hyperbolic/统计分布/Woods-Saxon/Gaussian/Yukawa 之间不互相嵌套；
# 6. inverse 类算子不互相套娃，例如 inv(inv(x))；
# 7. 幂次/根式结构限制自我嵌套，避免 square(square(square(x))) 这类无意义高幂。

basic_binary_ops = [
    "+",
    "-",
    "*",
    "/",
]

nuclear_binary_ops = [
    "symmetry_kernel",
    "isospin_asym",
    "coulomb_kernel",
    "fissility_kernel",
]

geometry_binary_ops = [
    "radius_sum13",
    "geom_cross_kernel",
]

alpha_binary_ops = [
    "alpha_gn_kernel",
    "alpha_brown_kernel",
]

level_density_binary_ops = [
    "sqrt_prod",
    "exp_sqrt_prod",
    "backshift",
]

lorentzian_binary_ops = [
    "lorentz_denom",
    "lorentz_kernel",
]

ratio_binary_ops = [
    "safe_ratio",
    "inv_sum",
    "product_over_sum",
]

relativity_binary_ops = [
    "beta_sq",
    "lorentz_gamma",
    "diff_inv",
    "hypot_2d",
    "inv_lorentz",
    "sum_sq",
    "cos_diff",
]

distribution_binary_ops = [
    "inv_one_minus_ratio",
    "geom_mean",
    "exp_neg_ratio",
    "log_ratio",
    "parallel_sum",
    "bose_einstein",
    "logistic_prod",
    "fermi_dirac",
    "bose_einstein_neg",
]

composite_binary_ops = (
    nuclear_binary_ops
    + geometry_binary_ops
    + alpha_binary_ops
    + level_density_binary_ops
    + lorentzian_binary_ops
    + ratio_binary_ops
    + relativity_binary_ops
    + distribution_binary_ops
)

simple_algebra_unary_ops = [
    "square",
    "cube",
    "cbrt",
    "sqrt",
    "abs",
    "sqrt_abs",
    "one_minus_sq",
    "sqrt_sq_minus_one",
    "rotor",
    "rotor_sq",
]

inverse_unary_ops = [
    "inv",
    "inv_abs",
    "inv2",
    "inv3",
    "inv_square_abs",
    "inv_one_plus",
    "inv_one_plus_signed",
]

power_unary_ops = [
    "pow13_abs",
    "pow23_abs",
    "pow23",
]

log_unary_ops = [
    "log",
    "log_abs",
    "log1p_abs",
]

exp_unary_ops = [
    "exp",
    "exp_neg_abs",
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

shape_potential_unary_ops = [
    "woods_saxon",
    "dws",
    "gauss_sq",
    "gauss_std",
    "yukawa",
    "morse_attr",
    "yukawa_signed",
    "morse_attr_signed",
]

elementary_complex_unary_ops = (
    exp_unary_ops
    + log_unary_ops
    + trig_unary_ops
    + hyperbolic_unary_ops
)

complex_unary_ops = (
    elementary_complex_unary_ops
    + shape_potential_unary_ops
)

complex_binary_ops = [
    "exp_sqrt_prod",
    "cos_diff",
    "exp_neg_ratio",
    "log_ratio",
    "bose_einstein",
    "logistic_prod",
    "fermi_dirac",
    "bose_einstein_neg",
]

high_risk_ops = (
    complex_unary_ops
    + complex_binary_ops
)


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


my_nested_constraints = {}


# ==============================================================
# 3.1 复杂一元函数内部禁止再出现复杂函数
# ==============================================================
# 禁止例子：
# - exp(exp(x))
# - exp(log(x))
# - log(exp(x))
# - sin(cos(x))
# - tanh(exp(x))
# - woods_saxon(gauss_sq(x))
# - yukawa(log_abs(x))

_forbid_nested(
    my_nested_constraints,
    outer_ops=complex_unary_ops,
    inner_ops=high_risk_ops,
)


# ==============================================================
# 3.2 复杂一元函数内部禁止出现物理宏二元算子
# ==============================================================
# 禁止例子：
# - exp(symmetry_kernel(N, Z))
# - log(coulomb_kernel(Z, A))
# - sin(lorentz_kernel(E, Gamma))
#
# 如果你确实需要 exp(x/y)，应该让 PySR 用基础 "/" 或专门的 exp_neg_ratio，
# 而不是 exp(safe_ratio(...)) 这种宏算子套宏算子。

_forbid_nested(
    my_nested_constraints,
    outer_ops=complex_unary_ops,
    inner_ops=composite_binary_ops,
)


# ==============================================================
# 3.3 物理宏二元算子内部禁止再出现物理宏二元算子
# ==============================================================
# 禁止例子：
# - coulomb_kernel(symmetry_kernel(N, Z), A)
# - lorentz_kernel(alpha_gn_kernel(Z, Q), Gamma)
# - bose_einstein(lorentz_kernel(E, G), T)
#
# 允许：
# - symmetry_kernel(N, Z)
# - coulomb_kernel(Z, A)
# - safe_ratio(x, y)
# - product_over_sum(x, y)
#
# 也允许用 +, -, *, / 在宏算子外面组合：
# - a * coulomb_kernel(Z, A) + b * symmetry_kernel(N, Z)

_forbid_nested(
    my_nested_constraints,
    outer_ops=composite_binary_ops,
    inner_ops=composite_binary_ops,
)


# ==============================================================
# 3.4 物理宏二元算子内部禁止出现复杂函数
# ==============================================================
# 禁止例子：
# - symmetry_kernel(exp(N), Z)
# - coulomb_kernel(log_abs(Z), A)
# - lorentz_kernel(sin(E), Gamma)
#
# 允许简单代数修饰：
# - symmetry_kernel(square(N), Z)
# - safe_ratio(abs(x), sqrt_abs(y))

_forbid_nested(
    my_nested_constraints,
    outer_ops=composite_binary_ops,
    inner_ops=high_risk_ops,
)


# ==============================================================
# 3.5 简单代数外壳禁止包裹复杂函数
# ==============================================================
# 禁止例子：
# - square(exp(x))
# - cube(log(x))
# - sqrt(sin(x))
# - pow23_abs(woods_saxon(x))
#
# 这样做的原因：
# - square(exp(x)) 本质上等价于 exp(2x)，容易隐藏复杂度；
# - square(sin(x))、cube(tanh(x)) 这类结构通常会迅速膨胀搜索空间。

_forbid_nested(
    my_nested_constraints,
    outer_ops=simple_algebra_unary_ops + power_unary_ops,
    inner_ops=high_risk_ops,
)


# ==============================================================
# 3.6 inverse 类算子禁止互相套娃
# ==============================================================
# 禁止例子：
# - inv(inv(x))
# - inv_abs(inv2(x))
# - inv2(inv_one_plus(x))
#
# 这类结构多数可以化简，且会制造不必要的奇异性。

_forbid_nested(
    my_nested_constraints,
    outer_ops=inverse_unary_ops,
    inner_ops=inverse_unary_ops,
)


# ==============================================================
# 3.7 inverse 类算子禁止包裹复杂函数
# ==============================================================
# 禁止例子：
# - inv(exp(x))
# - inv_abs(log_abs(x))
# - inv_one_plus(woods_saxon(x))
#
# 如果需要 1 / exp(x)，应让模型通过 exp(-x) 或专门阻尼结构表达。

_forbid_nested(
    my_nested_constraints,
    outer_ops=inverse_unary_ops,
    inner_ops=high_risk_ops,
)


# ==============================================================
# 3.8 幂次 / 根式结构之间限制嵌套
# ==============================================================
# 禁止或限制：
# - square(square(square(x)))
# - cube(cube(x))
# - cbrt(cbrt(x))
# - pow23(pow23(x))
# - pow13_abs(pow23_abs(x))
#
# 注意：
# - square(square(x)) 允许一次，用于表达 x^4；
# - cube(cube(x)) 不允许，因为 x^9 通常过高且不稳定；
# - cbrt / pow13 / pow23 之间不互相嵌套，避免分数幂过度复杂化。

root_power_ops = [
    "square",
    "cube",
    "cbrt",
    "sqrt",
    "sqrt_abs",
    "pow13_abs",
    "pow23_abs",
    "pow23",
]

_forbid_nested(
    my_nested_constraints,
    outer_ops=root_power_ops,
    inner_ops=root_power_ops,
)

# 例外：允许 square(square(x)) 一次，用来表示 x^4
if "square" in _active_operator_names:
    my_nested_constraints.setdefault("square", {})
    if "square" in _active_operator_names:
        my_nested_constraints["square"]["square"] = 1


# ==============================================================
# 3.9 log 类之间禁止互相嵌套
# ==============================================================
# 禁止例子：
# - log(log_abs(x))
# - log_abs(log1p_abs(x))
# - log1p_abs(log(x))

_forbid_nested(
    my_nested_constraints,
    outer_ops=log_unary_ops,
    inner_ops=log_unary_ops,
)


# ==============================================================
# 3.10 exp 类之间禁止互相嵌套
# ==============================================================
# 禁止例子：
# - exp(exp(x))
# - exp(exp_neg_abs(x))
# - exp_neg_abs(exp(x))

_forbid_nested(
    my_nested_constraints,
    outer_ops=exp_unary_ops,
    inner_ops=exp_unary_ops,
)


# ==============================================================
# 3.11 三角函数之间禁止互相嵌套
# ==============================================================
# 禁止例子：
# - sin(cos(x))
# - tan(sin(x))
# - asin(cos(x))

_forbid_nested(
    my_nested_constraints,
    outer_ops=trig_unary_ops,
    inner_ops=trig_unary_ops,
)


# ==============================================================
# 3.12 双曲函数之间禁止互相嵌套
# ==============================================================
# 禁止例子：
# - sinh(cosh(x))
# - tanh(sinh(x))
# - atanh(tanh(x))

_forbid_nested(
    my_nested_constraints,
    outer_ops=hyperbolic_unary_ops,
    inner_ops=hyperbolic_unary_ops,
)


# ==============================================================
# 3.13 三角函数与双曲函数之间禁止互相嵌套
# ==============================================================
# 禁止例子：
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


# ==============================================================
# 3.14 Woods-Saxon / Gaussian / Yukawa / Morse 结构之间禁止互相嵌套
# ==============================================================
# 禁止例子：
# - woods_saxon(gauss_sq(x))
# - yukawa(woods_saxon(x))
# - gauss_sq(yukawa(x))
# - morse_attr(gauss_std(x))

_forbid_nested(
    my_nested_constraints,
    outer_ops=shape_potential_unary_ops,
    inner_ops=shape_potential_unary_ops,
)


# ==============================================================
# 3.15 统计分布类二元算子内部禁止复杂函数
# ==============================================================
# 禁止例子：
# - bose_einstein(exp(x), y)
# - fermi_dirac(log_abs(x), y)
# - logistic_prod(woods_saxon(x), y)

statistical_binary_ops = [
    "bose_einstein",
    "logistic_prod",
    "fermi_dirac",
    "bose_einstein_neg",
]

_forbid_nested(
    my_nested_constraints,
    outer_ops=statistical_binary_ops,
    inner_ops=high_risk_ops + composite_binary_ops,
)


# ==============================================================
# 3.16 允许 rotor / rotor_sq 有限嵌套
# ==============================================================
# rotor(rotor(x)) 通常没有清晰物理含义，因此禁止。
# rotor_sq(rotor(x)) 也禁止。

rotor_ops = [
    "rotor",
    "rotor_sq",
]

_forbid_nested(
    my_nested_constraints,
    outer_ops=rotor_ops,
    inner_ops=rotor_ops,
)


# ==============================================================
# Final cleanup:
# Remove empty dictionaries and inactive inner operators
# ==============================================================
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

