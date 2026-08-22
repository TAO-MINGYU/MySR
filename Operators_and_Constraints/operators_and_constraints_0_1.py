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
    # "^",

    # # --- 物理与几何衍生算子 (新增自定义) ---
    # "beta_sq(x, y) = x^2 / y^2",                              # 相对论 Beta 因子平方
    # "lorentz_gamma(x, y) = 1 / sqrt(abs(1 - (x^2 / y^2)))",   # 洛伦兹 Gamma 因子
    # "diff_inv(x, y) = (1 / y) - (1 / x)",                     # 势能差/透镜公式
    # "hypot_2d(x, y) = sqrt(x^2 + y^2)",                       # 欧几里得距离/勾股定理
    # "inv_lorentz(x, y) = sqrt(abs(1 - (x^2 / y^2)))",         # 逆洛伦兹因子
    # "sum_sq(x, y) = x^2 + y^2",                               # 平方和
    # "cos_diff(x, y) = cos(x - y)",                            # 相位差余弦
    # "inv_one_minus_ratio(x, y) = 1 / (1 - (x / y))",          # 比例发散因子
    # "geom_mean(x, y) = sqrt(abs(x * y))",                     # 几何平均
    # "exp_neg_ratio(x, y) = exp(-x / y)",                      # 指数衰减 (比例)
    # "log_ratio(x, y) = log(abs(x / y))",                      # 对数比例
    # "parallel_sum(x, y) = (x * y) / (x + y)",                 # 并联公式/调和平均
    # "bose_einstein(x, y) = 1 / (exp(x / y) - 1)",             # 玻色-爱因斯坦统计因子
    # "logistic_prod(x, y) = 1 / (exp(-x * y) + 1)",            # 乘积逻辑斯蒂函数
    # "fermi_dirac(x, y) = 1 / (exp(x / y) + 1)",               # 费米-狄拉克统计因子
    # "bose_einstein_neg(x, y) = 1 / (exp(-x * y) - 1)"         # 负向玻色-爱因斯坦因子
]

# ==============================================================
# Unary Operators
# ==============================================================
my_unary_operators = [

    # --- 2.1 PySR/Julia 内置算子---
    # "square",    # 平方: x^2
    # "cube",      # 立方: x^3
    # "cbrt",      # 立方根: x^(1/3)
    # "sqrt",      # 平方根: √x (注意：如果数据可能为负，建议不用或改为 sqrt_abs)
    # "abs",       # 绝对值: |x|
    "exp",       # 自然指数: e^x
    # "log",       # 自然对数: ln(x)
    # "sin",       # 正弦: sin(x)
    # "cos",       # 余弦: cos(x)
    # "tan",       # 正切: tan(x)
    # "asin",      # 反正弦: arcsin(x)
    # "acos",      # 反余弦: arccos(x)
    # "atan",      # 反正切: arctan(x)
    # "sinh",      # 双曲正弦: sinh(x)
    # "cosh",      # 双曲余弦: cosh(x)
    # "tanh",      # 双曲正切: tanh(x)
    # "asinh",     # 反双曲正弦: arcsinh(x)
    # "acosh",     # 反双曲余弦: arccosh(x)
    # "atanh",     # 反双曲正切: arctanh(x)
    
    # # --- 物理与几何衍生算子 (新增自定义) ---
    # "inv(x) = 1 / x",                       # 倒数
    # "inv2(x) = 1 / (x^2)",                  # 平方反比
    # "inv3(x) = 1 / (x^3)",                  # 立方反比
    # "pow23(x) = cbrt(x)^2",                   # 2/3次方 (改写为cbrt的平方，完美避开负数底数报错)
    # "gauss_sq(x) = exp(-(x^2))",                 # 高斯分布 / 正态衰减
    # "log1p_abs(x) = log(abs(1 + x))",       # 平移绝对值对数 (你提供的范例)
    # "log_abs(x) = log(abs(x))",        # 绝对值对数 (加 1e-8 防止 log(0) 崩溃)
    # "woods_saxon(x) = 1 / (1 + exp(x))",  # Woods-Saxon 形式 / 费米分布
    # "yukawa(x) = exp(-x) / (abs(x))",  # 汤川势空间部分 (加防零保护)
    # "morse_attr(x) = 1 - exp(-x)",          # 莫尔斯势吸引项
    # "gauss_std(x) = exp(-(x^2)/2)",               # 标准正态分布核心
    # "inv_one_plus(x) = 1 / (1 + x)",              # 简单反比例/阻尼因子
    # "one_minus_sq(x) = 1 - x^2",                  # 倒抛物线
    # "sqrt_sq_minus_one(x) = sqrt(abs(x^2 - 1))"   # 双曲几何/相对论动量因子
]

# ==============================================================
# SymPy Mappings
# ==============================================================
my_extra_sympy_mappings = {

    # # ---一元算子翻译---
    # "inv": lambda x: 1 / x,
    # "inv2": lambda x: 1 / (x**2),
    # "inv3": lambda x: 1 / (x**3),
    # "pow23": lambda x: x**sp.Rational(2, 3),
    # "gauss_sq": lambda x: sp.exp(-(x**2)),
    # "log1p_abs": lambda x: sp.log(sp.Abs(1 + x)),
    # "log_abs": lambda x: sp.log(sp.Abs(x)),
    # "woods_saxon": lambda x: 1 / (1 + sp.exp(x)),
    # "yukawa": lambda x: sp.exp(-x) / (sp.Abs(x)),
    # "morse_attr": lambda x: 1 - sp.exp(-x),
    # "gauss_std": lambda x: sp.exp(-(x**2)/2),
    # "inv_one_plus": lambda x: 1 / (1 + x),
    # "one_minus_sq": lambda x: 1 - x**2,
    # "sqrt_sq_minus_one": lambda x: sp.sqrt(sp.Abs(x**2 - 1)),


    # # --- 二元算子翻译 ---
    # "beta_sq": lambda x, y: x**2 / y**2,
    # "lorentz_gamma": lambda x, y: 1 / sp.sqrt(sp.Abs(1 - x**2 / y**2)),
    # "diff_inv": lambda x, y: (1 / y) - (1 / x),
    # "hypot_2d": lambda x, y: sp.sqrt(x**2 + y**2),
    # "inv_lorentz": lambda x, y: sp.sqrt(sp.Abs(1 - x**2 / y**2)),
    # "sum_sq": lambda x, y: x**2 + y**2,
    # "cos_diff": lambda x, y: sp.cos(x - y),
    # "inv_one_minus_ratio": lambda x, y: 1 / (1 - (x / y)),
    # "geom_mean": lambda x, y: sp.sqrt(sp.Abs(x * y)),
    # "exp_neg_ratio": lambda x, y: sp.exp(-x / y),
    # "log_ratio": lambda x, y: sp.log(sp.Abs(x / y)),
    # "parallel_sum": lambda x, y: (x * y) / (x + y),
    # "bose_einstein": lambda x, y: 1 / (sp.exp(x / y) - 1),
    # "logistic_prod": lambda x, y: 1 / (sp.exp(-x * y) + 1),
    # "fermi_dirac": lambda x, y: 1 / (sp.exp(x / y) + 1),
    # "bose_einstein_neg": lambda x, y: 1 / (sp.exp(-x * y) - 1)
}
# endregion


# region constraints
# ==============================================================
# Operator Complexities
# ==============================================================
my_complexity_of_operators = {
    # ==========================================================
    # (Unary Operators)
    # ==========================================================
    
    # --- 1. PySR/Julia 内置标准一元算子 ---
    # "square": 1,               # 平方: x^2
    # "cube": 1,                 # 立方: x^3
    # "cbrt": 1,                 # 立方根: x^(1/3)
    # "sqrt": 1,                 # 平方根: √x
    # "abs": 1,                  # 绝对值: |x|
    "exp": 1,                  # 自然指数: e^x
    # "log": 1,                  # 自然对数: ln(x)
    
    # # --- 2. 三角函数与双曲函数 ---
    # "sin": 1,                  # 正弦: sin(x)
    # "cos": 1,                  # 余弦: cos(x)
    # "tan": 1,                  # 正切: tan(x)
    # "asin": 1,                 # 反正弦: arcsin(x)
    # "acos": 1,                 # 反余弦: arccos(x)
    # "atan": 1,                 # 反正切: arctan(x)
    # "sinh": 1,                 # 双曲正弦: sinh(x)
    # "cosh": 1,                 # 双曲余弦: cosh(x)
    # "tanh": 1,                 # 双曲正切: tanh(x)
    # "asinh": 1,                # 反双曲正弦: arcsinh(x)
    # "acosh": 1,                # 反双曲余弦: arccosh(x)
    # "atanh": 1,                # 反双曲正切: arctanh(x)

    # # --- 3. 简单自定义一元算子 (轻度打包，复杂度: 2) ---
    # "inv": 1,                  # 倒数 1/x (极为基础，给 1 鼓励使用)
    # "inv2": 2,                 # 1/x^2
    # "inv3": 2,                 # 1/x^3
    # "pow23": 2,                # cbrt(x)^2
    # "log_abs": 2,              # log(abs(x))
    # "morse_attr": 2,           # 1 - exp(-x)
    # "gauss_sq": 2,             # exp(-x^2)
    # "inv_one_plus": 2,         # 1 / (1 + x)
    # "one_minus_sq": 2,         # 1 - x^2

    # # --- 4. 中等物理衍生一元算子 (中度打包，复杂度: 3) ---
    # "gauss_std": 3,            # exp(-(x^2)/2)
    # "log1p_abs": 3,            # log(abs(1 + x))
    # "woods_saxon": 3,          # 1 / (1 + exp(x))
    # "yukawa": 3,               # exp(-x) / abs(x)
    # "sqrt_sq_minus_one": 3,    # sqrt(abs(x^2 - 1))


    # ==========================================================
    # (Binary Operators)
    # ==========================================================
    
    # --- 1. 基础二元算子 ---
    "+": 1, 
    "-": 1, 
    "*": 1, 
    "/": 1,
    # "^": 1,

    # # --- 2. 简单自定义二元算子 (轻度打包，复杂度: 2) ---
    # "beta_sq": 2,              # x^2 / y^2
    # "sum_sq": 2,               # x^2 + y^2
    # "cos_diff": 2,             # cos(x - y)
    # "exp_neg_ratio": 2,        # exp(-x / y)
    # "log_ratio": 2,            # log(abs(x / y))

    # # --- 3. 中等物理衍生二元算子 (中度打包，复杂度: 3) ---
    # "diff_inv": 3,             # (1 / y) - (1 / x)
    # "hypot_2d": 3,             # sqrt(x^2 + y^2)
    # "inv_lorentz": 3,          # sqrt(abs(1 - (x^2 / y^2)))
    # "inv_one_minus_ratio": 3,  # 1 / (1 - (x / y))
    # "geom_mean": 3,            # sqrt(abs(x * y))
    # "parallel_sum": 3,         # (x * y) / (x + y)
    # "bose_einstein": 3,        # 1 / (exp(x / y) - 1)
    # "logistic_prod": 3,        # 1 / (exp(-x * y) + 1)
    # "fermi_dirac": 3,          # 1 / (exp(x / y) + 1)
    # "bose_einstein_neg": 3,    # 1 / (exp(-x * y) - 1)

    # # --- 4. 复杂物理结构二元算子 (重度打包，复杂度: 4) ---
    # "lorentz_gamma": 4         # 1 / sqrt(abs(1 - (x^2 / y^2)))
}


# ==============================================================
# my_constraints
# ==============================================================
# 格式: '算子': 内部最大复杂度 或 '算子': (左参数最大, 右参数最大)
my_constraints = {
    # --- 二元约束 ---
    # "^": (-1, 1),
    "+": (-1, -1),
    "-": (-1, -1),
    "*": (-1, -1),
    "/": (-1, 5),
    # "beta_sq": (3, 3),
    # "sum_sq": (4, 4),
    # "cos_diff": (4, 4),
    # "exp_neg_ratio": (3, 3),
    # "log_ratio": (3, 3),
    # "diff_inv": (3, 3),
    # "hypot_2d": (4, 4),
    # "inv_lorentz": (3, 3),
    # "inv_one_minus_ratio": (3, 3),
    # "geom_mean": (4, 4),
    # "parallel_sum": (4, 4),
    # "bose_einstein": (3, 3),
    # "logistic_prod": (3, 3),
    # "fermi_dirac": (3, 3),
    # "bose_einstein_neg": (3, 3),
    # "lorentz_gamma": (2, 2),

    # # --- 一元约束 ---
    # "square": 4,
    # "cube": 4,
    # "cbrt": 4,
    # "sqrt": 4,
    # "abs": 4,
    "exp": 4,
    # "log": 4,
    # "sin": 4,
    # "cos": 4,
    # "tan": 4,
    # "asin": 4,
    # "acos": 4,
    # "atan": 4,
    # "sinh": 4,
    # "cosh": 4,
    # "tanh": 4,
    # "asinh": 4,
    # "acosh": 4,
    # "atanh": 4,
    # "inv": 4,
    # "inv2": 4,
    # "inv3": 4,
    # "pow23": 4,
    # "log_abs": 4,
    # "morse_attr": 4,
    # "gauss_sq": 4,
    # "inv_one_plus": 4,
    # "one_minus_sq": 4,
    # "gauss_std": 3,
    # "log1p_abs": 3,
    # "woods_saxon": 3,
    # "yukawa": 3,
    # "sqrt_sq_minus_one": 3
}

# ==============================================================
# my_nested_constraints
# ==============================================================
# 定义分类列表 (用于自动化生成嵌套字典)
class_b = [
    "square", 
    "cube", 
    "cbrt", 
    "sqrt", 
    "abs", 
    "exp", 
    "log",
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
    "inv", 
    "inv2", 
    "inv3", 
    "pow23", 
    "log_abs", 
    "one_minus_sq"
]

class_c = [
    "gauss_sq", 
    "gauss_std", 
    "inv_one_plus", 
    "sqrt_sq_minus_one", 
    "woods_saxon", 
    "yukawa", 
    "morse_attr", 
    "log1p_abs",
    "beta_sq", 
    "lorentz_gamma", 
    "diff_inv", 
    "hypot_2d", 
    "inv_lorentz", 
    "sum_sq", 
    "cos_diff", 
    "inv_one_minus_ratio", 
    "geom_mean", 
    "exp_neg_ratio", 
    "log_ratio", 
    "parallel_sum", 
    "bose_einstein", 
    "logistic_prod", 
    "fermi_dirac", 
    "bose_einstein_neg"
]


my_nested_constraints = {}

# --- 自动化填充逻辑 ---
# 规则: B不套B, C不套C, B可以套1个C, C可以套1个B
for op_out in (class_b + class_c):
    my_nested_constraints[op_out] = {}
    
    # 填充 B 类的限制
    for op_b in class_b:
        # 如果外层是B，内部禁B；如果外层是C，内部允许1个B
        my_nested_constraints[op_out][op_b] = 0 if op_out in class_b else 1
        
    # 填充 C 类的限制
    for op_c in class_c:
        # 如果外层是C，内部禁C；如果外层是B，内部允许1个C
        my_nested_constraints[op_out][op_c] = 0 if op_out in class_c else 1

# --- ^ 算子特殊限制：指数部分只允许常数，禁止任何算子进入 ---
# nested_constraints 从结构层面禁止，mutation 也无法绕过
# 注意：仅列出当前活跃的算子，未启用的算子不要加入（否则 Julia 报 UndefVarError）
my_nested_constraints["^"] = {
    # 二元算子禁入指数
    "+": 0, "-": 0, "*": 0, "/": 0, "^": 0,
    # 一元算子禁入指数（当前活跃：exp, log）
    "exp": 0, "log": 0,
}

# --- 显式追加 exp / log 嵌套限制 ---
# 自动生成已设 B类互不嵌套（my_nested_constraints["exp"]["exp"]=0），
# 此处显式追加二元算子 ^ 的禁止，并确保关键条目不被后续修改覆盖
my_nested_constraints["exp"].update({
    "^": 0,      # 禁止 exp(x^y)，无物理意义
    "exp": 0,    # 禁止 exp(exp(x))（与自动生成重复，显式确认）
    "log": 0,    # 禁止 exp(log(x))（与自动生成重复，显式确认）
})
my_nested_constraints["log"].update({
    "^": 0,      # 禁止 log(x^y)
    "exp": 0,    # 禁止 log(exp(x))
    "log": 0,    # 禁止 log(log(x))
})
# endregion


