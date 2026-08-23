# MySR

**MySR** — 面向核物理实验数据分析的符号回归（Symbolic Regression）工具库。

合并了 PySR（Python 前端）与 SymbolicRegression.jl（Julia 引擎）两条技术线，
采用单仓库双包（monorepo）结构：

```
MySR/
├── mysr/            # Python 包（Poetry 管理，可编辑安装）
│   ├── core/        #   搜索配置 / 编排 / 结果
│   ├── features/    #   特征工程（模块位）
│   ├── loss/        #   损失函数（模块位）
│   ├── operators/   #   算子与约束（模块位）
│   ├── export/      #   公式导出（计划中）
│   └── julia_bridge.py  # Python ↔ Julia 桥接
├── MySR.jl/         # Julia 包（SymbolicRegression.jl 封装）
│   ├── src/MySR.jl  #   search_from_files 入口
│   └── scripts/     #   桥接驱动脚本
├── Reference/       # PySR/SR.jl 蒸馏设计文档
├── Features/        # 特征设计笔记（md）
├── Loss Function/   # 损失函数设计笔记（md）
└── Operators_and_Constraints/  # 算子设计笔记（md）
```

## 环境

- conda env：`env_mysr`（Python 3.11 + Julia 1.12）
- Python 包管理：Poetry（可编辑/开发模式安装）
- Julia 包管理：Pkg（MySR.jl 依赖 SymbolicRegression.jl）

## 快速开始

```bash
conda activate env_mysr
cd /path/to/MySR
poetry install            # 开发模式（editable）安装 mysr
julia --project=MySR.jl -e 'using Pkg; Pkg.instantiate()'   # 安装 Julia 依赖

# 跑一次搜索
python - <<'PY'
import numpy as np
import mysr

X = np.random.randn(200, 2)
y = X[:, 0] ** 2 - X[:, 1] + 0.1 * np.random.randn(200)
hof = mysr.fit(X, y, mysr.SearchConfig(niterations=5, populations=4))
print(hof)
PY
```

## 状态

- v0.1：骨架 + 端到端桥接打通（Python 编排 → Julia 引擎 → 结果回传）
- 后续：特征/损失/算子模块实装、导出系统、核物理数据集支持

## 许可

MIT（与 PySR / SymbolicRegression.jl 一致）。
