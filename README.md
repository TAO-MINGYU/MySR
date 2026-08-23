# MySR

面向核物理实验数据分析的符号回归（Symbolic Regression）工具库——开发仓库。

本仓库 **monorepo 双包**，直接托管上游源码用于渐进式改造：

```
MySR/
├── pysr/                  # 前端：上游 PySR 源码（我们逐步修改）
├── SymbolicRegression.jl/ # 后端：上游 SRJL 源码（我们逐步修改）
├── pyproject.toml         # 前端打包配置（hatchling，vendored 自上游）
├── VENDORING.md           # 上游来源/版本/更新方式
├── LICENSE / LICENSE.PySR / LICENSE.SRJL
└── README.md
```

## 开发接线

**前端（Python）**：可编辑安装到 conda 环境 `env_mysr`（Python 3.11）：
```bash
conda activate env_mysr
pip install -e /home/taomingyu/projects/MySR   # 装的是 vendored pysr，改代码即时生效
python -c "import pysr; print(pysr.__version__)"
```

**后端（Julia）**：`env_mysr` 内置 Julia 1.12；使用本仓库的 SRJL 源码：
```bash
julia --project=SymbolicRegression.jl -e 'using Pkg; Pkg.instantiate()'
julia --project=SymbolicRegression.jl -e 'using SymbolicRegression; println("OK")'
```

## 工作流

1. 改前端 → `pysr/` 下的代码 → 直接 `import pysr` 生效（可编辑安装）
2. 改后端 → `SymbolicRegression.jl/src/` 下的代码 → 在依赖它的 Julia 环境里 `Pkg.develop(path="…/SymbolicRegression.jl")` 后生效
3. 前端调用后端的桥接沿用 PySR 自身的 juliacall 机制

## 许可

- 本项目自有代码：MIT（`LICENSE`）
- vendored 上游：PySR（Apache-2.0，`LICENSE.PySR`）、SymbolicRegression.jl（`LICENSE.SRJL`）
