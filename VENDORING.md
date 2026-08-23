# VENDORING

MySR 以 **monorepo 双包**形式直接托管上游源码，后续在此基础上逐步修改
（前端改 `pysr/`，后端改 `SymbolicRegression.jl/`）。

## 前端：PySR（Python）

| 项 | 值 |
|----|----|
| 上游仓库 | https://github.com/MilesCranmer/PySR |
| vendored 提交 | `85c2cc657d01362aaef667def6d590fae797da67`（clone 时的 main） |
| 版本 | 2.0.0-beta.3（见根 `pyproject.toml`，hatchling 构建，dist 名暂为 `pysr`） |
| 目录 | `pysr/` |
| 许可证 | Apache-2.0（见 `LICENSE.PySR`） |

## 后端：SymbolicRegression.jl（Julia）

| 项 | 值 |
|----|----|
| 来源 | 本机 Julia depot 中实际运行的包（~/.julia/packages/SymbolicRegression/jyjLZ） |
| 版本 | 1.13.4（见 `SymbolicRegression.jl/Project.toml`） |
| 目录 | `SymbolicRegression.jl/` |
| 许可证 | 见 `LICENSE.SRJL` |

## 更新上游

```bash
# 前端
git clone --depth 1 https://github.com/MilesCranmer/PySR /tmp/PySR_up
cp -a /tmp/PySR_up/pysr pysr/
# 后端
julia --project=SymbolicRegression.jl -e 'using Pkg; Pkg.update()'
# 然后更新本文件中的版本/提交记录
```

## 开发约定

- 修改 `pysr/` 后：`pip install -e .`（可编辑安装，改动即时生效）
- 修改 `SymbolicRegression.jl/` 后：`julia --project=SymbolicRegression.jl -e 'using Pkg; Pkg.instantiate()'`，
  再在任意 Julia 项目里 `Pkg.develop(path="…/SymbolicRegression.jl")` 使用本份源码
