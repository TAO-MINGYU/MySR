# Upstream source baseline

MySR 当前直接保留 PySR 前端和 SymbolicRegression.jl（SRJL）后端源码，供后续
逐步修改。两者的许可证分别见 `LICENSE.PySR` 和 `LICENSE.SRJL`。

| 组件 | 目录 | 上游版本 | 上游 Git 提交 |
|---|---|---|---|
| PySR | `pysr/` | 2.0.0-beta.3 开发快照 | `85c2cc657d01362aaef667def6d590fae797da67` |
| SRJL | `SymbolicRegression.jl/` | 2.0.0-beta.8 | `35d45fd625dc8df0067df60c72b615d83518ed44` |

上游仓库：

- PySR: https://github.com/MilesCranmer/PySR
- SRJL: https://github.com/astroautomata/SymbolicRegression.jl

PySR 快照的 `juliapkg.json` 原本要求 SRJL `v2.0.0-beta.8`。本仓库只把该
依赖改为相对路径 `../SymbolicRegression.jl`，从而在开发时直接使用仓库内后端。

后续开始修改任一上游源码后，更新上游必须通过逐项比较和合并完成，不能再用
整目录覆盖，以免丢失 MySR 自己的改动。
