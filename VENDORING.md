# Upstream source baseline

## Python frontend

| 字段 | 内容 |
| --- | --- |
| 当前目录 | `mysr/` |
| 上游项目 | PySR |
| 上游版本 | 2.0.0-beta.3 development snapshot |
| 上游提交 | `85c2cc657d01362aaef667def6d590fae797da67` |
| 上游仓库 | https://github.com/MilesCranmer/PySR |
| 许可证 | Apache License 2.0 |

## Julia backend

Julia 源码已经拆分到独立仓库 `TAO-MINGYU/MySRCore.jl`。该仓库基于
SymbolicRegression.jl 2.0.0-beta.8，提交
`35d45fd625dc8df0067df60c72b615d83518ed44`：

https://github.com/astroautomata/SymbolicRegression.jl

本地开发时，`mysr/juliapkg.json` 通过相对路径 `../../MySRCore.jl` 连接
相邻后端仓库。以后同步上游更新时必须逐项比较和合并，不能用整目录覆盖 MySR
自己的修改。
