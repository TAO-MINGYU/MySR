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

正式发行配置通过 `mysr/juliapkg.json` 中的 GitHub URL 和不可变版本标签
`v0.1.0` 解析 MySRCore.jl。这复用了固定基线 PySR 解析
SymbolicRegression.jl 的 JuliaPkg 模式，同时不要求 MySRCore.jl 预先注册到
Julia General registry。

双仓开发时，`mysr/test/generate_dev_juliapkg.py` 可将同一个 MySRCore 包条目
临时切换为本地 `path + dev: true`。本地路径不是发布配置的一部分，构建发行物前
必须恢复固定的 `url + rev` 配置。

以后同步 PySR 或 SymbolicRegression.jl 上游更新时，必须从这里记录的固定提交
开始逐项比较、合并和测试，不能用整目录覆盖 MySR 或 MySRCore.jl 自己的修改。
