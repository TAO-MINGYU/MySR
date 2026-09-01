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
`v1.0.0` 解析 MySRCore.jl。这复用了固定基线 PySR 解析
SymbolicRegression.jl 的 JuliaPkg 模式，同时不要求 MySRCore.jl 预先注册到
Julia General registry。

双仓开发时，`mysr/test/generate_dev_juliapkg.py` 可将同一个 MySRCore 包条目
临时切换为本地 `path + dev: true`。本地路径不是发布配置的一部分，构建发行物前
必须恢复固定的 `url + rev` 配置。

以后同步 PySR 或 SymbolicRegression.jl 上游更新时，必须从这里记录的固定提交
开始逐项比较、合并和测试，不能用整目录覆盖 MySR 或 MySRCore.jl 自己的修改。

## Algorithmic inspiration: AI Feynman

MySR 的自动特征工程代理分支参考了 AI Feynman 的代理插值、对称性、可分离性、
组合性和递归降维思想：

- 论文：Udrescu and Tegmark, *AI Feynman: A physics-inspired method for
  symbolic regression*, Science Advances 6 (2020)；
- 论文：Udrescu et al., *AI Feynman 2.0: Pareto-optimal symbolic regression
  exploiting graph modularity*, NeurIPS 2020；
- 官方仓库：https://github.com/SJ001/AI-Feynman （MIT License）。

`mysr/feature_engineering.py` 是面向 MySR 接口重新设计的独立实现，没有复制、
打包或运行官方 AI Feynman 源文件，因此 AI Feynman 不是 MySR 的 vendored dependency
或运行时依赖。若以后直接复制官方源码，必须另行保留其 MIT 版权与许可证文本。

## Algorithmic inspiration: FEAT

MySR 的轻量 FEAT-like 分支参考了 FEAT（Feature Engineering Automation Tool，
特征工程自动化工具）论文与公开文档中的表达式集合进化、下游线性模型、
ε-lexicase 父代选择及误差—复杂度多目标生存思想：

- 官方仓库：https://github.com/cavalab/feat （GNU GPLv3）；
- 本次核对的参考提交：`2967e6e5f7eee75ecf34062708e7b0b87c0b9145`。

`mysr/feat_engine.py` 是 Apache-2.0 下为 MySR 公共接口独立编写的轻量实现。
它没有复制、导入、编译、打包或 vendoring 官方 FEAT 的 GPLv3 源码，也不依赖
官方实现使用的 C++、Shogun 或 Eigen 运行栈。因此官方 FEAT 不是 MySR 的运行时
依赖；若未来直接接入官方实现，必须作为独立的许可证与发行决策重新审查。
