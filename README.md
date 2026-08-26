# MySR

MySR 是一款基于 PySR 和 SymbolicRegression.jl 开发的通用符号回归
（Symbolic Regression）工具。MySR 不绑定特定学科；未来的 NuSR 将在 MySR
之上提供核物理专用能力。

## 仓库结构

MySR 的源码分为两个独立仓库：

| 仓库 | 语言与职责 | 软件包身份 |
| --- | --- | --- |
| `TAO-MINGYU/MySR` | Python 前端 | distribution/import: `mysr` |
| `TAO-MINGYU/MySRCore.jl` | Julia 算法核心 | package/module: `MySRCore` |

本仓库是 Python 前端。主要源码位于 `mysr/`，入口类为 `MySRRegressor`；
当前实现保留 `PySRRegressor` 名称作为上游兼容别名。

## 开发安装

两个仓库应放在同一父目录：

```text
/home/taomingyu/projects/
|-- MySR/
`-- MySRCore.jl/
```

在 Conda 环境 `env_mysr` 中安装 Python 前端：

```bash
conda activate env_mysr
python -m pip install --editable /home/taomingyu/projects/MySR
python -c "from mysr import MySRRegressor; print(MySRRegressor)"
```

`mysr/juliapkg.json` 以开发路径连接相邻的 `MySRCore.jl`。

## 上游与许可证

Python 源码基于 PySR 2.0.0-beta.3；Julia 核心基于
SymbolicRegression.jl 2.0.0-beta.8。两个仓库均采用 Apache License 2.0。
详细来源和修改声明见 [VENDORING.md](VENDORING.md)、[NOTICE](NOTICE) 和
[FORK_CHANGES.md](FORK_CHANGES.md)。

MySR 是独立修改项目，不是 PySR 或 SymbolicRegression.jl 的官方发行版。
