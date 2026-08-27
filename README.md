# MySR

MySR 是从 PySR 独立派生并在其基础上持续改进和扩展的通用符号回归
（Symbolic Regression）Python 软件包。它保留 PySR 的成熟前端与
scikit-learn 接口作为可追溯基线，并使用从 SymbolicRegression.jl 派生的
MySRCore.jl 作为 Julia 算法后端。

MySR 不绑定特定学科；未来的 NuSR 将在 MySR 之上提供核物理专用能力。
当前 `0.1.0` 版本完成了包改名、前后端拆分和本地开发接线，尚未宣称已经实现
MySR 特有的符号回归算法改进。后续改进将逐项实现、测试并记录。

## 与 PySR 的关系

- MySR 的 Python 源码基于 PySR 2.0.0-beta.3 的固定快照。
- 保留的上游实现继续归属于 PySR 及其贡献者；MySR 的修改单独记录。
- MySR 是独立派生项目，不是 PySR 官方发行版，也不代表 PySR 上游团队。
- MySR 的目标是在保持来源可追溯和基线可复现的前提下，逐步形成自己的接口、
  工程能力和符号回归算法改进。

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
Apache License 2.0 的标准正文保留在 [LICENSE](LICENSE)；派生来源、版权归属
和修改记录分别保存在 [NOTICE](NOTICE)、[VENDORING.md](VENDORING.md) 和
[FORK_CHANGES.md](FORK_CHANGES.md)。
