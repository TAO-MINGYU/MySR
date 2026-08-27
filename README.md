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

## 安装与后端自动配置

从 GitHub 安装当前固定版本：

```bash
python -m pip install "git+https://github.com/TAO-MINGYU/MySR.git@v0.1.0"
python -c "from mysr import MySRRegressor; print(MySRRegressor)"
```

MySR 使用 JuliaPkg 管理 Julia 运行环境和依赖。首次导入时，JuliaPkg 读取随
Python wheel 一起发布的 `mysr/juliapkg.json`，从
`TAO-MINGYU/MySRCore.jl` 下载固定标签 `v0.1.0` 并完成依赖解析。这个机制与
固定基线 PySR 通过 GitHub URL 和版本标签解析 SymbolicRegression.jl 的方式
相同；普通用户不需要预先安装 MySRCore.jl，也不需要把两个仓库放在相邻目录，
并且当前安装不依赖 Julia General registry 中存在 MySRCore 条目。

## 双仓开发模式

只有需要同时编辑 Python 前端和 Julia 后端的开发者，才需要把两个仓库检出到
本地。例如：

```text
/home/taomingyu/projects/
|-- MySR/
`-- MySRCore.jl/
```

在 MySR 仓库中安装可编辑 Python 包（editable install）：

```bash
conda activate env_mysr
python -m pip install --editable /home/taomingyu/projects/MySR
```

然后把发布配置临时切换为本地 MySRCore 开发路径，并要求 JuliaPkg 重新解析：

```bash
cd /home/taomingyu/projects/MySR
python mysr/test/generate_dev_juliapkg.py \
  mysr/juliapkg.json \
  /home/taomingyu/projects/MySRCore.jl
python -m juliapkg resolve
python -c "from mysr import MySRRegressor; print(MySRRegressor)"
```

开发脚本只把 `MySRCore` 的来源从固定 `url + rev` 改为 `path + dev: true`，
不会改变包 UUID 或 Julia preferences。准备构建或提交发行版前，应恢复并重新解析
正式配置：

```bash
git restore mysr/juliapkg.json
python -m juliapkg resolve
```

## 上游与许可证

Python 源码基于 PySR 2.0.0-beta.3；Julia 核心基于
SymbolicRegression.jl 2.0.0-beta.8。两个仓库均采用 Apache License 2.0。
详细来源和修改声明见 [VENDORING.md](VENDORING.md)、[NOTICE](NOTICE) 和
[FORK_CHANGES.md](FORK_CHANGES.md)。

MySR 是独立修改项目，不是 PySR 或 SymbolicRegression.jl 的官方发行版。
Apache License 2.0 的标准正文保留在 [LICENSE](LICENSE)；派生来源、版权归属
和修改记录分别保存在 [NOTICE](NOTICE)、[VENDORING.md](VENDORING.md) 和
[FORK_CHANGES.md](FORK_CHANGES.md)。
