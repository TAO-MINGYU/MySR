# MySR

MySR 是从 PySR 独立派生并在其基础上持续改进和扩展的通用符号回归
（Symbolic Regression）Python 软件包。它保留 PySR 的成熟前端与
scikit-learn 接口作为可追溯基线，并使用从 SymbolicRegression.jl 派生的
MySRCore.jl 作为 Julia 算法后端。

MySR 不绑定特定学科；未来的 NuSR 将在 MySR 之上提供核物理专用能力。
`1.0.0` 版本将公开入口和实现类统一为 `MySRRegressor`，并新增默认关闭的
AI Feynman-inspired 自动特征工程（automated feature engineering）。该功能使用
多层感知机（Multilayer Perceptron, MLP）代理模型探测输入变量间的固定和参数化
广义对称性，再以 `suggest` 模式报告候选，或以 `augment` 模式把可重放候选追加到
MySRCore 的输入。当前证据来自聚焦合成测试，尚不构成性能优于 PySR 的声明。

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
当前公开回归器和实现类均命名为 `MySRRegressor`。MySR 仍明确承认其基于 PySR
发展，并在许可证、NOTICE、VENDORING 和 benchmark 中保留真实的 PySR 上游名称。

## 安装与后端自动配置

从 GitHub 安装当前固定版本：

```bash
python -m pip install "git+https://github.com/TAO-MINGYU/MySR.git@v1.0.0"
python -c "from mysr import MySRRegressor; print(MySRRegressor)"
```

MySR 使用 JuliaPkg 管理 Julia 运行环境和依赖。首次导入时，JuliaPkg 读取随
Python wheel 一起发布的 `mysr/juliapkg.json`，从
`TAO-MINGYU/MySRCore.jl` 下载固定标签 `v1.0.0` 并完成依赖解析。这个机制与
固定基线 PySR 通过 GitHub URL 和版本标签解析 SymbolicRegression.jl 的方式
相同；普通用户不需要预先安装 MySRCore.jl，也不需要把两个仓库放在相邻目录，
并且当前安装不依赖 Julia General registry 中存在 MySRCore 条目。

## 自动特征工程

除自动特征工程配置外，`MySRRegressor` 也可以直接接收
`formula_type="empirical"`、`"semi_theoretical"` 或 `"theoretical"`，并在 `fit`
时配合 `X_dimensions`/`y_dimensions` 将量纲搜索策略传给 MySRCore。未提供量纲元数据时，
只有 `empirical` 模式可以运行；半理论模式在开发版 MySRCore 中表示为唯一根外系数
`C_dim * f(X; θ)`，并要求启用常数优化。

该功能必须显式打开，并要求用户在 `MySRRegressor(formula_type=...)` 中显式声明公式类型。
`FeatureEngineeringConfig` 不再重复声明公式类型；前端特征工程和后端搜索始终使用同一个
`MySRRegressor.formula_type`。`empirical` 模式不做硬量纲筛选；
`semi_theoretical` 与 `theoretical` 模式会通过已加载的 MySRCore
DynamicQuantities 运行时传播量纲，拒绝内部量纲非法的候选，并把生成特征的量纲、
复杂度和表达式一起注入增广后的输入元数据。两种受约束模式的最终根输出规则仍由
MySRCore 搜索层执行：半理论模式使用外层 `C_dim`，理论模式要求根输出匹配
`y_dimensions`。候选可以由 AI Feynman-inspired
代理分支、轻量 FEAT-like 进化表征分支，或两个相互独立的分支共同生成。

```python
from mysr import MySRRegressor

model = MySRRegressor(
    formula_type="empirical",  # semi_theoretical/theoretical require dimensions
    auto_feature_engineering=True,
    feature_engineering_config={
        "mode": "augment",  # "suggest" only reports candidates
        "surrogate_engine": {"enabled": True},
        "feat_engine": {
            "enabled": True,
            "population_size": 24,
            "generations": 8,
            "max_evaluations": 240,
            "max_depth": 3,
            "max_bundle_size": 4,
        },
    },
)
model.fit(X, y, variable_names=["x1", "x2", "x3"])

print(model.feature_engineering_report_)
print(model.engineered_feature_expressions_)
predictions = model.predict(X)
```

当前代理分支可检索 `xi±xj`、`xi*xj`、`xi/xj`，以及
`xi±a*xj`、`xi*xj^a`、`xi/xj^a` 等参数化候选，并支持有预算的多层组合。
在递归组合阶段还可受控生成任意指定实数幂（例如 `(x1+x2)^1.5`）、
归一化差 `((x1-x2)/(x1+x2))`、欧氏型组合 `sqrt(x1^2+x2^2)`，以及
仅对无量纲比值应用的 `log`、`exp`、`sin` 和 `cos`。这些结构不是无条件
穷举。候选仍受 `max_composition_candidates`、`composition_beam_width`、
`max_composition_depth` 和定义域/量纲门控约束。需要自定义递归语法时，可在
`SurrogateEngineConfig(composition_operators=(...), power_exponents=(...))` 中指定。
它使用多个独立 MLP、多个扰动尺度及独立验证切分过滤不稳定关系。
FEAT-like 分支进化由多棵表达式树组成的特征集合（feature bundle），使用标准化
Ridge 回归评价集合的联合预测能力，以 ε-lexicase 选择父代，并通过误差—复杂度
非支配排序保留候选。`max_evaluations` 是硬评价预算；该分支默认关闭。两个分支
同时开启时都只读取原始 `X,y`，最终在一个全局生成列预算下去重和合并。
`auto_feature_engineering=False` 是默认值，因此升级不会自动改变既有预处理流程。

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
