# MySR

MySR 是面向核物理研究的符号回归（Symbolic Regression）开发仓库。

当前阶段只建立两块可修改、版本兼容的上游源码基石：

```text
MySR/
├── pysr/                  # Python 前端，包含 PySRRegressor
├── SymbolicRegression.jl/ # Julia 后端，简称 SRJL
├── pyproject.toml         # 当前 PySR 前端的开发配置
├── VENDORING.md           # 上游来源与版本
└── README.md
```

## 当前基线

- PySR 前端：`2.0.0-beta.3` 开发快照。
- SRJL 后端：`2.0.0-beta.8`。
- `pysr/juliapkg.json` 在本仓库中直接指向根目录下的 SRJL 源码。

这一步只保证 MySR 拥有可研究、可修改的 PySR 与 SRJL 基石，不提前建立
MySR 公共接口、发布流程或算法扩展。后续功能将逐项讨论、实现和验证。

## 开发环境

```bash
conda activate env_mysr
pip install -e /home/taomingyu/projects/MySR
python -c "from pysr import PySRRegressor; print(PySRRegressor)"
```

两块上游源码的精确来源见 [VENDORING.md](VENDORING.md)。
