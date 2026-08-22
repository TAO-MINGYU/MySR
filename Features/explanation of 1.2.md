# 核结构 Feature 函数说明
本文档对应 `nuclear_structure_features_individual.py`，它把原始 `features_1_1.py` 中 `make_nuclear_structure_features(n, z)` 生成的核结构 feature 拆成了“一 feature 一函数”的形式。
范围：**只包括由中子数 `N` 和质子数 `Z` 构造的核结构 feature**；核反应 feature 暂不包括。
重要提醒：这些 feature 是候选变量池，不代表每次 PySR 都应该全部使用。实际运行时仍应根据 target 和物理问题挑选小子集。
## 基本用法
```python
from nuclear_structure_features_individual import A, symmetry_term, coulomb_ZZ_over_A13

n = 92
z = 62

print(A(n, z))
print(symmetry_term(n, z))
print(coulomb_ZZ_over_A13(n, z))
```
也可以一次生成全表：
```python
from nuclear_structure_features_individual import make_nuclear_structure_feature_table

feature_table = make_nuclear_structure_feature_table(n=[92, 90, 82], z=[62, 60, 50])
print(feature_table)
```
## 符号约定
- `n` / `N`：中子数。
- `z` / `Z`：质子数。
- `A=N+Z`：质量数。
- `I=(N-Z)/A`：同位旋不对称度。
- 幻数集合使用：`[2, 8, 20, 28, 50, 82, 126, 184]`；其中 184 是超重核壳层 proxy。
- `EPS=1e-12` 用于数值保护，避免除零或对非正数取 log/sqrt。
## Feature 总览表
| 函数名 | 与 N,Z 的关系 | 物理意义 / 使用说明 |
|---|---|---|
| `N(n, z)` | $N$ | 中子数；最基础的核素标识量。 |
| `Z(n, z)` | $Z$ | 质子数；决定电荷、库仑效应和元素种类。 |
| `A(n, z)` | $A=N+Z$ | 质量数；核尺寸、体积项、表面项等全局尺度的基础变量。 |
| `N_minus_Z(n, z)` | $N-Z$ | 中子过剩量；描述中子-质子不平衡。 |
| `Z_minus_N(n, z)` | $Z-N$ | 质子过剩量；与中子过剩相反。 |
| `abs_N_minus_Z(n, z)` | $|N-Z|$ | 不区分方向的中子-质子不平衡强度。 |
| `N_over_Z(n, z)` | $N/Z$ | 中子-质子比例；对中子丰度敏感。 |
| `Z_over_N(n, z)` | $Z/N$ | 质子-中子比例；与 $N/Z$ 互补。 |
| `N_over_A(n, z)` | $N/A$ | 中子占总核子数比例。 |
| `Z_over_A(n, z)` | $Z/A$ | 质子占总核子数比例，也可看作电荷分数 proxy。 |
| `isospin_asymmetry(n, z)` | $I=(N-Z)/A$ | 同位旋不对称度；质量公式、对称能、滴线趋势中常见。 |
| `abs_isospin_asymmetry(n, z)` | $|I|$ | 只关心不对称强度，不区分富中子/富质子方向。 |
| `isospin_asymmetry_squared(n, z)` | $I^2$ | 对称能常见结构；符号上对 $N-Z$ 正负对称。 |
| `Tz(n, z)` | $T_z=(N-Z)/2$ | 同位旋第三分量 proxy；IMME 等关系中常见。 |
| `A_1_3(n, z)` | $A^{1/3}$ | 核半径尺度 proxy，常对应 $R\sim r_0A^{1/3}$。 |
| `A_2_3(n, z)` | $A^{2/3}$ | 核表面积尺度 proxy，常对应液滴模型表面项。 |
| `sqrt_A(n, z)` | $\sqrt{A}$ | 质量数平方根尺度；常用于经验缩放或 pairing 尺度变体。 |
| `inv_A(n, z)` | $1/A$ | 质量数反比尺度；常用于有限大小修正或归一化。 |
| `inv_A_1_3(n, z)` | $A^{-1/3}$ | 半径反比尺度；常出现在库仑项或几何尺度中。 |
| `inv_A_2_3(n, z)` | $A^{-2/3}$ | 表面积反比尺度。 |
| `inv_sqrt_A(n, z)` | $A^{-1/2}$ | 常见 pairing 缩放尺度之一。 |
| `log_A(n, z)` | $\log A$ | 缓慢变化的质量数尺度；只应在有趋势证据时使用。 |
| `volume_term(n, z)` | $A$ | 液滴模型体积项 proxy；与 `A` 数学上相同。 |
| `surface_term(n, z)` | $A^{2/3}$ | 液滴模型表面项 proxy；与 `A_2_3` 数学上相同。 |
| `radius_term(n, z)` | $A^{1/3}$ | 半径项 proxy；与 `A_1_3` 数学上相同。 |
| `coulomb_ZZ_over_A13(n, z)` | $Z(Z-1)/A^{1/3}$ | 库仑能 proxy；比 $Z^2$ 形式略去自相互作用。 |
| `coulomb_Z2_over_A13(n, z)` | $Z^2/A^{1/3}$ | 库仑能的简化 proxy。 |
| `symmetry_term(n, z)` | $(N-Z)^2/A$ | 对称能 proxy；半经验质量公式中的核心项之一。 |
| `symmetry_term_normalized(n, z)` | $I^2=((N-Z)/A)^2$ | 归一化对称能 proxy；与 `isospin_asymmetry_squared` 相同。 |
| `surface_symmetry_proxy(n, z)` | $A^{2/3}I^2$ | 表面对称能 proxy；用于描述表面项与同位旋不对称的耦合。 |
| `fissility_Z2_over_A(n, z)` | $Z^2/A$ | 裂变性/fissility 相关 proxy；重核中常有意义。 |
| `charge_density_proxy(n, z)` | $Z/A$ | 电荷分数 proxy；与 `Z_over_A` 相同。 |
| `is_even_N(n, z)` | $\mathbf{1}_{N\ \mathrm{even}}$ | N 为偶数的指示量；pairing 相关。 |
| `is_even_Z(n, z)` | $\mathbf{1}_{Z\ \mathrm{even}}$ | Z 为偶数的指示量；pairing 相关。 |
| `is_odd_N(n, z)` | $\mathbf{1}_{N\ \mathrm{odd}}$ | N 为奇数的指示量。 |
| `is_odd_Z(n, z)` | $\mathbf{1}_{Z\ \mathrm{odd}}$ | Z 为奇数的指示量。 |
| `is_even_even(n, z)` | $\mathbf{1}_{N\ even}\mathbf{1}_{Z\ even}$ | 偶偶核指示量；通常更强束缚、更稳定。 |
| `is_odd_odd(n, z)` | $\mathbf{1}_{N\ odd}\mathbf{1}_{Z\ odd}$ | 奇奇核指示量；pairing 能通常相反。 |
| `is_odd_A(n, z)` | $\mathbf{1}_{A\ odd}$ | 奇质量数核指示量；对应 pairing sign 为 0 的情况。 |
| `pairing_sign(n, z)` | $+1$ 偶偶，$0$ 奇A，$-1$ 奇奇 | 配对项符号；用于经验质量公式中的奇偶修正。 |
| `pairing_A_minus_1_2(n, z)` | $\delta_{\rm pair}A^{-1/2}$ | 按 $A^{-1/2}$ 缩放的配对 proxy。 |
| `pairing_A_minus_3_4(n, z)` | $\delta_{\rm pair}A^{-3/4}$ | 按 $A^{-3/4}$ 缩放的配对 proxy。 |
| `pairing_A_minus_1(n, z)` | $\delta_{\rm pair}A^{-1}$ | 按 $A^{-1}$ 缩放的配对 proxy。 |
| `lower_magic_N(n, z)` | $M_N^{\rm lower}$ | 不大于 $N$ 的最近下方幻数；壳层区间边界 proxy。 |
| `upper_magic_N(n, z)` | $M_N^{\rm upper}$ | 不小于 $N$ 的最近上方幻数；壳层区间边界 proxy。 |
| `lower_magic_Z(n, z)` | $M_Z^{\rm lower}$ | 不大于 $Z$ 的最近下方幻数。 |
| `upper_magic_Z(n, z)` | $M_Z^{\rm upper}$ | 不小于 $Z$ 的最近上方幻数。 |
| `shell_width_N(n, z)` | $M_N^{\rm upper}-M_N^{\rm lower}$ | N 所在壳区间宽度 proxy。 |
| `shell_width_Z(n, z)` | $M_Z^{\rm upper}-M_Z^{\rm lower}$ | Z 所在壳区间宽度 proxy。 |
| `distance_to_magic_N(n, z)` | $\min_M |N-M|$ | N 到最近幻数距离；壳闭合效应 proxy。 |
| `distance_to_magic_Z(n, z)` | $\min_M |Z-M|$ | Z 到最近幻数距离；质子壳闭合效应 proxy。 |
| `distance_to_magic_sum(n, z)` | $d_N+d_Z$ | 中子和质子离壳闭合的总距离。 |
| `distance_to_magic_product(n, z)` | $d_Nd_Z$ | 中子和质子离壳闭合距离的耦合 proxy。 |
| `valence_neutron_number(n, z)` | $\min(|N-M_N^{lower}|,|M_N^{upper}-N|)$ | 相对最近壳闭合的价中子数 proxy。 |
| `valence_proton_number(n, z)` | $\min(|Z-M_Z^{lower}|,|M_Z^{upper}-Z|)$ | 相对最近壳闭合的价质子数 proxy。 |
| `valence_neutron_fraction(n, z)` | $\mathrm{clip}(2N_v/W_N,0,1)$ | 归一化价中子 fraction；接近壳闭合为 0，中壳附近接近 1。 |
| `valence_proton_fraction(n, z)` | $\mathrm{clip}(2Z_v/W_Z,0,1)$ | 归一化价质子 fraction。 |
| `valence_product_NpNn(n, z)` | $N_vZ_v$ | $N_pN_n$ 型集体性 proxy；常与形变/集体性增强相关。 |
| `casten_P_factor(n, z)` | $P=N_vZ_v/(N_v+Z_v)$ | Casten P-factor proxy；常用于表征价核子集体性。 |
| `mid_shell_fraction_N(n, z)` | $4(N-M_l)(M_u-N)/(M_u-M_l)^2$ | N 的中壳程度；壳闭合附近约 0，中壳附近约 1。 |
| `mid_shell_fraction_Z(n, z)` | $4(Z-M_l)(M_u-Z)/(M_u-M_l)^2$ | Z 的中壳程度。 |
| `mid_shell_fraction_sum(n, z)` | $m_N+m_Z$ | 中子和质子中壳程度总和。 |
| `mid_shell_fraction_product(n, z)` | $m_Nm_Z$ | 中子和质子中壳程度耦合 proxy。 |
| `is_neutron_rich(n, z)` | $\mathbf{1}_{N>Z}$ | 富中子核区域指示量。 |
| `is_proton_rich(n, z)` | $\mathbf{1}_{Z>N}$ | 富质子核区域指示量。 |
| `is_N_equal_Z(n, z)` | $\mathbf{1}_{N=Z}$ | $N=Z$ 核指示量；轻核或同位旋对称性分析中可能有用。 |

## 建议的首轮小子集
对于质量、结合能、半径残差等核结构任务，第一轮通常不建议使用全部 66 个 feature。可以先从下列变量开始：
```python
first_round_features = [
    "A",
    "A_2_3",
    "coulomb_ZZ_over_A13",
    "symmetry_term",
    "pairing_A_minus_1_2",
]
```
如果第一轮残差显示明显壳效应或集体性结构，再考虑加入：
```python
second_round_features = [
    "distance_to_magic_N",
    "distance_to_magic_Z",
    "valence_product_NpNn",
    "casten_P_factor",
    "mid_shell_fraction_product",
]
```
## 注意事项
1. `volume_term` 与 `A` 数学上相同，`surface_term` 与 `A_2_3` 相同，`radius_term` 与 `A_1_3` 相同；它们是为了物理语义清楚而保留。
2. `symmetry_term_normalized` 与 `isospin_asymmetry_squared` 相同，`charge_density_proxy` 与 `Z_over_A` 相同。
3. magic-number、valence、mid-shell 相关变量是 proxy，不是微观壳模型计算。
4. 指示量类 feature，例如 `is_even_even`、`is_odd_A`、`is_neutron_rich`，适合作为分区或修正项，但容易导致分段式经验拟合，需要验证其外推稳定性。
