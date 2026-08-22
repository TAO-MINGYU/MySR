# NuSR / PySR 常用 Symbolic Regression Loss Function 调研与代码库

> 目的：为 NuSR 开发早期阶段提供一份可直接查阅、复制、修改的 PySR 自定义损失函数（loss function）资料。
>
> 使用场景：你希望通过 PySR / SymbolicRegression.jl 搜索物理经验公式时，不只是默认使用均方误差（MSE），而是根据核物理数据的误差结构、量纲尺度、极端值、相对误差、模型复杂度等因素选择更合适的目标函数。

---

## 0. PySR 中 loss function 的基本位置

PySR 的搜索目标不是“找到一个唯一正确公式”，而是在给定算子库、复杂度限制和数据集的条件下，搜索一批候选表达式，使它们在某个损失函数上尽量小，同时复杂度不要太高。

在 PySR 中，常见有两类 loss 写法：

1. **逐点损失函数（elementwise loss）**  
   只关心单个样本点的误差，例如：
   $$
   L_i = (y_i - \hat{y}_i)^2
   $$

2. **完整自定义损失函数（full loss function）**  
   直接接收整棵表达式树、整个数据集和 PySR 后端选项，例如：
   ```julia
   function my_loss(tree, dataset::Dataset{T,L}, options)::L where {T,L}
       prediction, flag = eval_tree_array(tree, dataset.X, options)
       ...
       return loss
   end
   ```

对 NuSR 来说，第二种更有用，因为它可以写：

- MSE、MAE、MAPE、RMSPE；
- COD / $R^2$；
- BIC / AIC 这类带复杂度惩罚的指标；
- 带实验误差权重的 Weighted MSE / Chi-square；
- 加入物理边界、渐近行为、单调性约束的 physics-informed loss。

---

# 1. 记号约定

设数据集有 $n$ 个样本：

$$
\{(x_i, y_i)\}_{i=1}^{n}
$$

其中：

- $x_i$：第 $i$ 个样本的输入变量，可以是一维，也可以是多维；
- $y_i$：第 $i$ 个样本的真实值；
- $\hat{y}_i = f(x_i)$：候选符号表达式给出的预测值；
- $e_i = y_i - \hat{y}_i$：残差（residual）；
- $|e_i|$：绝对误差；
- $n$：样本数量；
- $k$：模型参数数量或复杂度指标。对于 PySR，可以近似用公式中的常数个数、表达式节点数或自定义复杂度表示；
- $\epsilon$：极小正数，用来避免除以 0；
- $\sigma_i$：第 $i$ 个实验点的不确定度或标准差；
- $\bar{y}$：真实值的平均值：

$$
\bar{y} = \frac{1}{n}\sum_{i=1}^{n} y_i
$$

---

# 2. 常用 Loss Function 调研

## 2.1 MSE：Mean Squared Error，均方误差

### 表达式

$$
\mathrm{MSE}
=
\frac{1}{n}
\sum_{i=1}^{n}
(y_i - \hat{y}_i)^2
$$

### 各项意义

- $y_i$：真实值；
- $\hat{y}_i$：预测值；
- $y_i - \hat{y}_i$：残差；
- 平方：放大大误差；
- 求平均：得到整体平均误差水平。

### 特点

MSE 是最常见的回归损失函数，也是很多 Symbolic Regression 默认或常用的优化目标。

它的核心特点是：

1. **极度迁就极端值 / 离群点（outliers）**  
   因为误差被平方，大残差会被放大。例如误差从 1 变成 10，MSE 贡献从 1 变成 100。

2. **偏向让大数值区域拟合得更好**  
   如果数据中 $y$ 的量级跨度很大，大 $y$ 区域往往主导 MSE。

3. **数学性质好**  
   平方函数光滑，优化性质好，便于很多算法处理。

4. **适合噪声近似服从高斯分布（Gaussian noise）的情况**  
   如果残差可以看成独立同分布高斯噪声，最小化 MSE 与最大似然估计有联系。

### NuSR 使用建议

适合：

- 数据误差比较均匀；
- 离群点较少；
- 更关注绝对误差；
- 大数值区域确实更重要。

不适合：

- 核物理数据跨越多个数量级；
- 小数值区域同样重要；
- 数据中存在明显异常点；
- 你更关心相对误差。

---

## 2.2 RMSE：Root Mean Squared Error，均方根误差

### 表达式

$$
\mathrm{RMSE}
=
\sqrt{
\frac{1}{n}
\sum_{i=1}^{n}
(y_i - \hat{y}_i)^2
}
$$

### 各项意义

RMSE 是 MSE 的平方根，因此它和 $y$ 的量纲相同。

### 特点

1. **仍然迁就极端值**  
   因为内部仍然是平方误差。

2. **比 MSE 更直观**  
   MSE 的单位是 $y^2$，RMSE 的单位和 $y$ 一样。

3. **排序上与 MSE 等价**  
   对同一批候选公式，如果只比较大小，RMSE 和 MSE 的排序完全一致，因为平方根是单调函数。

### NuSR 使用建议

RMSE 更适合用作结果汇报指标，而不是非要作为搜索 loss。搜索时用 MSE 和 RMSE 通常差别不大。

---

## 2.3 MAE：Mean Absolute Error，平均绝对误差

### 表达式

$$
\mathrm{MAE}
=
\frac{1}{n}
\sum_{i=1}^{n}
|y_i - \hat{y}_i|
$$

### 各项意义

- $|y_i - \hat{y}_i|$：单个样本的绝对误差；
- 求平均：所有样本的平均绝对偏差。

### 特点

1. **比 MSE 更抗离群点**  
   大误差只线性增加，不会像 MSE 那样平方放大。

2. **对所有误差一视同仁**  
   误差 10 大约就是误差 1 的 10 倍，而不是 100 倍。

3. **不如 MSE 光滑**  
   在误差为 0 的地方绝对值函数不可导。不过 PySR 主要是演化搜索，不是单纯梯度下降，因此这通常不是大问题。

### NuSR 使用建议

适合：

- 数据中可能有异常点；
- 不希望少数点支配整个搜索；
- 希望整体拟合更加稳健。

不适合：

- 你确实需要强烈惩罚大误差；
- 大误差在物理上绝对不可接受。

---

## 2.4 Huber Loss，Huber 损失

### 表达式

设残差：

$$
e_i = y_i - \hat{y}_i
$$

Huber loss 定义为：

$$
L_{\delta}(e_i)
=
\begin{cases}
\frac{1}{2}e_i^2, & |e_i| \le \delta \\
\delta(|e_i| - \frac{1}{2}\delta), & |e_i| > \delta
\end{cases}
$$

整体损失：

$$
\mathrm{Huber}
=
\frac{1}{n}
\sum_{i=1}^{n}
L_{\delta}(e_i)
$$

### 各项意义

- $\delta$：阈值参数；
- 小误差区域使用平方误差；
- 大误差区域使用近似绝对误差。

### 特点

1. **小误差像 MSE，大误差像 MAE**  
   它是 MSE 和 MAE 的折中。

2. **比 MSE 抗离群点**  
   超过阈值 $\delta$ 后，大误差不再被平方放大。

3. **比 MAE 更光滑**  
   小误差区域是二次函数。

### NuSR 使用建议

适合：

- 数据整体可靠，但有少量异常点；
- 既希望拟合精细，又不想被极端值带偏；
- 实验数据存在少数不稳定点。

---

## 2.5 Log-Cosh Loss

### 表达式

$$
\mathrm{LogCosh}
=
\frac{1}{n}
\sum_{i=1}^{n}
\log \left( \cosh(y_i - \hat{y}_i) \right)
$$

### 各项意义

- $\cosh$：双曲余弦函数；
- $\log(\cosh(e))$：对小误差近似像 $\frac{e^2}{2}$，对大误差近似像 $|e|$。

### 特点

1. **小误差近似 MSE**
2. **大误差近似 MAE**
3. **整体光滑**
4. **比 Huber 少一个显式分段**

### NuSR 使用建议

适合用于稳健拟合，尤其是你想要类似 Huber 的性质，但不想手动处理分段函数。

---

## 2.6 MAPE：Mean Absolute Percentage Error，平均绝对百分比误差

### 表达式

$$
\mathrm{MAPE}
=
\frac{100\%}{n}
\sum_{i=1}^{n}
\left|
\frac{y_i - \hat{y}_i}{y_i}
\right|
$$

实际计算时常写成：

$$
\mathrm{MAPE}
=
\frac{100\%}{n}
\sum_{i=1}^{n}
\left|
\frac{y_i - \hat{y}_i}{y_i + \epsilon}
\right|
$$

### 各项意义

- 分子：绝对误差；
- 分母：真实值；
- 整体表示相对误差百分比。

### 特点

1. **关注相对误差，而不是绝对误差**
2. **对小 $y_i$ 极度敏感**
3. **当 $y_i$ 接近 0 时可能爆炸**
4. **对不同量级的数据更公平**
5. **不适合真实值可以为 0 或正负混合的数据**

### NuSR 使用建议

适合：

- 物理量始终为正；
- 不同数据点跨越数量级；
- 你更关心“相对偏差多少百分比”。

不适合：

- $y$ 经常接近 0；
- $y$ 有正有负；
- 小值区域噪声本来就很大。

---

## 2.7 RMSPE：Root Mean Squared Percentage Error，均方根百分比误差

### 表达式

$$
\mathrm{RMSPE}
=
100\%
\sqrt{
\frac{1}{n}
\sum_{i=1}^{n}
\left(
\frac{y_i - \hat{y}_i}{y_i + \epsilon}
\right)^2
}
$$

### 各项意义

RMSPE 是“百分比误差”的均方根版本。

### 特点

1. **比 MAPE 更惩罚大的相对误差**
2. **同样对 $y_i \approx 0$ 敏感**
3. **适合相对误差很重要的场景**

### NuSR 使用建议

适合：

- 你希望控制相对误差；
- 大相对误差必须被强烈惩罚；
- 数据值始终远离 0。

---

## 2.8 SMAPE：Symmetric Mean Absolute Percentage Error，对称平均绝对百分比误差

### 表达式

常见形式：

$$
\mathrm{SMAPE}
=
\frac{100\%}{n}
\sum_{i=1}^{n}
\frac{
2|y_i - \hat{y}_i|
}{
|y_i| + |\hat{y}_i| + \epsilon
}
$$

### 各项意义

- 分子：两倍绝对误差；
- 分母：真实值和预测值绝对值之和；
- $\epsilon$：防止分母为 0。

### 特点

1. **比 MAPE 更对称**
2. **不会只由真实值 $y_i$ 控制分母**
3. **当 $y_i$ 和 $\hat{y}_i$ 都接近 0 时仍然不稳定**
4. **数值范围通常更容易控制**

### NuSR 使用建议

适合：

- 数据有不同量级；
- 想用相对误差，但 MAPE 对小 $y$ 太敏感；
- 预测值和真实值都应参与尺度归一化。

---

## 2.9 MSLE：Mean Squared Logarithmic Error，均方对数误差

### 表达式

$$
\mathrm{MSLE}
=
\frac{1}{n}
\sum_{i=1}^{n}
\left[
\log(1+y_i) - \log(1+\hat{y}_i)
\right]^2
$$

### 各项意义

- 对真实值和预测值先取 $\log(1+\cdot)$；
- 再计算平方误差。

### 特点

1. **更关注相对比例，而不是绝对差值**
2. **适合正值且跨数量级的数据**
3. **不适合负值数据**
4. **对低估和高估的惩罚不完全对称**

### NuSR 使用建议

适合：

- 截面、寿命、强度等跨数量级的正物理量；
- 希望公式捕捉数量级趋势；
- 不希望大数值区域完全支配训练。

不适合：

- $y$ 或 $\hat{y}$ 可能小于 $-1$；
- 目标值有正负号变化；
- 物理上绝对误差比比例误差更重要。

---

## 2.10 RMSLE：Root Mean Squared Logarithmic Error，均方根对数误差

### 表达式

$$
\mathrm{RMSLE}
=
\sqrt{
\frac{1}{n}
\sum_{i=1}^{n}
\left[
\log(1+y_i) - \log(1+\hat{y}_i)
\right]^2
}
$$

### 特点

RMSLE 是 MSLE 的平方根，数值尺度比 MSLE 更直观。它仍然适合正值、跨数量级、关注比例关系的数据。

---

## 2.11 COD / $R^2$：Coefficient of Determination，决定系数

### 表达式

$$
R^2
=
1 -
\frac{
\sum_{i=1}^{n}(y_i - \hat{y}_i)^2
}{
\sum_{i=1}^{n}(y_i - \bar{y})^2
}
$$

如果作为 loss 使用，通常写成：

$$
\mathrm{CODLoss}
=
1 - R^2
=
\frac{
\sum_{i=1}^{n}(y_i - \hat{y}_i)^2
}{
\sum_{i=1}^{n}(y_i - \bar{y})^2 + \epsilon
}
$$

### 各项意义

- 分子：模型残差平方和；
- 分母：真实数据相对均值的总波动；
- $R^2$：模型解释了多少数据方差；
- $R^2=1$：完美拟合；
- $R^2=0$：和直接预测均值差不多；
- $R^2<0$：比预测均值还差。

### 特点

1. **是归一化后的 MSE 型指标**
2. **便于不同数据集之间比较**
3. **本质仍然受平方误差影响**
4. **当 $y$ 几乎是常数时，分母接近 0，会不稳定**
5. **更常用作评价指标，而不是搜索 loss**

### NuSR 使用建议

适合：

- 用来汇报模型好坏；
- 比较不同数据集上的解释能力；
- 在 PySR 中也可作为搜索 loss，但要注意常数目标值问题。

---

## 2.12 Relative L2 Error，相对二范数误差

### 表达式

$$
\mathrm{RelativeL2}
=
\frac{
\sqrt{\sum_{i=1}^{n}(y_i - \hat{y}_i)^2}
}{
\sqrt{\sum_{i=1}^{n}y_i^2} + \epsilon
}
$$

### 特点

1. **整体相对误差**
2. **不像 MAPE 那样逐点除以 $y_i$**
3. **对单个接近 0 的点不那么敏感**
4. **仍然会受大误差影响**

### NuSR 使用建议

适合：

- 整体量级跨越较大；
- 不想让小 $y_i$ 单点支配 loss；
- 希望衡量整体函数曲线的相对偏差。

---

## 2.13 Weighted MSE，加权均方误差

### 表达式

$$
\mathrm{WeightedMSE}
=
\frac{
\sum_{i=1}^{n}
w_i (y_i - \hat{y}_i)^2
}{
\sum_{i=1}^{n} w_i
}
$$

### 各项意义

- $w_i$：第 $i$ 个样本的权重；
- 权重大，说明这个点更重要；
- 权重小，说明这个点影响较弱。

### 特点

1. **可以表达实验点可信度**
2. **可以让高精度数据点更重要**
3. **可以人为强调某些物理区域**
4. **权重设计会张�物质近似不可压缩、密度近似常数时最直接的几何结果，也是许多体积、表面积、库仑能和几何截面公式的共同起点。
**适用范围** 适用于全局尺度估计；对轻核、halo 核、强形变核或精细电荷半径系统学不足。
**NuSR 用途** 应当作为核物理 operator library 中的基础原子项；同时也是许多更复杂模型的 shared latent scale。
**适合抽取的先验** $A^{1/3}$、$A^{2/3}$、$A^{-1/3}$、$\pi R^2$、$R^2$、$R^3$。
**实现建议** 在 NuSR 中不应把 $A^{1/3}$ 当作搜索偶然发现的结果，而应把它当作原生 token。

**二参数 Fermi 分布与 Woods–Saxon 密度分布**

**公式**
$$
\rho(r)=\frac{\rho_0}{1+\exp\!\left(\frac{r-c}{a}\right)}.
$$

**物理含义** 这是描述有限核表面弥散最经典的 logistic 轮廓。作为“电荷密度分布”时，人们常说 two-parameter Fermi distribution；作为“势或密度轮廓”时，人们常说 Woods–Saxon form。对 NuSR 来说，它们在数学上几乎就是同一个核心算子。
**变量解释** $c$ 是半密度半径，$a$ 是表面弥散参数，$\rho_0$ 是中心密度。常把表面厚度写成 $t=4a\ln 3$。
**适用范围** 电子散射、平均场势、光学模型、反应和散射建模。
**NuSR 用途** 非常适合作为 custom operator：`ws(x)=1/(1+exp(x))` 与 `dws(x)=ws(x)(1-ws(x))`。
**适合抽取的先验** $(r-R)/a$、$e^{(r-R)/a}$、$ws((r-R)/a)$、$\mathrm d ws/\mathrm dr$。

**谐振子半径、电荷半径经验式与中子皮关系**

**公式**
$$
\langle r^2\rangle_{n\ell}=b^2\left(2n+\ell+\frac{3}{2}\right),
$$
而经验电荷半径常写为
$$
R_c \approx r_0A^{1/3}+r_1+r_2 I+r_3/A,
$$
中子皮最常见的经验写法则是
$$
\Delta r_{np}\equiv R_n-R_p\approx a+bI,
$$
或更一般地与对称能斜率 $L$ 建立相关。

**物理含义** 谐振子半径给出与壳层主量子数直接关联的平均尺度；电荷半径经验式把 $A^{1/3}$ 的主导趋势与壳效应、同位旋和 odd-even staggering 分离出来；中子皮关系则把核结构问题与核物质 EOS 的对称能斜率联结起来。
**适用范围** 电荷半径与中子皮不是单一闭式公式可以完全统治的领域，参数形式高度拟合依赖。
**NuSR 用途** 更适合作为 feature engineering 与 future-task target，而不是第一阶段主任务。
**适合抽取的先验** $b$、$2n+\ell+3/2$、$I$、$I^2$、$A^{1/3}$、$1/A$、magic-distance、odd-even indicators。

**谐振子势、Woods–Saxon 势、自旋轨道项与有限核 Coulomb 势**

**公式**
$$
V_{HO}(r)=\frac{1}{2}m\omega^2r^2,\qquad
V_{WS}(r)=-V_0\,f(r),\quad f(r)=\frac{1}{1+\exp\!\left(\frac{r-R}{a}\right)},
$$
$$
V_{ls}(r)=V_{so}\frac{1}{r}\frac{\mathrm df}{\mathrm dr}\,\mathbf l\!\cdot\!\mathbf s,
$$
$$
V_C(r)=
\begin{cases}
\dfrac{Ze^2}{2R_C}\left(3-\dfrac{r^2}{R_C^2}\right), & r<R_C,\$$
4pt]
\dfrac{Ze^2}{r}, & r\ge R_C.
\end{cases}
$$

**物理含义** 谐振子势是最简单的壳模型平均场；Woods–Saxon 势提供更真实的有限表面；自旋轨道项是 magic numbers 得以正确出现的关键；有限核 Coulomb 势给出核内外不同的电静势轮廓。
**适用范围** 单粒子平均场、壳模型直觉、近似能级与反应计算。
**NuSR 用途** 最适合做 operator family，而不是直接做拟合终点。
**适合抽取的先验** $r^2$、$f(r)$、$f'(r)$、$l\cdot s$、$1/r$、分段 Coulomb 轮廓。
**实现建议** 对 NuSR 来说，应优先原生支持 `ws`, `dws`, `rotor`, `protected_inv_r` 等保护算子。

**Nilsson 模型、形变 Woods–Saxon 与光学模型势**

**公式**
$$
H_{\rm Nilsson}=H_{\rm osc}-2\kappa\hbar\omega\,\mathbf l\!\cdot\!\mathbf s-\kappa\mu\hbar\omega\left(\mathbf l^2-\langle \mathbf l^2\rangle_N\right)+V_{\rm def},
$$
形变 Woods–Saxon 常写成
$$
V(r,\theta)=-\frac{V_0}{1+\exp\!\left(\frac{r-R(\theta)}{a}\right)},\qquad
R(\theta)=R_0\!\left[1+\sum_\lambda \beta_\lambda Y_{\lambda 0}(\theta)\right],
$$
而光学模型势的通式是
$$
U(r,E)=V(r,E)+iW(r,E)+V_{so}(r)\,\mathbf l\!\cdot\!\mathbf s+V_C(r).
$$

**物理含义** Nilsson 模型把形变与自旋轨道耦合放到单粒子谱里，是研究转动带、壳闭合与形变壳效应的经典语言；光学模型则把弹性与非弹性道的平均效应压缩成复势，是核反应建模的核心输入。
**适用范围** 这是高度有用、但参数空间和隐变量都比第一阶段质量拟合复杂得多的模型族。
**NuSR 用途** 最适合作为 teacher model、operator family 和将来 reaction module 的知识后端。
**为什么不宜第一阶段原生纳入** 因为它们通常需要 $E$、$\ell$、$j$、$\beta_2,\beta_4$、道耦合与目标自旋等额外上下文，超出“只做质量/结合能”的最小闭环。

这一节最值得直接软件化的不是“完整 Nilsson/OMP 计算器”，而是 Woods–Saxon 轮廓、其导数、$A^{1/3}$ 标度、形变半径表示 $R(\theta)$ 以及有限核 Coulomb 轮廓。它们是未来很多核物理 symbolic search 的共享基础。

## 能级密度与衰变经验公式

这一组公式有一个共同特征：它们包含大量对 NuSR 很有价值的非线性结构，例如平方根、指数、对数半衰期、奇偶修正和分段匹配。相比完整反应模型，它们更容易成为高可解释度的第二阶段任务。

**Bethe 费米气体级密度公式**

**公式**
$$
\rho(U)\approx \frac{\exp\!\left(2\sqrt{aU}\right)}{12\sqrt{2}\,a^{1/4}U^{5/4}},
$$
其中 $U$ 为有效激发能，$a$ 为级密度参数。

**物理含义** 这是高激发能级密度最经典的统计表达，来自把高度激发的核视作费米气体。核心结构不是参数值，而是 $e^{2\sqrt{aU}}$ 这一非常强的增长律。
**适用范围** 更适合中高激发能区；低激发区通常需要 constant-temperature 或匹配模型。
**NuSR 用途** 很适合作为 level-density baseline，也适合作为 operator template。
**适合抽取的先验** $\sqrt{U}$、$\sqrt{aU}$、$U^{-5/4}$、$\exp(2\sqrt{aU})$。
**注意** 对 NuSR 来说，$a$ 最好首先当作可学习系数或由 $A$ 派生的 feature，而不是自由函数。

**Constant Temperature、BSFG、Gilbert–Cameron 与自旋截断因子**

**公式**
$$
\rho_{CT}(U)=\frac{1}{T}\exp\!\left(\frac{U-E_0}{T}\right),
$$
$$
\rho_{BSFG}(U)\propto \frac{\exp\!\left(2\sqrt{a(U-\Delta)}\right)}{a^{1/4}(U-\Delta)^{5/4}},
$$
$$
\rho(U,J)\approx \rho(U)\frac{2J+1}{2\sigma^2}\exp\!\left[-\frac{(J+1/2)^2}{2\sigma^2}\right].
$$

**物理含义** CT 模型描述低激发能区的近指数增长；BSFG 把配对/壳效应通过 back-shift $\Delta$ 掺入费米气体表达；Gilbert–Cameron 模型则在低能用 CT、高能用 Fermi-gas，并通过匹配能量把两者拼接。自旋截断因子负责从总级密度分解到固定 $J$ 的部分级密度。
**适用范围** 这是核反应实际计算中最常用的一组经验模型，也是 IAEA RIPL 推荐与整理的主流输入形式。
**NuSR 用途** 它们非常适合做 phase-2 baseline 与 residual symbolic regression；其中最适合先实现的其实不是“自己拟全局参数”，而是先把这些公式作为标准模板导入。
**适合抽取的先验** $U-\Delta$、$1/T$、$\exp((U-E_0)/T)$、$\sigma^2$、$(2J+1)e^{-(J+1/2)^2/2\sigma^2}$。

**α 衰变经验公式族**

**公式**
$$
\log_{10}T_{1/2}\approx a\frac{Z_d}{\sqrt{Q_\alpha}}+b
\qquad\text{(Geiger–Nuttall)},
$$
$$
\log_{10}T_{1/2}=(aZ+b)Q_\alpha^{-1/2}+cZ+d+h_{\log}
\qquad\text{(Viola–Seaborg)},
$$
$$
\log_{10}T_{1/2}=9.54\,\frac{Z_d^{0.6}}{\sqrt{Q_\alpha}}-51.37
\qquad\text{(Brown, e-e)},
$$
$$
\log_{10}T_{1/2}=a+bA^{1/6}\sqrt{Z}+c\frac{Z}{\sqrt{Q_\alpha}}
\qquad\text{(Royer)}.
$$

**物理含义** 这一族公式是 NuSR 做衰变的最佳切入口，因为它们结构简单、变量很少，而且物理直觉清晰：半衰期对 $Q_\alpha^{-1/2}$ 和电荷数高度敏感，奇偶阻碍项 $h_{\log}$ 或奇偶分组参数捕捉未配对核子的影响。
**适用范围** 重核、超重核尤为常见；不同公式对奇偶类和壳闭合区的表现不同。
**NuSR 用途** 这是最适合做 residual symbolic regression 的衰变任务。先用 Royer 或 Viola–Seaborg 做 baseline，再学习对 magic numbers、形变或局域结构的修正，往往比“从零搜索 $\log T_{1/2}$”更稳定。
**适合抽取的先验** $Q_\alpha^{-1/2}$、$Z_d/\sqrt{Q_\alpha}$、$Z_d^{0.6}/\sqrt{Q_\alpha}$、$A^{1/6}\sqrt Z$、hindrance indicators。

**β 衰变 $ft$ 值关系与 Fermi 理论**

**公式**
$$
ft=\frac{K}{g_V^2B_F+g_A^2B_{GT}},
$$
$$
\lambda=\frac{\ln 2}{t_{1/2}}\propto \int F(Z,W)\,pW(W_0-W)^2\,{\rm d}W,
$$
更精密的超允许 $0^+\!\to 0^+$ 关系常写成
$$
\mathcal Ft=ft(1+\delta_R')(1+\delta_{NS}-\delta_C).
$$

**物理含义** 与 α 衰变不同，β 衰变的主要难点不在于经验函数的非线性，而在于矩阵元、选择定则和辐射修正。
**适用范围** 如果只做“经验规律发现”，β 衰变比 α 衰变复杂得多；但 $ft$ 值关系仍然是很好的结构化入口。
**NuSR 用途** 它更适合作为 future constrained task：用已经评估的 $ft$ 数据训练结构化修正项，而不是从实验半衰期直接无约束拟合。
**适合抽取的先验** $\log ft$、$Q_\beta$、Fermi function $F(Z,W)$、allowed / forbidden 标签、$B_F$、$B_{GT}$ 代理量。
**为什么不宜过早纳入** 因为如果没有高质量选择定则与矩阵元标签，符号回归很容易把核结构未建模效应错误吸收到经验系数里。

**自发裂变半衰期经验式**

**公式**
$$
\log_{10}T_{SF}=a+b\chi+c\chi^2+h_{odd}+d\,\delta W_{\rm shell},
\qquad \chi=\frac{Z^2}{A},
$$
其中具体系数取决于 Swiatecki、Sasakawa 或更现代的拟合形式；某些版本只显式依赖 $Z^2/A$，某些版本还加入壳修正、配对或 $Q_\alpha$ 相关信息。

**物理含义** 自发裂变半衰期系统学强烈依赖 fissility、壳修正与 odd-even blocking。
**适用范围** 主要用于锕系和超重核。
**NuSR 用途** 它适合作为中期任务：先实现文献中的经验式，再把壳修正或局域结构残差交给符号回归。
**适合抽取的先验** $Z^2/A$、$(Z^2/A)^2$、壳修正代理、奇偶阻碍项、与 $Q_\alpha$ 的联合特征。
**注意** 与 α 衰变相比，这个任务更容易出现数据稀疏和类别不平衡，因此更建议做 residual learning 而不是从头搜索。

从 NuSR 视角看，能级密度与 α 衰变是非常自然的第二阶段扩展：两者既有经典 baseline，又有明确的非线性结构和足够强的物理可解释性；β 衰变和自发裂变则更适合在完成标签清理与结构化约束后再进入标准训练管线。

## 反应、截面、裂变与集体模型

核反应和集体模型的经验公式，通常比质量公式更依赖额外上下文：能量、角动量、宇称、入射道、出射道、形变与靶核自旋。因而它们往往不适合作为 NuSR 的最初主战场，但非常适合作为后续 operator library 与 physics-aware teacher model。

**Breit–Wigner、Hauser–Feshbach 与 Weisskopf 蒸发公式**

**公式**
$$
\sigma_{BW}(E)=\frac{\pi}{k^2}\frac{2J+1}{(2s_a+1)(2s_A+1)}
\frac{\Gamma_{in}\Gamma_{out}}{(E-E_R)^2+\Gamma^2/4},
$$
$$
\sigma_{ab}^{HF}\propto
\sum_{J,\pi}\frac{2J+1}{(2I_a+1)(2I_A+1)}
\frac{T_a^{J\pi}T_b^{J\pi}}{\sum_c T_c^{J\pi}}W_{ab},
$$
$$
\Gamma_j^{W}(E^*)\propto \frac{1}{\rho_P(E^*)}
\int_0^{E^*-B_j}\sigma_j^{inv}(\epsilon)\,
\rho_D(E^*-B_j-\epsilon)\,\epsilon\,{\rm d}\epsilon.
$$

**物理含义** Breit–Wigner 是单共振截面的标准局域表达；Hauser–Feshbach 是复合核统计反应的核心框架；Weisskopf 公式给出高激发核蒸发粒子宽度的统计表达。
**适用范围** 它们对应的不是单一“全球经验律”，而是从局域共振到统计平均再到蒸发链的不同层次。
**NuSR 用途** 最适合做 operator/template，而不是最初的 end-to-end 回归目标。尤其是 $k^{-2}$、$\Gamma_{in}\Gamma_{out}$、$(E-E_R)^2+\Gamma^2/4$ 与 $T_aT_b/\sum T_c$ 这些结构，非常适合编码成受保护模板。
**适合抽取的先验** $k^{-2}$、Lorentzian 分母、宽度乘积、transmission coefficients、$\rho(E)$ 比值、逆反应截面。

**$S$-factor、Gamow 因子、几何截面与经验反应截面**

**公式**
$$
\sigma(E)=\frac{S(E)}{E}e^{-2\pi\eta},
\qquad
\eta=\frac{Z_1Z_2e^2}{\hbar v},
$$
$$
\sigma_g\sim \pi R^2,
$$
重离子总反应截面的经验式常可概括为
$$
\sigma_R\approx \pi r_0^2\left(A_p^{1/3}+A_t^{1/3}-c\right)^2\Phi(E),
$$
而光学模型里的总反应截面则可写成
$$
\sigma_R=\frac{\pi}{k^2}\sum_\ell (2\ell+1)\left(1-|S_\ell|^2\right).
$$

**物理含义** $S$-factor 与 Gamow 因子把带库仑势垒的低能带电反应截面分解为“强相互作用平滑部分”和“库仑穿透指数部分”；几何截面给出核半径控制下的第一近似；Kox/Shen/Tripathi 一类公式则把重离子总反应截面压缩成可计算的经验参数化。
**适用范围** 天体核物理、低能带电反应、重离子传输和总反应截面系统学。
**NuSR 用途** 这组公式非常适合做 custom operator library。最重要的不是记住某一组经验系数，而是把 $e^{-2\pi\eta}$、$S(E)/E$、$\pi R^2$ 和 optical $S_\ell$ 结构变成可复用数学模块。
**适合抽取的先验** $\eta$、$e^{-2\pi\eta}$、$S(E)$、$1/E$、$A_p^{1/3}+A_t^{1/3}$、$\sqrt{\sigma/\pi}$。

**Bohr–Wheeler 裂变宽度、液滴裂变势垒、转动谱、振动谱与转动惯量**

**公式**
$$
\Gamma_f^{BW}(E,J)\approx \frac{1}{2\pi\rho_{gs}(E,J)}
\int_0^{E-B_f}\rho_{sp}(E-B_f-\epsilon,J)\,{\rm d}\epsilon
\approx \frac{T}{2\pi}\frac{\rho_{sp}(E-B_f,J)}{\rho_{gs}(E,J)},
$$
$$
E_{rot}(J)=\frac{\hbar^2}{2\mathcal J}J(J+1),
\qquad
E(J)=AJ(J+1)-B[J(J+1)]^2+\cdots,
$$
$$
E_{vib}\approx \left(n+\frac12\right)\hbar\omega.
$$
经验可变转动惯量模型还常写成
$$
E_I=\frac{\hbar^2I(I+1)}{2\mathcal J_I}+\frac{1}{2}C(\mathcal J_I-\mathcal J_0)^2.
$$

**物理含义** Bohr–Wheeler 把裂变看成过势垒的过渡态问题；液滴裂变势垒受表面能与库仑能竞争控制，常以 fissility $Z^2/A$ 或更细致的形状函数来组织；转动能谱的 $J(J+1)$ 律是形变核最常见、也最可解释的经验规律之一；振动谱则在小振幅极限下给出近似谐振子结构。
**适用范围** 锕系与重核的转动带、裂变宽度与势垒建模。
**NuSR 用途** 这一组公式很适合做 post-hoc validation 与 specialized templates。尤其是 $J(J+1)$ 及其高阶修正，对“公式是否像核结构公式”有极强的判别力。
**适合抽取的先验** $J(J+1)$、$[J(J+1)]^2$、$\mathcal J^{-1}$、$B_f$、$Z^2/A$、$\rho_{sp}/\rho_{gs}$。

这一节对 NuSR 的核心启示是：反应与集体模型并不缺公式，缺的是“可复用算子层”。如果将来 NuSR 要扩展到截面与能谱，最好先建设 Gamow、Lorentzian、Woods–Saxon、$J(J+1)$、transmission coefficient、level-density ratio 这些数学接口。

## 面向 NuSR 的实现建议

对 NuSR 而言，最重要的不是把某篇论文的全部参数逐字搬进 Python，而是把文献中的结构拆成四类软件对象：平滑 baseline、局部关系层、显式特征、受保护算子。PySR / SymbolicRegression.jl 已支持用户自定义算子、复杂度惩罚和人类可解释表达式搜索，因此合理的工作流应当是：先把物理公认的结构写成 searchable prior，再用符号回归学习“剩余且可压缩的部分”。

**最适合作为 baseline model 的公式族**

最适合直接做 baseline 的，是闭式、低维、覆盖广、物理解释直观的公式：核质量中的 SEMF/LDM；能级密度中的 Bethe Fermi-gas、BSFG 与 Gilbert–Cameron；α 衰变中的 Viola–Seaborg 与 Royer；截面中的几何截面与局域 Breit–Wigner。它们有两个共同优点：一是变量少，二是残差本身往往仍保留明确物理结构，因此非常利于第二层符号回归。

**最适合作为 residual symbolic regression 基准的公式族**

最适合做 residual baseline 的，不一定是最简单的公式，而是已经足够强、但仍保留可解释残差的公式：质量上的 FRDM 与 Duflo–Zuker；α 衰变上的 Royer、Brown 与 Viola–Seaborg；级密度上的 BSFG / Gilbert–Cameron；自发裂变上的 Swiatecki 型系统学。NuSR 若在这些 baseline 之上学习残差，往往会自动把学习重点集中到壳闭合、奇偶效应、形变和局域异常上。

**最适合拆解成 feature terms 的结构**

最值得显式特征工程化的，不是“整条公式”，而是公式里的原子项。就质量任务而言，最核心的是 $A$、$Z$、$N$、$I=(N-Z)/A$、$A^{1/3}$、$A^{2/3}$、$A^{-1/3}$、$Z(Z-1)$、$(N-Z)^2/A$、奇偶指示量、距 magic numbers 的距离。就衰变而言，最核心的是 $Q_\alpha^{-1/2}$、$Z_d/\sqrt{Q_\alpha}$、$Z_d^{0.6}/\sqrt{Q_\alpha}$、$\log ft$ 和阻碍因子。就反应而言，最核心的是 $\eta$、$e^{-2\pi\eta}$、$k^{-2}$、$J(J+1)$、$A_p^{1/3}+A_t^{1/3}$ 与 level-density ratio。

**最适合转化为 custom operators 的结构**

NuSR 中最值得原生支持的 custom operators，不是 abstruse 的论文公式，而是跨多个模型重复出现的“共享数学块”：`cbrt(x)`、`ws(x)=1/(1+exp(x))`、`dws(x)`、`symmetry(A,N,Z)=((N-Z)^2)/A`、`gamow(E,Z1,Z2,mu)=exp(-2*pi*eta)`、`rotor(J)=J*(J+1)`、`lorentz(E,Er,Gamma)`、`protected_log`、`protected_sqrt`、`finite_coulomb(r,Rc)`。这些算子一旦进入 operator library，就能同时服务质量、密度、衰变和反应四类任务。

**最适合实现为 hard constraints 的规则**

最明确的 hard constraints 包括：$A=N+Z$；质量、宽度、级密度、反应截面与半衰期必须为正；2pF / Woods–Saxon 密度应有界且随半径单调下降；IMME 在固定多重态中应首先满足二次 $T_z$ 结构；Gilbert–Cameron 模型在匹配能量处应连续；同位素链上的某些推导量，例如 $Q$-value、$S_n$、$S_p$，必须与最终质量自洽。对 NuSR 来说，这些规则更适合放进 constraint layer，而不是等模型训练完再“解释”。

**最适合实现为 soft penalty / custom loss 的规则**

软惩罚更适合处理那些“物理上应大致成立，但不必逐点严格成立”的规律。例如：大 $A$ 极限下 $B/A$ 不应发散；离 magic numbers 很远时壳修正不应异常放大；对固定 $Z$ 的 α 衰变链，$\log T_{1/2}$ 对 $Q_\alpha$ 应有稳定单调趋势；级密度拟合不应在低能或阈值附近出现非物理奇点；截面与宽度拟合应避免在能窗边界处爆炸式外推。复杂度惩罚也应对“无物理意义但能降低训练误差”的高阶拼接项保持敏感。

**最适合作为 post-hoc validation rule 的规律**

最强的 post-hoc validation 规则包括：Garvey–Kelson 关系；IMME 二次律；质量—分离能—衰变能之间的代数一致性； α 衰变链上 $\log T_{1/2}$ 与 $Q_\alpha^{-1/2}$ 的近线性；转动带中的 $E(J)\approx AJ(J+1)$；低能带电粒子反应中的 $S(E)$ 比原始截面更平滑。这些规则既能筛查候选表达式，也能作为模型择优的第二评价指标。

**不建议第一阶段纳入的模型与原因**

不建议在第一阶段直接纳入的，是完整 FRDM、完整 Duflo–Zuker、Nilsson/形变 WS 的全参数版、完整光学模型、完整 Hauser–Feshbach，以及高度结构依赖的 β 衰变矩阵元系统学。原因不是它们“不重要”，恰恰相反，它们太重要、也太复杂：需要额外隐变量、壳占据、道耦合、自旋宇称标签、丰富的目标核上下文与更严格的数据清洗。如果在 NuSR 第一阶段就把这些对象作为自由符号空间的一部分，搜索会被不必要地稀释。

**如果第一阶段只做核质量 / 结合能，最推荐的实现栈**

最推荐的最小实现栈是：
第一，AMe2020 作为标准数据后端，并统一派生 $S_n$、$S_p$、$Q_\alpha$ 等二级量；
第二，SEMF/LDM 作为 baseline；
第三，把对称能、对能、Wigner 项、magic-distance 作为显式特征；
第四，把 Garvey–Kelson 与 IMME 作为关系/验证层；
第五，FRDM 与 Duflo–Zuker 仅作为 teacher baseline，用于 residual target 或蒸馏比较。
这样做可以在不把搜索空间做得过大的前提下，已经把“全球平滑 + 局域关系 + 局部物理修正”三层结构建起来。

**未来扩展路线图**

最合理的扩展顺序是：
先做核质量 / 结合能；
再做 α 衰变与简单电荷半径；
随后做能级密度与 Gilbert–Cameron/BSFG residual；
再进入自发裂变与经整理的 β 衰变 $ft$ 数据；
最后再进入反应截面、光学模型与 Hauser–Feshbach。
配套数据源则应分别绑定 AME2020、ENSDF/NuDat、IAEA RIPL-3 和 EXFOR/反应数据库，而不是混用未经评估的网页数据。

**可转化为 NuSR 特征库的数学项**

| 类别 | 推荐数学项 |
|---|---|
| 质量与结合能 | $A,Z,N,I,A^{1/3},A^{2/3},A^{-1/3},A^{-1},Z(Z-1),(N-Z)^2/A,|N-Z|$ |
| 奇偶与壳效应 | `is_even_N`, `is_even_Z`, `is_even_A`, $d_N^{\rm magic}, d_Z^{\rm magic}$, shell-closure flags |
| 半径 / 密度 | $R=r_0A^{1/3}$, $(r-R)/a$, $ws((r-R)/a)$, $dws/dr$, $R_c$ proxy, $\Delta r_{np}$ proxy |
| 单粒子势 | $r^2$, $l\cdot s$, $1/r$, finite Coulomb inside/outside pieces |
| 级密度 | $\sqrt{U}$, $\exp(2\sqrt{aU})$, $U^{-5/4}$, $U-\Delta$, $1/T$, $\sigma^2$, $J(J+1)$ |
| 衰变 | $Q_\alpha^{-1/2}$, $Z_d/\sqrt{Q_\alpha}$, $Z_d^{0.6}/\sqrt{Q_\alpha}$, $\log ft$, hindrance flags |
| 反应 | $\eta$, $e^{-2\pi\eta}$, $S(E)/E$, $k^{-2}$, $A_p^{1/3}+A_t^{1/3}$, $\sqrt{\sigma/\pi}$ |
| 集体模型 | $J(J+1)$, $[J(J+1)]^2$, $\mathcal J^{-1}$, $Z^2/A$, $B_f$, $\rho_{sp}/\rho_{gs}$ |

这些项大多并不需要 NuSR 去“发现”；它们应优先进入 `feature_registry` 或 `operator_registry`。真正值得交给符号回归去学习的，是这些项之间的组合方式、修正强度和局域偏离。

**可转化为 NuSR 约束库的物理规则**

| 物理规则 | 建议类型 | 说明 |
|---|---|---|
| $A=N+Z$ | hard constraint | 质量任务基础守恒关系 |
| $B>0,\ \rho>0,\ \sigma\ge 0,\ \Gamma>0,\ T_{1/2}>0$ | hard constraint | 正值物理量不能越界 |
| 2pF / WS 密度单调下降 | hard / soft | 取决于是否允许局域表面修正 |
| IMME 在固定多重态中为二次 $T_z$ | hard / validation | 少数异常再交给高阶项 |
| Gilbert–Cameron 在匹配能处连续 | hard | 模型定义要求 |
| 大 $A$ 极限下 $B/A$ 有界 | soft penalty | 抑制发散表达式 |
| α 衰变链上 $\log T_{1/2}$ 对 $Q_\alpha$ 单调 | validation | 针对固定链或固定 $Z$ |
| 转动带低能级满足近 $J(J+1)$ 结构 | validation | 检查像不像“核谱公式” |
| mass → $S_n,S_p,Q$ 自洽 | hard / validation | 由同一质量表导出的量需一致 |

这组规则的意义在于，它们能把 NuSR 从“拟合数字的表达式搜索器”变成“受核物理守则约束的表达式发现器”。

## 推荐后续阅读与数据源

如果把这份文档继续扩写成 NuSR `docs/` 的常驻知识库，我建议后续阅读遵循“原始论文 → 现代综述 → 官方评估数据库”的顺序，而不是反过来。下面这份清单优先保留那些对软件实现最有直接价值的来源。

**质量与宏观—微观模型**

- C. F. von Weizsäcker, *Zur Theorie der Kernmassen* (1935), Zeitschrift für Physik, DOI: 10.1007/BF01337700.
- H. A. Bethe, R. F. Bacher, *Nuclear Physics. A. Stationary States of Nuclei* (1936), Rev. Mod. Phys. 8, 82–229, DOI: 10.1103/RevModPhys.8.82.
- W. D. Myers, W. J. Świątecki, *Nuclear masses and deformations* (1966), Nuclear Physics 81, 1–60, DOI: 10.1016/0029-5582(66)90639-0.
- W. D. Myers, *Droplet Model of Atomic Nuclei* (1977), monograph.
- P. Möller, J. R. Nix, W. D. Myers, W. J. Świątecki, *Nuclear Ground-State Masses and Deformations* (1995), Atomic Data and Nuclear Data Tables 59, 185–381, DOI: 10.1006/adnd.1995.1002.
- J. Duflo, A. P. Zuker, *Microscopic mass formulas* (1995), Phys. Rev. C 52, R23, DOI: 10.1103/PhysRevC.52.R23.
- G. T. Garvey et al., *Set of nuclear-mass relations and a resultant mass table* (1969), Rev. Mod. Phys. 41, S1, DOI: 10.1103/RevModPhys.41.S1.
- J. MacCormick, G. Audi, *The isobaric multiplet mass equation for $A\le 71$ revisited* (2013), Atomic Data and Nuclear Data Tables 99, 680–703, DOI: 10.1016/j.adt.2012.11.002.

**半径、单粒子势与平均场**

- R. D. Woods, D. S. Saxon, *Diffuse Surface Optical Model for Nucleon-Nuclei Scattering* (1954), Phys. Rev. 95, 577, DOI: 10.1103/PhysRev.95.577.
- S. G. Nilsson, *Binding states of individual nucleons in strongly deformed nuclei* (1955), Dan. Mat. Fys. Medd., DOI: 不确定。
- Maria Goeppert Mayer, *Nuclear Configurations in the Spin-Orbit Coupling Model. I. Empirical Evidence* (1950), Phys. Rev. 78, 16, DOI: 10.1103/PhysRev.78.16.
- A. J. Koning, J. P. Delaroche, *Local and global nucleon optical models from 1 keV to 200 MeV* (2003), Nuclear Physics A 713, 231–310, DOI: 10.1016/S0375-9474(02)01321-0.
- F. Hofmann, *Nuclear Charge Radii* (Springer reference entry, 2022).
- Toshio Suzuki, *The relationship of the neutron skin thickness to the symmetry energy and its slope* (2022), PTEP 063D01, DOI: 10.1093/ptep/ptac083.

**能级密度与衰变**

- H. A. Bethe, *An Attempt to Calculate the Number of Energy Levels of a Heavy Nucleus* (1936), Phys. Rev. 50, 332, DOI: 10.1103/PhysRev.50.332.
- A. Gilbert, A. G. W. Cameron, *A Composite Nuclear-Level Density Formula with Shell Corrections* (1965), Can. J. Phys. 43, 1446, DOI: 10.1139/p65-139.
- A. V. Ignatyuk et al., RIPL / level-density handbooks and recommended files.
- V. E. Viola Jr., G. T. Seaborg, *Nuclear Systematics of the Heavy Elements. II Lifetimes for Alpha, Beta and Spontaneous Fission Decay* (1966), J. Inorg. Nucl. Chem. 28, 741–761, DOI: 10.1016/0022-1902(66)80412-8.
- B. A. Brown, *Simple relation for alpha decay half-lives* (1992), Phys. Rev. C 46, 811, DOI: 10.1103/PhysRevC.46.811.
- G. Royer, *Alpha emission and spontaneous fission through quasi-molecular shapes* (2000), J. Phys. G 26, 1149, DOI: 10.1088/0954-3899/26/8/305.
- J. C. Hardy, I. S. Towner, *Superallowed $0^+\to0^+$ nuclear beta decays* (2005), Phys. Rev. C 71, 055501, DOI: 10.1103/PhysRevC.71.055501.
- W. J. Świątecki, *Systematics of Spontaneous Fission Half-Lives* (1955), Phys. Rev. 100, 937, DOI: 10.1103/PhysRev.100.937.

**核反应、裂变与集体模型**

- G. Breit, E. Wigner, *Capture of Slow Neutrons* (1936), Phys. Rev. 49, 519, DOI: 10.1103/PhysRev.49.519.
- W. Hauser, H. Feshbach, *The Inelastic Scattering of Neutrons* (1952), Phys. Rev. 87, 366, DOI: 10.1103/PhysRev.87.366.
- V. Weisskopf, *Statistics and Nuclear Reactions* (1937), Phys. Rev. 52, 295, DOI: 10.1103/PhysRev.52.295.
- N. Bohr, J. A. Wheeler, *The Mechanism of Nuclear Fission* (1939), Phys. Rev. 56, 426, DOI: 10.1103/PhysRev.56.426.
- A. Bohr, B. Mottelson, *Rotational States in Even-Even Nuclei* (1953), Phys. Rev. 90, 717, DOI: 10.1103/PhysRev.90.717.2.
- M. A. J. Mariscotti et al., *Phenomenological Analysis of Ground-State Bands in Even-Even Nuclei* (1969), Phys. Rev. 178, 1864, DOI: 10.1103/PhysRev.178.1864.
- L. F. Canto et al., *The total reaction cross section of heavy-ion reactions induced by stable and unstable exotic beams: the low-energy regime* (2020), EPJ A 56, 281, DOI: 10.1140/epja/s10050-020-00277-8.
- Thomas Rauscher, *Relevant energy ranges for astrophysical reaction rates* (2010), Phys. Rev. C 81, 045807, DOI: 10.1103/PhysRevC.81.045807.

**官方数据库与评估后端**

- AME2020 / NUBASE2020：核质量与衰变能标准后端。
- ENSDF / NuDat：结构与衰变评估数据。
- IAEA RIPL-3：质量、形变、光学模型、级密度、γ 强度函数与裂变输入库。
- EXFOR：实验核反应截面。
这些资源应成为 NuSR 的数据后端，而不是可有可无的附属下载页。没有标准后端，先验库再好，也很难形成稳定的可复现管线。

这份文档如果继续演化成 NuSR 的长期知识库，我建议把每个条目最终落成统一 schema：`name / formula / variables / domain / baseline_use / residual_use / feature_terms / operators / constraints / references / caveats`。这样后续无论是给 PySR 生成 grammar、给训练脚本自动注入 physical priors，还是给文档系统自动渲染，都能保持一致。
