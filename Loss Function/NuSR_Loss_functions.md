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
4. **权重设计会强烈影响结果**

### NuSR 使用建议

非常适合核物理实验数据，因为不同实验点的不确定度可能不同。

常见选择：

$$
w_i = \frac{1}{\sigma_i^2}
$$

其中 $\sigma_i$ 是实验误差。

---

## 2.14 Chi-square Loss，卡方损失

### 表达式

$$
\chi^2
=
\sum_{i=1}^{n}
\left(
\frac{y_i - \hat{y}_i}{\sigma_i}
\right)^2
$$

也可以写成：

$$
\chi^2
=
\sum_{i=1}^{n}
\frac{(y_i - \hat{y}_i)^2}{\sigma_i^2}
$$

### 各项意义

- $\sigma_i$：第 $i$ 个实验点的不确定度；
- 残差除以实验误差，表示“偏离了几个标准差”。

### 特点

1. **非常适合带实验误差的数据**
2. **高精度点被赋予更大权重**
3. **低精度点不会过度支配拟合**
4. **前提是 $\sigma_i$ 合理可信**

### NuSR 使用建议

这是 NuSR 中非常重要的一类 loss，因为核物理实验数据天然带有不确定度。

---

## 2.15 Reduced Chi-square，约化卡方

### 表达式

$$
\chi^2_{\nu}
=
\frac{\chi^2}{n-k}
$$

其中：

- $n$：样本数；
- $k$：模型参数个数；
- $n-k$：自由度。

### 特点

1. **考虑了模型参数数量**
2. **可用于判断拟合是否与实验误差相称**
3. **常用于物理实验数据分析**
4. **如果 $k$ 定义不合理，解释会变差**

### NuSR 使用建议

适合作为模型评价指标，也可以尝试作为 PySR 搜索 loss。但在 Symbolic Regression 中，$k$ 不一定容易定义，可以用公式中常数个数或表达式节点数近似。

---

## 2.16 AIC：Akaike Information Criterion，赤池信息准则

### 表达式

常见形式：

$$
\mathrm{AIC}
=
n\log(\mathrm{MSE}) + 2k
$$

### 各项意义

- $n$：样本数；
- $\mathrm{MSE}$：均方误差；
- $k$：模型参数数量或复杂度；
- $2k$：复杂度惩罚项。

### 特点

1. **同时考虑拟合误差和模型复杂度**
2. **比 BIC 对复杂模型更宽容**
3. **更偏向预测性能**
4. **绝对值意义不强，主要用于同一数据集上的模型比较**

### NuSR 使用建议

适合在候选公式之间做模型选择，尤其是在你希望避免公式过度复杂时。

---

## 2.17 BIC：Bayesian Information Criterion，贝叶斯信息准则

### 表达式

$$
\mathrm{BIC}
=
n\log(\mathrm{MSE}) + k\log(n)
$$

### 各项意义

- $n\log(\mathrm{MSE})$：拟合误差项；
- $k\log(n)$：复杂度惩罚项；
- $k$：参数数量或复杂度指标。

### 特点

1. **惩罚复杂度比 AIC 更强**
2. **样本数越大，对复杂模型惩罚越明显**
3. **适合选择更简洁的经验公式**
4. **非常符合物理经验公式发现的审美：简单、可解释、误差可接受**

### NuSR 使用建议

非常适合用于 NuSR 的候选公式筛选，尤其是你想避免 PySR 给出过度复杂的“数学拟合怪物”。

注意：BIC 的数值可能为负。如果 PySR 当前版本或你的使用方式强烈假设 loss 非负，可以考虑返回：

$$
\mathrm{BICShifted}
=
\mathrm{BIC} + C
$$

但模型排序只要不变，常数平移不影响比较。

---

## 2.18 Complexity-Regularized MSE，复杂度正则化 MSE

### 表达式

$$
\mathrm{Loss}
=
\mathrm{MSE}
+
\lambda C(f)
$$

其中：

- $C(f)$：公式复杂度，例如节点数、常数数、运算符数量；
- $\lambda$：复杂度惩罚强度。

### 特点

1. **直接惩罚复杂公式**
2. **比 AIC/BIC 更灵活**
3. **需要手动选择 $\lambda$**
4. **可以根据物理审美设计复杂度**

### NuSR 使用建议

适合 NuSR 后期加入物理先验时使用。例如：

- 惩罚过多嵌套；
- 惩罚不希望出现的算子；
- 惩罚不满足量纲结构的表达式；
- 惩罚不满足渐近极限的表达式。

---

# 3. 粗略选择建议

| 目标 | 推荐 loss |
|---|---|
| 默认回归拟合 | MSE / RMSE |
| 抗离群点 | MAE / Huber / Log-Cosh |
| 关注相对误差 | MAPE / RMSPE / SMAPE |
| 正值且跨数量级 | MSLE / RMSLE |
| 整体相对函数误差 | Relative L2 |
| 带实验不确定度 | Weighted MSE / Chi-square |
| 评价解释方差 | COD / $R^2$ |
| 惩罚复杂公式 | AIC / BIC / Complexity-Regularized MSE |
| 物理实验数据拟合 | Chi-square / Weighted MSE / Huber |
| 经验公式发现 | BIC / AIC / Relative L2 / MAPE |

---

# 4. PySR 自定义 Loss Function 代码库

下面的代码风格按照你的要求组织：

- 外层是 Python 变量；
- 每个变量是一个 Julia 后端 loss function 字符串；
- 可以传给 PySRRegressor 的 `loss_function` 参数；
- 所有函数都使用 `eval_tree_array(tree, dataset.X, options)`；
- 遇到非法表达式直接返回极大惩罚；
- 尽量保持 `L(...)` 类型一致，兼容 PySR 的 Float32 / Float64 精度。

```python
# region loss functions
# ==============================================================
# loss function
# ==============================================================

# 1. BIC 函数
# 它在 Julia 后端运行，计算：n * log(MSE) + k * log(n)
# 其中 k 是公式中常数的个数
bic_loss_script = """
# 1. 定义一个辅助函数来统计常数节点数量
function get_k_constants(tree)
    count = 0
    # 遍历树的所有节点
    for node in tree
        # degree == 0 表示叶子节点（常数或变量）
        # .constant 属性判断是否为常数
        if node.degree == 0 && node.constant
            count += 1
        end
    end
    return count
end

# 2. 主损失函数
function my_bic_loss(tree, dataset::Dataset{T,L}, options)::L where {T,L}
    prediction, flag = eval_tree_array(tree, dataset.X, options)

    if !flag
        return L(1e10)
    end

    n = dataset.n

    # 使用我们自己定义的函数统计 k
    k = get_k_constants(tree)
    if k == 0
        k = 1
    end

    mse = sum((prediction .- dataset.y) .^ 2) / n
    mse = max(mse, L(1e-12))

    # BIC 公式
    bic = n * log(mse) + k * log(n)

    return L(bic)
end
"""

# 2. AIC 函数
# 它在 Julia 后端运行，计算：n * log(MSE) + 2k
# 相比 BIC，AIC 对复杂公式的惩罚弱一些
aic_loss_script = """
function get_k_constants(tree)
    count = 0
    for node in tree
        if node.degree == 0 && node.constant
            count += 1
        end
    end
    return count
end

function my_aic_loss(tree, dataset::Dataset{T,L}, options)::L where {T,L}
    prediction, flag = eval_tree_array(tree, dataset.X, options)

    if !flag
        return L(1e10)
    end

    n = dataset.n

    k = get_k_constants(tree)
    if k == 0
        k = 1
    end

    mse = sum((prediction .- dataset.y) .^ 2) / n
    mse = max(mse, L(1e-12))

    aic = n * log(mse) + L(2.0) * k

    return L(aic)
end
"""

# 3. MSE 函数：计算每个数据点的平方误差的均值
# MSE 会强烈惩罚大误差，因此容易迁就极端值
mse_loss_script = """
function my_mse_loss(tree, dataset::Dataset{T,L}, options)::L where {T,L}
    prediction, flag = eval_tree_array(tree, dataset.X, options)

    if !flag
        return L(1e10)
    end

    n = dataset.n

    squared_error = (dataset.y .- prediction) .^ 2
    mse = sum(squared_error) / n

    return L(mse)
end
"""

# 4. RMSE 函数：计算均方根误差
# RMSE 和 y 的量纲一致，但本质上仍然强烈惩罚大误差
rmse_loss_script = """
function my_rmse_loss(tree, dataset::Dataset{T,L}, options)::L where {T,L}
    prediction, flag = eval_tree_array(tree, dataset.X, options)

    if !flag
        return L(1e10)
    end

    n = dataset.n

    squared_error = (dataset.y .- prediction) .^ 2
    mse = sum(squared_error) / n
    rmse = sqrt(max(mse, L(0.0)))

    return L(rmse)
end
"""

# 5. MAE 函数：计算每个数据点的绝对误差的均值
# MAE 比 MSE 更抗离群点
mae_loss_script = """
function my_mae_loss(tree, dataset::Dataset{T,L}, options)::L where {T,L}
    # 1. 评估树的当前表达式，获取预测值。
    prediction, flag = eval_tree_array(tree, dataset.X, options)

    # 2. 遇到非法数学操作（如除零、负数开根号），直接赋予极大的死亡惩罚
    if !flag
        return L(1e10)
    end

    # 获取样本数量
    n = dataset.n

    # 3. 核心计算：直接将真实值与预测值相减，并取绝对值
    # 注意使用 .- 和 abs. 进行数组的逐元素操作
    absolute_error = abs.(dataset.y .- prediction)

    # 4. 求所有样本绝对误差的平均值
    mae = sum(absolute_error) / n

    # 5. 返回结果，确保是 L 类型 (与你的 precision 匹配)
    return L(mae)
end
"""

# 6. Huber 函数
# 小误差区间像 MSE，大误差区间像 MAE
# delta 越小，越接近 MAE；delta 越大，越接近 MSE
huber_loss_script = """
function my_huber_loss(tree, dataset::Dataset{T,L}, options)::L where {T,L}
    prediction, flag = eval_tree_array(tree, dataset.X, options)

    if !flag
        return L(1e10)
    end

    n = dataset.n
    delta = L(1.0)

    residual = dataset.y .- prediction
    abs_residual = abs.(residual)

    # ifelse. 是 Julia 的逐元素条件选择
    loss_each = ifelse.(
        abs_residual .<= delta,
        L(0.5) .* residual .^ 2,
        delta .* (abs_residual .- L(0.5) * delta)
    )

    huber = sum(loss_each) / n

    return L(huber)
end
"""

# 7. Log-Cosh 函数
# 小误差近似 MSE，大误差近似 MAE，并且整体光滑
log_cosh_loss_script = """
function my_log_cosh_loss(tree, dataset::Dataset{T,L}, options)::L where {T,L}
    prediction, flag = eval_tree_array(tree, dataset.X, options)

    if !flag
        return L(1e10)
    end

    n = dataset.n

    residual = dataset.y .- prediction

    # log(cosh(x)) 对大 x 可能溢出
    # 这里使用一个数值稳定形式：
    # log(cosh(x)) = abs(x) + log1p(exp(-2abs(x))) - log(2)
    abs_residual = abs.(residual)
    loss_each = abs_residual .+ log1p.(exp.(-L(2.0) .* abs_residual)) .- log(L(2.0))

    log_cosh = sum(loss_each) / n

    return L(log_cosh)
end
"""

# 8. MAPE 函数：计算每个数据点的绝对百分比误差的均值
# 注意：当 y 接近 0 时，MAPE 会变得非常敏感
mape_loss_script = """
function my_mape_loss(tree, dataset::Dataset{T,L}, options)::L where {T,L}
    prediction, flag = eval_tree_array(tree, dataset.X, options)

    # 遇到非法数学操作，直接返回极大惩罚
    if !flag
        return L(1e10)
    end

    n = dataset.n
    eps = L(1e-10)

    # 核心：使用 abs.() 对数组逐元素求绝对值
    # 注意：这里使用 abs.(dataset.y) .+ eps，而不是 dataset.y .+ eps
    # 这样可以避免 y 为负时分母符号干扰百分比误差
    percentage_error = abs.(dataset.y .- prediction) ./ (abs.(dataset.y) .+ eps)

    # 求均值，乘以 100 变成百分比
    mape = (sum(percentage_error) / n) * L(100.0)

    return L(mape)
end
"""

# 9. RMSPE 函数：计算每个数据点的百分比误差的均方根
# 相比 MAPE，RMSPE 会更强烈惩罚大的相对误差
rmspe_loss_script = """
function my_rmspe_loss(tree, dataset::Dataset{T,L}, options)::L where {T,L}
    # 1. 评估树的当前表达式，获取预测值。
    # flag 返回 false 表示计算过程中出现了非法数学操作（如除以0，负数开偶次根等）
    prediction, flag = eval_tree_array(tree, dataset.X, options)

    # 如果计算非法，赋予极大的死亡惩罚，让该个体直接被淘汰
    if !flag
        return L(1e10)
    end

    # 获取样本数量
    n = dataset.n

    # 2. 定义极小值 eps。使用 L() 确保数据类型与 PySR 当前的精度 (32或64位) 匹配
    eps = L(1e-10)

    # 3. 计算相对误差
    # 注意：Julia 中对数组进行逐元素操作（如相减、相除、乘方）必须在运算符前加点号（.-, ./, .^）
    percentage_error = (dataset.y .- prediction) ./ (abs.(dataset.y) .+ eps)

    # 4. 求均方，开根号，再乘以 100 转为百分比
    mean_sq_err = sum(percentage_error .^ 2) / n
    rmspe = sqrt(mean_sq_err) * L(100.0)

    # 5. 返回必须是 L 类型
    return L(rmspe)
end
"""

# 10. SMAPE 函数：对称平均绝对百分比误差
# 分母同时使用真实值和预测值，因此比 MAPE 更对称
smape_loss_script = """
function my_smape_loss(tree, dataset::Dataset{T,L}, options)::L where {T,L}
    prediction, flag = eval_tree_array(tree, dataset.X, options)

    if !flag
        return L(1e10)
    end

    n = dataset.n
    eps = L(1e-10)

    numerator = L(2.0) .* abs.(dataset.y .- prediction)
    denominator = abs.(dataset.y) .+ abs.(prediction) .+ eps

    smape_each = numerator ./ denominator
    smape = (sum(smape_each) / n) * L(100.0)

    return L(smape)
end
"""

# 11. MSLE 函数：均方对数误差
# 适合正值且跨数量级的数据；如果 y 或 prediction <= -1，则返回死亡惩罚
msle_loss_script = """
function my_msle_loss(tree, dataset::Dataset{T,L}, options)::L where {T,L}
    prediction, flag = eval_tree_array(tree, dataset.X, options)

    if !flag
        return L(1e10)
    end

    n = dataset.n

    # MSLE 要求 1 + y > 0 且 1 + prediction > 0
    if any(dataset.y .<= L(-1.0)) || any(prediction .<= L(-1.0))
        return L(1e10)
    end

    log_error = log1p.(dataset.y) .- log1p.(prediction)
    msle = sum(log_error .^ 2) / n

    return L(msle)
end
"""

# 12. RMSLE 函数：均方根对数误差
# 是 MSLE 的平方根，数值尺度更直观
rmsle_loss_script = """
function my_rmsle_loss(tree, dataset::Dataset{T,L}, options)::L where {T,L}
    prediction, flag = eval_tree_array(tree, dataset.X, options)

    if !flag
        return L(1e10)
    end

    n = dataset.n

    if any(dataset.y .<= L(-1.0)) || any(prediction .<= L(-1.0))
        return L(1e10)
    end

    log_error = log1p.(dataset.y) .- log1p.(prediction)
    msle = sum(log_error .^ 2) / n
    rmsle = sqrt(max(msle, L(0.0)))

    return L(rmsle)
end
"""

# 13. COD / R^2 Loss 函数
# R^2 = 1 - SSE/SST
# 作为 loss 使用时，最小化 1 - R^2，也就是 SSE/SST
cod_loss_script = """
function my_cod_loss(tree, dataset::Dataset{T,L}, options)::L where {T,L}
    prediction, flag = eval_tree_array(tree, dataset.X, options)

    if !flag
        return L(1e10)
    end

    eps = L(1e-12)

    y_mean = sum(dataset.y) / dataset.n

    sse = sum((dataset.y .- prediction) .^ 2)
    sst = sum((dataset.y .- y_mean) .^ 2)

    # 如果 y 几乎是常数，则 SST 接近 0，此时 R^2 不稳定
    cod_loss = sse / (sst + eps)

    return L(cod_loss)
end
"""

# 14. Relative L2 函数
# 计算整体二范数相对误差：||y - yhat||_2 / ||y||_2
# 相比 MAPE，它不会因为单个 y_i 接近 0 而爆炸
relative_l2_loss_script = """
function my_relative_l2_loss(tree, dataset::Dataset{T,L}, options)::L where {T,L}
    prediction, flag = eval_tree_array(tree, dataset.X, options)

    if !flag
        return L(1e10)
    end

    eps = L(1e-12)

    numerator = sqrt(sum((dataset.y .- prediction) .^ 2))
    denominator = sqrt(sum(dataset.y .^ 2)) + eps

    relative_l2 = numerator / denominator

    return L(relative_l2)
end
"""

# 15. Weighted MSE 函数
# 如果 dataset 中有 weights，则使用 dataset.weights
# 如果没有 weights，则退化为普通 MSE
# 常见物理选择：weights = 1 / sigma^2
weighted_mse_loss_script = """
function get_dataset_weights(dataset, n, L)
    weights = try
        getproperty(dataset, :weights)
    catch
        nothing
    end

    if weights === nothing
        return ones(L, n)
    else
        return weights
    end
end

function my_weighted_mse_loss(tree, dataset::Dataset{T,L}, options)::L where {T,L}
    prediction, flag = eval_tree_array(tree, dataset.X, options)

    if !flag
        return L(1e10)
    end

    n = dataset.n
    weights = get_dataset_weights(dataset, n, L)

    squared_error = (dataset.y .- prediction) .^ 2
    weighted_mse = sum(weights .* squared_error) / (sum(weights) + L(1e-12))

    return L(weighted_mse)
end
"""

# 16. Chi-square 函数
# 如果 dataset 中有 weights，则默认 weights = 1 / sigma^2
# 此时 chi-square = sum(weights * residual^2)
# 如果没有 weights，则退化为普通 SSE
chi_square_loss_script = """
function get_dataset_weights(dataset, n, L)
    weights = try
        getproperty(dataset, :weights)
    catch
        nothing
    end

    if weights === nothing
        return ones(L, n)
    else
        return weights
    end
end

function my_chi_square_loss(tree, dataset::Dataset{T,L}, options)::L where {T,L}
    prediction, flag = eval_tree_array(tree, dataset.X, options)

    if !flag
        return L(1e10)
    end

    n = dataset.n
    weights = get_dataset_weights(dataset, n, L)

    residual = dataset.y .- prediction

    chi_square = sum(weights .* residual .^ 2)

    return L(chi_square)
end
"""

# 17. Reduced Chi-square 函数
# reduced chi-square = chi-square / (n - k)
# 这里 k 用公式中的常数数量近似
reduced_chi_square_loss_script = """
function get_k_constants(tree)
    count = 0
    for node in tree
        if node.degree == 0 && node.constant
            count += 1
        end
    end
    return count
end

function get_dataset_weights(dataset, n, L)
    weights = try
        getproperty(dataset, :weights)
    catch
        nothing
    end

    if weights === nothing
        return ones(L, n)
    else
        return weights
    end
end

function my_reduced_chi_square_loss(tree, dataset::Dataset{T,L}, options)::L where {T,L}
    prediction, flag = eval_tree_array(tree, dataset.X, options)

    if !flag
        return L(1e10)
    end

    n = dataset.n
    weights = get_dataset_weights(dataset, n, L)

    k = get_k_constants(tree)
    if k == 0
        k = 1
    end

    dof = max(n - k, 1)

    residual = dataset.y .- prediction
    chi_square = sum(weights .* residual .^ 2)

    reduced_chi_square = chi_square / dof

    return L(reduced_chi_square)
end
"""

# 18. Complexity-Regularized MSE 函数
# loss = MSE + lambda * complexity
# 这里 complexity 用 length(tree) 近似表示表达式树节点数
complexity_regularized_mse_loss_script = """
function my_complexity_regularized_mse_loss(tree, dataset::Dataset{T,L}, options)::L where {T,L}
    prediction, flag = eval_tree_array(tree, dataset.X, options)

    if !flag
        return L(1e10)
    end

    n = dataset.n

    mse = sum((dataset.y .- prediction) .^ 2) / n

    # lambda 越大，越偏好简单公式
    lambda = L(1e-4)

    # 用表达式树节点数近似复杂度
    complexity = length(tree)

    loss = mse + lambda * complexity

    return L(loss)
end
"""

# 19. Hybrid MSE + MAPE 函数
# 同时考虑绝对误差和相对误差
# 适合既不想忽略大数值区域，也不想忽略小数值区域的情况
hybrid_mse_mape_loss_script = """
function my_hybrid_mse_mape_loss(tree, dataset::Dataset{T,L}, options)::L where {T,L}
    prediction, flag = eval_tree_array(tree, dataset.X, options)

    if !flag
        return L(1e10)
    end

    n = dataset.n
    eps = L(1e-10)

    mse = sum((dataset.y .- prediction) .^ 2) / n

    percentage_error = abs.(dataset.y .- prediction) ./ (abs.(dataset.y) .+ eps)
    mape = sum(percentage_error) / n

    # alpha 控制 MSE 与 MAPE 的混合比例
    # 注意：MSE 和 MAPE 的量纲不同，实际使用时最好先做归一化或调参
    alpha = L(0.5)

    loss = alpha * mse + (L(1.0) - alpha) * mape

    return L(loss)
end
"""

# 20. Safe Relative Error 函数
# 使用 max(abs(y), y_scale) 作为分母，避免小 y 点支配 loss
# y_scale 可以理解为人为设置的物理尺度
safe_relative_error_loss_script = """
function my_safe_relative_error_loss(tree, dataset::Dataset{T,L}, options)::L where {T,L}
    prediction, flag = eval_tree_array(tree, dataset.X, options)

    if !flag
        return L(1e10)
    end

    n = dataset.n

    # 物理尺度，需要根据你的 y 的典型量级调整
    y_scale = L(1.0)

    denominator = max.(abs.(dataset.y), y_scale)

    safe_relative_error = abs.(dataset.y .- prediction) ./ denominator
    loss = sum(safe_relative_error) / n

    return L(loss)
end
"""

# endregion
```

---

# 5. 在 PySR 中使用这些 loss

示意代码如下：

```python
from pysr import PySRRegressor

model = PySRRegressor(
    niterations=1000,
    binary_operators=["+", "-", "*", "/"],
    unary_operators=["exp", "log", "sqrt"],
    loss_function=mape_loss_script,
)

model.fit(X, y)
```

如果你要使用 BIC：

```python
model = PySRRegressor(
    niterations=1000,
    binary_operators=["+", "-", "*", "/"],
    unary_operators=["exp", "log", "sqrt"],
    loss_function=bic_loss_script,
)

model.fit(X, y)
```

---

# 6. 对 NuSR 的实际建议

## 6.1 初学 PySR 阶段

建议先用：

1. MSE；
2. MAE；
3. MAPE；
4. RMSPE；
5. BIC。

原因是这五个足够覆盖最基本的拟合行为：

| loss | 观察目的 |
|---|---|
| MSE | 默认情况下 PySR 会找到什么 |
| MAE | 去掉极端点支配后会找到什么 |
| MAPE | 按相对误差看会找到什么 |
| RMSPE | 强烈惩罚大相对误差后会找到什么 |
| BIC | 惩罚复杂度后会保留什么 |

---

## 6.2 做核物理实验数据时

建议重点考虑：

1. Weighted MSE；
2. Chi-square；
3. Reduced Chi-square；
4. Huber；
5. BIC。

原因：

- 核物理实验数据通常有实验误差；
- 不同实验点的可信度不同；
- 经验公式不应该只追求误差小，也应该追求简洁和可解释；
- Huber 可以避免个别异常实验点彻底带偏搜索。

---

## 6.3 做经验公式发现时

建议不要只看一个 loss。

更合理的流程是：

1. 用 MSE 跑一批；
2. 用 MAE 跑一批；
3. 用 MAPE / RMSPE 跑一批；
4. 用 BIC 跑一批；
5. 比较不同 loss 下反复出现的结构；
6. 找出稳定出现的子表达式；
7. 再结合物理量纲、极限行为、已知经验公式判断。

真正值得相信的符号结构，往往不是某一次 PySR 输出的第一名，而是在多种 loss、多次随机种子、多种复杂度限制下反复出现的结构。

---

# 7. 参考资料

- PySR Documentation: Custom loss functions and API options.
- SymbolicRegression.jl Documentation: Loss functions, full objective customization, tree evaluation.
- scikit-learn Documentation: Regression metrics including MSE, MAE, MAPE, MSLE, and $R^2$.
- Standard statistical model selection criteria: AIC and BIC.
