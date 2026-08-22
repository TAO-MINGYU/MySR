# region loss functions
# ==============================================================
# weighted loss function
# ==============================================================
# 说明：
# - 这些 loss_function 会读取 PySR 的 dataset.weights。
# - Python 端用 model.fit(X, y, weights=weights) 传入。
# - 加权规则按你的标准：
#       loss = sum(weights .* loss_values)
# - 如果 weights = 1 / sigma^2，则：
#       loss = sum(loss_values / sigma^2)
# - 如果 fit 时没有传 weights，则退化为 sum(loss_values)。

weighted_loss_helpers_script = """
function my_weighted_sum(values, dataset::Dataset{T,L})::L where {T,L}
    weights = dataset.weights

    if weights === nothing
        return L(sum(values))
    end

    return L(sum(weights .* values))
end

function my_weighted_mean_y(dataset::Dataset{T,L})::L where {T,L}
    weights = dataset.weights

    if weights === nothing
        return L(sum(dataset.y) / dataset.n)
    end

    weight_sum = sum(weights)

    if weight_sum <= L(0.0)
        return L(NaN)
    end

    return L(sum(weights .* dataset.y) / weight_sum)
end
"""


# 1. MSE 函数：Mean Squared Error，均方误差
# 加权计算：sum(w_i * (y_i - yhat_i)^2)
mse_loss_script = """
function my_mse_loss(tree, dataset::Dataset{T,L}, options)::L where {T,L}
    prediction, flag = eval_tree_array(tree, dataset.X, options)

    if !flag
        return L(1e10)
    end

    squared_error = (dataset.y .- prediction) .^ 2
    mse = (w = dataset.weights; w === nothing ? sum(squared_error) : sum(w .* squared_error))

    return L(mse)
end
"""


# 2. RMSE 函数：Root Mean Squared Error，均方根误差
# 加权计算：sqrt(sum(w_i * (y_i - yhat_i)^2))
rmse_loss_script = """
function my_rmse_loss(tree, dataset::Dataset{T,L}, options)::L where {T,L}
    prediction, flag = eval_tree_array(tree, dataset.X, options)

    if !flag
        return L(1e10)
    end

    squared_error = (dataset.y .- prediction) .^ 2
    mse = (w = dataset.weights; w === nothing ? sum(squared_error) : sum(w .* squared_error))
    rmse = sqrt(max(mse, L(1e-12)))

    return L(rmse)
end
"""


# 3. MAE 函数：Mean Absolute Error，平均绝对误差
# 加权计算：sum(w_i * |y_i - yhat_i|)
mae_loss_script = """
function my_mae_loss(tree, dataset::Dataset{T,L}, options)::L where {T,L}
    prediction, flag = eval_tree_array(tree, dataset.X, options)

    if !flag
        return L(1e10)
    end

    absolute_error = abs.(dataset.y .- prediction)
    mae = (w = dataset.weights; w === nothing ? sum(absolute_error) : sum(w .* absolute_error))

    return L(mae)
end
"""


# 4. Huber 函数：Huber Loss，MSE 与 MAE 的折中
# 加权计算：sum(w_i * huber_i)
huber_loss_script = """
function my_huber_loss(tree, dataset::Dataset{T,L}, options)::L where {T,L}
    prediction, flag = eval_tree_array(tree, dataset.X, options)

    if !flag
        return L(1e10)
    end

    delta = L(1.0)

    error = dataset.y .- prediction
    abs_error = abs.(error)

    loss_values = ifelse.(
        abs_error .<= delta,
        L(0.5) .* error .^ 2,
        delta .* (abs_error .- L(0.5) * delta)
    )

    huber = (w = dataset.weights; w === nothing ? sum(loss_values) : sum(w .* loss_values))

    return L(huber)
end
"""


# 5. Log-Cosh 函数：log(cosh(error))
# 加权计算：sum(w_i * log(cosh(error_i)))
log_cosh_loss_script = """
function my_log_cosh_loss(tree, dataset::Dataset{T,L}, options)::L where {T,L}
    prediction, flag = eval_tree_array(tree, dataset.X, options)

    if !flag
        return L(1e10)
    end

    error = dataset.y .- prediction

    # 为了数值稳定，使用 log(cosh(x)) 的稳定形式：
    # log(cosh(x)) = |x| + log(1 + exp(-2|x|)) - log(2)
    abs_error = abs.(error)
    loss_values = abs_error .+ log.(L(1.0) .+ exp.(-L(2.0) .* abs_error)) .- log(L(2.0))

    log_cosh = (w = dataset.weights; w === nothing ? sum(loss_values) : sum(w .* loss_values))

    return L(log_cosh)
end
"""


# 6. MAPE 函数：Mean Absolute Percentage Error，平均绝对百分比误差
# 加权计算：sum(w_i * |(y_i - yhat_i) / y_i|) / n * 100
mape_loss_script = """
function my_mape_loss(tree, dataset::Dataset{T,L}, options)::L where {T,L}
    prediction, flag = eval_tree_array(tree, dataset.X, options)

    if !flag
        return L(1e10)
    end

    eps = L(1e-10)
    n = dataset.n

    percentage_error = abs.((dataset.y .- prediction) ./ (dataset.y .+ eps))
    mape = (w = dataset.weights; w === nothing ? sum(percentage_error) / n : sum(w .* percentage_error) / n) * L(100.0)

    return L(mape)
end
"""


# 7. RMSPE 函数：Root Mean Squared Percentage Error，均方根百分比误差
# 加权计算：sqrt(sum(w_i * percentage_error_i^2) / n) * 100
rmspe_loss_script = """
function my_rmspe_loss(tree, dataset::Dataset{T,L}, options)::L where {T,L}
    prediction, flag = eval_tree_array(tree, dataset.X, options)

    if !flag
        return L(1e10)
    end

    eps = L(1e-10)
    n = dataset.n

    percentage_error = (dataset.y .- prediction) ./ (dataset.y .+ eps)

    mean_sq_percentage_error = (w = dataset.weights; w === nothing ? sum(percentage_error .^ 2) / n : sum(w .* percentage_error .^ 2) / n)
    rmspe = sqrt(max(mean_sq_percentage_error, L(1e-12))) * L(100.0)

    return L(rmspe)
end
"""


# 8. SMAPE 函数：Symmetric Mean Absolute Percentage Error，对称平均绝对百分比误差
# 加权计算：sum(w_i * 2 * |y_i - yhat_i| / (|y_i| + |yhat_i| + eps)) / n * 100
smape_loss_script = """
function my_smape_loss(tree, dataset::Dataset{T,L}, options)::L where {T,L}
    prediction, flag = eval_tree_array(tree, dataset.X, options)

    if !flag
        return L(1e10)
    end

    eps = L(1e-10)
    n = dataset.n

    numerator = L(2.0) .* abs.(dataset.y .- prediction)
    denominator = abs.(dataset.y) .+ abs.(prediction) .+ eps

    smape_values = numerator ./ denominator
    smape = (w = dataset.weights; w === nothing ? sum(smape_values) / n : sum(w .* smape_values) / n) * L(100.0)

    return L(smape)
end
"""


# 9. MSLE 函数：Mean Squared Logarithmic Error，均方对数误差
# 加权计算：sum(w_i * (log(1+y_i) - log(1+yhat_i))^2)
msle_loss_script = """
function my_msle_loss(tree, dataset::Dataset{T,L}, options)::L where {T,L}
    prediction, flag = eval_tree_array(tree, dataset.X, options)

    if !flag
        return L(1e10)
    end

    # MSLE 要求 y > -1 且 prediction > -1
    if any(dataset.y .<= L(-1.0)) || any(prediction .<= L(-1.0))
        return L(1e10)
    end

    log_error = log.(L(1.0) .+ dataset.y) .- log.(L(1.0) .+ prediction)
    msle = (w = dataset.weights; w === nothing ? sum(log_error .^ 2) : sum(w .* log_error .^ 2))

    return L(msle)
end
"""


# 10. RMSLE 函数：Root Mean Squared Logarithmic Error，均方根对数误差
# 加权计算：sqrt(sum(w_i * log_error_i^2))
rmsle_loss_script = """
function my_rmsle_loss(tree, dataset::Dataset{T,L}, options)::L where {T,L}
    prediction, flag = eval_tree_array(tree, dataset.X, options)

    if !flag
        return L(1e10)
    end

    if any(dataset.y .<= L(-1.0)) || any(prediction .<= L(-1.0))
        return L(1e10)
    end

    log_error = log.(L(1.0) .+ dataset.y) .- log.(L(1.0) .+ prediction)
    msle = (w = dataset.weights; w === nothing ? sum(log_error .^ 2) : sum(w .* log_error .^ 2))
    rmsle = sqrt(max(msle, L(1e-12)))

    return L(rmsle)
end
"""


# 11. COD / R2 函数：Coefficient of Determination，决定系数
# 加权计算：sum(w_i * residual_i^2) / sum(w_i * total_i^2)
cod_loss_script = """
function my_cod_loss(tree, dataset::Dataset{T,L}, options)::L where {T,L}
    prediction, flag = eval_tree_array(tree, dataset.X, options)

    if !flag
        return L(1e10)
    end

    eps = L(1e-12)

    y_mean = (w = dataset.weights; w === nothing ? sum(dataset.y)/dataset.n : (ws=sum(w); ws <= L(0.0) ? L(NaN) : sum(w .* dataset.y)/ws))

    if isnan(y_mean)
        return L(1e10)
    end

    ss_res_values = (dataset.y .- prediction) .^ 2
    ss_tot_values = (dataset.y .- y_mean) .^ 2

    ss_res = (w = dataset.weights; w === nothing ? sum(ss_res_values) : sum(w .* ss_res_values))
    ss_tot = (w = dataset.weights; w === nothing ? sum(ss_tot_values) : sum(w .* ss_tot_values))

    if ss_tot < eps
        return L(1e10)
    end

    # 这里是 1 - R2 = SS_res / SS_tot
    cod_loss = ss_res / ss_tot

    return L(cod_loss)
end
"""


# 12. Relative L2 函数：相对 L2 误差
# 加权计算：(sum(w_i * (y_i - yhat_i)^2) / n) / (sum(w_i * y_i^2) / n + eps)
relative_l2_loss_script = """
function my_relative_l2_loss(tree, dataset::Dataset{T,L}, options)::L where {T,L}
    prediction, flag = eval_tree_array(tree, dataset.X, options)

    if !flag
        return L(1e10)
    end

    eps = L(1e-12)
    n = dataset.n

    numerator = (w = dataset.weights; w === nothing ? sum((dataset.y .- prediction) .^ 2) / n : sum(w .* (dataset.y .- prediction) .^ 2) / n)
    denominator = (w = dataset.weights; w === nothing ? sum(dataset.y .^ 2) / n : sum(w .* dataset.y .^ 2) / n) + eps

    relative_l2 = numerator / denominator

    return L(relative_l2)
end
"""


# 13. Weighted MSE 函数：加权均方误差
# 加权计算：sum(w_i * (y_i - yhat_i)^2)
weighted_mse_loss_script = """
function my_weighted_mse_loss(tree, dataset::Dataset{T,L}, options)::L where {T,L}
    prediction, flag = eval_tree_array(tree, dataset.X, options)

    if !flag
        return L(1e10)
    end

    squared_error = (dataset.y .- prediction) .^ 2
    weighted_mse = (w = dataset.weights; w === nothing ? sum(squared_error) : sum(w .* squared_error))

    return L(weighted_mse)
end
"""


# 14. Chi-square 函数：卡方损失
# 如果 weights = 1 / sigma^2，则这里就是 sum((y_i - yhat_i)^2 / sigma_i^2)
chi_square_loss_script = """
function my_chi_square_loss(tree, dataset::Dataset{T,L}, options)::L where {T,L}
    prediction, flag = eval_tree_array(tree, dataset.X, options)

    if !flag
        return L(1e10)
    end

    squared_error = (dataset.y .- prediction) .^ 2
    chi_square = (w = dataset.weights; w === nothing ? sum(squared_error) : sum(w .* squared_error))

    return L(chi_square)
end
"""


# 15. Reduced Chi-square 函数：约化卡方损失
# 如果 weights = 1 / sigma^2，则这里是 sum((y_i - yhat_i)^2 / sigma_i^2) / dof
reduced_chi_square_loss_script = """
function get_k_constants_for_reduced_chi_square(tree)
    count = 0

    for node in tree
        if node.degree == 0 && node.constant
            count += 1
        end
    end

    return count
end

function my_reduced_chi_square_loss(tree, dataset::Dataset{T,L}, options)::L where {T,L}
    prediction, flag = eval_tree_array(tree, dataset.X, options)

    if !flag
        return L(1e10)
    end

    n = dataset.n

    k = get_k_constants_for_reduced_chi_square(tree)

    dof = n - k
    if dof <= 0
        return L(1e10)
    end

    squared_error = (dataset.y .- prediction) .^ 2
    chi_square = (w = dataset.weights; w === nothing ? sum(squared_error) : sum(w .* squared_error))

    reduced_chi_square = chi_square / dof

    return L(reduced_chi_square)
end
"""


# 16. AIC 函数：Akaike Information Criterion，赤池信息准则
# 加权计算：用 sum(w_i * squared_error_i) 替代原来的 MSE 部分
aic_loss_script = """
function get_k_constants_for_aic(tree)
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

    k = get_k_constants_for_aic(tree)
    if k == 0
        k = 1
    end

    mse = (w = dataset.weights; w === nothing ? sum((dataset.y .- prediction) .^ 2) : sum(w .* (dataset.y .- prediction) .^ 2))
    mse = max(mse, L(1e-12))

    aic = n * log(mse) + L(2.0) * k

    return L(aic)
end
"""


# 17. BIC 函数：Bayesian Information Criterion，贝叶斯信息准则
# 加权计算：用 sum(w_i * squared_error_i) 替代原来的 MSE 部分
bic_loss_script = """
function get_k_constants_for_bic(tree)
    count = 0

    for node in tree
        if node.degree == 0 && node.constant
            count += 1
        end
    end

    return count
end

function my_bic_loss(tree, dataset::Dataset{T,L}, options)::L where {T,L}
    prediction, flag = eval_tree_array(tree, dataset.X, options)

    if !flag
        return L(1e10)
    end

    n = dataset.n

    k = get_k_constants_for_bic(tree)
    if k == 0
        k = 1
    end

    mse = (w = dataset.weights; w === nothing ? sum((dataset.y .- prediction) .^ 2) : sum(w .* (dataset.y .- prediction) .^ 2))
    mse = max(mse, L(1e-12))

    bic = n * log(mse) + k * log(n)

    return L(bic)
end
"""


# 18. Complexity-Regularized MSE 函数：复杂度正则化 MSE
# 加权计算：sum(w_i * squared_error_i) + lambda * complexity
complexity_regularized_mse_loss_script = """
function get_tree_size_for_complexity_regularized_mse(tree)
    count = 0

    for node in tree
        count += 1
    end

    return count
end

function my_complexity_regularized_mse_loss(tree, dataset::Dataset{T,L}, options)::L where {T,L}
    prediction, flag = eval_tree_array(tree, dataset.X, options)

    if !flag
        return L(1e10)
    end

    mse = (w = dataset.weights; w === nothing ? sum((dataset.y .- prediction) .^ 2) : sum(w .* (dataset.y .- prediction) .^ 2))

    complexity = get_tree_size_for_complexity_regularized_mse(tree)

    lambda = L(1e-4)

    loss = mse + lambda * complexity

    return L(loss)
end
"""


# 19. Hybrid MSE + MAPE 函数：绝对误差与相对误差混合损失
# 加权计算：alpha * sum(w_i * squared_error_i) / n + (1-alpha) * sum(w_i * percentage_error_i) / n * 100
hybrid_mse_mape_loss_script = """
function my_hybrid_mse_mape_loss(tree, dataset::Dataset{T,L}, options)::L where {T,L}
    prediction, flag = eval_tree_array(tree, dataset.X, options)

    if !flag
        return L(1e10)
    end

    eps = L(1e-10)
    n = dataset.n

    alpha = L(0.5)

    mse = (w = dataset.weights; w === nothing ? sum((dataset.y .- prediction) .^ 2) / n : sum(w .* (dataset.y .- prediction) .^ 2) / n)

    percentage_error = abs.((dataset.y .- prediction) ./ (dataset.y .+ eps))
    mape = (w = dataset.weights; w === nothing ? sum(percentage_error) / n : sum(w .* percentage_error) / n) * L(100.0)

    loss = alpha * mse + (L(1.0) - alpha) * mape

    return L(loss)
end
"""


# 20. Safe Relative Error 函数：安全相对误差
# 加权计算：sum(w_i * |y_i - yhat_i| / max(|y_i|, scale_floor)) / n * 100
safe_relative_error_loss_script = """
function my_safe_relative_error_loss(tree, dataset::Dataset{T,L}, options)::L where {T,L}
    prediction, flag = eval_tree_array(tree, dataset.X, options)

    if !flag
        return L(1e10)
    end

    scale_floor = L(1e-6)
    n = dataset.n

    denominator = max.(abs.(dataset.y), scale_floor)

    relative_error = abs.(dataset.y .- prediction) ./ denominator

    loss = (w = dataset.weights; w === nothing ? sum(relative_error) / n : sum(w .* relative_error) / n) * L(100.0)

    return L(loss)
end
"""


# 21. Clipped MSE 函数：截断均方误差
# 加权计算：sum(w_i * min(|e_i|, threshold)^2)
clipped_mse_loss_script = """
function my_clipped_mse_loss(tree, dataset::Dataset{T,L}, options)::L where {T,L}
    prediction, flag = eval_tree_array(tree, dataset.X, options)

    if !flag
        return L(1e10)
    end

    threshold = L(10.0)

    abs_error = abs.(dataset.y .- prediction)
    clipped_error = min.(abs_error, threshold)

    clipped_mse = (w = dataset.weights; w === nothing ? sum(clipped_error .^ 2) : sum(w .* clipped_error .^ 2))

    return L(clipped_mse)
end
"""


# 22. Max Absolute Error 函数：最大绝对误差
# 加权计算：maximum(w_i * |y_i - yhat_i|)
max_absolute_error_loss_script = """
function my_max_absolute_error_loss(tree, dataset::Dataset{T,L}, options)::L where {T,L}
    prediction, flag = eval_tree_array(tree, dataset.X, options)

    if !flag
        return L(1e10)
    end

    absolute_error = abs.(dataset.y .- prediction)

    weights = dataset.weights
    if weights === nothing
        max_error = maximum(absolute_error)
    else
        max_error = maximum(weights .* absolute_error)
    end

    return L(max_error)
end
"""


# 23. Quantile Loss 函数：分位数损失
# 加权计算：sum(w_i * quantile_loss_i)
quantile_loss_script = """
function my_quantile_loss(tree, dataset::Dataset{T,L}, options)::L where {T,L}
    prediction, flag = eval_tree_array(tree, dataset.X, options)

    if !flag
        return L(1e10)
    end

    q = L(0.5)

    error = dataset.y .- prediction

    loss_values = max.(q .* error, (q .- L(1.0)) .* error)

    quantile_loss = (w = dataset.weights; w === nothing ? sum(loss_values) : sum(w .* loss_values))

    return L(quantile_loss)
end
"""


# 24. Cauchy Loss 函数：柯西损失
# 加权计算：sum(w_i * cauchy_loss_i)
cauchy_loss_script = """
function my_cauchy_loss(tree, dataset::Dataset{T,L}, options)::L where {T,L}
    prediction, flag = eval_tree_array(tree, dataset.X, options)

    if !flag
        return L(1e10)
    end

    c = L(1.0)

    error = dataset.y .- prediction

    loss_values = (c ^ 2 / L(2.0)) .* log.(L(1.0) .+ (error ./ c) .^ 2)

    cauchy_loss = (w = dataset.weights; w === nothing ? sum(loss_values) : sum(w .* loss_values))

    return L(cauchy_loss)
end
"""


# 25. Physics-Scale Normalized MSE 函数：物理尺度归一化 MSE
# 加权计算：sum(w_i * ((y_i - yhat_i) / y_scale)^2)
physics_scale_normalized_mse_loss_script = """
function my_physics_scale_normalized_mse_loss(tree, dataset::Dataset{T,L}, options)::L where {T,L}
    prediction, flag = eval_tree_array(tree, dataset.X, options)

    if !flag
        return L(1e10)
    end

    # 你需要根据具体物理量修改这个尺度。
    # 例如结合能可以取 8 MeV，质量残差可以取 1 MeV，截面可取典型截面尺度。
    y_scale = L(1.0)

    normalized_error = (dataset.y .- prediction) ./ y_scale

    loss = (w = dataset.weights; w === nothing ? sum(normalized_error .^ 2) : sum(w .* normalized_error .^ 2))

    return L(loss)
end
"""
# endregion
