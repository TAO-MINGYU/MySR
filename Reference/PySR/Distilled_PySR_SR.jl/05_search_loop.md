# 05 — 搜索主循环 (equation_search)

## 1. equation_search 入口

**文件**: `src/SymbolicRegression.jl` (行469-597)

```julia
equation_search(X::AbstractMatrix, y; kws...) -> HallOfFame
```

**参数**:
- `X` — 特征矩阵 (features × samples)
- `y` — 目标向量或矩阵
- `niterations` — 搜索代数
- `options` — `Options` 结构体
- `parallelism` — `:serial` | `:multithreading` | `:multiprocessing`
- `numprocs` — multiprocessing 的进程数
- `return_state` — 是否返回完整状态 (用于 warm_start)
- `guesses` — 种子方程 (作为初始猜测)

## 2. 内部流水线

### Phase 1: `_validate_options`
- 调用 `test_option_configuration`: 检查所有运算符可评测
- 调用 `test_dataset_configuration`: 检查形状一致性
- 检查并发约束 (deterministic 必须 serial)

### Phase 2: `_create_workers` → `SearchState`
```
search_state:
  procs:             worker 进程 ID 列表
  worker_output:     (nout × npop) 的 WorkerOutputType 矩阵
  channels:          对应的 Channel 矩阵 (非阻塞通信)
  tasks:             @sr_spawner 返回的 Task/Future
  task_order:        打乱的 (out, pop) 对 (负载均衡)
  worker_assignment: 每个 worker 的分配计数
  halls_of_fame:     每个输出的 HallOfFame
  last_pops:         最近接收的群体
  best_sub_pops:     每个群体的最佳子集 (用于迁移)
  cur_maxsizes:      每个群体的当前最大方程大小
  running_search_statistics: 自适应简约跟踪器
  cycles_remaining:  剩余进化周期数
  num_evals:         总评估次数
  seed_members:      种子方程成员
```

并行模式:
- `:serial`: 直接在主线程执行
- `:multithreading`: `Threads.@spawn` 到其他线程
- `:multiprocessing`: `@spawnat worker_idx` 到远程进程

### Phase 3: `_initialize_search!`
1. 如有 saved_state: 反序列化恢复 HallOfFame 和 Population
2. 解析 guesses (字符串/表达式 → PopMember)
3. 对每个 (out, pop) pair: 创建初始随机群体
4. 评估并填充 baseline_loss

### Phase 4: `_warmup_search!`
- 每个群体运行 1 个周期
- `cur_maxsize` 初始为 3，逐渐增加到 maxsize
- `get_cur_maxsize()` 线性插值:
  ```
  cur_maxsize = maxsize - (maxsize-3) * (cycles_remaining / total_cycles)
  ```

### Phase 5: `_main_search_loop!` (核心)
```
while sum(cycles_remaining) > 0:
    for (out, pop) in shuffled task_order:
        # 非阻塞检查是否有群体准备好
        if isready(channels[out, pop]):
            # 1. 提取结果
            population = extract_from_worker(worker_output[out, pop])
            
            # 2. 更新追踪
            best_sub_pops[out, pop] = best_sub_pop(population, topn=10)
            last_pops[out, pop] = copy(population)
            
            # 3. 更新名人堂
            update_hall_of_fame!(halls_of_fame[out], best_seen, options)
            dominating = calculate_pareto_frontier(halls_of_fame[out])
            
            # 4. 迁移
            migrate!(population, best_sub_pops[out, other_pop])  # 群体间
            migrate!(population, dominating, frac=fraction_replaced_hof)  # 名人堂
            migrate!(population, seed_members, frac=fraction_replaced_guesses)  # 种子
            
            # 5. 更新动态参数
            cur_maxsizes[out, pop] = get_cur_maxsize(...)
            move_window!(running_search_statistics)
            
            # 6. 发送到 worker 继续进化
            tasks[out, pop] = @sr_spawner _dispatch_s_r_cycle(
                dataset, population, cur_maxsize, ...
            )
            
            cycles_remaining -= 1
            num_evals += ...
            
        # 早停检查
        check_for_user_quit(reader)   # 用户按 q
        check_for_loss_threshold(hof)  # 达到目标损失
        check_for_timeout(start_time)  # 超时
        check_max_evals(num_evals)     # 超出最大评估数
```

### Phase 6: `_tear_down!`
- 关闭 stdin reader
- 关闭 channels
- `rmprocs` 释放 worker 进程

### Phase 7: `_info_dump`
- 打印最终结果
- `save_to_file()` 输出 CSV

### Phase 8: `_format_output`
- 返回 `HallOfFame` (或 `(HallOfFame, SearchState)`)

## 3. _dispatch_s_r_cycle (在 worker 上执行)

```julia
function _dispatch_s_r_cycle(dataset, pop, ncycles, curmaxsize, ...)
    # 1. 进化循环
    pop, best_seen, num_evals = s_r_cycle(
        dataset, pop, ncycles, curmaxsize, ...
    )
    
    # 2. 后处理
    pop, tmp_evals = optimize_and_simplify_population(
        dataset, pop, options, curmaxsize
    )
    num_evals += tmp_evals
    
    # 3. 如果用批处理，用全量数据重评估
    if options.batching
        pop, tmp_evals = finalize_costs(full_dataset, pop, options)
        num_evals += tmp_evals
    end
    
    return (pop, best_seen, record, num_evals)
end
```

## 4. s_r_cycle (单次迭代)

**文件**: `src/SingleIteration.jl`

```
s_r_cycle(dataset, pop, ncycles, curmaxsize, ...):
    temperatures = LinRange(1.0, annealing ? 0.0 : 1.0, ncycles)
    best_seen = HallOfFame()  # 本次迭代的最佳
    
    for T in temperatures:
        pop, evals = reg_evol_cycle(dataset, pop, T, curmaxsize, ...)
        # 跟踪每个复杂度的最佳
        for member in pop:
            if loss < best_seen[complexity]:
                best_seen[complexity] = copy(member)
    
    return (pop, best_seen, evals)
```

## 5. 批处理机制

- `SubDataset` 持有对 `BasicDataset` 的 views
- 进化时只用 batch_size 个子样本
- `optimize_and_simplify_population` 后调用 `finalize_costs` 用全量数据
- 平衡速度和精度
