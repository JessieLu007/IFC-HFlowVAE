
import itertools
import numpy as np
import pandas as pd


def _effect(subset, df, dv, cache):
    
    subset = tuple(subset)
    if subset in cache:
        return cache[subset]
    if len(subset) == 0:
        val = pd.Series(df[dv].mean(), index=df.index)
    else:
        val = df.groupby(list(subset))[dv].transform("mean")
        for r in range(len(subset)):
            for sub in itertools.combinations(subset, r):
                val = val - _effect(sub, df, dv, cache)
    cache[subset] = val
    return val


def compute_all_effects(df, dv, factor_cols):
    """计算 factor_cols 各子集(含空集=总均值，含全集=残差项)的效应}"""
    df = df.reset_index(drop=True)
    cache = {}
    all_subsets = []
    for r in range(len(factor_cols) + 1):
        all_subsets.extend(itertools.combinations(factor_cols, r))
    for s in all_subsets:
        _effect(s, df, dv, cache)

    
    total = sum(cache[s] for s in all_subsets)
    resid = (df[dv] - total).abs().max()
    if resid > 1e-6:
        raise RuntimeError(f"效应分解重构校验失败，最大误差={resid:.2e}，请检查数据是否为完全交叉的饱和设计")
    return cache, all_subsets


def aligned_rank_for_effect(df, dv, target_effect, factor_cols):
    
    df = df.reset_index(drop=True).copy()
    cache, all_subsets = compute_all_effects(df, dv, factor_cols)
    target = tuple(target_effect)
    full_set = tuple(factor_cols)
    keep = {(), target, full_set}  # 保留：总均值 + 目标效应本身 + 残差项，其余全部移除
    other_sum = sum(cache[s] for s in all_subsets if s not in keep)
    df["aligned"] = df[dv] - other_sum
    df["aligned_rank"] = df["aligned"].rank(method="average")
    return df