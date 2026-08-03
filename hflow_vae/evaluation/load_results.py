
"""
从实验目录里组装单个数据集的长表
目录结构约定：
    {root}/experiment_runs/{mix_ratio}/per_fold/{dataset}__seed{seed}__fold{fold}.parquet
mix_ratios = ['mix_37_235', 'mix_55_532']
"""
import re
from pathlib import Path
import pandas as pd

EXPECTED_COLUMNS = {"Method", "Dataset", "Seed", "Fold", "Classifier", "Recall", "F1_score", "G_mean"}


def load_dataset_long_table(root, dataset, methods_expected=None, verbose=True):
    """
    加载 {root}/results/per_fold/ 下某个 dataset 的所有
    seed x fold parquet 文件，拼接成一张长表。
    """
    root = Path(root)
    per_fold_dir = root / "results" / "per_fold"
    if not per_fold_dir.exists():
        raise FileNotFoundError(f"目录不存在: {per_fold_dir}")

    files = sorted(per_fold_dir.glob(f"{dataset}__seed*__fold*.parquet"))
    if not files:
        raise FileNotFoundError(f"没有找到匹配文件: {per_fold_dir}/{dataset}__seed*__fold*.parquet")

    pattern = re.compile(rf"^{re.escape(dataset)}__seed(.+)__fold(.+)\.parquet$")
    dfs = []
    for f in files:
        m = pattern.match(f.name)
        if m is None:
            raise ValueError(f"文件名不符合约定格式 {{dataset}}__seed{{seed}}__fold{{fold}}.parquet: {f.name}")
        seed_from_name, fold_from_name = m.group(1), m.group(2)

        d = pd.read_parquet(f)
        missing_cols = EXPECTED_COLUMNS - set(d.columns)
        if missing_cols:
            raise ValueError(f"{f.name} 缺少列: {missing_cols}")

        # 文件名 vs 内容一致性校验，防止路径和内容对不上导致 seed/fold 标错
        if not (d["Seed"].astype(str) == seed_from_name).all():
            raise ValueError(f"{f.name}: 文件名解析出的 seed={seed_from_name} 与内容 Seed 列不一致")
        if not (d["Fold"].astype(str) == fold_from_name).all():
            raise ValueError(f"{f.name}: 文件名解析出的 fold={fold_from_name} 与内容 Fold 列不一致")
        if not (d["Dataset"] == dataset).all():
            bad_vals = d.loc[d["Dataset"] != dataset, "Dataset"].unique()
            raise ValueError(f"{f.name}: Dataset 列出现了非预期值 {bad_vals}")

        dfs.append(d[list(EXPECTED_COLUMNS)])

    out = pd.concat(dfs, ignore_index=True)

    # 重复检查
    dup_mask = out.duplicated(subset=["Method", "Classifier", "Seed", "Fold"], keep=False)
    if dup_mask.any():
        n_dup = out.loc[dup_mask, ["Method", "Classifier", "Seed", "Fold"]].drop_duplicates().shape[0]
        raise ValueError(f"发现 {n_dup} 组 (Method,Classifier,Seed,Fold) 重复记录，请检查是否重复加载")

    if methods_expected is not None:
        actual = set(out["Method"].unique())
        missing = set(methods_expected) - actual
        extra = actual - set(methods_expected)
        if missing and verbose:
            print(f"⚠️ 缺失 method: {missing}")
        if extra and verbose:
            print(f"⚠️ 出现未预期的 method: {extra}")

    if verbose:
        n_seed = out["Seed"].nunique()
        n_fold = out["Fold"].nunique()
        n_method = out["Method"].nunique()
        n_clf = out["Classifier"].nunique()
        print(f"{dataset} 加载完成: {len(out)}行 "
              f"= {n_seed}seed x {n_fold}fold(实际按block计) x {n_method}method x {n_clf}classifier ")
    return out


def compare_mix_ratios_for_shared_methods(root, dataset, shared_methods=None,
                                            mix_ratios=("mix_37_235", "mix_55_532"),
                                            metric="G_mean"):
    if shared_methods is None:
        shared_methods = ["raw", "smote", "ctgan", "tvae", "tabddpm",
                           "forestdiffusion", "tabkde", "none_noft", "none_ft"]

    tables = {mr: load_dataset_long_table(root,dataset, mr, verbose=False) for mr in mix_ratios}
    merged = None
    for mr, df in tables.items():
        sub = df[df["Method"].isin(shared_methods)][["Method", "Classifier", "Seed", "Fold", metric]]
        sub = sub.rename(columns={metric: f"{metric}__{mr}"})
        merged = sub if merged is None else merged.merge(
            sub, on=["Method", "Classifier", "Seed", "Fold"], how="outer")

    cols = [c for c in merged.columns if c.startswith(f"{metric}__")]
    merged["max_abs_diff"] = merged[cols].max(axis=1) - merged[cols].min(axis=1)
    return merged.sort_values("max_abs_diff", ascending=False)