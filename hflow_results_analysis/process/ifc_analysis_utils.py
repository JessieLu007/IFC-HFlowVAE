import os
import re
import glob
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import page_trend_test
from statsmodels.stats.multitest import multipletests
import scikit_posthocs as sp

from evaluation.inner_dataset_analysis import (
    validate_and_prepare, parse_ifc_cells, friedman_nemenyi, IFC_CELLS
)
# import constant variables
from utils.common import (
    DATASETS, CLASSIFIERS, BASELINE_METHODS, MIX_RATIOS, STAGES, HEADLINE_MIX_RATIO,
    HEADLINE_STAGE, NONE_METHODS, SELF_LOOP_METHODS, RAW_COLS
    )


# 1. 原始文件读取 + 预处理 + 合并
def load_per_fold_dir(per_fold_dir):
    files = sorted(glob.glob(os.path.join(per_fold_dir, "**", "*.parquet"), recursive=True))
    reader = pd.read_parquet
    if not files:
        files = sorted(glob.glob(os.path.join(per_fold_dir, "**", "*.csv"), recursive=True))
        reader = pd.read_csv
    if not files:
        raise FileNotFoundError(f"{per_fold_dir} 下没找到任何 parquet/csv 文件,检查路径或后缀")

    df = pd.concat([reader(f) for f in files], ignore_index=True)
    missing = set(RAW_COLS) - set(df.columns)
    if missing:
        raise ValueError(f"{per_fold_dir} 读出来的表缺少列: {missing}")
    return df[RAW_COLS].copy()


def parse_prior_finetune(method_series):
    prior_type = pd.Series(np.nan, index=method_series.index, dtype=object)
    finetune = pd.Series(np.nan, index=method_series.index, dtype=object)

    is_baseline = method_series.isin(BASELINE_METHODS)
    non_baseline = method_series[~is_baseline]
    if len(non_baseline) > 0:
        parsed = non_baseline.str.rsplit("_", n=1, expand=True)
        prior_type.loc[~is_baseline] = parsed[0]
        finetune.loc[~is_baseline] = parsed[1].map({"ft": True, "noft": False})

    return prior_type, finetune


def build_all_results(data_root="experiments"):
    """
    读取 mix_37_235 / mix_55_532 × stage2 / stage3 共4个目录
    """
    pieces = []
    for mix_ratio in MIX_RATIOS:
        for stage_depth in STAGES:
            per_fold_dir = os.path.join(data_root, mix_ratio, f"stage{stage_depth}", "results", "per_fold")
            df = load_per_fold_dir(per_fold_dir)

            is_headline_dir = (mix_ratio == HEADLINE_MIX_RATIO and stage_depth == HEADLINE_STAGE)
            keep_methods = (BASELINE_METHODS | NONE_METHODS | SELF_LOOP_METHODS) if is_headline_dir else SELF_LOOP_METHODS
            df = df[df["Method"].isin(keep_methods)].copy()

            prior_type, finetune = parse_prior_finetune(df["Method"])
            df["prior_type"] = prior_type
            df["finetune"] = finetune

            is_self_loop = df["Method"].isin(SELF_LOOP_METHODS)
            df["stage_depth"] = np.where(is_self_loop, stage_depth, np.nan)
            mix_ratio_col = pd.Series(np.nan, index=df.index, dtype=object)
            mix_ratio_col[is_self_loop] = mix_ratio
            df["mix_ratio"] = mix_ratio_col

            pieces.append(df)

    all_results = pd.concat(pieces, ignore_index=True)
    _sanity_check(all_results)
    return all_results


def _sanity_check(all_results):
    dup_key = ["Dataset", "Seed", "Fold", "Classifier", "Method", "stage_depth", "mix_ratio"]
    dup_mask = all_results.duplicated(subset=dup_key, keep=False)
    if dup_mask.any():
        raise ValueError(
            f"发现 {dup_mask.sum()} 条重复记录,先检查目录读取逻辑:\n"
            f"{all_results[dup_mask].sort_values(dup_key).head(20)}"
        )

    n_blocks = all_results[["Dataset", "Seed", "Fold", "Classifier"]].drop_duplicates().shape[0]
    n_raw = (all_results["Method"] == "raw").sum()
    if n_raw != n_blocks:
        print(f"⚠️ raw记录数={n_raw},期望={n_blocks}(每个Dataset×Seed×Fold×Classifier一条),请检查")

    self_loop = all_results[all_results["Method"].isin(SELF_LOOP_METHODS)]
    counts = self_loop.groupby(["Dataset", "Seed", "Fold", "Classifier", "Method"]).size()
    bad = counts[counts != 4]
    if len(bad) > 0:
        print(f"⚠️ 有{len(bad)}个(Dataset,Seed,Fold,Classifier,Method)组合不是恰好4条记录,请检查:")
        print(bad.head(20))

    print(f"✅ build_all_results 完成: 共{len(all_results)}行, "
          f"baseline={all_results['Method'].isin(BASELINE_METHODS).sum()}, "
          f"none={all_results['Method'].isin(NONE_METHODS).sum()}, "
          f"self_loop={all_results['Method'].isin(SELF_LOOP_METHODS).sum()}")


def select_mix_ratio_view(all_results, mix_ratio):
    is_universal = all_results["mix_ratio"].isna()
    is_selected = all_results["mix_ratio"] == mix_ratio
    return all_results[is_universal | is_selected].copy()


def run_main_ablation(df, metric, mix_ratio=None, datasets=None, classifiers=None):
    datasets = datasets or DATASETS
    classifiers = classifiers or CLASSIFIERS

    if mix_ratio is not None:
        df = select_mix_ratio_view(df, mix_ratio)

    sub = df[
        df["Method"].isin(IFC_CELLS)
        & ((df["stage_depth"] == 3) | (df["prior_type"] == "none"))
    ].copy()

    rank_rows = []
    for ds in datasets:
        df_ds = validate_and_prepare(sub, dataset=ds)
        df_ds = parse_ifc_cells(df_ds)
        for clf in classifiers:
            fn_res = friedman_nemenyi(df_ds, metric, clf)
            r = fn_res["avg_rank"].rename("avg_rank").reset_index()
            r.columns = ["config", "avg_rank"]
            r["Dataset"], r["Classifier"] = ds, clf
            rank_rows.append(r)
    rank_long = pd.concat(rank_rows, ignore_index=True)

    cross_rank = (
        rank_long.groupby("config")["avg_rank"]
        .agg(mean_rank="mean", std_rank="std", n_blocks="count")
        .reset_index()
        .sort_values("mean_rank")
    )

    wide_cross = rank_long.pivot(index=["Dataset", "Classifier"], columns="config", values="avg_rank")
    omnibus_stat, omnibus_p = stats.friedmanchisquare(*[wide_cross[c].values for c in wide_cross.columns])
    nemenyi = sp.posthoc_nemenyi_friedman(wide_cross.values)
    nemenyi.index = nemenyi.columns = wide_cross.columns

    return {
        "rank_long": rank_long,
        "cross_rank": cross_rank,
        "omnibus_stat": omnibus_stat,
        "omnibus_p": omnibus_p,
        "nemenyi": nemenyi,
    }


def paired_test(df, metric, compare_col, level_a, level_b,
                 configs=None, datasets=None, classifiers=None, fixed_filters=None):
    configs = configs or sorted(SELF_LOOP_METHODS)
    datasets = datasets or DATASETS
    classifiers = classifiers or CLASSIFIERS

    d = df.copy()
    if fixed_filters:
        for k, v in fixed_filters.items():
            d = d[d[k] == v]

    d = d[d["prior_type"].isin(["bootstrap", "smote"])].copy()
    d["config"] = d["prior_type"] + "_" + d["finetune"].map({True: "ft", False: "noft"})
    d["block"] = d["Seed"].astype(str) + "_" + d["Fold"].astype(str)

    rows = []
    for ds in datasets:
        for clf in classifiers:
            sub = d[(d["Dataset"] == ds) & (d["Classifier"] == clf)]
            for cfg in configs:
                wide = sub[sub["config"] == cfg].pivot(index="block", columns=compare_col, values=metric)
                if not {level_a, level_b}.issubset(wide.columns) or wide[[level_a, level_b]].isna().any().any():
                    continue
                stat, p = stats.wilcoxon(wide[level_b], wide[level_a])
                rows.append({
                    "Dataset": ds, "Classifier": clf, "config": cfg,
                    "diff": (wide[level_b] - wide[level_a]).median(),
                    "n_pos": int((wide[level_b] > wide[level_a]).sum()),
                    "n_neg": int((wide[level_b] < wide[level_a]).sum()),
                    "p_unc_block": p,
                })
    block_results = pd.DataFrame(rows)
    if block_results.empty:
        raise ValueError("没有任何(Dataset,Classifier,config)满足完整覆盖条件")

    final_rows = []
    for cfg, g in block_results.groupby("config"):
        stat, p = stats.wilcoxon(g["diff"])
        final_rows.append({
            "config": cfg, "n_blocks": len(g),
            "n_pos_total": g["n_pos"].sum(), "n_neg_total": g["n_neg"].sum(),
            "median_of_diffs": g["diff"].median(), "p_unc": p,
        })
    summary = pd.DataFrame(final_rows)
    summary["p_holm"] = multipletests(summary["p_unc"], method="holm")[1]

    overall_stat, overall_p = stats.wilcoxon(block_results["diff"])

    return {
        "block_results": block_results,
        "summary": summary,
        "overall_stat": overall_stat,
        "overall_p": overall_p,
    }


def smote_depth_trend(all_results, metric, mix_ratio, datasets=None, classifiers=None):
    if datasets is None:
        datasets = DATASETS
    if classifiers is None:
        classifiers = CLASSIFIERS

    df = all_results[
        all_results["Dataset"].isin(datasets) &
        all_results["Classifier"].isin(classifiers)
    ]

    d1 = df[df["Method"] == "smote"][
        ["Dataset", "Classifier", "Seed", "Fold", metric]
    ].rename(columns={metric: "d1"})

    results = {}
    for cfg in ["smote_noft", "smote_ft"]:
        sub2 = all_results[
            (all_results["Method"] == cfg) & (all_results["stage_depth"] == 2) & (all_results["mix_ratio"] == mix_ratio)
        ]
        sub3 = all_results[
            (all_results["Method"] == cfg) & (all_results["stage_depth"] == 3) & (all_results["mix_ratio"] == mix_ratio)
        ]
        d2 = sub2[["Dataset", "Classifier", "Seed", "Fold", metric]].rename(columns={metric: "d2"})
        d3 = sub3[["Dataset", "Classifier", "Seed", "Fold", metric]].rename(columns={metric: "d3"})

        trend_df = d1.merge(d2, on=["Dataset", "Classifier", "Seed", "Fold"]).merge(
            d3, on=["Dataset", "Classifier", "Seed", "Fold"]
        )
        per_block = trend_df.groupby(["Dataset", "Classifier"])[["d1", "d2", "d3"]].mean().reset_index()

        res_inc = page_trend_test(per_block[["d1", "d2", "d3"]].values, predicted_ranks=[1, 2, 3])
        res_dec = page_trend_test(per_block[["d1", "d2", "d3"]].values, predicted_ranks=[3, 2, 1])

        results[cfg] = {
            "means": per_block[["d1", "d2", "d3"]].mean().to_dict(),
            "n_blocks": len(per_block),
            "p_increasing": res_inc.pvalue,
            "p_decreasing": res_dec.pvalue,
        }
    return results


def _list_real_fold_dirs(seed_dir, seed_num):
    result = []
    for fold_dir in sorted(glob.glob(os.path.join(seed_dir, "fold*"))):
        fold_num = int(os.path.basename(fold_dir).replace("fold", ""))
        if fold_num == seed_num:
            continue
        result.append((fold_num, fold_dir))
    return result


def build_artifacts_path_table(data_root="experiments"):
    rows = []
    for mix_ratio in MIX_RATIOS:
        for stage_depth in STAGES:
            artifacts_root = os.path.join(data_root, mix_ratio, f"stage{stage_depth}", "artifacts")
            is_headline_dir = (mix_ratio == HEADLINE_MIX_RATIO and stage_depth == HEADLINE_STAGE)
            candidate_methods = (
                (BASELINE_METHODS - {"raw"}) | NONE_METHODS | SELF_LOOP_METHODS
                if is_headline_dir else SELF_LOOP_METHODS
            )

            for dataset_dir in sorted(glob.glob(os.path.join(artifacts_root, "*"))):
                dataset = os.path.basename(dataset_dir)
                for seed_dir in sorted(glob.glob(os.path.join(dataset_dir, "seed*"))):
                    seed_num = int(os.path.basename(seed_dir).replace("seed", ""))
                    for fold_num, fold_dir in _list_real_fold_dirs(seed_dir, seed_num):
                        for method in candidate_methods:
                            method_dir = os.path.join(fold_dir, method)
                            fidelity_path = os.path.join(method_dir, "fidelity.parquet")
                            final_path = os.path.join(method_dir, "final.parquet")

                            prior_type, finetune = parse_prior_finetune(pd.Series([method]))
                            is_self_loop = method in SELF_LOOP_METHODS

                            rows.append({
                                "Dataset": dataset,
                                "Seed": seed_num,
                                "Fold": fold_num,
                                "Method": method,
                                "prior_type": prior_type.iloc[0],
                                "finetune": finetune.iloc[0],
                                "stage_depth": stage_depth if is_self_loop else np.nan,
                                "mix_ratio": mix_ratio if is_self_loop else np.nan,
                                "fidelity_path": fidelity_path,
                                "final_path": final_path,
                                "fidelity_exists": os.path.exists(fidelity_path),
                                "final_exists": os.path.exists(final_path),
                            })

    table = pd.DataFrame(rows)
    missing = table[~(table["fidelity_exists"] & table["final_exists"])]
    if len(missing) > 0:
        print(f"⚠️ 有{len(missing)}条记录缺少 fidelity.parquet 或 final.parquet,先检查:")
        print(missing[["Dataset", "Seed", "Fold", "Method", "mix_ratio", "stage_depth"]].head(20))
    return table

FOLD_DATA_FILES = {
    "X_train": "X_train.parquet",
    "y_train": "y_train.parquet",
    "X_valid": "X_valid.parquet",
    "y_valid": "y_valid.parquet",
    "X_test": "X_test.parquet",
    "y_test": "y_test.parquet",
}


def build_original_data_path_table(data_root="experiments",
                                    canonical_mix_ratio=HEADLINE_MIX_RATIO,
                                    canonical_stage=HEADLINE_STAGE):
    artifacts_root = os.path.join(data_root, canonical_mix_ratio, f"stage{canonical_stage}", "artifacts")
    rows = []
    for dataset_dir in sorted(glob.glob(os.path.join(artifacts_root, "*"))):
        dataset = os.path.basename(dataset_dir)
        for seed_dir in sorted(glob.glob(os.path.join(dataset_dir, "seed*"))):
            seed_num = int(os.path.basename(seed_dir).replace("seed", ""))
            for fold_num, fold_dir in _list_real_fold_dirs(seed_dir, seed_num):
                fold_data_dir = os.path.join(fold_dir, "_fold_data")

                row = {"Dataset": dataset, "Seed": seed_num, "Fold": fold_num}
                for key, fname in FOLD_DATA_FILES.items():
                    p = os.path.join(fold_data_dir, fname)
                    row[f"{key}_path"] = p
                    row[f"{key}_exists"] = os.path.exists(p)
                rows.append(row)

    table = pd.DataFrame(rows)
    exist_cols = [c for c in table.columns if c.endswith("_exists")]
    missing = table[~table[exist_cols].all(axis=1)]
    if len(missing) > 0:
        print(f"⚠️ 有{len(missing)}个(Dataset,Seed,Fold)缺少部分_fold_data文件,先检查:")
        print(missing[["Dataset", "Seed", "Fold"] + exist_cols].head(20))
    return table


def load_pos_subset(X_path, y_path, pos_label=1, y_col=None):
    X = pd.read_parquet(X_path)
    y = pd.read_parquet(y_path)

    if y_col is None:
        if y.shape[1] != 1:
            raise ValueError(f"{y_path} 有多列,需要显式传入 y_col 指定标签列")
        y_col = y.columns[0]

    mask = (y[y_col] == pos_label).values
    return X.loc[mask].reset_index(drop=True)


def add_pos_paths(original_data_table, pos_label=1, y_col=None, save=True):
    table = original_data_table.copy()
    train_pos_paths, test_pos_paths = [], []
    train_pos_n, test_pos_n = [], []

    for _, row in table.iterrows():
        fold_data_dir = os.path.dirname(row["X_train_path"])

        X_train_pos = load_pos_subset(row["X_train_path"], row["y_train_path"], pos_label, y_col)
        X_test_pos = load_pos_subset(row["X_test_path"], row["y_test_path"], pos_label, y_col)

        train_pos_n.append(len(X_train_pos))
        test_pos_n.append(len(X_test_pos))

        if save:
            train_pos_path = os.path.join(fold_data_dir, "X_train_pos.parquet")
            test_pos_path = os.path.join(fold_data_dir, "X_test_pos.parquet")
            X_train_pos.to_parquet(train_pos_path)
            X_test_pos.to_parquet(test_pos_path)
            train_pos_paths.append(train_pos_path)
            test_pos_paths.append(test_pos_path)
        else:
            train_pos_paths.append(np.nan)
            test_pos_paths.append(np.nan)

    table["X_train_pos_path"] = train_pos_paths
    table["X_test_pos_path"] = test_pos_paths
    table["n_train_pos"] = train_pos_n
    table["n_test_pos"] = test_pos_n
    return table


# =================================================================================

DATASET_ABBREV = {
    'breast_cancer_coimbra': 'BCC',
    'PIMA_diabetes': 'PIMA',
    'TCGA_InfoWithGrade': 'TCGA',
    'Thyroid_Diff': 'THY',
    'diabetes_risk_prediction': 'DRP',
    'framingham': 'FRM',
    'hepatitis': 'HEP',
} 

METHOD_DISPLAY = {
    'raw': 'Raw', 'smote': 'SMOTE', 'ctgan': 'CTGAN', 'tvae': 'TVAE',
    'forestdiffusion': 'ForestDiff', 'tabddpm': 'TabDDPM', 'tabkde': 'TabKDE', 'IFC': 'IFC',
}

def fmt_num(x, decimals=2, drop_leading_zero=True):
    s = f"{x:.{decimals}f}"
    if drop_leading_zero:
        s = s[1:] if s.startswith("0.") else ("-" + s[2:] if s.startswith("-0.") else s)
    return s


def format_cell(mean, std, rank, close_call=False, drop_leading_zero=True):
    m, s = fmt_num(mean, drop_leading_zero=drop_leading_zero), fmt_num(std, drop_leading_zero=drop_leading_zero)
    cell = f"{m} $\\pm$ {s}" + (r"$^\dagger$" if close_call else "")
    if rank == 1:
        return f"\\textbf{{{cell}}}"
    elif rank == 2:
        return f"\\underline{{{cell}}}"
    return cell


def build_raw_reference_table(all_results, datasets=None, classifiers=None, drop_leading_zero=True):
    datasets, classifiers = datasets or DATASETS, classifiers or CLASSIFIERS
    rows = []
    for ds in datasets:
        sub = all_results[(all_results["Dataset"] == ds) & (all_results["Method"] == "raw")]
        row = [DATASET_ABBREV.get(ds, ds)]
        for clf in classifiers:
            clf_sub = sub[sub["Classifier"] == clf]
            for metric in ["F1_score", "G_mean"]:
                mean, std = clf_sub[metric].mean(), clf_sub[metric].std()
                row.append(f"{fmt_num(mean, drop_leading_zero=drop_leading_zero)} $\\pm$ {fmt_num(std, drop_leading_zero=drop_leading_zero)}")
        rows.append(" & ".join(row) + r" \\")
    return "\n".join(rows)


def build_comparison_table(all_results, extra_rows, datasets=None, classifiers=None,
                            baseline_methods=None, drop_leading_zero=True,
                            mark_close_calls=False, close_call_eps=0.01):
    
    datasets, classifiers = datasets or DATASETS, classifiers or CLASSIFIERS
    baseline_methods = baseline_methods or sorted(BASELINE_METHODS - {"raw"})
    metrics = ["F1_score", "G_mean"]
    row_defs = [(METHOD_DISPLAY.get(m, m), m, None, None) for m in baseline_methods] + extra_rows

    latex_rows = []
    for idx, ds in enumerate(datasets):
        cell_stats = {}
        for label, method, mix_ratio, stage_depth in row_defs:
            sub = all_results[(all_results["Dataset"] == ds) & (all_results["Method"] == method)]
            if mix_ratio is not None:
                sub = sub[sub["mix_ratio"] == mix_ratio]
            if stage_depth is not None:
                sub = sub[sub["stage_depth"] == stage_depth]
            for clf in classifiers:
                clf_sub = sub[sub["Classifier"] == clf]
                for metric in metrics:
                    cell_stats[(label, clf, metric)] = (clf_sub[metric].mean(), clf_sub[metric].std())

        rank_map, close_map = {}, {}
        labels = [r[0] for r in row_defs]
        for clf in classifiers:
            for metric in metrics:
                col_means = {l: cell_stats[(l, clf, metric)][0] for l in labels}
                ordered = sorted(col_means, key=lambda l: -col_means[l])
                for i, l in enumerate(ordered[:2]):
                    rank_map[(l, clf, metric)] = i + 1
                if mark_close_calls and len(ordered) >= 2:
                    gap = col_means[ordered[0]] - col_means[ordered[1]]
                    if gap < close_call_eps:
                        close_map[(ordered[0], clf, metric)] = True
                        close_map[(ordered[1], clf, metric)] = True

        for i, (label, method, mix_ratio, stage_depth) in enumerate(row_defs):
            row = [DATASET_ABBREV.get(ds, ds) if i == 0 else "", label]
            for clf in classifiers:
                for metric in metrics:
                    mean, std = cell_stats[(label, clf, metric)]
                    row.append(format_cell(mean, std, rank_map.get((label, clf, metric), 0),
                                            close_call=close_map.get((label, clf, metric), False),
                                            drop_leading_zero=drop_leading_zero))
            latex_rows.append(" & ".join(row) + r" \\")
        if idx != len(datasets) -1:
            latex_rows.append(r"\midrule")
    return "\n".join(latex_rows)


def summarize_method_stats(all_results, method, mix_ratio=None, stage_depth=None,
                            datasets=None, classifiers=None):
    sub = all_results[all_results["Method"] == method]
    if mix_ratio is not None: sub = sub[sub["mix_ratio"] == mix_ratio]
    if stage_depth is not None: sub = sub[sub["stage_depth"] == stage_depth]
    return sub.groupby(["Dataset", "Classifier"])[["F1_score", "G_mean"]].agg(["mean", "std", "count"])
