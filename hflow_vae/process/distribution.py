import seaborn as sns
from matplotlib import pyplot as plt
from scipy.spatial.distance import jensenshannon
from scipy.stats import skew, gaussian_kde
from scipy.signal import find_peaks
import numpy as np, pandas as pd

from process.ifc_fidelity_analysis import load_synthetic_stage, pick_representative_fold


def numeric_peak_count(data, prominence_ratio=0.05, pad_ratio=0.15, grid_size=500):
    data = np.asarray(data).flatten()
    data = data[~np.isnan(data)]
    if len(data) < 5:
        return np.nan
    x_min, x_max = data.min(), data.max()
    x_range = x_max - x_min if x_max > x_min else 1.0
    x_grid = np.linspace(x_min - pad_ratio*x_range, x_max + pad_ratio*x_range, grid_size)
    y = gaussian_kde(data)(x_grid)
    peaks, _ = find_peaks(y, prominence=prominence_ratio * y.max())
    return len(peaks)


def categorical_shift(real_series, synthetic_series):
    real_counts = real_series.value_counts(normalize=True)
    syn_counts = synthetic_series.value_counts(normalize=True)
    all_cats = real_counts.index.union(syn_counts.index)
    p = real_counts.reindex(all_cats, fill_value=0).values
    q = syn_counts.reindex(all_cats, fill_value=0).values
    js = jensenshannon(p, q, base=2)
    coverage = syn_counts.index.nunique() / max(real_counts.index.nunique(), 1)
    return round(js, 3), round(coverage, 3)


def build_stage_evolution_table(real_df, stage_frames, numerical_cols, categorical_cols):
    rows = []
    real_row_base = {"stage": "Real"}
    for col in numerical_cols:
        rows.append({**real_row_base, "column": col, "type": "numeric",
                     "skewness": round(skew(real_df[col].dropna()), 3),
                     "n_peaks": numeric_peak_count(real_df[col])})
    for label, df in stage_frames:
        for col in numerical_cols:
            rows.append({"stage": label, "column": col, "type": "numeric",
                         "skewness": round(skew(df[col].dropna()), 3),
                         "n_peaks": numeric_peak_count(df[col])})
        for col in categorical_cols:
            js, coverage = categorical_shift(real_df[col], df[col])
            rows.append({"stage": label, "column": col, "type": "categorical",
                         "js_divergence": js, "category_coverage": coverage})
    return pd.DataFrame(rows)


def compute_composite_fidelity(real_df, synthetic_df, numerical_cols, categorical_cols):
    real_skew = {c: skew(real_df[c].dropna()) for c in numerical_cols}
    real_peaks = {c: numeric_peak_count(real_df[c]) for c in numerical_cols}
    num_dev, peak_mismatch, cat_js = [], [], []
    for c in numerical_cols:
        num_dev.append(abs(skew(synthetic_df[c].dropna()) - real_skew[c]))
        peak_mismatch.append(int(numeric_peak_count(synthetic_df[c]) != real_peaks[c]))
    for c in categorical_cols:
        js, _ = categorical_shift(real_df[c], synthetic_df[c])
        cat_js.append(js)
    return {
        "mean_abs_skew_dev": np.mean(num_dev) if num_dev else np.nan,
        "peak_mismatch_rate": np.mean(peak_mismatch) if peak_mismatch else np.nan,
        "mean_js_divergence": np.mean(cat_js) if cat_js else np.nan,
    }


def build_fidelity_summary(all_results, orig_table, artifacts_table, datasets, methods_to_show,
                            datasets_info):
    rows = []
    for ds in datasets:
        numerical_cols = datasets_info[ds]["numerical"]
        categorical_cols = datasets_info[ds]["categorical"]

        seed, fold = pick_representative_fold(all_results, ds)
        orig_row = orig_table[(orig_table["Dataset"]==ds)&(orig_table["Seed"]==seed)&(orig_table["Fold"]==fold)].iloc[0]
        real_df = pd.read_parquet(orig_row["X_train_pos_path"])
        for label, spec in methods_to_show.items():
            syn_df = load_synthetic_stage(artifacts_table, ds, seed, fold, **spec)
            rows.append({"Dataset": ds, "Method": label,
                         **compute_composite_fidelity(real_df, syn_df, numerical_cols, categorical_cols)})
    return pd.DataFrame(rows)


def plot_fidelity_heatmap(summary_df, value_col, dataset_map=None):
    plot_df = summary_df.copy()

    if dataset_map is not None:
        plot_df["Dataset"] = plot_df["Dataset"].replace(dataset_map)

    pivot = plot_df.pivot(index="Dataset", columns="Method", values=value_col)
    fig, ax = plt.subplots(figsize=(1.2*len(pivot.columns)+2, 0.6*len(pivot.index)+2))
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="Blues", ax=ax)
    fig.tight_layout()
    return fig


def summarize_across_datasets(summary_df, value_col):
    return summary_df.groupby("Method")[value_col].agg(["mean", "std"]).sort_values("mean")


from scipy.spatial.distance import jensenshannon
from scipy.stats import skew
import numpy as np, pandas as pd

def categorical_imbalance(series):
    counts = series.value_counts(normalize=True)
    k = len(counts)
    if k <= 1:
        return 0.0
    uniform = np.full(k, 1.0 / k)
    return round(jensenshannon(counts.values, uniform, base=2), 3)


def real_data_distribution_summary(datasets_dict, numerical_cols_map, categorical_cols_map):
    rows = []
    for ds, df in datasets_dict.items():
        num_cols, cat_cols = numerical_cols_map.get(ds, []), categorical_cols_map.get(ds, [])
        row = {"Dataset": ds, "n_numeric": len(num_cols), "n_categorical": len(cat_cols)}
        if num_cols:
            skews = [abs(skew(df[c].dropna())) for c in num_cols]
            row["mean_abs_skew"] = round(np.mean(skews), 3)
            row["pct_severe_skew"] = round(np.mean([s > 2 for s in skews]), 3)
            row["pct_multimodal"] = round(np.mean([numeric_peak_count(df[c]) >= 2 for c in num_cols]), 3)
        else:
            row["mean_abs_skew"] = row["pct_severe_skew"] = row["pct_multimodal"] = np.nan
        row["mean_cat_imbalance"] = round(np.mean([categorical_imbalance(df[c]) for c in cat_cols]), 3) if cat_cols else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


# ---- 第一步: 标记real data里"难建模"的列 ----
def flag_real_problem_columns(real_df, numerical_cols, categorical_cols,
                               skew_threshold=1, imbalance_threshold=0.3):
    rows = []
    for c in numerical_cols:
        s = skew(real_df[c].dropna())
        n_peaks = numeric_peak_count(real_df[c])
        rows.append({"column": c, "type": "numeric", "skewness": round(s, 3),
                     "n_peaks": n_peaks, "flagged": abs(s) > skew_threshold or n_peaks >= 2})
    for c in categorical_cols:
        js = categorical_imbalance(real_df[c])
        rows.append({"column": c, "type": "categorical", "js_vs_uniform": js,
                     "flagged": js > imbalance_threshold})
    return pd.DataFrame(rows)


# ---- 第二步: 只在"困难列"上比较各个synthetic方法对real的保留 ----
def build_mitigation_table(real_df, problem_cols_df, group_specs):
    flagged = problem_cols_df[problem_cols_df["flagged"]]
    rows = []
    for label, syn_df in group_specs:
        num_devs, peak_matches, cat_js = [], [], []
        for _, r in flagged.iterrows():
            c = r["column"]
            if r["type"] == "numeric":
                num_devs.append(abs(skew(syn_df[c].dropna()) - r["skewness"]))
                peak_matches.append(int(numeric_peak_count(syn_df[c]) == r["n_peaks"]))
            else:
                js, _ = categorical_shift(real_df[c], syn_df[c])
                cat_js.append(js)
        rows.append({
            "group": label,
            "mean_skew_dev_on_flagged": round(np.mean(num_devs), 3) if num_devs else np.nan,
            "peak_match_rate_on_flagged": round(np.mean(peak_matches), 3) if peak_matches else np.nan,
            "mean_js_on_flagged_categorical": round(np.mean(cat_js), 3) if cat_js else np.nan,
        })
    return pd.DataFrame(rows)


# ---- 第三步: 跨数据集拼总表 ----
def build_mitigation_summary_all_datasets(all_results, orig_table, artifacts_table, datasets,
                                           methods_to_show, numerical_cols_map, categorical_cols_map):
    all_rows = []
    for ds in datasets:
        seed, fold = pick_representative_fold(all_results, ds)
        orig_row = orig_table[(orig_table["Dataset"] == ds) & (orig_table["Seed"] == seed)
                               & (orig_table["Fold"] == fold)].iloc[0]
        real_df = pd.read_parquet(orig_row["X_train_pos_path"])
        num_cols = numerical_cols_map.get(ds, [])
        cat_cols = categorical_cols_map.get(ds, [])

        problem_cols = flag_real_problem_columns(real_df, num_cols, cat_cols)
        group_specs_ds = [(label, load_synthetic_stage(artifacts_table, ds, seed, fold, **spec))
                           for label, spec in methods_to_show.items()]
        mitigation = build_mitigation_table(real_df, problem_cols, group_specs_ds)
        mitigation.insert(0, "Dataset", ds)
        all_rows.append(mitigation)
    return pd.concat(all_rows, ignore_index=True).rename(columns={"group": "Method"})