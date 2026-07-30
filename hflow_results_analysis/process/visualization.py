from pathlib import Path
from scipy.stats import chi2_contingency, kendalltau
import numpy as np, pandas as pd
from sklearn.manifold import TSNE
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import matplotlib.pyplot as plt
import seaborn as sns


def compute_grid_shape(n):
    ncols = int(np.ceil(np.sqrt(n)))
    nrows = int(np.ceil(n / ncols))
    return nrows, ncols


def build_preprocessor(real_X_train, numeric_cols, categorical_cols):
    return ColumnTransformer([
        ("num", StandardScaler(), numeric_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
    ]).fit(real_X_train[numeric_cols + categorical_cols])


def get_method_dir(artifacts_table, dataset, seed, fold, method, mix_ratio=None, stage_depth=None):
    row = artifacts_table[(artifacts_table["Dataset"] == dataset) & (artifacts_table["Seed"] == seed)
                           & (artifacts_table["Fold"] == fold) & (artifacts_table["Method"] == method)]
    if mix_ratio is not None: row = row[row["mix_ratio"] == mix_ratio]
    if stage_depth is not None: row = row[row["stage_depth"] == stage_depth]
    assert len(row) == 1, f"定位到{len(row)}行,应为1"
    return row.iloc[0]


def load_synthetic_stage(artifacts_table, dataset, seed, fold, method, mix_ratio=None, stage_depth=None, stage="last"):
    row = get_method_dir(artifacts_table, dataset, seed, fold, method, mix_ratio, stage_depth)
    if stage == "__fidelity__":
        return pd.read_parquet(row["fidelity_path"])
    method_dir = Path(row["fidelity_path"]).parent
    v_files = sorted(method_dir.glob("v*.parquet"), key=lambda p: int(p.stem[1:]))
    cluster_files = sorted(method_dir.glob("cluster*.parquet"), key=lambda p: int(p.stem.replace("cluster", "")))
    if stage == "__all_clusters__":
        assert cluster_files, f"{method_dir}下没有cluster*文件"
        return pd.concat([pd.read_parquet(p) for p in cluster_files], ignore_index=True)
    stage_files = {p.stem: p for p in v_files + cluster_files}
    if stage == "last":
        return pd.read_parquet(list(stage_files.values())[-1])
    return pd.read_parquet(stage_files[stage])


def pick_representative_fold(all_results, dataset, method="raw", metric="F1_score"):
    sub = all_results[(all_results["Dataset"] == dataset) & (all_results["Method"] == method)]
    per_fold = sub.groupby(["Seed", "Fold"])[metric].mean().sort_values()
    return per_fold.index[len(per_fold) // 2]


def subsample(df, max_points=150, random_state=0):
    return df if len(df) <= max_points else df.sample(n=max_points, random_state=random_state)


def build_combined_embedding(real_df, group_specs, feature_cols, preprocessor, random_state=0):
    frames = [real_df.assign(__group__="__real__")] + [df.assign(__group__=label) for label, df in group_specs]
    combined = pd.concat(frames, ignore_index=True)
    X = preprocessor.transform(combined[feature_cols])
    X = X.toarray() if hasattr(X, "toarray") else X
    emb = TSNE(n_components=2, random_state=random_state, init="pca",
               perplexity=min(30, max(5, len(X)//4))).fit_transform(X)
    combined["tsne_x"], combined["tsne_y"] = emb[:, 0], emb[:, 1]
    return combined


def render_subplot_grid(combined, labels, show_labels, max_scatter_points=150, random_state=0,
                         real_style=dict(color="gray", s=8, alpha=0.35)):
    real_emb = combined[combined["__group__"] == "__real__"]
    real_plot = subsample(real_emb, max_scatter_points, random_state)
    n = len(labels)
    nrows, ncols = compute_grid_shape(n)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4*ncols, 4*nrows), sharex=True, sharey=True)
    axes = np.array(axes).reshape(-1)
    palette = sns.color_palette("tab10", n_colors=n)

    for ax, label, show_label, color in zip(axes, labels, show_labels, palette):
        ax.scatter(real_plot["tsne_x"], real_plot["tsne_y"], **real_style, label="Real (train pos)")
        sns.kdeplot(x=real_emb["tsne_x"], y=real_emb["tsne_y"], ax=ax, levels=2, color='gray', linewidths=1, linestyles='--', alpha=0.7)
        sub = combined[combined["__group__"] == label]
        sub_plot = subsample(sub, max_scatter_points, random_state)
        ax.scatter(sub_plot["tsne_x"], sub_plot["tsne_y"], s=12, alpha=0.7, color=color, label=label)
        if len(sub) >= 5:
            sns.kdeplot(x=sub["tsne_x"], y=sub["tsne_y"], ax=ax, levels=2, color=color, linewidths=1)  # 全量估计密度
        ax.set_title(show_label, fontsize=12)
        ax.legend(fontsize=10)
    for ax in axes[n:]:
        ax.axis("off")
    fig.tight_layout()
    return fig

# ================================ kendall correlation related ======================
from scipy.stats.contingency import crosstab

def cramers_v(x, y):
    table = pd.crosstab(x, y)
    if table.shape[0] <= 1 or table.shape[1] <= 1:
        return 0.0
    chi2 = chi2_contingency(table, correction=False)[0]
    n = table.sum().sum()
    phi2 = chi2 / n
    r, k = table.shape
    phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))
    rcorr = r - ((r-1)**2)/(n-1)
    kcorr = k - ((k-1)**2)/(n-1)

    return np.sqrt(phi2corr / min((kcorr-1), (rcorr-1)))

# nun_cat correlation ratio η
def correlation_ratio(categories, measurements):
    categories = pd.Series(categories)
    measurements = pd.Series(measurements)
    if categories.nunique() <= 1:
        return 0.0
    grand_mean = measurements.mean()
    numerator = sum(
        len(group) * (group.mean() - grand_mean) ** 2
        for _, group in measurements.groupby(categories)
    )

    denominator = sum((measurements - grand_mean) ** 2)
    if denominator == 0:
        return 0.0

    return np.sqrt(numerator / denominator)


def kendall_corr_matrix(df,numerical_cols, categorical_cols=None):
   
    if categorical_cols is None:
        categorical_cols = []
    cols = numerical_cols + categorical_cols

    n = len(cols)
    mat = np.eye(n)

    for i in range(n):
        for j in range(i + 1, n):
            col_i = cols[i]
            col_j = cols[j]
            if (col_i in numerical_cols and col_j in numerical_cols):
                value, _ = kendalltau(df[col_i], df[col_j])
            elif (col_i in categorical_cols and col_j in categorical_cols):
                value = cramers_v(df[col_i], df[col_j])
            else:
                if col_i in numerical_cols:
                    num_col = col_i
                    cat_col = col_j
                else:
                    num_col = col_j
                    cat_col = col_i
                value = correlation_ratio(df[cat_col], df[num_col])

            if np.isnan(value):
                value = 0.0

            mat[i, j] = value
            mat[j, i] = value

    return pd.DataFrame(mat, index=cols, columns=cols)


'''def kendall_corr_matrix(df, numerical_cols):
    n = len(numerical_cols)
    mat = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            tau, _ = kendalltau(df[numerical_cols[i]], df[numerical_cols[j]])
            mat[i, j] = mat[j, i] = tau
    return pd.DataFrame(mat, index=numerical_cols, columns=numerical_cols)'''


def plot_correlation_grid(real_df, group_specs, numerical_cols, categorical_cols=None):
    
    n_total = len(group_specs) + 1
    ncols = int(np.ceil(np.sqrt(n_total)))
    nrows = int(np.ceil(n_total / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.2*ncols, 3*nrows))
    axes = np.array(axes).reshape(-1)

    real_corr = kendall_corr_matrix(real_df, numerical_cols, categorical_cols)
    sns.heatmap(real_corr, ax=axes[0], cmap="RdBu_r", vmin=-1, vmax=1, cbar=False, square=True,
                xticklabels=False, yticklabels=True)
    axes[0].set_title("Real", fontsize=10)

    for ax, (label, df) in zip(axes[1:], group_specs):
        sns.heatmap(kendall_corr_matrix(df, numerical_cols, categorical_cols), ax=ax, cmap="RdBu_r", vmin=-1, vmax=1,
                    cbar=False, square=True, xticklabels=False, yticklabels=False)
        ax.set_title(label, fontsize=10)
    for ax in axes[n_total:]:
        ax.axis("off")
    fig.tight_layout()
    return fig


def plot_correlation_deviation_grid(real_df, group_specs, numerical_cols, categorical_cols=None):

    real_corr = kendall_corr_matrix(real_df, numerical_cols, categorical_cols)
    n = len(group_specs)
    #ncols = int(np.ceil(np.sqrt(n))); nrows = int(np.ceil(n / ncols))
    ncols = 6; nrows = 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(10, 4))
    axes = np.array(axes).reshape(-1)
    cols = numerical_cols + categorical_cols
    iu = np.triu_indices(len(cols), k=1)
    #iu = np.triu_indices(len(numerical_cols), k=1)

    for ax, (label, df) in zip(axes, group_specs):
        diff = kendall_corr_matrix(df, numerical_cols, categorical_cols) - real_corr
        sns.heatmap(diff, ax=ax, cmap="RdBu_r", vmin=-1, vmax=1, cbar=False, square=False,
                    xticklabels=False, yticklabels=False)
        mad = diff.values[iu].__abs__().mean()
        ax.set_title(f"{label}\n(CAD={mad:.3f})", fontsize=12)
    for ax in axes[n:]:
        ax.axis("off")
    #fig.suptitle(title + " — deviation from Real"); fig.tight_layout()
    fig.tight_layout()
    return fig


def correlation_deviation_summary(real_df, group_specs, numerical_cols, categorical_cols=None):
    real_corr = kendall_corr_matrix(real_df, numerical_cols, categorical_cols)
    cols = numerical_cols + categorical_cols
    iu = np.triu_indices(len(cols), k=1)
    #iu = np.triu_indices(len(numerical_cols), k=1)
    rows = []
    for label, df in group_specs:
        diff = np.abs((kendall_corr_matrix(df, numerical_cols, categorical_cols) - real_corr).values[iu])
        rows.append({"group": label, "mean_abs_corr_dev": round(diff.mean(), 3), "max_abs_corr_dev": round(diff.max(), 3)})
    return pd.DataFrame(rows)