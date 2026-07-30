
import warnings
import numpy as np
import pandas as pd
import pingouin as pg
import scikit_posthocs as sp
from scipy import stats
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt
import seaborn as sns

from .art_core import aligned_rank_for_effect

BASELINE_METHODS = ["raw", "smote", "ctgan", "tvae", "tabddpm", "forestdiffusion", "tabkde"]
IFC_CELLS = ["none_noft", "none_ft", "bootstrap_noft", "bootstrap_ft", "smote_noft", "smote_ft"]
ALL_METHODS = BASELINE_METHODS + IFC_CELLS
CORE_METRICS = ["F1_score", "G_mean"]


# ------------------------------------------------------------------
# 0. 数据校验与预处理
# ------------------------------------------------------------------
def validate_and_prepare(df, dataset=None):
    """
    基本校验 + 生成 block 列（Seed x Fold 作为配对设计的重复测量单元）。
    """
    df = df.copy()
    if dataset is not None:
        df = df[df["Dataset"] == dataset].copy()

    unknown = set(df["Method"].unique()) - set(ALL_METHODS)
    if unknown:
        raise ValueError(f"发现未在 ALL_METHODS 中定义的 Method 取值: {unknown}，")

    df["block"] = df["Seed"].astype(str) + "_" + df["Fold"].astype(str)

    
    completeness = (df.groupby(["Classifier", "block"])["Method"]
                       .apply(lambda s: set(s) == set(df["Method"].unique())))
    if not completeness.all():
        bad = completeness[~completeness].index.tolist()
        raise ValueError(f"以下 (Classifier, block) 组合下 Method 不完整（数据缺失/未对齐），"
                          f"配对检验要求完全交叉设计，请先补齐或从分析中剔除: {bad[:5]}"
                          f"{' ...' if len(bad) > 5 else ''}")

    
    dup_mask = df.duplicated(subset=["Method", "Classifier", "block"], keep=False)
    if dup_mask.any():
        n_dup = df.loc[dup_mask, ["Method", "Classifier", "block"]].drop_duplicates().shape[0]
        raise ValueError(f"发现 {n_dup} 组 (Method,Classifier,Seed,Fold) 重复记录，"
                          f"配对设计要求每组恰好一条记录。常见原因：混合了不同 mix_ratio 目录的结果，"
                          f"或同一批实验被重复加载了两次。")
    return df


def parse_ifc_cells(df):
   
    df = df.copy()
    is_ifc = df["Method"].isin(IFC_CELLS)
    df["is_ifc_cell"] = is_ifc
    parts = df["Method"].where(is_ifc).str.rsplit("_", n=1, expand=True)
    df["prior_type"] = parts[0]
    df["use_finetune"] = parts[1].map({"ft": True, "noft": False})
    return df



def descriptive_table(df, metric, n_boot=2000, ci_alpha=0.05, seed=0):
    
    rng = np.random.default_rng(seed)
    rows = []
    for (method, clf), g in df.groupby(["Method", "Classifier"]):
        x = g[metric].values
        boot_means = rng.choice(x, size=(n_boot, len(x)), replace=True).mean(axis=1)
        lo, hi = np.percentile(boot_means, [100 * ci_alpha / 2, 100 * (1 - ci_alpha / 2)])
        rows.append({
            "Method": method, "Classifier": clf, "n": len(x),
            "mean": x.mean(), "std": x.std(ddof=1),
            "ci_low": lo, "ci_high": hi,
            "is_ifc_cell": method in IFC_CELLS,
        })
    out = pd.DataFrame(rows)
    out["rank"] = out.groupby("Classifier")["mean"].rank(ascending=False, method="min").astype(int)
    return out.sort_values(["Classifier", "rank"]).reset_index(drop=True)


def add_delta_vs_reference(desc_table, reference_method="raw"):
   
    out = desc_table.copy()
    ref = out[out["Method"] == reference_method].set_index("Classifier")["mean"]
    out["delta_vs_" + reference_method] = out.apply(
        lambda r: r["mean"] - ref.get(r["Classifier"], np.nan), axis=1)
    return out


def descriptive_table_by_group(df, metric, classifier, group_col="Fold"):
    
    rows = []
    for (method, gval), g in df[df["Classifier"] == classifier].groupby(["Method", group_col]):
        x = g[metric].values
        rows.append({"Method": method, group_col: gval, "n": len(x),
                     "mean": x.mean(), "std": x.std(ddof=1) if len(x) > 1 else np.nan})
    out = pd.DataFrame(rows)
    out["rank"] = out.groupby(group_col)["mean"].rank(ascending=False, method="min").astype(int)
    return out.sort_values([group_col, "rank"]).reset_index(drop=True)


def method_rank_frequency(df, metric, classifier, top_k=3):
    
    sub = df[df["Classifier"] == classifier]
    wide = sub.pivot(index="block", columns="Method", values=metric)
    ranks = wide.rank(axis=1, ascending=False)  # 每个block内部排名
    out = pd.DataFrame({
        "avg_rank": ranks.mean(),
        "rank_std": ranks.std(),
        "best_rank": ranks.min(),
        "worst_rank": ranks.max(),
        f"pct_top{top_k}": (ranks <= top_k).mean(),
    })
    return out.sort_values("avg_rank")


def kendalls_w(df, metric, classifier):
    
    sub = df[df["Classifier"] == classifier]
    wide = sub.pivot(index="block", columns="Method", values=metric)
    stat, p = stats.friedmanchisquare(*[wide[c].values for c in wide.columns])
    n_blocks, n_methods = wide.shape
    w = stat / (n_blocks * (n_methods - 1))
    return {"kendalls_w": w, "friedman_stat": stat, "friedman_p": p,
            "n_blocks": n_blocks, "n_methods": n_methods}



def art_anova_3x2(df, metric, classifier):
    
    sub = df[(df["Classifier"] == classifier) & (df["is_ifc_cell"])].copy()
    factor_cols = ["prior_type", "use_finetune", "block"]

    results = []
    for target in [("prior_type",), ("use_finetune",), ("prior_type", "use_finetune")]:
        aligned_df = aligned_rank_for_effect(sub, metric, target, factor_cols)
        within = [f for f in factor_cols if f != "block"]
        with warnings.catch_warnings():
            
            warnings.simplefilter("ignore", category=RuntimeWarning)
            anova_res = pg.rm_anova(data=aligned_df, dv="aligned_rank",
                                     within=within, subject="block", detailed=True)
        effect_name = " * ".join(target) if len(target) > 1 else target[0]
        row = anova_res[anova_res["Source"] == effect_name].copy()
        row.insert(0, "target_effect", effect_name)
        results.append(row)

    out = pd.concat(results, ignore_index=True)
    out.insert(0, "Classifier", classifier)
    out.insert(0, "metric", metric)
    return out[["metric", "Classifier", "target_effect", "SS", "ddof1", "ddof2",
                "F", "p_unc", "ng2"]]


def art_anova_summary_all_classifiers(df, metric):
    return pd.concat([art_anova_3x2(df, metric, clf) for clf in sorted(df["Classifier"].unique())],
                      ignore_index=True)



def friedman_nemenyi(df, metric, classifier):
    
    sub = df[df["Classifier"] == classifier]
    wide = sub.pivot(index="block", columns="Method", values=metric)
    wide = wide[[m for m in ALL_METHODS if m in wide.columns]]  # 固定顺序

    stat, p = stats.friedmanchisquare(*[wide[c].values for c in wide.columns])
    nemenyi = sp.posthoc_nemenyi_friedman(wide.values)
    nemenyi.index = wide.columns
    nemenyi.columns = wide.columns

    avg_rank = wide.rank(axis=1, ascending=False).mean().sort_values()
    return {
        "friedman_stat": stat, "friedman_p": p,
        "avg_rank": avg_rank, "nemenyi_pvalues": nemenyi,
        "wide_data": wide,
    }


def wilcoxon_holm_vs_reference(df, metric, classifier, reference_method, compare_methods=None):
    
    sub = df[df["Classifier"] == classifier]
    wide = sub.pivot(index="block", columns="Method", values=metric)
    if compare_methods is None:
        compare_methods = [m for m in ALL_METHODS if m != reference_method and m in wide.columns]

    rows = []
    for m in compare_methods:
        if m not in wide.columns:
            continue
        stat, p = stats.wilcoxon(wide[reference_method], wide[m])
        median_diff = (wide[reference_method] - wide[m]).median()
        rows.append({"reference": reference_method, "compare_to": m,
                      "median_diff": median_diff, "wilcoxon_stat": stat, "p_unc": p})
    out = pd.DataFrame(rows)
    if len(out) > 0:
        out["p_holm"] = multipletests(out["p_unc"], method="holm")[1]
    return out.sort_values("p_holm") if "p_holm" in out.columns else out



METHOD_PALETTE_GROUPS = {
    "baseline": "#8c8c8c",
    "ifc": "#2f6fb0",
}


def _method_group(m):
    return "ifc" if m in IFC_CELLS else "baseline"


def plot_boxplot_by_classifier(df, metric, dataset_name="", save_path=None):
   
    classifiers = sorted(df["Classifier"].unique())
    fig, axes = plt.subplots(1, len(classifiers), figsize=(6 * len(classifiers), 5), sharey=True)
    if len(classifiers) == 1:
        axes = [axes]

    for ax, clf in zip(axes, classifiers):
        sub = df[df["Classifier"] == clf]
        order = sub.groupby("Method")[metric].mean().sort_values(ascending=False).index.tolist()
        palette = {m: METHOD_PALETTE_GROUPS[_method_group(m)] for m in order}
        sns.boxplot(data=sub, x="Method", y=metric, order=order, hue="Method",
                    palette=palette, legend=False, ax=ax,
                    showfliers=False, boxprops=dict(alpha=0.6))
        sns.stripplot(data=sub, x="Method", y=metric, order=order, ax=ax,
                       color="black", size=3, alpha=0.4, jitter=0.2)
        ax.set_title(f"{clf}")
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
        ax.set_xlabel("")
    fig.suptitle(f"{dataset_name} - {metric}  (gray=baseline, blue=IFC cell)")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_interaction_effect(df, metric, dataset_name="", save_path=None):
    
    ifc = df[df["is_ifc_cell"]]
    classifiers = sorted(ifc["Classifier"].unique())
    fig, axes = plt.subplots(1, len(classifiers), figsize=(5 * len(classifiers), 4.5), sharey=True)
    if len(classifiers) == 1:
        axes = [axes]

    prior_order = ["none", "bootstrap", "smote"]
    for ax, clf in zip(axes, classifiers):
        sub = ifc[ifc["Classifier"] == clf]
        agg = sub.groupby(["prior_type", "use_finetune"])[metric].agg(["mean", "std", "count"]).reset_index()
        agg["se"] = agg["std"] / np.sqrt(agg["count"])
        for ft_val, label, marker in [(False, "no finetune", "o"), (True, "finetune", "s")]:
            d = agg[agg["use_finetune"] == ft_val].set_index("prior_type").reindex(prior_order)
            ax.errorbar(prior_order, d["mean"], yerr=d["se"], marker=marker, label=label, capsize=4)
        ax.set_title(clf)
        ax.set_xlabel("prior_type")
        ax.legend()
    axes[0].set_ylabel(metric)
    fig.suptitle(f"{dataset_name} - {metric} interaction (prior_type x use_finetune)")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig