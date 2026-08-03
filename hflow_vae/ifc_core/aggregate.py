
from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon, friedmanchisquare

METRICS = ["Recall", "F1_score", "G_mean"]


def load_results(results_dir: Path) -> pd.DataFrame:
    files = sorted(results_dir.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"no result files under {results_dir}")
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


def summarize(results: pd.DataFrame) -> pd.DataFrame:
    """按 (Dataset, Method, Classifier) 聚合 mean±std,聚合维度是 Seed x Fold。"""
    agg = results.groupby(["Dataset", "Method", "Classifier"])[METRICS].agg(["mean", "std"])
    agg.columns = [f"{metric}_{stat}" for metric, stat in agg.columns]
    return agg.reset_index()

def compare_two_methods(
    results: pd.DataFrame, method_a: str, method_b: str, alpha: float = 0.05) -> pd.DataFrame:

    rows = []
    for (dataset, classifier), group in results.groupby(["Dataset", "Classifier"]):
        methods_present = group["Method"].unique()
        if method_a not in methods_present or method_b not in methods_present:
            continue

        pivot = group.pivot_table(index=["Seed", "Fold"], columns="Method", values=METRICS)

        for metric in METRICS:
            if (metric, method_a) not in pivot.columns or (metric, method_b) not in pivot.columns:
                continue

            paired = pd.concat(
                [pivot[(metric, method_a)], pivot[(metric, method_a)]],
                axis=1, keys=["a", "b"]
            ).dropna()

            if len(paired) < 3 or (paired["a"] == paired["b"]).all():
                p_value = np.nan
            else:
                try:
                    _, p_value = wilcoxon(paired["a"], paired["b"])
                except ValueError:
                    p_value = np.nan

            rows.append({
                "Dataset": dataset, "Classifier": classifier, "Metric": metric, 
                "Method_A": method_a, "Method_B": method_b,
                "Mean_A": paired["a"].mean() if len(paired) else np.nan,
                "Mean_B": paired["b"].mean() if len(paired) else np.nan, 
                "n_pairs": len(paired), "p_value": p_value,
                "significant": bool(p_value < alpha) if pd.notna(p_value) else False,
            })

    return pd.DataFrame(rows)


def pairwise_significance(
    results: pd.DataFrame, reference_method: str="smote_tf", alpha: float = 0.05,
) -> pd.DataFrame:
    other_methods = [m for m in results["Method"].unique() if m != reference_method]
    frames = [compare_two_methods(results, reference_method, m, alpha=alpha) for m in other_methods]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def friedman_nemenyi(
    results: pd.DataFrame, classifier: str, metric: str,
) -> tuple[float, float, pd.DataFrame | None]:
  
    sub = results[results["Classifier"] == classifier]
    per_dataset_method = sub.groupby(["Dataset", "Method"])[metric].mean().reset_index()
    wide = per_dataset_method.pivot(index="Dataset", columns="Method", values=metric).dropna()

    stat, p_value = friedmanchisquare(*[wide[c].values for c in wide.columns])

    nemenyi = None
    if p_value < 0.05:
        try:
            import scikit_posthocs as sp
            nemenyi = sp.posthoc_nemenyi_friedman(wide.values)
            nemenyi.columns = wide.columns
            nemenyi.index = wide.columns
        except ImportError:
            print("scikit-posthocs 未安装,跳过 Nemenyi post-hoc")

    return stat, p_value, nemenyi