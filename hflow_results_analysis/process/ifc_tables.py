import numpy as np
import pandas as pd
from utils.common import DATASETS, CLASSIFIERS, BASELINE_METHODS

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
    m = fmt_num(mean, drop_leading_zero=drop_leading_zero)
    s = fmt_num(std, drop_leading_zero=drop_leading_zero)
    cell = f"{m} $\\pm$ {s}" + (r"$^\dagger$" if close_call else "")
    if rank == 1:
        return f"\\textbf{{{cell}}}"
    elif rank == 2:
        return f"\\underline{{{cell}}}"
    return cell


def build_raw_reference_table(all_results, datasets, classifiers, drop_leading_zero=True):
    datasets = datasets or DATASETS
    classifiers = classifiers or CLASSIFIERS
    rows = []
    for ds in datasets:
        sub = all_results[(all_results["Dataset"] == ds) & (all_results["Method"] == "raw")]
        row = [DATASET_ABBREV.get(ds, ds)]
        for clf in classifiers:
            clf_sub = sub[sub["Classifier"] == clf]
            for metric in ["F1_score", "G_mean"]:
                mean, std = clf_sub[metric].mean(), clf_sub[metric].std()
                row.append(
                    f"{fmt_num(mean, drop_leading_zero=drop_leading_zero)} $\\pm$ "
                    f"{fmt_num(std, drop_leading_zero=drop_leading_zero)}"
                )
        rows.append(" & ".join(row) + r" \\")
    return "\n".join(rows)


def build_comparison_table(all_results, extra_rows, datasets, classifiers, baseline_methods,
                            drop_leading_zero=True, mark_close_calls=False, close_call_eps=0.01):

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
                    row.append(format_cell(
                        mean, std, rank_map.get((label, clf, metric), 0),
                        close_call=close_map.get((label, clf, metric), False),
                        drop_leading_zero=drop_leading_zero,
                    ))
            latex_rows.append(" & ".join(row) + r" \\")
        if idx != len(datasets) -1:
            latex_rows.append(r"\midrule")
    return "\n".join(latex_rows)


def summarize_method_stats(all_results, method, mix_ratio=None, stage_depth=None):
    sub = all_results[all_results["Method"] == method]
    if mix_ratio is not None:
        sub = sub[sub["mix_ratio"] == mix_ratio]
    if stage_depth is not None:
        sub = sub[sub["stage_depth"] == stage_depth]
    return sub.groupby(["Dataset", "Classifier"])[["F1_score", "G_mean"]].agg(["mean", "std", "count"])