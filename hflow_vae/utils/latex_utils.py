# [1] latex table

import pandas as pd


def format_metric(mean, std, bold=False, underline=False):
    """
    格式化:
    0.74 ± .03
    """
    mean_str = f"{mean:.2f}"
    std_str = f"{std:.2f}".replace("0.", ".")

    text = f"{mean_str} $\\pm$ {std_str}"

    if bold:
        text = f"\\textbf{{{text}}}"
    elif underline:
        text = f"\\underline{{{text}}}"

    return text


def generate_latex_table(
    summary_df,
    datasets,
    methods=None
):
    """
    summary_df:
        Dataset Method Classifier F1_mean F1_std G_mean G_std
    datasets:
        要展示的数据集列表
    methods:
        要展示的方法顺序
    """

    df = summary_df[
        summary_df["Dataset"].isin(datasets)
    ].copy()

    if methods is not None:
        df = df[df["Method"].isin(methods)]
        df["Method"] = pd.Categorical(
            df["Method"],
            categories=methods,
            ordered=True
        )


    rows = []
    for dataset in datasets:

        sub = df[df["Dataset"] == dataset]
        # 每个classifier分别计算rank
        rank_info = {}

        for clf in sub["Classifier"].unique():
            clf_data = sub[sub["Classifier"] == clf]
            for metric in ["F1_mean", "G_mean"]:
                values = clf_data.set_index("Method")[metric]
                sorted_values = values.sort_values(
                    ascending=False
                )

                rank_info[(clf, metric)] = {
                    "best": sorted_values.index[0],
                    "second": sorted_values.index[1]
                }


        for method in sub["Method"].unique():
            row = {
                "Dataset": dataset if method == sub["Method"].iloc[0] else ""
            }
            row["Method"] = method

            for clf in ["SVC", "MLP", "XGB"]:
                item = sub[
                    (sub["Method"] == method) &
                    (sub["Classifier"] == clf)
                ]

                if len(item) == 0:
                    row[f"{clf}\_F1"] = "-"
                    row[f"{clf}\_G"] = "-"
                    continue

                item = item.iloc[0]

                for metric, col in [
                    ("F1_mean", "F1"),
                    ("G_mean", "G")
                ]:

                    bold = (
                        method ==
                        rank_info[(clf, metric)]["best"]
                    )

                    underline = (
                        method ==
                        rank_info[(clf, metric)]["second"]
                    )


                    row[f"{clf}_{col}"] = format_metric(
                        item[metric],
                        item[
                            metric.replace(
                                "_mean",
                                "_std"
                            )
                        ],
                        bold,
                        underline
                    )
            rows.append(row)

    result = pd.DataFrame(rows)

    # 转latex
    latex = result.to_latex(
        index=False,
        escape=False
    )

    return latex