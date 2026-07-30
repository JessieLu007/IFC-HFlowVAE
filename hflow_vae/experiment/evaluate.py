# experiment/evaluate.py
"""
把"生成的少数类合成数据"接入分类评估:构造增强训练集 -> 编码 -> 送入
ClassificationEvaluator -> 返回带 Method/Dataset/Seed/Fold 标识的结果表。
"""

from __future__ import annotations
import numpy as np
import pandas as pd

from evaluation.classifier import ClassificationEvaluator

_evaluator = ClassificationEvaluator()

def _onehot_expand_categoricals(df_enc: pd.DataFrame, processor) -> pd.DataFrame:

    df_out = df_enc[processor.num_cols].copy()
    for col in processor.cat_cols:
        n_classes = processor.meta[col]["classes"]
        codes = df_enc[col].to_numpy()
        codes = np.clip(codes, -1, n_classes - 1)
        for k in range(n_classes):
            df_out[f"{col}__{k}"] = (codes == k).astype(np.float32)
        df_out[f"{col}__unknown"] = (codes == -1).astype(np.float32)
    return df_out


def evaluate_raw(fold_data, random_state: int) -> pd.DataFrame:
    """不做任何数据增强，直接用原始（不平衡）训练集训练分类器，作为基线对照"""
    #X_train_enc = fold_data.processor.transform(fold_data.X_train)
    #X_test_enc  = fold_data.processor.transform(fold_data.X_test)

    X_train_enc = _onehot_expand_categoricals(
        fold_data.processor.transform(fold_data.X_train), fold_data.processor
    )
    X_test_enc = _onehot_expand_categoricals(
        fold_data.processor.transform(fold_data.X_test), fold_data.processor
    )

    result = _evaluator.evaluate(
        X_train_enc, fold_data.y_train, X_test_enc, fold_data.y_test,
        random_state=random_state
    )
    result.insert(0, "Method", "raw")
    result.insert(1, "Dataset", fold_data.dataset)
    result.insert(2, "Seed", fold_data.seed)
    result.insert(3, "Fold", fold_data.fold)
    return result


def build_augmented_train(fold_data, X_syn: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    
    X_aug = pd.concat(
        [fold_data.X_train.reset_index(drop=True), X_syn.reset_index(drop=True)],
        ignore_index=True,
    )
    y_syn = pd.Series(np.ones(len(X_syn), dtype=fold_data.y_train.to_numpy().dtype))
    y_aug = pd.concat(
        [fold_data.y_train.reset_index(drop=True), y_syn], ignore_index=True,
    )
    return X_aug, y_aug


def evaluate_method(
    fold_data, X_syn: pd.DataFrame, method_name: str, random_state: int,
) -> pd.DataFrame:

    X_train_aug, y_train_aug = build_augmented_train(fold_data, X_syn)

    X_train_enc = _onehot_expand_categoricals(
        fold_data.processor.transform(X_train_aug), fold_data.processor)
    
    X_test_enc = _onehot_expand_categoricals(
        fold_data.processor.transform(fold_data.X_test), fold_data.processor)

    result = _evaluator.evaluate(
        X_train_enc, y_train_aug, X_test_enc, fold_data.y_test,
        random_state=random_state,
    )
    result.insert(0, "Method", method_name)
    result.insert(1, "Dataset", fold_data.dataset)
    result.insert(2, "Seed", fold_data.seed)
    result.insert(3, "Fold", fold_data.fold)
    return result