import os
import json
from pathlib import Path
import pandas as pd
import numpy as np


# 训练数据集名称
DATASETS = ['breast_cancer_coimbra', 'PIMA_diabetes','TCGA_InfoWithGrade', 'Thyroid_Diff',
            'diabetes_risk_prediction', 'framingham', 'hepatitis']
# 对比方法名称
BASELINE_MODELS = ['smotenc', 'ctgan', 'tvae', 'tabddpm', 'tabkde']
SEEDS = [42, 456, 1024, 2026, 3407]

ROOT = Path(__file__).parent.parent


# get the per_fold classification results of dataset_name
def get_per_fold_result(dataset_name, seed, fold):
    path = ROOT / 'results' / 'per_fold' / f'{dataset_name}__seed{seed}__fold{fold}.parquet'
    return pd.read_parquet(path)


# get X_train_pos of certain seedxfold 
def get_train_and_test_pos(dataset_name, seed, fold):
    path = ROOT / 'artifacts' / dataset_name / f'seed{seed}' / f'fold{fold}' / '_fold_data'
    X_train = pd.read_parquet(path / 'X_train.parquet')
    y_train = pd.read_parquet(path / 'y_train.parquet').squeeze()
    X_test = pd.read_parquet(path / 'X_test.parquet')
    y_test = pd.read_parquet(path / 'y_test.parquet').squeeze() 

    train_pos_mask = (y_train == 1)
    train_pos = X_train[train_pos_mask]
    test_pos_mask = (y_test == 1)
    test_pos = X_test[test_pos_mask]

    return train_pos, test_pos


# get the generated {X_train_pos} sample for fidelity evaluation
def get_gens(dataset_name, seed, fold, method_tag, gen_type):
    path = ROOT / 'artifacts'/ dataset_name / f'seed{seed}' / f'fold{fold}'/ method_tag / f'{gen_type}.parquet'
    return pd.read_parquet(path)


def set_seed(seed):
    import random
    import torch

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def create_path (path):
    
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"路径不存在，已创建：{path}")
    else:
        print(f"路径已存在：{path}")


def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        context = json.load(f)
    return context

def load_data_and_info(dataset_name):
    """
    return df, X, y, num_cols, cat_cols, target
    """

    ROOT = Path.cwd().parent
    DATA_LOC = ROOT / "data" / "original"
    RESULTS_DIR = ROOT / "results" / "per_fold"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    datasets_info = load_json(ROOT / "data" / "datasets_info.json")
    df = pd.read_csv(DATA_LOC / f"{dataset_name}.csv")
    num_cols = datasets_info[dataset_name]["numerical"]
    cat_cols = datasets_info[dataset_name]["categorical"]
    target = datasets_info[dataset_name]["target"]
    X = df.drop(columns = target)
    y = df[target]
    return df, X, y, num_cols, cat_cols, target, datasets_info


# ============= 返回要访问的结果路径 ======================
from pathlib import Path

current_dir = Path(__file__).resolve().parent.parent
ARTIFACT_ROOT = current_dir / "artifacts"


def artifact_dir(dataset: str, seed: int, fold: int, method_tag: str) -> Path:
    return ARTIFACT_ROOT / dataset / f"seed{seed}" / f"fold{fold}" / method_tag


import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, SimpleImputer


def impute_missing_values(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
    num_cols: list[str],
    cat_cols: list[str],
    random_state: int = 42,
    max_iter: int = 20,
):
    train = train_df.copy()
    valid = valid_df.copy()
    test  = test_df.copy()

    if len(num_cols) > 0:
        num_imputer = IterativeImputer(
            random_state=random_state,
            max_iter=max_iter,
            sample_posterior=False,
            initial_strategy="median"
        )
        train[num_cols] = num_imputer.fit_transform(train[num_cols])
        valid[num_cols] = num_imputer.transform(valid[num_cols])
        test[num_cols]  = num_imputer.transform(test[num_cols])
    
    if len(cat_cols) > 0:
        cat_imputer = SimpleImputer(
            strategy="most_frequent"
        )

        train[cat_cols] = cat_imputer.fit_transform(train[cat_cols])
        valid[cat_cols] = cat_imputer.transform(valid[cat_cols])
        test[cat_cols] = cat_imputer.transform(test[cat_cols])


def detect_precision(series):
    """
    检测一个数值列中所有非缺失值的小数位数，返回最大小数位数。
    """
    clean_vals = series.dropna()
    if len(clean_vals) == 0:
        return 0
    
    max_decimals = 0
    for v in clean_vals:
        # 如果值是整数或浮点，转为字符串，去掉末尾的0，再分割
        s = f"{v:.12f}".rstrip('0').rstrip('.')
        if '.' in s:
            decimals = len(s.split('.')[1])
        else:
            decimals = 0
        if decimals > max_decimals:
            max_decimals = decimals
    return max_decimals

def get_precision_dict(X_train, categorical_features):
    """
    为训练数据中的每个数值列计算精度。
    """
    precision_dict = {}
    for col in X_train.columns:
        if col not in categorical_features:
            precision_dict[col] = detect_precision(X_train[col])
    return precision_dict


def restore_precision(X_generated, precision_dict):
    """
    将生成的DataFrame的数值列四舍五入到原始精度。
    
    """
    X_restored = X_generated.copy()
    for col, decimals in precision_dict.items():
        if col in X_restored.columns:
            X_restored[col] = X_restored[col].round(decimals)
    return X_restored

