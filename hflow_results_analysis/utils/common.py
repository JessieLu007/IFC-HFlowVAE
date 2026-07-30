import os
import json
from pathlib import Path
import pandas as pd
import numpy as np


# 训练数据集名称
DATASETS = ['breast_cancer_coimbra', 'PIMA_diabetes','TCGA_InfoWithGrade', 'Thyroid_Diff',
            'diabetes_risk_prediction', 'framingham', 'hepatitis']
CLASSIFIERS = ['SVC', 'MLP', 'XGB']
# 对比方法名称
BASELINE_MODELS = ['smotenc', 'ctgan', 'tvae', 'tabddpm', 'tabkde']
BASELINE_METHODS = {"raw", "smote", "ctgan", "tvae", "forestdiffusion", "tabddpm", "tabkde"}
SEEDS = [42, 456, 1024, 2026, 3407]


MIX_RATIOS = ["mix_37_235", "mix_55_532"]
STAGES = [2, 3]
HEADLINE_MIX_RATIO = "mix_37_235"
HEADLINE_STAGE = 3

NONE_METHODS = {"none_noft", "none_ft"}
SELF_LOOP_METHODS = {"bootstrap_noft", "bootstrap_ft", "smote_noft", "smote_ft"}

RAW_COLS = ["Method", "Dataset", "Seed", "Fold", "Classifier", "Recall", "F1_score", "G_mean"]


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
    """创建路径，如果已经存在就跳过"""
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
    X_train: DataFrame
    categorical_features: 分类特征列名列表（或索引）
    返回: 字典 {列名: 精度小数位数}
    """
    precision_dict = {}
    for col in X_train.columns:
        if col not in categorical_features:
            precision_dict[col] = detect_precision(X_train[col])
    return precision_dict


def restore_precision(X_generated, precision_dict):
    """
    将生成的DataFrame的数值列四舍五入到原始精度。
    返回新的DataFrame。
    """
    X_restored = X_generated.copy()
    for col, decimals in precision_dict.items():
        if col in X_restored.columns:
            X_restored[col] = X_restored[col].round(decimals)
    return X_restored


    def fit_claude(
        self, train_tensors, val_tensors, save_dir: Optional[str] = None,
        loader_seed: Optional[int] = None,
        method: str = "pq",              # "pq" 或 "patience"，切换对比用
        strip_k: int = 5, pq_alpha: float = 1.0, ema_alpha: float = 0.3,
    ):
        if self.model is None:
            self.build_model()
        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)

        train_loader = self._make_loader(train_tensors, self.config.batch_size, shuffle=True, seed=loader_seed)
        val_loader = self._make_loader(val_tensors, self.config.batch_size, shuffle=False)

        self.best_val_total = float("inf")
        self.best_state_dict = None
        self.best_epoch = None
        self.history = []

        patience_counter = 0          # 仅 method="patience" 用
        train_loss_window = []        # 仅 method="pq" 用
        val_ema = None                # 仅 method="pq" 用

        for epoch in range(1, self.config.epochs + 1):
            train_metrics = self._run_epoch(train_loader, train=True)
            val_metrics = self._run_epoch(val_loader, train=False)

            row = {"epoch": epoch}
            row.update({f"train_{k}": v for k, v in train_metrics.items()})
            row.update({f"val_{k}": v for k, v in val_metrics.items()})

            if method == "pq":
                train_loss_window.append(train_metrics["total"])
                if len(train_loss_window) > strip_k:
                    train_loss_window.pop(0)
                val_ema = val_metrics["total"] if val_ema is None else (
                    ema_alpha * val_metrics["total"] + (1 - ema_alpha) * val_ema
                )
                row["val_ema"] = val_ema
                current_score = val_ema
            else:  # "patience"
                current_score = val_metrics["total"]

            self.history.append(row)
            print(f"[epoch {epoch:04d}] train_total={train_metrics['total']:.4f} "
                f"val_total={val_metrics['total']:.4f}"
                + (f" val_ema={val_ema:.4f}" if method == "pq" else ""))

            # ---- 更新best（两种method共用同一套判据）----
            if current_score < self.best_val_total - self.config.min_delta:
                self.best_val_total = current_score
                self.best_epoch = epoch
                self.best_state_dict = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1

            # ---- 停止判据，按method分叉 ----
            if method == "patience":
                if patience_counter >= self.config.patience1:
                    print(f"早停触发(patience)：验证集 total loss 已连续 {self.config.patience1} 轮未提升。")
                    break
            else:  # "pq"
                if len(train_loss_window) == strip_k:
                    min_train = min(train_loss_window)
                    progress = 1000 * (sum(train_loss_window) / (strip_k * min_train) - 1) if min_train > 1e-6 else 0.0
                    gen_loss = 100 * (val_ema / max(self.best_val_total, 1e-6) - 1)
                    should_stop = (gen_loss > 0) if progress <= 1e-8 else (gen_loss / progress) > pq_alpha
                    if should_stop:
                        print(f"早停触发(PQ, alpha={pq_alpha})：epoch={epoch}, gen_loss={gen_loss:.2f}, progress={progress:.2f}")
                        break

        if self.best_state_dict is not None:
            self.model.load_state_dict(self.best_state_dict)
        if save_dir is not None:
            pd.DataFrame(self.history).to_csv(os.path.join(save_dir, "training_log.csv"), index=False)
            self.save(save_dir)
        return self.history