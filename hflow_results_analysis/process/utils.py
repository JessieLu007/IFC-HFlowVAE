import glob
import pandas as pd
from pathlib import Path


# ---------- 配置 ----------
OURS_METHODS = {"none_noft", "none_ft", "bootstrap_noft", "bootstrap_ft", "smote_noft", "smote_ft"}
STAGE2_OURS_METHODS = {"bootstrap_noft", "bootstrap_ft", "smote_noft", "smote_ft"}  # 2-stage实际只跑了这4个
BASELINE_METHODS = {"smote", "ctgan", "tvae", "forestdiffusion", "tabddpm", "tabkde", "raw"}

FOLD_COUNT = {"breast_cancer_coimbra": 3, "hepatitis": 3}   # 未列出的数据集默认5折
DEFAULT_FOLDS = 5
N_SEEDS = 5


def expected_rows(dataset):
    return FOLD_COUNT.get(dataset, DEFAULT_FOLDS) * N_SEEDS


def load_dir(path, pattern="*.parquet"):
    files = glob.glob(f"{path}/{pattern}")
    if not files:
        raise FileNotFoundError(f"No files found under {path}")
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


def expand_method(df, stage_depth, mix_ratio): 
    """根据Method 补充prior_type和finetune, 同时增加stage_depth和mix_ratio"""

    df = df.copy()

    df["stage_depth"] = stage_depth
    df["mix_ratio"]   = mix_ratio

    prior_type = []
    finetune   = []

    for m in df["Method"]:
        if m in OURS_METHODS:
            p, ft = m.rsplit("_", 1)

            prior_type.append(p)
            finetune.append(ft == "ft")
        else:
            prior_type.append(None)
            finetune.append(None)

    df["prior_type"] = prior_type
    df["finetune"]   = finetune

    return df

# ==========================================================================

def is_valid_fold_dir(fold_dir, dataset):
    """过滤掉 fold<seed_number> 这种checkpoint目录。"""
    try:
        fold_num = int(fold_dir.name.replace("fold", ""))
    except ValueError:
        return False
    max_folds = FOLD_COUNT.get(dataset, DEFAULT_FOLDS)
    return 1 <= fold_num <= max_folds

def build_manifest(artifacts_root, methods_filter, stage_depth, mix_ratio):
    rows = []
    skipped_methods = set()

    for dataset_dir in artifacts_root.iterdir():
        if not dataset_dir.is_dir():
            continue

        dataset = dataset_dir.name
        for seed_dir in dataset_dir.glob("seed*"):
            seed = int(seed_dir.name.replace("seed", ""))
            for fold_dir in seed_dir.glob("fold*"):
                if not is_valid_fold_dir(fold_dir, dataset):
                    continue

                fold = int(fold_dir.name.replace("fold", ""))
                for method_dir in fold_dir.iterdir():
                    if not method_dir.is_dir():
                        continue

                    method = method_dir.name
                    if method not in methods_filter:
                        skipped_methods.add(method) # 保存的ifc_v1/ifc_ft0等训练中间产物
                        continue

                    fidelity_path = method_dir / "fidelity.parquet"
                    final_path = method_dir / "final.parquet"

                    rows.append({
                        "Dataset": dataset,
                        "Seed": seed,
                        "Fold": fold,
                        "Method": method,
                        "fidelity_path": str(fidelity_path) if fidelity_path.exists() else None,
                        "final_path": str(final_path) if final_path.exists() else None,
                        "stage_depth": stage_depth,
                        "mix_ratio": mix_ratio
                    })
    if skipped_methods:
        pass
        #print(f"跳过的非Method子目录: {sorted(skipped_methods)}")
    return pd.DataFrame(rows)


def get_all_manifest(root: Path):

    manifests = []

    configs = [
        (root / "experiments" / "mix_37_235" / "stage3" / "artifacts", 3, "mix_37_235", OURS_METHODS | BASELINE_METHODS),
        (root / "experiments" / "mix_37_235" / "stage2" / "artifacts", 2, "mix_37_235",  STAGE2_OURS_METHODS),
        (root / "experiments" / "mix_55_532" / "stage3" / "artifacts", 3, "mix_55_532",  STAGE2_OURS_METHODS),
        (root / "experiments" / "mix_55_532" / "stage2" / "artifacts", 2, "mix_55_532",  STAGE2_OURS_METHODS),
    ]

    for artifacts_root, depth, ratio, methods in configs:
        df = build_manifest(artifacts_root, methods, depth, ratio)
        manifests.append(df)

    manifest_all = pd.concat(manifests, ignore_index=True)
    manifest_all = expand_method(manifest_all, stage_depth=None, mix_ratio=None)

    return manifest_all


def build_fold_data_manifest(artifacts_root):
    rows = []
    for dataset_dir in artifacts_root.iterdir():
        if not dataset_dir.is_dir():
            continue
        dataset = dataset_dir.name
        for seed_dir in dataset_dir.glob("seed*"):
            seed = int(seed_dir.name.replace("seed", ""))
            for fold_dir in seed_dir.glob("fold*"):
                if not is_valid_fold_dir(fold_dir, dataset):
                    continue
                fold = int(fold_dir.name.replace("fold", ""))
                fold_data_dir = fold_dir / "_fold_data"
                if not fold_data_dir.exists():
                    continue
                row = {"Dataset": dataset, "Seed": seed, "Fold": fold}
                for split in ["X_train", "X_valid", "X_test", "y_train", "y_valid", "y_test"]:
                    p = fold_data_dir / f"{split}.parquet"
                    row[f"{split}_path"] = str(p) if p.exists() else None
                rows.append(row)
    return pd.DataFrame(rows)
