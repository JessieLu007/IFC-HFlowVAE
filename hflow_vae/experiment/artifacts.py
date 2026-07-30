"""
目录结构:artifacts/{dataset}/seed{seed}/fold{fold}/{method_tag}/*.parquet
method_tag:
    - baseline: "ctgan" / "smote" / "tabddpm" / "tabkde" / "tvae" / "forestdiffusion" 
    - IFC cell: config.tag,例如 "smote_ft" / "none_noft" 

"""

from __future__ import annotations
from dataclasses import asdict
from pathlib import Path
import json
import joblib

import pandas as pd

from experiment.ifc_pipeline import IFCResult

current_dir = Path(__file__).resolve().parent.parent
ARTIFACT_ROOT = current_dir / "artifacts"


def artifact_dir(dataset: str, seed: int, fold: int, method_tag: str) -> Path:
    return ARTIFACT_ROOT / dataset / f"seed{seed}" / f"fold{fold}" / method_tag


def save_df(df: pd.DataFrame, d: Path, name: str) -> None:
    d.mkdir(parents=True, exist_ok=True)
    df.to_parquet(d / f"{name}.parquet")


def load_df(dataset: str, seed: int, fold: int, method_tag: str, name: str) -> pd.DataFrame:
    path = artifact_dir(dataset, seed, fold, method_tag) / f"{name}.parquet"
    return pd.read_parquet(path)


def save_fold_data(fold_data) -> None:
    d = artifact_dir(fold_data.dataset, fold_data.seed, fold_data.fold, "_fold_data")
    save_df(fold_data.X_train, d, "X_train")
    save_df(fold_data.y_train.to_frame("y"), d, "y_train")
    save_df(fold_data.X_valid, d, "X_valid")
    save_df(fold_data.y_valid.to_frame("y"), d, "y_valid")
    save_df(fold_data.X_test, d, "X_test")
    save_df(fold_data.y_test.to_frame("y"), d, "y_test")

    d.mkdir(parents=True, exist_ok=True)
    joblib.dump(fold_data.processor, d / "processor.joblib")


def load_processor(dataset: str, seed: int, fold: int):
    d = artifact_dir(dataset, seed, fold , "_fold_data")
    return joblib.load(d / "processor.joblib")


def save_baseline_result(
    X_syn: pd.DataFrame, dataset: str, seed: int, fold: int, method_name: str,
) -> None:
    d = artifact_dir(dataset, seed, fold, method_name)
    save_df(X_syn, d, "final")


def save_ifc_result(result: IFCResult, dataset: str, seed: int, fold: int) -> None:
    d = artifact_dir(dataset, seed, fold, result.config.tag)

    save_df(result.X_final, d, "final")
    save_df(result.X_fidelity, d, "fidelity")

    if result.prior_pool is not None:
        save_df(result.prior_pool, d, "prior_pool")

    for stage_name, out in result.stage_outputs.items():
        save_df(out.X_syn, d, stage_name)          # v0 / v1 / v2 / v3

    if result.cluster_outputs is not None:
        for i, out in result.cluster_outputs.items():
            save_df(out.X_syn, d, f"cluster{i}")

    config_dict = asdict(result.config)
    (d / "config.json").write_text(json.dumps(config_dict, default=str, indent=2))


def list_available_methods(dataset: str, seed: int, fold: int) -> list[str]:
    
    d = ARTIFACT_ROOT / dataset / f"seed{seed}" / f"fold{fold}"
    if not d.exists():
        return []
    return sorted(p.name for p in d.iterdir() if p.is_dir())