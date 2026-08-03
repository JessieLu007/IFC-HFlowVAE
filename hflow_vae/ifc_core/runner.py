"""

"""
# experiment/runner.py
from __future__ import annotations
from pathlib import Path
from typing import Optional

import pandas as pd

from ifc_core.ifc_pipeline import IFCConfig
import utils.common as common
from utils.common import set_seed, load_json
from ifc_core.prepare_folds import prepare_folds
from ifc_core.run_single_fold import run_single_fold

ROOT = Path.cwd().parent
DATA_LOC = ROOT / "data" / "original"
RESULTS_DIR = ROOT / "results" / "per_fold"

FOLD_THRESHOLD = 300
VAL_RATIO_OF_TRAIN = 0.15


def _result_file(dataset: str, seed: int, fold: int) -> Path:
    return RESULTS_DIR / f"{dataset}__seed{seed}__fold{fold}.parquet"


def run_all(
    datasets: list[str] = common.DATASETS,
    seeds: list[int] = common.SEEDS,
    resume: bool = True,
    ifc_overrides:Optional[dict] = None,
    ifc_ft_overrides:Optional[dict] = None,
    ifc_cells:Optional[list[IFCConfig]] | None = None,
) -> None:
    datasets_info = load_json(ROOT / "data" / "datasets_info.json")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    for dataset in datasets:
        df = pd.read_csv(DATA_LOC / f"{dataset}.csv")
        target = datasets_info[dataset]["target"]
        X = df.drop(columns=[target])
        y = df[target]

        for seed in seeds:
            set_seed(seed)

            for fold_data in prepare_folds(
                X=X, y=y, dataset_name=dataset, datasets_info=datasets_info,
                seed=seed, fold_threshold=FOLD_THRESHOLD, val_ratio=VAL_RATIO_OF_TRAIN,
            ):
                out_path = _result_file(dataset, seed, fold_data.fold)

                if resume and out_path.exists():
                    print(f"[skip] {dataset} seed={seed} fold={fold_data.fold}")
                    continue
                print(f"[run]  {dataset} seed={seed} fold={fold_data.fold}")
               
                fold_results = run_single_fold(fold_data, ifc_cells=ifc_cells)
                fold_results.to_parquet(out_path)
                

if __name__ == "__main__":
    run_all()