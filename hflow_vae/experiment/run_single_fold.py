# experiment/run_single_fold.py

from __future__ import annotations
import time
import pandas as pd

from generators.baselines import (
    smote_generate, ctgan_generate, tvae_generate, 
    tabddpm_generate, tabkde_generate, forestDiffusion_generate
    )

from experiment.ifc_pipeline import IFCConfig, run_ifc_pipeline, IFC_CELLS
from experiment.artifacts import save_baseline_result, save_ifc_result
from experiment.evaluate import evaluate_method, evaluate_raw
from experiment.artifacts import save_df, artifact_dir, save_fold_data


BASELINE_GENERATORS = {
    "smote": smote_generate,
    "ctgan": ctgan_generate,
    "tvae": tvae_generate,
    "forestdiffusion": forestDiffusion_generate,
    "tabddpm": tabddpm_generate,
    "tabkde": tabkde_generate,
}



def run_single_fold(
    fold_data, 
    n_samples_final: int | None = None,
    ifc_cells: list[IFCConfig] | None = None
    ) -> pd.DataFrame:
    n_pos = len(fold_data.X_train_pos)
    n_neg = len(fold_data.X_train) - n_pos
    ratio = n_neg / n_pos if n_pos > 0 else float("inf")
    n_final = n_samples_final if n_samples_final is not None else (
        (n_neg - n_pos) if ratio > 2 else n_pos
    )

    save_fold_data(fold_data)

    all_results = [evaluate_raw(fold_data, random_state=fold_data.seed)]
    for name, gen_fn in BASELINE_GENERATORS.items():
        t0 = time.perf_counter()
        X_syn, X_fidelity = gen_fn(fold_data, n_samples_list=[n_final, n_pos])
        X_syn = fold_data.processor.canonicalize(X_syn)
        assert list(X_syn.columns) == fold_data.processor.original_columns
        X_fidelity = fold_data.processor.canonicalize(X_fidelity)
        print(f"[timing] {name}: {time.perf_counter() - t0:.1f}s")

        save_baseline_result(X_syn, fold_data.dataset, fold_data.seed, fold_data.fold, name)
        save_df(X_fidelity, artifact_dir(fold_data.dataset, fold_data.seed, fold_data.fold, name), "fidelity")
        all_results.append(evaluate_method(fold_data, X_syn, method_name=name, random_state=fold_data.seed))

    cells = ifc_cells if ifc_cells is not None else IFC_CELLS 
    for config in cells:
        config.set_seed(fold_data.seed)
        result = run_ifc_pipeline(fold_data, config=config, n_samples_final=n_final)
        save_ifc_result(result, fold_data.dataset, fold_data.seed, fold_data.fold)
        all_results.append(
            evaluate_method(fold_data, result.X_final, method_name=config.tag, random_state=fold_data.seed)
        )

    return pd.concat(all_results, ignore_index=True)