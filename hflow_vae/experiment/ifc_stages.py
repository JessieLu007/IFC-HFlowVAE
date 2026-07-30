
# experiment/ifc_stages.py
from __future__ import annotations
import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Optional,Union
import pandas as pd
import numpy as np
import torch
from sklearn.cluster import KMeans
from data.data_processor import TabularDataProcessor
from experiment.data_objects import FoldData
from generators.baselines import smote_generate as _smote_generate
from hetero_vae.train import VAETrainer
from hetero_vae.config import VAEConfig


@dataclass
class StageOutput:
    trainer: VAETrainer
    X_syn: pd.DataFrame


'''def sample_from_model(trainer, n_samples: int) -> pd.DataFrame:
    """基于已经训练好的 trainer 采样"""
    return trainer.generate_samples(num_samples=n_samples, inverse_transform=True)'''


# --------------------------------------------------
# 1. Prior pool : self-circulating cold start
# --------------------------------------------------

def make_prior_pool(
    X_real_pos: pd.DataFrame, 
    prior_type: str, 
    n_samples: int, 
    fold_data=None,
    ) -> Optional[pd.DataFrame]:
    """
    prior_type:
        none      -> no self-circulating, return None
        bootstrap -> X_real_pos samples n_samples data without putting back，testing self-circulating
        smote     -> use smote's generation samples to start model training
    """
    processor = fold_data.processor
    if prior_type == "none":
        return None
    if prior_type == "bootstrap":
        rng = np.random.RandomState(fold_data.get_seed("prior_bootstrap"))
        idx = rng.choice(len(X_real_pos), size=n_samples, replace=True)
        return X_real_pos.iloc[idx].reset_index(drop=True)
    if prior_type == "smote":
        assert fold_data is not None, "prior_type='smote' fold_data is needed"
        X_prior = _smote_generate(fold_data, [n_samples])[0].reset_index(drop=True)
        return processor.canonicalize(X_prior)
    raise ValueError (f"unknown prior_type: {prior_type}")


# -----------------------------------------------------------------------------
# 2. Mixture: sample from pools according to weights, and concat with real data
# ----------------------------------------------------------------------------- 
def build_mixture(
    X_real_pos: pd.DataFrame, 
    weighted_pools: list[tuple[pd.DataFrame, float]],
    n_target: int, 
    seed: int
    ) -> pd.DataFrame:
    """
    weighted_pools: [(df, weight), ...], weight must sums to 1.0, 
    so the number of generated data equals positive training data.
    """
    rng = np.random.RandomState(seed)
    parts = [X_real_pos]

    total_weight = sum(w for _, w in weighted_pools)
    assert abs(total_weight - 1.0) < 1e-6, f"weights must sum to 1.0, got {total_weight}"

    allocated = 0
    n_pools = len(weighted_pools)
    for i, (df, weight) in enumerate(weighted_pools):
        if i == n_pools - 1:
            k = n_target - allocated
        else:
            k = int(round(n_target * weight))
            allocated += k

        replace = k > len(df)
        idx = rng.choice(len(df), size=k, replace=replace)
        parts.append(df.iloc[idx].reset_index(drop=True))

    mixture = pd.concat(parts, ignore_index=True)
    return mixture.sample(frac=1.0, random_state=seed).reset_index(drop=True)


# ----------------------------------------------------------------------
# 3. train and sample using this projects' generative model, FlowVAE
# ----------------------------------------------------------------------

MIN_CLUSTER_VALID = 5

def train_one_stage( 
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    n_samples: int,
    config: VAEConfig,
    fold_data: FoldData,
    stage_name: str,
    warm_start: Optional[Union[VAETrainer, VAETrainer]] = None,
    save_dir: Path = None,
    fallback_epochs: Optional[int]=None,
)-> StageOutput:
    
    num_cols  = fold_data.num_cols
    cat_cols  = fold_data.cat_cols
    processor = fold_data.processor
    
    trainer = VAETrainer(config)
    trainer.config.seed = fold_data.get_seed(f"{stage_name}_model_init")

    train_tensors, val_tensors = trainer.prepare_data(
        train_df, valid_df, num_cols, cat_cols, processor=processor, processor_kwargs=None)
    
    trainer.build_model()
    if warm_start is not None:
        trainer.warm_start_from(warm_start)

    if len(valid_df) < MIN_CLUSTER_VALID and fallback_epochs is not None:
        fixed_config = copy.deepcopy(trainer.config)
        fixed_config.epochs = fallback_epochs
        trainer.config = fixed_config
        trainer.fit(train_tensors=train_tensors, val_tensors=val_tensors, save_dir=save_dir, 
                     pq_alpha=float("inf"), 
                    loader_seed=fold_data.get_seed(f"{stage_name}_less_loss_fit"))
    else:
        trainer.fit(train_tensors=train_tensors, val_tensors=val_tensors, save_dir=save_dir,
                    loader_seed=fold_data.get_seed(f"{stage_name}_loader"))

    X_syn = trainer.generate_samples(num_samples=n_samples, inverse_transform=True, 
                                     seed=fold_data.get_seed(f"{stage_name}_sample"))
    
    assert list(X_syn.columns) == processor.original_columns, (
        f"[{trainer.config.num_cols+trainer.config.cat_cols}] "
        "生成数据列序与processor.original_columns不一致，"
        "说明generate_samples/canonicalize的列序契约被破坏"
    )

    return StageOutput(trainer=trainer, X_syn=X_syn)


# ---------------------------------------------------------------------------
# 4. Finetune clustering
# ---------------------------------------------------------------------------

from sklearn.mixture import GaussianMixture

def cluster_split_latent(
    X_real_pos: pd.DataFrame,
    trainer: VAETrainer,
    n_clusters: int,
    fold_data: FoldData,
    batch_size: int = 64,
) -> tuple[Optional[list[pd.DataFrame]], Optional[GaussianMixture]]:
    """
    在 VAE latent space 中对 X_real_pos 聚类, 使用trainer中的Processor和model
    """
    model = trainer.model
    config = trainer.config
    device = next(model.parameters()).device

    model.eval()
    x_num_t, x_cat_t = trainer.encode_dataframe(X_real_pos)
    n = len(X_real_pos)

    latents = []
    with torch.no_grad():
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            x_num_b = x_num_t[start:end].to(device) if config.num_cols else None
            x_cat_b = x_cat_t[start:end].to(device) if config.cat_cols else None
            z_mean, _ = model.encoder(
                x_num_b if config.num_cols else None,
                x_cat_b if config.cat_cols else None
            )
            latents.append(z_mean.detach().cpu().numpy())

    Z = np.concatenate(latents, axis=0)
    MIN_SAMPLES_PER_CLUSTER = 4
    n_clusters_eff = max(1, min(n_clusters, len(X_real_pos) // MIN_SAMPLES_PER_CLUSTER))

    if n_clusters_eff <= 1:
        return None, None
    else:
        gmm = GaussianMixture(
            n_components=n_clusters_eff,
            covariance_type="diag",          # 见下方说明，不建议用默认的'full'
            random_state=fold_data.get_seed("cluster_gmm"),
            n_init=10,
            reg_covar=1e-4,
        )
        labels = gmm.fit_predict(Z)
        clusters = [X_real_pos[labels == k].reset_index(drop=True) for k in range(n_clusters_eff)]

        return clusters, gmm


def assign_clusters(df, trainer, gmm, num_cols, cat_cols, batch_size=64):
    """用已经fit好的gmm, 把valid_pos之类的新数据映射到相同的簇标签"""
    model = trainer.model
    device = next(model.parameters()).device
    model.eval()
    x_num_t, x_cat_t = trainer.encode_dataframe(df)
    n = len(df)
    latents = []
    with torch.no_grad():
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            x_num_b = x_num_t[start:end].to(device) if num_cols else None
            x_cat_b = x_cat_t[start:end].to(device) if cat_cols else None
            z_mean, _ = model.encoder(x_num_b, x_cat_b)
            latents.append(z_mean.detach().cpu().numpy())
    Z = np.concatenate(latents, axis=0)
    labels = gmm.predict(Z)

    cluster_dfs = [df[labels == k].reset_index(drop=True) for k in range(gmm.n_components)]

    non_empty_sizes = [len(cdf) for cdf in cluster_dfs if len(cdf) > 0]
    fill_size = non_empty_sizes[0] if non_empty_sizes else max(1, n // gmm.n_components)

    for i, cdf in enumerate(cluster_dfs):
        if len(cdf) == 0:
            filled = df.sample(n=fill_size, replace=True, random_state=None)
            cluster_dfs[i] = filled.reset_index(drop=True)

    return cluster_dfs