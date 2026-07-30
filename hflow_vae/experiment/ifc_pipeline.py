"""
IFC-HFlowVAE 的 3x2 析因消融流水线:
    因子一 prior_type   ∈ {"none", "bootstrap", "smote"}
    因子二 use_finetune  ∈ {False, True}
6 个 config 对应论文消融表的 6 个 cell。
"""
# experiment/ifc_pipeline.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path
import pandas as pd


from .ifc_stages import (
    StageOutput, make_prior_pool, build_mixture, train_one_stage, cluster_split_latent
)
from hetero_vae.config import VAEConfig
from .ifc_stages import assign_clusters


# 为了避免循环引入，将artifact_dir在这里重复实现
current_dir = Path(__file__).resolve().parent.parent
ARTIFACT_ROOT = current_dir / "artifacts"


def artifact_dir(dataset: str, seed: int, fold: int, method_tag: str) -> Path:
    return ARTIFACT_ROOT / dataset / f"seed{seed}" / f"fold{fold}" / method_tag

@dataclass
class IFCConfig:
    prior_type: str = "smote"          # "none" | "bootstrap" | "smote"
    use_finetune: bool = True
    n_clusters: int = 5
    seed: int = 0

    # self-circulating 各阶段是否从上一阶段权重继续训练, False = 每阶段全新初始化（当前默认）
    warm_start_stages: bool = False

    # 基础VAE配置（所有阶段共用，除非finetune单独指定）
    vae_config: VAEConfig = field(default_factory=VAEConfig)
    finetune_vae_config: Optional[VAEConfig] = None

    def set_seed(self, seed: int):
        self.seed = seed
        self.vae_config.seed = seed
        if self.finetune_vae_config is not None:
            self.finetune_vae_config.seed = seed

    @property
    def tag(self) -> str:
        return f"{self.prior_type}_{'ft' if self.use_finetune else 'noft'}"


@dataclass
class IFCResult:
    config: IFCConfig
    prior_pool: Optional[pd.DataFrame]
    stage_outputs: dict[str, StageOutput]        # {"v0"} 或 {"v1","v2","v3"}
    cluster_outputs: Optional[dict[int, StageOutput]]
    X_final: pd.DataFrame # 平衡目标数量 n_final，给下游分类器用
    X_fidelity: pd.DataFrame # 固定N条，专给fidelity/t-SNE等评估用


def run_ifc_pipeline(fold_data, config: IFCConfig, n_samples_final: int) -> IFCResult:
    
    X_real_pos = fold_data.X_train_pos.reset_index(drop=True)
    valid_pos = fold_data.X_valid_pos.reset_index(drop=True)

    N = len(X_real_pos)

    stage_outputs: dict[str, StageOutput] = {}

    base_vae_config = config.vae_config
    ft_vae_config = config.finetune_vae_config or base_vae_config

    if config.prior_type == "none":
        v0 = train_one_stage(
            X_real_pos, valid_pos, N, base_vae_config,
            fold_data=fold_data, stage_name="v0",
            )
        stage_outputs["v0"] = v0
        
        prior_pool = None
        base_trainer = v0.trainer

    else:
        prior_pool = make_prior_pool(
            X_real_pos, config.prior_type, N, fold_data=fold_data
        )

        def _next_warm_start(prev: Optional[StageOutput]):
            return prev.trainer if (config.warm_start_stages and prev is not None) else None
        
        mix1 = build_mixture(X_real_pos, [(prior_pool, 1.0)], N, fold_data.get_seed("mix_v1"))
        v1 = train_one_stage(mix1, valid_pos, N, base_vae_config, 
                             fold_data=fold_data, stage_name="v1", warm_start=_next_warm_start(None),
                             save_dir=artifact_dir(fold_data.dataset, fold_data.seed, fold_data.seed, "ifc_v1"))
        
        stage_outputs["v1"] = v1
        
        mix2 = build_mixture(X_real_pos, [(prior_pool, 0.3), (v1.X_syn, 0.7)], N, fold_data.get_seed("mix_v2"))
        v2 = train_one_stage(mix2, valid_pos, N, base_vae_config,
                             fold_data=fold_data, stage_name="v2", warm_start=_next_warm_start(v1),
                             save_dir=artifact_dir(fold_data.dataset, fold_data.seed, fold_data.seed, "ifc_v2"))
        
        stage_outputs["v2"] = v2

        mix3 = build_mixture(X_real_pos, [(prior_pool, 0.2), (v1.X_syn, 0.3), (v2.X_syn, 0.5)], N, fold_data.get_seed("mix_v3"))
        v3 = train_one_stage(mix3, valid_pos, N, base_vae_config, 
                             fold_data=fold_data, stage_name="v3", warm_start=_next_warm_start(v2),
                             save_dir=artifact_dir(fold_data.dataset, fold_data.seed, fold_data.seed, "ifc_v3"))
        
        stage_outputs["v3"] = v3

        base_trainer = v3.trainer

    if not config.use_finetune:
        X_final = base_trainer.generate_samples(n_samples_final, 1.0, True, fold_data.get_seed("sample_final"))
        X_fidelity = base_trainer.generate_samples(N,1.0, True, fold_data.get_seed("sample_fidelity"))
        cluster_outputs = None

    else:
        clusters, km = cluster_split_latent(X_real_pos, base_trainer, config.n_clusters,
                                            fold_data, batch_size=config.vae_config.batch_size)
        # 如果数据太少，跳过微调
        if clusters is None:
            X_final = base_trainer.generate_samples(n_samples_final, 1.0, True, fold_data.get_seed("sample_final"))
            X_fidelity = base_trainer.generate_samples(N,1.0, True, fold_data.get_seed("sample_fidelity"))
            cluster_outputs = None
        else:
            clusters = [c for c in clusters if len(c) > 0]
            valid_clusters = assign_clusters(valid_pos, base_trainer, km, fold_data.num_cols, fold_data.cat_cols)

            cluster_outputs, parts_final, parts_fidelity = {}, [], []
            allocated_final, allocated_fidelity = 0, 0

            for i, cluster_df in enumerate(clusters):
                if len(cluster_df) == 0:
                    continue

                is_last = (i == len(clusters) - 1)
                k_final = (n_samples_final - allocated_final) if is_last else round(n_samples_final * len(cluster_df) / N)
                allocated_final += k_final

                k_fidelity = (N - allocated_fidelity) if is_last else round(N * len(cluster_df) / N)
                allocated_fidelity += k_fidelity

                out = train_one_stage(cluster_df, valid_clusters[i], k_final, ft_vae_config,
                                    fold_data=fold_data, stage_name=f"ft{i}", warm_start=base_trainer,
                                    save_dir=artifact_dir(fold_data.dataset, fold_data.seed, fold_data.seed, method_tag=f"ifc_ft{i}")
                                    )
                cluster_outputs[i] = out
                parts_final.append(out.X_syn)

                X_fid_cluster = out.trainer.generate_samples(k_fidelity, 1.0, True, seed=fold_data.get_seed(f"fidelity_ft{i}"))
                parts_fidelity.append(X_fid_cluster)

        X_final = pd.concat(parts_final, ignore_index=True)
        X_fidelity = pd.concat(parts_fidelity, ignore_index=True)

    return IFCResult(config, prior_pool, stage_outputs, cluster_outputs, X_final, X_fidelity)


def make_ifc_cell(
    prior_type: str, 
    use_finetune: bool,
    overrides: Optional[dict]=None,
    ft_overrides: Optional[dict]=None,
    ) -> IFCConfig:
    """
    overrides: overrides base_cfg for all common training, e.g.:
        {"layer2": 128, "kl_warmup_steps":50, "use_uncertainty_weighting":True}
    ft_overrides: overrides for finetuning
    """

    base_cfg = VAEConfig()
    if overrides: base_cfg.update_from_dict(overrides)

    if use_finetune:
        ft_deaults = dict(learning_rate=1e-4, patience1=20, epochs=500)
        if ft_overrides:
            ft_deaults.update(ft_overrides)
        ft_cfg = base_cfg.for_finetune(**ft_deaults)
    else:
        ft_cfg = None

    return IFCConfig(
        prior_type=prior_type, use_finetune=use_finetune,
        vae_config=base_cfg, finetune_vae_config=ft_cfg,
    )


def build_ifc_cells(
    overrides: Optional[dict] = None,
    ft_overrides: Optional[dict] = None,
) -> list[IFCConfig]:
    return [
        make_ifc_cell("none", False, overrides=overrides, ft_overrides=ft_overrides),
        make_ifc_cell("none", True, overrides=overrides, ft_overrides=ft_overrides),
        make_ifc_cell("bootstrap", False, overrides=overrides, ft_overrides=ft_overrides),
        make_ifc_cell("bootstrap", True, overrides=overrides, ft_overrides=ft_overrides),
        make_ifc_cell("smote", False, overrides=overrides, ft_overrides=ft_overrides),
        make_ifc_cell("smote", True, overrides=overrides, ft_overrides=ft_overrides),
    ]

IFC_CELLS: list[IFCConfig] = build_ifc_cells()