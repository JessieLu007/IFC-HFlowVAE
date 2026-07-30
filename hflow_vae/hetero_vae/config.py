# hetero_vae/config.py

from dataclasses import dataclass, field
from typing import List, Optional
import copy
import json
import os


_ARCHITECTURE_FIELDS = {
    "num_cols", "cat_cols", "cat_cardinalities", "token_dim",
    "latent_dim", "layer1", "layer2", "two_layers",
    "flow_type", "n_flows",
}

@dataclass
class VAEConfig:
    num_cols: List[str] = field(default_factory=list)
    cat_cols: List[str] = field(default_factory=list)
    cat_cardinalities: List[int] = field(default_factory=list)

    token_dim: int = 16
    latent_dim: int = 16
    layer1: int = 128
    layer2: int = 128
    two_layers: bool = True

    numeric_tokenizer: str = "linear" # "linear" | "periodic"
    periodic_n_frequencies: int = 16
    periodic_sigma: float = 1.0

    flow_type: Optional[str] = "planar" # "planar" | None
    n_flows: int = 2
    kl_flow_correction: bool = True
    beta: float = 0.3 
    kl_warmup_steps: int = 50
    free_bits: float = 0.02

    learning_rate: float = 1e-3
    batch_size: int = 64
    epochs: int = 10000
    patience1: int = 50
    n_clusters: int = 5
    min_delta: float = 1e-4
    use_uncertainty_weighting: bool = True

    num_transform_method: str = "quantile"
    cat_encoding_method: str = "ordinal"

    def for_finetune(self, **training_overrides) -> "VAEConfig":
        bad_keys = set(training_overrides) & _ARCHITECTURE_FIELDS
        if bad_keys:
            raise ValueError(f"for_finetune() are not allowed to change the following keys: {bad_keys}")
        new_config = copy.deepcopy(self)
        new_config.update_from_dict(training_overrides)
        return new_config

    def update_from_dict(self, d: dict):
        for k, v in d.items():
            if not hasattr(self, k):
                raise KeyError(f"VAEConfig doesn't have key: {k}")
            setattr(self, k, v)

    def save(self, save_dir: str) -> str:
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, "config.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.__dict__, f, ensure_ascii=False, indent=2)
        return path

    def load(self, config_path: str) -> "VAEConfig":
        with open(config_path, "r", encoding="utf-8") as f:
            d = json.load(f)
        self.update_from_dict(d)
        return self

    @classmethod
    def from_file(cls, config_path: str) -> "VAEConfig":
        obj = cls()
        obj.load(config_path)
        return obj