# hetero_vae/model.py
from typing import List, Optional
import torch.nn.functional as F
import torch
import math
import torch.nn as nn
from hetero_vae.config import VAEConfig
from hetero_vae.flows import NormalizingFlow


class NumericFeatureEmbedding(nn.Module):

    def __init__(self, num_features: int, token_dim: int):
        super().__init__()
        self.num_features = num_features
        self.token_dim = token_dim
        self.weight = nn.Parameter(torch.randn(num_features, token_dim) * 0.02)
        self.bias = nn.Parameter(torch.zeros(num_features, token_dim))

    def forward(self, x_num: torch.Tensor) -> torch.Tensor:
        # x_num: (B, num_features) -> (B, num_features, token_dim)
        return x_num.unsqueeze(-1) * self.weight + self.bias


class CategoricalFeatureEmbedding(nn.Module):

    def __init__(self, cardinalities: List[int], token_dim: int):
        super().__init__()
        self.embeddings = nn.ModuleList([nn.Embedding(card, token_dim) for card in cardinalities])

    def forward(self, x_cat: torch.Tensor) -> torch.Tensor:
        # x_cat: (B, n_cat) long -> (B, n_cat, token_dim)
        outs = [emb(x_cat[:, i]) for i, emb in enumerate(self.embeddings)]
        return torch.stack(outs, dim=1)


class PeriodicEmbedding(nn.Module):
    """暂时没有启用"""
    def __init__(self, num_features: int, token_dim: int, n_frequencies: int = 16, sigma: float = 1.0):
        super().__init__()
        self.coefficients = nn.Parameter(torch.randn(num_features, n_frequencies) * sigma)
        self.linear_weight = nn.Parameter(torch.randn(num_features, 2 * n_frequencies, token_dim) * 0.02)
        self.linear_bias = nn.Parameter(torch.zeros(num_features, token_dim))

    def forward(self, x_num: torch.Tensor) -> torch.Tensor:
        v = 2 * math.pi * self.coefficients.unsqueeze(0) * x_num.unsqueeze(-1)
        periodic = torch.cat([torch.sin(v), torch.cos(v)], dim=-1)          
        out = torch.einsum('bfk,fkt->bft', periodic, self.linear_weight) + self.linear_bias
        return F.relu(out)


class HeteroEncoder(nn.Module):
    def __init__(self, config: VAEConfig):
        super().__init__()
        self.n_num = len(config.num_cols)
        self.n_cat = len(config.cat_cols)
        assert self.n_num + self.n_cat > 0, "num_cols 和 cat_cols 不能同时为空"

        if config.numeric_tokenizer == "periodic":
            self.num_embed = PeriodicEmbedding(
                self.n_num, config.token_dim, config.periodic_n_frequencies, config.periodic_sigma
            ) if self.n_num > 0 else None
        else:
            self.num_embed = NumericFeatureEmbedding(self.n_num, config.token_dim) if self.n_num > 0 else None
        self.cat_embed = (
            CategoricalFeatureEmbedding(config.cat_cardinalities, config.token_dim) if self.n_cat > 0 else None
        )

        fused_dim = (self.n_num + self.n_cat) * config.token_dim
        layers = [nn.Linear(fused_dim, config.layer1), nn.SiLU()]
        if config.two_layers:
            layers += [nn.Linear(config.layer1, config.layer2), nn.SiLU()]
        self.backbone = nn.Sequential(*layers)

        out_dim = config.layer2 if config.two_layers else config.layer1
        self.fc_mu = nn.Linear(out_dim, config.latent_dim)
        self.fc_logvar = nn.Linear(out_dim, config.latent_dim)

    def forward(self, x_num: Optional[torch.Tensor], x_cat: Optional[torch.Tensor]):
        tokens = []
        if self.num_embed is not None:
            tokens.append(self.num_embed(x_num))
        if self.cat_embed is not None:
            tokens.append(self.cat_embed(x_cat))
        h = torch.cat(tokens, dim=1).flatten(start_dim=1)
        h = self.backbone(h)
        z_mean = self.fc_mu(h)
        z_logvar = self.fc_logvar(h)
        return z_mean, z_logvar


class HeteroDecoder(nn.Module):
    def __init__(self, config: VAEConfig):
        super().__init__()
        self.n_num = len(config.num_cols)
        self.n_cat = len(config.cat_cols)
        self.cat_cardinalities = config.cat_cardinalities

        self.num_logvar = nn.Parameter(torch.zeros(self.n_num)) if self.n_num > 0 else None

        layers = [nn.Linear(config.latent_dim, config.layer1), nn.SiLU()]
        if config.two_layers:
            layers += [nn.Linear(config.layer1, config.layer2), nn.SiLU()]
        self.backbone = nn.Sequential(*layers)
        out_dim = config.layer2 if config.two_layers else config.layer1

        self.num_head = nn.Linear(out_dim, self.n_num) if self.n_num > 0 else None
        self.cat_heads = (
            nn.ModuleList([nn.Linear(out_dim, card) for card in self.cat_cardinalities])
            if self.n_cat > 0
            else None
        )

    def forward(self, z: torch.Tensor):
        h = self.backbone(z)
        num_out = self.num_head(h) if self.num_head is not None else None
        cat_logits = [head(h) for head in self.cat_heads] if self.cat_heads is not None else []
        return num_out, cat_logits


class UncertaintyWeighting(nn.Module):
    """Kendal et al. 2018 同方差不确定性加权
    """
    def __init__(self, n_tasks: int):
        super().__init__()
        self.log_vars = nn.Parameter(torch.zeros(n_tasks))

    def forward(self, losses: List[torch.Tensor]) -> torch.Tensor:
        total = 0.0
        for i, loss in enumerate(losses):
            # clamp
            log_var = torch.clamp(self.log_vars[i], min=-5.0, max=5.0)
            precision = torch.exp(-log_var)
            total = total + precision * loss + log_var
        return total


class FlowVAE(nn.Module):
    def __init__(self, config: VAEConfig):
        super().__init__()
        self.config = config
        self.encoder = HeteroEncoder(config)
        self.decoder = HeteroDecoder(config)
        self.use_flow = config.flow_type is not None
        self.flow = NormalizingFlow(config.latent_dim, config.flow_type, config.n_flows) if self.use_flow else None
        
        self._train_steps = 0
        self._skip_kl_warmup = False

        self.use_uncertainty_weighting = config.use_uncertainty_weighting
        if self.use_uncertainty_weighting:
            self.loss_weighting = UncertaintyWeighting(n_tasks=2)  # 0: num, 1: cat

    def current_beta(self) -> float:
        
        if self.config.kl_warmup_steps <=0 or self._skip_kl_warmup:
            return self.config.beta
        progress = min(1.0, self._train_steps / self.config.kl_warmup_steps)
        return self.config.beta * progress
    
    def _kl_per_sample(self, z_mean, z_logvar, z0, zk, log_det) -> torch.Tensor:
        kl_per_dim = -0.5 * (1 + z_logvar - z_mean.pow(2) - z_logvar.exp())  # (B, D)
        if self.config.free_bits > 0:
            kl_per_dim = torch.clamp(kl_per_dim, min=self.config.free_bits)
        kl_closed = kl_per_dim.sum(dim=1)

        if not (self.use_flow and self.config.kl_flow_correction):
            return kl_closed

        correction = 0.5 * torch.sum(zk.pow(2) - z0.pow(2), dim=1)
        kl_total =kl_closed + correction - log_det
        if True:
            if self.config.free_bits > 0:
                kl_total = torch.clamp(kl_total, min=self.config.free_bits * self.config.latent_dim)
        return kl_total

    @staticmethod
    def reparameterize(z_mean: torch.Tensor, z_logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * z_logvar)
        eps = torch.randn_like(std)
        return z_mean + eps * std

    def forward(self, x_num: Optional[torch.Tensor], x_cat: Optional[torch.Tensor]):
        z_mean, z_logvar = self.encoder(x_num, x_cat)
        z0 = self.reparameterize(z_mean, z_logvar)

        if self.use_flow:
            zk, log_det = self.flow(z0)
        else:
            zk, log_det = z0, torch.zeros(z0.shape[0], device=z0.device, dtype=z0.dtype)

        num_out, cat_logits = self.decoder(zk)

        return {
            "z_mean": z_mean,
            "z_logvar": z_logvar,
            "z0": z0,
            "zk": zk,
            "log_det": log_det,
            "num_out": num_out,
            "cat_logits": cat_logits,
        }

    @torch.no_grad()
    def sample(self, num_samples: int, device: Optional[torch.device] = None, 
               temperature: float = 1.0, seed: Optional[int] = None):
        device = device or next(self.parameters()).device
        gen = torch.Generator(device=device).manual_seed(seed) if seed is not None else None
        z0 = torch.randn(num_samples, self.config.latent_dim, device=device, generator=gen) * temperature
        if self.use_flow:
            zk, _ = self.flow(z0)
        else:
            zk = z0
        num_out, cat_logits = self.decoder(zk)
        ## ===============================================修改点NLL-----------------------------
        if False:
            if num_out is not None:
                std = torch.exp(0.5 * self.decoder.num_logvar).unsqueeze(0)
                num_out = num_out + torch.randn_like(num_out) * std

        return num_out, cat_logits

    
    def compute_loss(self, x_num, x_cat, beta: Optional[float] = None) -> dict:
        
        if self.training:
            self._train_steps += 1

        out = self.forward(
            x_num if self.config.num_cols else None,
            x_cat if self.config.cat_cols else None,
        )

        batch_size = out["z_mean"].shape[0]
        device = out["z_mean"].device

        num_loss = torch.zeros(batch_size, device=device)
        ## ===============================================修改点NLL-----------------------------
        if out["num_out"] is not None:
            if False:
                logvar = self.decoder.num_logvar.clamp(min=-1, max=1)
                num_loss = 0.5 * (
                    (x_num - out["num_out"]).pow(2) / logvar.exp() + logvar
                ).sum(dim=1)
            num_loss = F.mse_loss(out["num_out"], x_num, reduction="none").sum(dim=1)

        cat_loss = torch.zeros(batch_size, device=device)
        for i, logits in enumerate(out["cat_logits"]):
            cat_loss = cat_loss + F.cross_entropy(logits, x_cat[:, i], reduction="none")

        if self.use_uncertainty_weighting and out["num_out"] is not None:
            recon_loss = self.loss_weighting([num_loss, cat_loss])
        else:
            recon_loss = num_loss + cat_loss

        kl = self._kl_per_sample(out["z_mean"], out["z_logvar"], out["z0"], out["zk"], out["log_det"])
        beta = self.current_beta()
        total = recon_loss.mean() + beta * kl.mean()

        if self.use_flow and not self.config.kl_flow_correction:
            total = total - out["log_det"].mean()

        return {
            "total": total, "recon": recon_loss.mean(),
            "num_recon": num_loss.mean(), "cat_recon": cat_loss.mean(),
            "kl": kl.mean(), "beta": torch.as_tensor(beta, device=recon_loss.device), "log_det": out["log_det"].mean(),
        }