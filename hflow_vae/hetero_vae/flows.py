# hetero_vae/flows.py

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class PlanarFlow(nn.Module):

    def __init__(self, latent_dim: int):
        super().__init__()
        self.latent_dim = latent_dim
        self.u = nn.Parameter(torch.empty(1, latent_dim))
        self.w = nn.Parameter(torch.empty(1, latent_dim))
        self.b = nn.Parameter(torch.zeros(1))
        nn.init.normal_(self.u, std=0.01)
        nn.init.normal_(self.w, std=0.01)

    def forward(self, z: torch.Tensor):
        
        w_u = torch.sum(self.w * self.u)
        u_hat = self.u + (-1.0 + F.softplus(-w_u)) * self.w / (torch.sum(self.w ** 2) + 1e-8)

        z_w = torch.sum(z * self.w, dim=1, keepdim=True)          
        h = torch.tanh(z_w + self.b)                              
        h_prime = 1.0 - h ** 2                                     

        z_transformed = z + u_hat * h                             

        uw = torch.sum(u_hat * self.w, dim=1)                      
        det_term = 1.0 + h_prime.squeeze(-1) * uw                   
        log_det = torch.log(torch.abs(det_term) + 1e-8)        

        return z_transformed, log_det


class NormalizingFlow(nn.Module):
    """当前 flow_type 仅支持 'planar' 或 None。"""

    def __init__(self, latent_dim: int, flow_type: Optional[str] = "planar", n_flows: int = 4):
        super().__init__()
        self.flow_type = flow_type
        if flow_type == "planar":
            self.flows = nn.ModuleList([PlanarFlow(latent_dim) for _ in range(n_flows)])
        elif flow_type is None:
            self.flows = nn.ModuleList()
        else:
            raise ValueError(f"当前仅支持 flow_type='planar' 或 None，got '{flow_type}'")

    def forward(self, z: torch.Tensor):
        if len(self.flows) == 0:
            return z, torch.zeros(z.shape[0], device=z.device, dtype=z.dtype)

        log_det_total = torch.zeros(z.shape[0], device=z.device, dtype=z.dtype)
        z_k = z
        for flow in self.flows:
            z_k, log_det = flow(z_k)
            log_det_total = log_det_total + log_det
        return z_k, log_det_total