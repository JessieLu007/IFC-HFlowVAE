# hetero_vae/train.py


import os
import pickle
import copy
from typing import List, Optional, Tuple, Union
from tqdm import tqdm
import torch.nn.functional as F

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from hetero_vae.config import VAEConfig
from hetero_vae.model import FlowVAE
from data.data_processor import TabularDataProcessor


def build_cat_cardinalities(processor, cat_cols: List[str]) -> List[int]:
   
    return [processor.meta[col]["classes"] + 1 for col in cat_cols]


def encode_cat_with_unknown(arr: np.ndarray, cardinalities: List[int]) -> np.ndarray:
   
    arr = arr.astype(np.int64).copy()
    for i, card in enumerate(cardinalities):
        arr[arr[:, i] < 0, i] = card - 1
    return arr


class VAETrainer:
    def __init__(self, config: VAEConfig, device: Optional[str] = None):
        self.config = copy.deepcopy(config)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model: Optional[FlowVAE] = None
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.processor = None
        self.history: List[dict] = []
        self.best_state_dict = None
        self.best_val_total = float("inf")

    def warm_start_from(self, source: Union["VAETrainer", FlowVAE]):
        if self.model is None:
            self.build_model()
        source_model = source.model if isinstance(source, VAETrainer) else source
        self.model.load_state_dict(source_model.state_dict())
        self.model._skip_kl_warmup = True
        return self
    
    def encode_dataframe(self, df: pd.DataFrame, already_transformed: bool=False,
                         device: Optional[torch.device] = None):
        df_t = df if already_transformed else self.processor.transform(df)
        x_num, x_cat = self._df_to_tensors(df_t)
        return x_num.to(device), x_cat.to(device)

   
    def prepare_data(
        self,
        df_train: pd.DataFrame,
        df_val: pd.DataFrame,
        num_cols: List[str],
        cat_cols: List[str],
        processor: Optional[TabularDataProcessor] = None,
        processor_kwargs: Optional[dict] = None,
    ):
        
        if processor is not None:
            self.processor = processor
        else:
            processor_kwargs = processor_kwargs or {}
            self.processor = TabularDataProcessor(num_cols=num_cols, cat_cols=cat_cols, **processor_kwargs)
            self.processor.fit(df_train)

        df_train_t = self.processor.transform(df_train)
        df_val_t = self.processor.transform(df_val)

        self.config.num_cols = num_cols
        self.config.cat_cols = cat_cols
        self.config.cat_cardinalities = build_cat_cardinalities(self.processor, cat_cols) if cat_cols else []
        self.config.num_transform_method = self.processor.num_transform_method
        self.config.cat_encoding_method = self.processor.cat_encoding_method

        train_tensors = self._df_to_tensors(df_train_t)
        val_tensors = self._df_to_tensors(df_val_t)
        return train_tensors, val_tensors

    def _df_to_tensors(self, df_t: pd.DataFrame) -> Tuple[torch.Tensor, torch.Tensor]:
        n = len(df_t)
        if self.config.num_cols:
            x_num = df_t[self.config.num_cols].to_numpy(dtype=np.float32)
        else:
            x_num = np.zeros((n, 0), dtype=np.float32)

        if self.config.cat_cols:
            x_cat_raw = df_t[self.config.cat_cols].to_numpy()
            x_cat = encode_cat_with_unknown(x_cat_raw, self.config.cat_cardinalities)
        else:
            x_cat = np.zeros((n, 0), dtype=np.int64)

        return torch.tensor(x_num), torch.tensor(x_cat)

    def _make_loader(self, tensors, batch_size, shuffle, seed: Optional[int] = None):
        x_num, x_cat = tensors
        ds = TensorDataset(x_num, x_cat)
        gen = torch.Generator().manual_seed(seed) if (shuffle and seed is not None) else None
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=False, generator=gen)

    # model building
    def build_model(self) -> FlowVAE:
        torch.manual_seed(self.config.seed)
        self.model = FlowVAE(self.config).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.learning_rate)
        return self.model

    def _batch_step(self, x_num, x_cat, train: bool, beta: Optional[float] = None) -> dict:
        x_num = x_num.to(self.device, non_blocking=True)
        x_cat = x_cat.to(self.device, non_blocking=True)
        self.model.train(mode=train)

        with torch.set_grad_enabled(train):
            losses = self.model.compute_loss(
                x_num if self.config.num_cols else None,
                x_cat if self.config.cat_cols else None,
                beta=beta,
            )
            if train:
                self.optimizer.zero_grad(set_to_none=True)
                losses["total"].backward()
                self.optimizer.step()
        return losses


    def _run_epoch(self, loader, train: bool) -> dict:
        agg = {}
        n_batches = 0
        for x_num, x_cat in loader:
            losses = self._batch_step(x_num, x_cat, train=train)
            for k, v in losses.items():
                agg[k] = agg.get(k, 0.0) + v.detach()
            n_batches += 1
        return {k: (v / max(n_batches, 1)).item() for k, v in agg.items()}
    
    
    def fit(self, train_tensors, val_tensors, save_dir: Optional[str] = None, 
            strip_k: int = 5, pq_alpha:float=1.0, ema_alpha: float=0.3,
            loader_seed: Optional[int]=None):
        
        if self.model is None:
            self.build_model()
        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
        train_loader = self._make_loader(train_tensors, self.config.batch_size, shuffle=True, seed=loader_seed)
        val_loader = self._make_loader(val_tensors, self.config.batch_size, shuffle=False)

        self.best_val_total = float("inf") # best val_ema
        self.best_state_dict = None
        self.best_epoch = None
        self.history = []
        train_loss_window = []
        val_ema = None

        epoch_iter = tqdm(range(1, self.config.epochs + 1), desc="Training Progress", leave=True)
        for epoch in epoch_iter: #range(1, self.config.epochs+1): 
            train_metrics = self._run_epoch(train_loader, train=True)
            val_metrics = self._run_epoch(val_loader, train=False)

            train_loss_window.append(train_metrics["total"])
            if len(train_loss_window) > strip_k:
                train_loss_window.pop(0)

            val_ema = val_metrics["total"] if val_ema is None else (
                ema_alpha * val_metrics["total"] + (1 - ema_alpha) * val_ema
            )

            row = {"epoch": epoch, "val_ema": val_ema}
            row.update({f"train_{k}": v for k, v in train_metrics.items()})
            row.update({f"val_{k}": v for k, v in val_metrics.items()})
            self.history.append(row)

            '''if epoch % 1 == 0 or epoch == 1:
                print(f"[epoch {epoch:04d}] train_total={train_metrics['total']:.4f} "
                      f"val_total={val_metrics['total']:.4f} val_ema={val_ema:.4f} "
                      f"kl={val_metrics['kl']:.4f}")'''

            if val_ema < self.best_val_total - self.config.min_delta:
                self.best_val_total = val_ema
                self.best_epoch = epoch
                self.best_state_dict = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
                patience_counter = 0

            # ----- PQ_alhpa 早停 -----
            if len(train_loss_window) == strip_k:
                min_train = min(train_loss_window)
                progress = 1000 * (sum(train_loss_window) / (strip_k * min_train) -1) if min_train > 1e-6 else 0.0
                gen_loss = 100 * (val_ema / max(self.best_val_total, 1e-6) - 1)
                should_stop = (gen_loss > 0) if progress <= 1e-8 else (gen_loss / progress) > pq_alpha
                if should_stop:
                    print(f"早停触发(PQ准则, alpha={pq_alpha})：epoch={epoch}, gen_loss={gen_loss:.2f}, progress={progress:.2f}")
                    break

        if self.best_state_dict is not None:
            self.model.load_state_dict(self.best_state_dict)

        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            pd.DataFrame(self.history).to_csv(os.path.join(save_dir, "training_log.csv"), index=False)
            self.save(save_dir)
        return self.history



    # VAETrainer 采样 / 生成
    def generate_samples(self, num_samples: int = 100, temperature: float = 1.0, 
                         inverse_transform: bool = True, seed: Optional[int] = None,):
        if self.model is None:
            raise ValueError("模型尚未构建/加载。")
        self.model.eval()

        gen = torch.Generator(device=self.device).manual_seed(seed) if seed is not None else None
        num_out, cat_logits = self.model.sample(num_samples, device=self.device, 
                                                temperature=temperature, seed=seed)       

        data = {}
        if num_out is not None:
            ## ===============================================修改点NLL-----------------------------
            if False:
                std = torch.exp(0.5 * self.model.decoder.num_logvar).unsqueeze(0)
                num_out = num_out + torch.randn_like(num_out) * std * 0.1
            num_arr = num_out.detach().cpu().numpy()
            for i, col in enumerate(self.config.num_cols):
                data[col] = num_arr[:, i]

        for i, col in enumerate(self.config.cat_cols):
            ## ===============================================修改点multinomial-----------------------------
            if False:
                probs = F.softmax(cat_logits[i], dim=1)
                pred_idx = torch.multinomial(probs, num_samples=1).squeeze(1).detach().cpu().numpy()
            else:
                pred_idx = torch.argmax(cat_logits[i], dim=1).detach().cpu().numpy()
                
            card = self.config.cat_cardinalities[i]
            pred_idx = np.clip(pred_idx, 0, card - 2)
            data[col] = pred_idx

        df_generated = pd.DataFrame(data)

        if inverse_transform:
            if self.processor is None:
                raise ValueError("没有可用的 processor，无法做逆变换。")
            df_generated = self.processor.inverse_transform(df_generated)

        df_generated = self.processor.canonicalize(df_generated)

        return df_generated

    
    def save(self, save_dir: str):
        os.makedirs(save_dir, exist_ok=True)
        self.config.save(save_dir)
        torch.save(self.model.state_dict(), os.path.join(save_dir, "model_state.pt"))
        if self.processor is not None:
            with open(os.path.join(save_dir, "processor.pkl"), "wb") as f:
                pickle.dump(self.processor, f)

    @classmethod
    def load(cls, save_dir: str, device: Optional[str] = None) -> "VAETrainer":
        config = VAEConfig.from_file(os.path.join(save_dir, "config.json"))
        trainer = cls(config, device=device)
        trainer.build_model()
        state_dict = torch.load(os.path.join(save_dir, "model_state.pt"), map_location=trainer.device)
        trainer.model.load_state_dict(state_dict)

        processor_path = os.path.join(save_dir, "processor.pkl")
        if os.path.exists(processor_path):
            with open(processor_path, "rb") as f:
                trainer.processor = pickle.load(f)

        return trainer