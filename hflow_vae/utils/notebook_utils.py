import matplotlib.pyplot as plt
import numpy as np
from models.config import VAEConfig
from models.trainer import VAETrainer

def setup_vae_training(data, config_params=None, override_params=None):
    """在notebook中设置VAE训练 - 支持参数覆盖"""
    
    config = VAEConfig()
    
   
    if config_params:
        for key, value in config_params.items():
            setattr(config, key, value)
    config.update_from_data(data)
    
   
    if override_params:
        config.update_from_dict(override_params)
    
    trainer = VAETrainer(config)
    
    model = trainer.build_models(data.shape[1])
    
    return trainer, model, config



def plot_training_history(history):
   
    metrics = list(history.history.keys())
    n_metrics = len([m for m in metrics if not m.startswith('val_')])
    
    fig, axes = plt.subplots(1, n_metrics, figsize=(5*n_metrics, 5))
    if n_metrics == 1:
        axes = [axes]
    
    metric_idx = 0
    for metric in metrics:
        if not metric.startswith('val_'):
            axes[metric_idx].plot(history.history[metric], label=f'训练 {metric}')
            val_metric = f'val_{metric}'
            if val_metric in history.history:
                axes[metric_idx].plot(history.history[val_metric], label=f'验证 {metric}')
            axes[metric_idx].set_title(metric)
            axes[metric_idx].legend()
            axes[metric_idx].grid(True)
            metric_idx += 1
    
    plt.tight_layout()
    plt.show()

'''def compare_flow_types():
    
    flow_info = {
        "real_nvp": {
            "description": "RealNVP - 非体积保持流，使用耦合层",
            "advantages": ["精确的似然计算", "高效的采样", "稳定的训练"],
            "disadvantages": ["需要维度为偶数", "可能表达能力有限"],
            "best_for": ["图像数据", "需要精确密度估计的任务"]
        },
        "iaf": {
            "description": "逆自回归流 - 强大的自回归流",
            "advantages": ["高表达能力", "适合顺序数据"],
            "disadvantages": ["实现复杂", "采样较慢"],
            "best_for": ["顺序数据", "音频/时间序列", "需要高表达能力的任务"]
        },
        "planar": {
            "description": "Planar Flow - 简单但有效的流",
            "advantages": ["实现简单", "计算高效", "适合简单分布"],
            "disadvantages": ["表达能力有限", "可能训练不稳定"],
            "best_for": ["简单分布", "计算资源有限", "初步实验"]
        }
    }
    
    for flow_type, info in flow_info.items():
        print(f"\n=== {flow_type.upper()} ===")
        print(f"描述: {info['description']}")
        print(f"优势: {', '.join(info['advantages'])}")
        print(f"劣势: {', '.join(info['disadvantages'])}")
        print(f"适用场景: {', '.join(info['best_for'])}")
    
    return flow_info'''

def print_model_info(model):
    print("=== 编码器 ===")
    model.encoder.summary()
    print("\n=== 解码器 ===")
    model.decoder.summary()
    print(f"\n=== 配置信息 ===")
    model.config.print_config()

def get_available_params():
    config = VAEConfig()
    params = list(config.__dict__.keys())
    print("可配置的参数列表:")
    for param in params:
        print(f"  - {param}")
    return params






