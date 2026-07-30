import warnings
import os
import sys
import tensorflow as tf
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.exceptions import ConvergenceWarning
from sklearn.exceptions import DataConversionWarning

def setup_jupyter_env():
    # 1. 忽略特定类型的警告
    warnings.filterwarnings('ignore', category=FutureWarning)
    warnings.filterwarnings('ignore', category=UserWarning)
    warnings.filterwarnings("ignore", category=DataConversionWarning)
    warnings.filterwarnings('ignore', category=ConvergenceWarning)

    root_dir = os.path.dirname(os.path.abspath(__file__))
    
    os.chdir(root_dir)
    
    # 将根目录添加到 sys.path 
    if root_dir not in sys.path:
        sys.path.append(root_dir)

    # 3. 字体设置
    try:
        from utils.set_fonts import apply_times_with_cn
        apply_times_with_cn(cn_font_path=r"C:\Windows\Fonts\msyh.ttc")
    except ImportError:
        print("⚠️ Warning: Could not import font setup.")

    print(f"🚀 环境初始化完成。")
    #print(f"📂 当前工作目录: {os.getcwd()}")
    #print(f"📦 已添加搜索路径: {root_dir}")

if __name__ != "__main__":
    setup_jupyter_env()