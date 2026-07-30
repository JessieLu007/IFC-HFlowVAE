from dataclasses import dataclass
import numpy as np
import zlib
import pandas as pd
from data.data_processor import TabularDataProcessor


@dataclass
class FoldData:
    """
    一个交叉验证Fold的全部实验数据
    """

    # -------- 实验信息 --------
    dataset: str # 数据集名称
    seed: int
    fold: int

    # -------- Processor --------
    processor: TabularDataProcessor

    # -------- 属性信息 --------
    num_cols: list[str]
    cat_cols: list[str]
    target_col: str
    has_missing: bool # 是否包含缺失值

    # -------- Data ---------
    X_train: pd.DataFrame
    y_train: pd.Series
    X_train_pos: pd.DataFrame
    y_train_pos: pd.Series

    X_valid: pd.DataFrame
    y_valid: pd.Series
    X_valid_pos: pd.DataFrame
    y_valid_pos: pd.Series

    X_test: pd.DataFrame
    y_test: pd.Series
    X_test_pos: pd.DataFrame
    y_test_pos: pd.Series

    def get_seed(self, role: str) -> int:
        
        role_hash = zlib.crc32(role.encode("utf-8"))
        ss = np.random.SeedSequence([self.seed, self.fold, role_hash])
        return int(ss.generate_state(1)[0])
        