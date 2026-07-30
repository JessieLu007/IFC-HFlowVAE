# core/data_processor.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, QuantileTransformer, StandardScaler, MinMaxScaler
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, SimpleImputer


class TabularDataProcessor:
    """表格数据双向处理器：MICE 插补 + 分布映射 + 数值结构约束恢复"""

    def __init__(self, num_cols, cat_cols, num_transform_method='quantile', cat_encoding_method='ordinal', random_state=42):
        assert isinstance(num_cols, list) and isinstance(cat_cols, list)
        assert num_transform_method in ('quantile', 'standard', 'minmax', 'none'), (
            f"num_transform_method 必须是 'quantile'/'standard'/'minmax'/'none'，got '{num_transform_method}'")
        assert cat_encoding_method in ('label', 'ordinal', 'none'), (
            f"cat_encoding_method 必须是 'label'/'ordinal'/'none'，got '{cat_encoding_method}'")

        self.num_cols = num_cols
        self.cat_cols = cat_cols
        self.num_transform_method = num_transform_method
        self.cat_encoding_method = cat_encoding_method
        self.random_state = random_state
        self.original_columns = None
        self.meta = {}

        self.num_imputer = IterativeImputer(
            max_iter=10,
            random_state=random_state,
            initial_strategy='median',
        )

        self.cat_imputer = SimpleImputer(
            strategy="most_frequent"
        )

    # =========================================================
    # 类别编码辅助
    # =========================================================
    def _cat_encode(self, enc, x_str: str):
        if isinstance(enc, OrdinalEncoder):
            return enc.transform([[x_str]])[0, 0]
        else:
            return enc.transform([x_str])[0]

    def _cat_inverse_transform(self, col: str, vals_1d: np.ndarray) -> np.ndarray:
        enc = self.meta[col]['encoder'] 
        if isinstance(enc, OrdinalEncoder):
            return enc.inverse_transform(vals_1d.reshape(-1, 1)).flatten()
        else:
            return enc.inverse_transform(vals_1d)
        
    def _restore_original_order(self, df:pd.DataFrame) -> pd.DataFrame:
        missing = set(self.original_columns) - set(df.columns)
        assert not missing, f"回复序列前发现缺列： {missing}"
        return df[self.original_columns]

    # =========================================================
    # 数值结构检测
    # =========================================================
    def _detect_quantize_step(self, unique_values, candidates=(0.5,), tol=1e-6):
        for q in candidates:
            remainders = np.abs(unique_values % q)
            if np.all(np.minimum(remainders, np.abs(remainders - q)) < tol):
                return q
        return None

    def _analyze_numeric_structure(self, values, is_int: bool, tolerance=1e-9):
        result = {'step': None, 'quantize_step': None}
        if is_int:
            return result
        unique = np.sort(np.unique(pd.Series(values).dropna().values))
        if len(unique) <= 1:
            return result
        result['quantize_step'] = self._detect_quantize_step(unique)
        return result

    def _infer_precision(self, series: pd.Series) -> int:
        sample = series.dropna().sample(min(1000, len(series)), random_state=self.random_state)
        decimals = [
            len(f"{v:.10f}".rstrip('0').split('.')[1]) if '.' in f"{v:.10f}".rstrip('0') else 0
            for v in sample
        ]
        return int(min(np.percentile(decimals, 95), 4))

    # =========================================================
    # fit
    # =========================================================
    def fit(self, df: pd.DataFrame):
        df_num = df[self.num_cols].copy()
        df_cat = df[self.cat_cols].copy()
        self.original_columns = df.columns.tolist()

        if len(self.num_cols) > 0:
            # 1. 保存数值属性的精度等统计信息
            for col in self.num_cols:
                valid_data = df[col].dropna()
                is_int = bool(np.allclose(valid_data, np.round(valid_data)))
                structure = self._analyze_numeric_structure(valid_data, is_int=is_int)
                quantize_step = structure['quantize_step']

                if quantize_step is not None:
                    q_str = f"{quantize_step:.10f}".rstrip('0')
                    prec = len(q_str.split('.')[1]) if '.' in q_str else 0
                else:
                    prec = self._infer_precision(valid_data)

                value_range = valid_data.max() - valid_data.min()
                self.meta[col] = {
                    'type': 'num',
                    'is_int': is_int,
                    'min': float(valid_data.min() - 0.05 * value_range),
                    'max': float(valid_data.max() + 0.05 * value_range),
                    'mean': float(valid_data.mean()),
                    'std': float(valid_data.std()) + 1e-8,
                    'quantize_step': quantize_step,
                    'prec': prec
                }

            # 3. 没有缺失值也要操作，否则num_imputer不fit，后续transform时会报错？
            #if df_num.isnull().any().any():
            df_num = pd.DataFrame(
                self.num_imputer.fit_transform(df_num),
                columns=self.num_cols,
                index=df.index,
            )
            
            # 4. 对数值列进行fit, 并保存训练后得到的Encoder到每列的meta信息中
            for col in self.num_cols:
                if self.num_transform_method == "quantile":
                    enc = QuantileTransformer(
                        n_quantiles=min(len(df_num),1000),
                        output_distribution="normal",
                        random_state=self.random_state,
                    )
                elif self.num_transform_method == "standard":
                    enc = StandardScaler()
                elif self.num_transform_method == "minmax":
                    enc = MinMaxScaler()
                else:
                    enc = None
                if enc is not None:
                    enc.fit(df_num[[col]])
                self.meta[col]['encoder'] = enc

        if len(self.cat_cols) > 0:
            # 1. 即便没有缺失值，也进行操作
            df_cat = pd.DataFrame(
                self.cat_imputer.fit_transform(df_cat),
                columns=self.cat_cols,
                index=df.index,
            )
            # 2. 对类别列进行编码 
            for col in self.cat_cols:
                valid_str = df_cat[[col]]

                if self.cat_encoding_method == 'label':
                    cat_encoder = LabelEncoder()
                    cat_encoder.fit(valid_str)
                    n_classes = len(cat_encoder.classes_)
                elif self.cat_encoding_method == 'ordinal':
                    cat_encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
                    cat_encoder.fit(valid_str)
                    n_classes = len(cat_encoder.categories_[0])
                else:
                    cat_encoder = None
                    n_classes = valid_str.nunique()

                self.meta[col] = {
                    'type': 'cat',
                    'encoder': cat_encoder,
                    'classes': n_classes,
                }

        return self

    # =========================================================
    # transform
    # =========================================================
    def transform(self, df:pd.DataFrame) -> pd.DataFrame:
        df_out = df.copy()

        # =========================
        # Numerical
        # =========================
        if len(self.num_cols) > 0:
            df_num = df_out[self.num_cols].copy()

            #1.  MICE补全
            df_num = pd.DataFrame(
                self.num_imputer.transform(df_num),
                columns=self.num_cols,
                index=df.index
            )

            # 2. 数值变换
            for col in self.num_cols:
                enc = self.meta[col]["encoder"]
                if enc is not None:
                    df_num[col] = enc.transform(df_num[[col]]).ravel()

            df_out[self.num_cols] = df_num

        # =========================
        # Categorical
        # =========================
        if len(self.cat_cols) > 0:
            df_cat = df_out[self.cat_cols].copy()

            # 1. 缺失值补全
            df_cat = pd.DataFrame(
                self.cat_imputer.transform(df_cat),
                columns=self.cat_cols,
                index=df.index,
            )

            # 2. 编码
            for col in self.cat_cols:
                enc = self.meta[col]["encoder"]
                if enc is None:
                    continue
                if isinstance(enc, LabelEncoder):
                    df_cat[col] = enc.transform(df_cat[col])
                elif isinstance(enc, OrdinalEncoder):
                    df_cat[col] = enc.transform(df_cat[[col]]).ravel()
            
            df_out[self.cat_cols] = df_cat

        return df_out
    
    # =================================
    # 提供缺失值补全以后的数据
    # =================================
    def transform_readable(self, df:pd.DataFrame) -> pd.DataFrame:
        assert hasattr(self, 'meta') and self.meta, "必须先调用 fit() 方法"
        df_out = df.copy()

        # Numerical
        if self.num_cols:
            df_num = pd.DataFrame(self.num_imputer.transform(df_out[self.num_cols]),
                                  columns=self.num_cols, index=df.index)
            df_out[self.num_cols] = df_num
        
        # Categorical
        if self.cat_cols:
            df_cat = pd.DataFrame(self.cat_imputer.transform(df_out[self.cat_cols]),
                                  columns=self.cat_cols, index=df.index)
            df_out[self.cat_cols] = df_cat

        # 将填充的缺失值还原到之前的精确度
        df_out = self.canonicalize(df_out)

        return df_out
    
    def canonicalize(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        对已经做过缺失值处理的（预填充）数据，
        仅对数值列执行 clipping/rounding/类型恢复。
        cat 列已经是原始标签，直接透传。
        """
        df_out = pd.DataFrame(index=df.index)

        for col in self.num_cols:
            meta = self.meta[col]
            vals = np.clip(df[col].values.astype(float), meta['min'], meta['max'])
            q = meta['quantize_step']
            if q is not None and q > 0:
                vals = np.round(vals / q) * q
                vals = np.round(vals, 6)
            prec = meta['prec']#self.precisions.get(col, 2)
            vals = np.round(vals, prec)
            df_out[col] = vals.astype(np.int64) if (prec == 0 or meta['is_int']) else vals

        for col in self.cat_cols:
            df_out[col] = df[col].values

        extra_cols = [c for c in df.columns if c not in self.num_cols + self.cat_cols]
        for col in extra_cols:
            df_out[col] = df[col].values

        return self._restore_original_order(df_out)

    # =========================================================

    def inverse_transform(self, df_or_tensor):
        ordered_cols = self.num_cols + self.cat_cols

        if not isinstance(df_or_tensor, pd.DataFrame):
            df = pd.DataFrame(df_or_tensor, columns=ordered_cols)
        else:
            df = df_or_tensor.copy()

        df_out = pd.DataFrame(index=df.index)

        # ====================
        # Numerical
        # ====================
        for col in self.num_cols:
            meta = self.meta[col]
            vals = df[[col]]

            if meta["encoder"] is not None:
                inversed = meta["encoder"].inverse_transform(vals).ravel()
            else:
                inversed = vals.ravel()
            # 限制到训练集附近
            inversed = np.clip(inversed, meta['min'], meta['max'])
            # 恢复量化步长
            q = meta['quantize_step']
            if q is not None and q > 0:
                inversed = np.round(inversed / q) * q
                inversed = np.round(inversed, 6)
            # 恢复原始精确度
            prec = meta["prec"]
            inversed = np.round(inversed, prec)

            if meta["is_int"]:
                df_out[col] = pd.Series(pd.array(inversed, dtype=pd.Int64Dtype()), index=df.index)
            else:
                df_out[col] = inversed

        # ====================
        # Categorical
        # ====================
        for col in self.cat_cols:
            enc = self.meta[col]["encoder"]
            vals = df[col].to_numpy()
            vals = np.nan_to_num(vals, nan=0)
            vals = np.round(vals).astype(int)
            vals = np.clip(vals, 0, self.meta[col]["classes"]-1)

            if enc is None:
                labels = vals
            elif isinstance(enc, LabelEncoder):
                labels = enc.inverse_transform(vals)
            elif isinstance(enc, OrdinalEncoder):
                labels = enc.inverse_transform(vals.reshape(-1, 1)).ravel()
            else:
                labels = vals

            # 尝试恢复数值类别
            try:
                df_out[col] = pd.to_numeric(labels)
            except Exception:
                df_out[col] = labels

        return self._restore_original_order(df_out)

    # =========================================================
    def get_data_info(self):
        return {
            'num_numerical': len(self.num_cols),
            'num_classes':   [self.meta[col]['classes'] for col in self.cat_cols],
        }