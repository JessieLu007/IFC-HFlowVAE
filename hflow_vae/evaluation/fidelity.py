from scipy.stats import ks_2samp
import numpy as np
import pandas as pd


def marginal_fidelity(X_real: pd.DataFrame, X_syn: pd.DataFrame, num_cols, cat_cols) -> dict:
    scores = {}
    for col in num_cols:
        ks_stat = ks_2samp(X_real[col].dropna(), X_syn[col].dropna()).statistic
        scores[col] = 1.0 - ks_stat

    for col in cat_cols:
        p_real = X_real[col].value_counts(normalize=True)
        p_syn = X_syn[col].value_counts(normalize=True)
        all_cats = p_real.index.union(p_syn.index)
        p_real = p_real.reindex(all_cats, fill_value=0.0)
        p_syn = p_syn.reindex(all_cats, fill_value=0.0)
        tvd = 0.5 * np.abs(p_real - p_syn).sum()
        scores[col] = 1.0 - tvd

    scores["__num_avg__"] = np.mean([scores[c] for c in num_cols]) if num_cols else np.nan
    scores["__cat_avg__"] = np.mean([scores[c] for c in cat_cols]) if cat_cols else np.nan
    scores["__overall__"] = np.mean([v for k, v in scores.items() if not k.startswith("__")])
    return scores



from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict

def discriminative_fidelity(X_real_enc, X_syn_enc, random_state=0) -> float:
    """返回pMSE，越接近0越好；c是合并样本里"假"样本的比例"""
    X_all = pd.concat([X_real_enc, X_syn_enc], ignore_index=True)
    label = np.concatenate([np.zeros(len(X_real_enc)), np.ones(len(X_syn_enc))])
    c = len(X_syn_enc) / len(X_all)

    clf = LogisticRegression(max_iter=1000, random_state=random_state)
    probs = cross_val_predict(clf, X_all, label, cv=3, method="predict_proba")[:, 1]
    pmse = np.mean((probs - c) ** 2)
    return pmse


def correlation_fidelity(X_real, X_syn, num_cols):
    corr_real = X_real[num_cols].corr().values
    corr_syn = X_syn[num_cols].corr().values
    return np.linalg.norm(corr_real - corr_syn, ord='fro')   # Frobenius距离，越小越好