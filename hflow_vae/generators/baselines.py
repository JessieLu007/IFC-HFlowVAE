from ctgan import CTGAN
from synthcity.plugins import Plugins
from imblearn.over_sampling import SMOTE, SMOTENC
from collections import Counter


def ctgan_generate(fold_data, n_samples_list):
    model = CTGAN()
    model.fit(fold_data.X_train_pos, discrete_columns=fold_data.cat_cols)
    return [model.sample(n) for n in n_samples_list]


def tvae_generate(fold_data, n_samples_list):
    model = Plugins().get("tvae", n_iter=1000)
    model.fit(fold_data.X_train_pos)
    return [model.generate(count=n).dataframe() for n in n_samples_list]


def tabddpm_generate(fold_data, n_samples_list):

    model = Plugins().get("ddpm")
    model.fit(fold_data.X_train_pos)
    return [model.generate(count=n).dataframe() for n in n_samples_list]


def smote_generate(fold_data, n_samples_list):

    X = fold_data.X_train
    y = fold_data.y_train

    counts = Counter(y)
    count_0 = counts[0]
    count_1 = counts[1]

    results = []
    
    for n_samples in n_samples_list:
        target_counts = {0: count_0, 1: (count_1 + n_samples)}
        if len(fold_data.cat_cols) > 0:
            model = SMOTENC(
                categorical_features=fold_data.cat_cols,
                sampling_strategy=target_counts,
                random_state=fold_data.seed
                )
        else:
            model = SMOTE(
                sampling_strategy=target_counts,
                random_state=fold_data.seed
            )
    
        X_resampled, y_resampled = model.fit_resample(X, y)
        X_syn = X_resampled[len(X):]
        X_syn = fold_data.processor.canonicalize(X_syn)

        results.append(X_syn)

    return results


from tabkde.process_dataset import process_data
from tabkde.copula_encoding.main import main as train_main
from tabkde.sample import main as sample_main
from types import SimpleNamespace
from pathlib import Path


def tabkde_generate(fold_data, n_samples_list):

    # 先对数据进行预处理，并保存预处理后的结果
    process_data(fold_data.dataset, fold_data.X_train_pos, fold_data.X_test_pos)

    current_dir = Path(__file__).resolve().parent
    save_path = current_dir / 'synthetic' / fold_data.dataset

    def _make_args(n_samples):
        return SimpleNamespace(
            dataname=fold_data.dataset,
            device="cpu",
            method="TabKDE",
            mode="train",
            decorrelation=False,
            cat_encoding_ordinal="ours",
            copula_encoder=1,
            latent_encoding="vae",
            rank_encoding=1,
            steps=50,
            num_samples=n_samples, 
            save_path=save_path # 这里虽然传入了保存路径，但是实现代码已经删除相关保存部分，所以这里只是个形式
        )

    train_main(_make_args(n_samples_list[0]))
    results = []

    for n in n_samples_list:
        X_syn = sample_main(_make_args(n))
        #X_syn = fold_data.processor.canonicalize(X_syn)
        results.append(X_syn)

    return results


from ForestDiffusion.diffusion_with_trees_class import ForestDiffusionModel
import pandas as pd

def forestDiffusion_generate(fold_data, n_samples_list):
    cat_indexes = [fold_data.X_train.columns.get_loc(col) for col in fold_data.cat_cols]
    model = ForestDiffusionModel(fold_data.X_train_pos.to_numpy(), cat_indexes=cat_indexes, n_t=10, duplicate_K=1)
    results = []
    for n in n_samples_list:
        X_syn = model.generate(batch_size=n)
        X_syn = pd.DataFrame(X_syn, columns=fold_data.X_train.columns)
        results.append(X_syn)
    
    return results