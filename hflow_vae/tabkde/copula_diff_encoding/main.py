import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import argparse
import warnings

import os
from tqdm import tqdm
import json
import time
import pandas as pd
from joblib import dump


from tabkde.tools_1 import DataProcessor, EmpiricalTransformer
from tabkde.tools import fit_mix_gaussian_to_

def load_files(data_path, *file_names):
    """
    Load multiple CSV files into DataFrames.

    Parameters:
    - data_path (str): Base directory path where the files are stored.
    - file_names (str): Variable number of file names to be loaded.

    Returns:
    - dict: A dictionary where keys are file names and values are DataFrames. If loading fails, the value is None for that file.
    """
    dataframes = {}
    #print(f'loading data from {data_path}')

    for file_name in file_names:
        file_path = data_path + file_name
        try:
            dataframes[file_name[:-4]] = pd.read_csv(file_path) #:-4 is to remove .csv
        except Exception as e:
            print(f"Error loading {file_name}: {e}")
            dataframes[file_name[:-4]] = None

    return dataframes

def save_model(model, name, path):
    """
    Save a model under the given name at the specified path.

    Parameters:
        model: The object to be saved.
        name (str): The filename for the saved model (without extension).
        path (str): The directory where the model should be saved.

    Returns:
        str: The full path to the saved model file.
    """
    # Ensure the directory exists
    os.makedirs(path, exist_ok=True)

    # Construct the full file path
    full_path = os.path.join(path, f"{name}.joblib")

    # Save the model using joblib
    dump(model, full_path)

    return full_path

warnings.filterwarnings('ignore')


def main(args):
    dataname = args.dataname
    data_dir =  f'data/{dataname}/'
 
    decorrelation = args.decorrelation
    cat_encoding = args.cat_encoding_ordinal
    copula_encoder = args.copula_encoder
    #print(f'copula_encoder is {copula_encoder}')

    device =  args.device


    info_path = data_dir + 'info.json'

    with open(info_path, 'r') as f:
        info = json.load(f)

    curr_dir = os.path.dirname(os.path.abspath(__file__))
    ckpt_dir = f'{curr_dir}/ckpt/{dataname}' 
    #print(ckpt_dir)
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)


 
    list_of_files = ['train.csv']
    #print(data_dir)
    loaded_data = load_files(data_dir, *list_of_files)
    train_data = loaded_data.get("train")
    
    ordinal_encoder_model = DataProcessor()
    df_train_encoded, _ = ordinal_encoder_model.fit(train_data, decorrelation = decorrelation, json_path = info_path, cat_encoding = cat_encoding)
    save_model(ordinal_encoder_model, name = 'data_aware_model', path = ckpt_dir)
 
    if copula_encoder:
        copula_model = EmpiricalTransformer(df_train_encoded)
        df_train_encoded = copula_model.fit(method = 'average') 
        save_model(copula_model, name = 'copula_model', path = ckpt_dir)
     
    np.save(f'{ckpt_dir}/train_z.npy', df_train_encoded.to_numpy())

    #print(f"Successfully saved the embeddings on disk!: {ckpt_dir}/train_z.npy")  

        

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Copula Encoder')

    parser.add_argument('--dataname', type=str, default='adult_equal', help='Name of dataset.')
    parser.add_argument('--gpu', type=int, default=0, help='GPU index.')

    args = parser.parse_args()

    if args.gpu != -1 and torch.cuda.is_available():
        args.device = 'cuda:{}'.format(args.gpu)
    else:
        args.device = 'cpu'