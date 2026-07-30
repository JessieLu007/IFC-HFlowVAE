import argparse
import os
import pandas as pd
import numpy as np
import torch
from numpy.linalg import solve, svd, norm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
from scipy.stats import gaussian_kde
from collections import Counter

from scipy.stats import entropy

import time
from tqdm import tqdm
# import hickle
from sklearn.neighbors import KernelDensity
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

import random
from sklearn.preprocessing import StandardScaler

from scipy.stats import gaussian_kde
from collections import Counter
from scipy.stats import entropy




# from sdv.single_table import GaussianCopulaSynthesizer
# from sdv.metadata import SingleTableMetadata
# from sdv.metadata import Metadata
# from sdv.evaluation.single_table import get_column_plot
# from rfm import LaplaceRFM


import plotly.graph_objects as go

from sklearn.mixture import GaussianMixture

def convert_one_hot(y, c):
    o = np.zeros((y.size, c))
    o[np.arange(y.size), y] = 1
    return o

def print_red_bold(text):
    html_text = f'<span style="color:red; font-weight:bold;">{text}</span>'
    display(HTML(html_text))
    
def cprint(text, color = None, bold=False):
    """
    Print text in the specified color and optionally in bold using ANSI escape codes.

    Arguments:
    text (str): The text to print.
    color (str): The color name ('red', 'green', 'yellow', 'blue', 'purple', 'cyan', 'white').
    bold (bool): Whether to print the text in bold (default: False).
    """
    
    if (color is None) and not bold:
        print(text)
    else:
        colors = {
            'red': '\033[91m',
            'green': '\033[92m',
            'yellow': '\033[93m',
            'blue': '\033[94m',
            'purple': '\033[95m',
            'cyan': '\033[96m',
            'white': '\033[97m'
        }

        bold_code = '\033[1m' if bold else ''
        reset = '\033[0m'
        if color is None: 
            colored_text = f"{bold_code}{text}{reset}"
            print(colored_text)
        elif color in colors:
            colored_text = f"{bold_code}{colors[color]}{text}{reset}"
            print(colored_text)
        else:
            cprint(f"Invalid color '{color}'. Available colors: {', '.join(colors.keys())}, lets use red color instead", color = 'yellow', bold = True)
            cprint(text, color = 'red', bold = True)


def load_data(dataset, datadir):
    if dataset == '.DS_Store': 
        #print(".DS_Store is not valid")
        return 
    if not os.path.isdir(datadir + "/" + dataset):
        #print("Warning: there is no diractory:" + datadir + "/" + dataset)
        return
    if not os.path.isfile(datadir + "/" + dataset + "/" + dataset + ".txt"):
        #print("Warning: there is no file:"+ datadir + "/" + dataset + "/" + dataset + ".txt")
        return
        
    dic = {}
    for k, v in map(lambda x : x.split(), open(datadir + "/" + dataset + "/" + dataset + ".txt", "r").readlines()):
        dic[k] = v
    c = int(dic["n_clases="])
    d = int(dic["n_entradas="])
    n_train = int(dic["n_patrons_entrena="])
    n_val = int(dic["n_patrons_valida="])
    n_train_val = int(dic["n_patrons1="])
    n_test = 0
    if "n_patrons2=" in dic:
        n_test = int(dic["n_patrons2="])
    n_tot = n_train_val + n_test
    #print ("\t Total number of data point:", n_tot, "number of training points:", n_train,
           #"number of validation points:", n_val, "number of test points:", n_test, 
           #"\t n_features:", d, "\tn_clases:", c, "\n\n\n")    
    details = [n_train, n_val, n_test, n_tot, d, c]
    f = open(datadir +'/' + dataset + "/" + dic["fich1="], "r").readlines()[1:]
    X = np.asarray(list(map(lambda x: list(map(float, x.split()[1:-1])), f)))
    y = np.asarray(list(map(lambda x: int(x.split()[-1]), f)))
    
    
    fold = list(map(lambda x: list(map(int, x.split())),
                    open(datadir + "/" + dataset + "/" + "conxuntos.dat", "r").readlines()))
    train_fold, val_fold = fold[0], fold[1]
    #print(f"training and validation sizes: {len(train_fold), len(val_fold)}")
    
    fold_ = list(map(lambda x: list(map(int, x.split())),
                    open(datadir + "/" + dataset + "/" + "conxuntos_kfold.dat", "r").readlines()))
    
    train_fold_list = []
    test_fold_list = []
    for repeat in range(4):
        train_fold_, test_fold_ = fold_[repeat * 2], fold_[repeat * 2 + 1]        
        train_fold_list.append(train_fold_)
        test_fold_list.append(test_fold_)
    return X, y, train_fold, val_fold, train_fold_list, test_fold_list, details



def augment_dataset(data, num_features, cat_features, noise_std=0.1, num_purtub = False):
    """
    Augments the dataset by adding noise to numerical features and randomly changing categorical features.
    
    Parameters:
    - data : The input dataset (pd.DataFrame).
    - num_features: List of numerical feature names.
    - cat_features: List of categorical feature names.
    - noise_std (float): Standard deviation for normal noise added to numerical features.
    
    Returns:
    - pd.DataFrame: A new dataset with original and augmented data points, with labels.
    """

    columns_list = data.columns.tolist()
    drop_columns = [feature_name for feature_name in cat_features if len(data[feature_name].unique().tolist()) <= 1]
    remained_columns = [col for col in columns_list if col not in drop_columns]
    remained_cat_features = [col for col in cat_features if col not in drop_columns]
    if num_purtub:
        index_map = {remained_columns[i]:i+1 for i in range(len(remained_columns))}
    else: 
        index_map = {remained_cat_features[i]:i+1 for i in range(len(remained_cat_features))}
        


    
    # print(index_map)
    categories = {feature_name:data[remained_columns][feature_name].unique().tolist() for feature_name in remained_cat_features}
    # print(categories)

    
    augmented_data = []
    
    # Step 1: Add original data points with label 0
    for _, row in data[remained_columns].iterrows():
        original_point = row.copy()
        original_point['label'] = 0
        augmented_data.append(original_point)
    
    # Step 2: Generate augmented points
    for _, row in data[remained_columns].iterrows():
        # original_point = row.copy()
        
        if num_purtub:
            for feature_name in num_features:
                new_point = row.copy()
                label = index_map[feature_name]
                original_value = new_point[feature_name]
                new_value = original_value + np.random.normal(0, noise_std)
                new_point[feature_name] = new_value
                new_point['label'] = label
                augmented_data.append(new_point)
                
        for feature_name in remained_cat_features:
            new_point = row.copy()
            label = index_map[feature_name]
            original_value = new_point[feature_name]
            C = categories[feature_name].copy()
            C.remove(original_value)
            new_value = random.choice(C)
            new_point[feature_name] = new_value
            new_point['label'] = label
            augmented_data.append(new_point)           
            
    # Convert augmented data to DataFrame
    augmented_df = pd.DataFrame(augmented_data)
    return augmented_df


def preprocess_data(df, normalize=True):
    
    df_copy = df.copy()
    # Encode categorical columns to numerical (ordinal encoding)
    for col in df.select_dtypes(include=['object']).columns:
        df_copy[col] = df_copy[col].astype('category').cat.codes + 1  # Start encoding from 1 instead of 0
    df_copy = df_copy.astype(float)
    if normalize:
        scaler = StandardScaler()
        # numerical_cols = df.select_dtypes(include=['float64', 'int']).columns
        df_copy[:] = scaler.fit_transform(df_copy)
        return df_copy, scaler.mean_, scaler.var_
    return df_copy, None, None


import numpy as np

def sample_l2_laplace(M, mu, T, n_samples=1):
    """
    Samples from a multivariate Laplace distribution with l2-norm scaling,
    proportional to f(X) = exp(||XA - mu A|| / T),
    where M = AA^T and A is the Cholesky decomposition of M.
    
    Parameters:
        M: d x d positive semi-definite matrix.
        mu: 1 x d mean vector.
        T (float): Scaling parameter.
        n_samples (int): Number of samples to draw. Defulat = 1
    
    Returns:
        np.ndarray: n_samples x d array of samples.
    """
    d = M.shape[0]
    
    # Cholesky decomposition of M
    A = np.linalg.cholesky(M)
    
    # Sample direction from a standard Gaussian for each sample
    gaussian_samples = np.random.normal(size=(n_samples, d))
    gaussian_samples /= np.linalg.norm(gaussian_samples, axis=1, keepdims=True)
    
    # Sample the scaling factor from an exponential distribution
    radial_component = np.random.exponential(scale=T, size=(n_samples, 1))
    
    # Combine radial component with direction
    laplace_samples = radial_component * gaussian_samples
    
    # Apply the linear transformation with matrix A and add the mean
    # A_inv = np.linalg.inv(A.T)
    transformed_samples = laplace_samples @ A + mu
    
    return transformed_samples


def cholesky_decomposition(M, threshold = 0, rank_report = False):
    """
    Perform Cholesky decomposition of a positive semi-definite matrix M.

    Parameters:
    M (numpy.ndarray): Input positive semi-definite matrix of shape (n, n).
    
    threshold: 
    Instead of M, we use threshold * I + M to prevent numerical error (defult = 1e-10)

    Returns:
    A (numpy.ndarray): Lower triangular matrix A such that M = AA^T.
    """
    # Check if M is a square matrix
    if M.shape[0] != M.shape[1]:
        raise ValueError("Input matrix must be square.")

    # Check if M is symmetric (approximately, due to numerical precision)
    if not np.allclose(M, M.T):
        raise ValueError("Input matrix must be symmetric.")

    # Perform Cholesky decomposition
    try:
        A = np.linalg.cholesky(threshold*np.eye(M.shape[0]) + M)
        lambda_min, lambda_max, eigens = eigen_report(M, threshold)
        #cprint(f'min eigenvalie of M: {lambda_min}', color = 'blue', bold = True)
        #cprint(f'max eigenvalie of M: {lambda_max}', color = 'blue', bold = True)
        r = erank(eigens)
        #cprint(f'1. effective rank of M: {r}', color = 'blue', bold = True)
        er = erank(eigens, Type = 'entropy')
        #cprint(f'2. Entropy based effective rank of M: {er}', color = 'blue', bold = True)
        return A, threshold, r, er, lambda_min, lambda_max
    except np.linalg.LinAlgError:
        text = f"Warning: Matrix M is not positive semi-definite."
        #cprint(text, color= 'blue', bold = True)
        lamba_min = np.min(np.real(np.linalg.eigvals(M)))
        #cprint(f'Least eigenvalue of M: {lamba_min}', color = 'cyan', bold = True)
        #cprint(f'M + (-lambda_min + 1e-5)*I is used insted of M')
        return cholesky_decomposition(M, threshold = -lamba_min + 1e-5, rank_report = rank_report) 
        # raise ValueError("Matrix M is not positive semi-definite.")

def erank(eigens, Type = None, Lambda = 1):
    if Type == 'entropy':
        P = eigens/eigens.sum()
        return np.exp2(np.sum(-1*P * np.log2(P)))
    if Type == None:
        return np.sum(eigens/ (Lambda+eigens))


def eigen_report(matrix, threshold):
    eigens = np.real(np.linalg.eigvals(threshold*np.eye(matrix.shape[0]) + matrix))
    lambda_min = np.min(eigens)
    lambda_max = np.max(eigens)
    return lambda_min, lambda_max, eigens


def compute_min_distances_cpu(A, B, k=1):
    # Build a KDTree for dataset B
    # print(A.shape, B.shape)
    tree = KDTree(B)
    
    # Query the tree with each point in A to find the minimum distance to B
    distances, _ = tree.query(A, k = k)
    
    return distances



def compute_min_distances(A, B, k=1, device=None):
    """
    Computes the minimum distances from each point in A to B using GPU if available.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device)

    if device.type == 'cpu':
        return compute_min_distances_cpu(A, B, k)

    # Ensure A and B are 2D before converting to torch
    A = np.asarray(A)
    B = np.asarray(B)

    if A.ndim == 1:
        A = A.reshape(1, -1)
    if B.ndim == 1:
        B = B.reshape(1, -1)

    A = torch.tensor(A, dtype=torch.float32, device=device)
    B = torch.tensor(B, dtype=torch.float32, device=device)

    dists = torch.cdist(A, B, p=2)
    topk_dists, _ = torch.topk(dists, k, largest=False, dim=1)

    return topk_dists.cpu().numpy()


    


def compare_two_dist(array1, array2 = None, label_1 = '', label_2 = '', 
                     title = "Distribution Comparison", kde_bw = .3, 
                     x_min = None, x_max = None, color_1 = None, 
                     color_2 = None, show_plot = True):
    # Compute density estimates
    # kde1 = gaussian_kde(array1, bw_method= kde_bw)
    kde1 = gaussian_kde(array1)
    if array2 is not None:
        # kde2 = gaussian_kde(array2, bw_method= kde_bw)
        kde2 = gaussian_kde(array2)
    
    # Define a range of values for the x-axis
    if x_min is None:
        if array2  is not None:
            x_min_1 = array1.min() 
            x_min_2 = array2.min()
        else:
            x_min_1 = array1.min()
    else: x_min_1 = x_min_2 = x_min

    if x_max is None:
        if array2  is not None:
            x_max_1 = array1.max()
            x_max_2 = array2.max()
        else:
            x_max_1 = array1.max()
    else: x_max_1 = x_max_2 = x_max
    

        
    x_values_1 = np.linspace(x_min_1, x_max_1, 1000)
    x_values_2 = np.linspace(x_min_2, x_max_2, 1000)

  
  
    # Plot the KDEs
    plt.figure(figsize=(10, 6))
    plt.plot(x_values_1, kde1(x_values_1), label= label_1, color = color_1)
    if array2  is not None:
        plt.plot(x_values_2, kde2(x_values_2), label= label_2, color = color_2)
    plt.xlabel("Distance to Closest point")
    plt.ylabel("pdf value")
    plt.title(title)
    plt.legend()
    plt.show()
    


def prepare_metadata_and_synthesizer(df, table_name= "table_name", metadata_file_path = "metadata.json"):
    """
    Prepare metadata from the DataFrame, save it to a JSON file, and initialize a synthesizer.
    
    Parameters:
    - df (pd.DataFrame): The DataFrame to extract metadata from.
    - table_name (str): The name to assign to the table in metadata.
    - metadata_file_path (str): Path to the metadata JSON file.
    
    Returns:
    - synthesizer (GaussianCopulaSynthesizer): An initialized synthesizer with the generated metadata.
    """
    # Check if the metadata file already exists and delete it if it does
    if os.path.exists(metadata_file_path):
        os.remove(metadata_file_path)  # Delete the existing metadata file

    # Detect metadata from the DataFrame
    metadata = Metadata()
    metadata.detect_table_from_dataframe(table_name=table_name, data=df)

    # Save the new metadata to the JSON file
    metadata.save_to_json(metadata_file_path)

    # Load metadata from the JSON file for reproducibility
    metadata = Metadata.load_from_json(metadata_file_path)
    
    return metadata


def sdv_full_report(data_name, datadir, synthesizer = None, 
                    metadata = None, columns_to_compare = None, train_portion = 0.5):
    df, _, _, _, _, _, _ = load_data(dataset = data_name, datadir = datadir)
    df = np.random.permutation(df)
    if not isinstance(df, pd.DataFrame):
        df = pd.DataFrame(df)
    df.columns = df.columns.astype(str)
    n = len(df)
    m = int(n * train_portion)
    train = df[:m]
    hold_out = df[m:]

    if metadata is None:
        metadata  = prepare_metadata_and_synthesizer(df = df, 
                                                   table_name= data_name, 
                                                   metadata_file_path = datadir + "/"+ data_name + "/metadata.json")
    
    synthesizer = synthesizer(metadata)
    #print('Start fitting')
    synthesizer.fit(train)
    #print('Start to synthesize')
    synthetic_data = synthesizer.sample(num_rows=len(hold_out))

    D_real_real = compute_min_distances(hold_out, train)
    D_synth_to_real = compute_min_distances(synthetic_data, train) 

    # %matplotlib inline
    compare_two_dist(D_real_real, D_synth_to_real, label_1 = 'real', label_2 = 'syntn')
 
    # F = [None for _ in range(len(columns_to_compare))]
    # for i in range(len(columns_to_compare)):
    #     c = columns_to_compare[i]
    #     F[i] = get_column_plot(
    #         real_data=hold_out,
    #         synthetic_data=synthetic_data,
    #         column_name= c,
    #         metadata=metadata
    #     )
        
    #     F[i].show()
    
    df_real_real = pd.DataFrame(D_real_real)
    df_real_real.columns = df_real_real.columns.astype(str)
    df_synth_to_real = pd.DataFrame(D_synth_to_real)
    df_synth_to_real.columns = df_synth_to_real.columns.astype(str)

    df_real_real.rename(columns={'0': 'Distance to Closest point'}, inplace=True)
    df_synth_to_real.rename(columns={'0': 'Distance to Closest point'}, inplace=True)

    metadata_DCP  = prepare_metadata_and_synthesizer(df = df_real_real, 
                                                   table_name= "DCP", 
                                                   metadata_file_path = datadir + "/"+ data_name + "/metadata_DCP.json")
    
    fig = get_column_plot(
        real_data=df_real_real,
        synthetic_data=df_synth_to_real,
        column_name= 'Distance to Closest point',
        metadata=metadata_DCP
    )
    
    fig.show()
    
    return df, train, hold_out, synthetic_data, metadata

def eigen_values_plot(H, shade = False):
    fig = go.Figure()
    fill_option = 'tozeroy' if shade else None
    
    for M, bw in H:
        # Assuming `eigen_report` returns the eigenvalues and min/max lambda values
        lambda_min, lambda_max, eigens = eigen_report(M, threshold=0)
        rank = sum(np.array(eigens) > 1e-5)
        
        # Add trace with shading below the curve
        fig.add_trace(go.Scatter(
            x=[i + 1 for i in range(len(eigens))],
            y=eigens,
            mode='lines',
            name=f"Eigenvalues for bw = {bw}, rank = {rank}",
            fill= fill_option  # Fills the area below the curve
        ))
    
    # Update layout
    fig.update_layout(
        title="Eigenvalues with Shaded Area Below the Curves",
        xaxis_title="Index",
        yaxis_title="Eigenvalue",
        legend_title="Bandwidth and Rank"
    )
    
    # Show the plot
    fig.show()

def RFM_report_diff_bw (X_train, y_train, X_test, y_test, bw_list, device, mem_gb, diag= False):
    History = []
    for bandwidth in bw_list:
        #cprint(f'Staring RFM training for bandwidth = {bandwidth} \n', color = 'blue', bold = True)
        model = LaplaceRFM(bandwidth=bandwidth, device = device, mem_gb = mem_gb, diag = diag)
        model.fit(
            (X_train, y_train), 
            (X_test, y_test), 
            loader=False, 
            iters=5,
            classif=True
        )
        History.append((model.M.numpy(), bandwidth))
        #cprint('End of training \n')
    return History

def synth_data_report_diff_bw(X_train, X_held_out, M, bw_list = None, threshold = 0):
    n = len(X_train)
    DCP = []
    n_sample = len(X_held_out)
    held_out_to_train = compute_min_distances(X_held_out, X_train)
    if bw_list is None: bw_list = [1.]
    S = []
    for bw in bw_list:   
        synth = []
        for i in range(n_sample):
            index = np.random.randint(0, n)
            x_1 = sample_l2_laplace(M + threshold*np.eye(M.shape[0]), X_train[index], T = bw, n_samples=1).reshape(-1)
            synth.append(x_1)
        S.append(np.array(synth))
        D_synth_to_real = compute_min_distances(np.array(synth), X_train)
        DCP.append((D_synth_to_real, bw))
    return held_out_to_train, DCP, S




def get_dis_plots(held_out_to_train, DCP_real_to_synthetic_list):
    
    # Create a histogram for each dataset
    fig = go.Figure()
    
    # Real data histogram
    fig.add_trace(go.Histogram(
        x= held_out_to_train,
        histnorm='probability',  # Normalise to show probability distributions
        name='DCP: held out to real',
        opacity=0.6
    ))
    for dcp, bw in  DCP_real_to_synthetic_list:
    # Synthetic data histogram
        fig.add_trace(go.Histogram(
            x= dcp,
            histnorm='probability',  # Normalise to show probability distributions
            name= f'DCP: synth to real for bw = {bw}',
            opacity=0.6
        ))
    # Update layout
    fig.update_layout(
        title="Density Comparison Plot",
        xaxis_title="Distance",
        yaxis_title="Density",
        legend_title="Dataset"
    )
    return fig
    


def get_dis_plots_KDE(held_out_to_train, DCP_real_to_synthetic_list):
    
    # Create a figure
    fig = go.Figure()
    
    # Estimate KDE for real data and plot as a line
    kde_real = gaussian_kde(held_out_to_train)
    x_real = np.linspace(min(held_out_to_train), max(held_out_to_train), 1000)
    y_real = kde_real(x_real)
    
    fig.add_trace(go.Scatter(
        x=x_real,
        y=y_real,
        mode='lines',
        name='DCP: held out to real'
    ))
    
    # Loop over each synthetic dataset and plot its KDE
    for dcp, bw in DCP_real_to_synthetic_list:
        kde_synth = gaussian_kde(dcp)
        x_synth = np.linspace(min(dcp), max(dcp), 1000)
        y_synth = kde_synth(x_synth)
        
        fig.add_trace(go.Scatter(
            x=x_synth,
            y=y_synth,
            mode='lines',
            name=f'DCP: synth to real for bw = {bw}'
        ))
    
    # Update layout
    fig.update_layout(
        title="Density Comparison Plot",
        xaxis_title="Distance",
        yaxis_title="Density",
        legend_title="Dataset",
        # width= 1000,  # Set the width of the figure in pixels
        height=600  # Set the height of the figure in pixels
    )
    
    return fig

def columns_dist_comparison(real_data, list_synt_data, column_name, bw_list = None, names = None, shade = False):
    if isinstance(real_data, pd.DataFrame):
        if column_name not in real_data.columns:
            raise KeyError(f"Column '{column_name}' does not exist in the real DataFrame.")

        # Get the column as a NumPy array
        R = real_data[column_name].to_numpy().reshape(-1)
    else:
        R = real_data[:, column_name].reshape(-1)
    
    # Estimate KDE for real data
    kde_real = gaussian_kde(R)
 
    x_real = np.linspace(min(R), max(R), 1000)
    y_real = kde_real(x_real)
    # Create a figure
    fig = go.Figure()
        
    fig.add_trace(go.Scatter(
        x=x_real,
        y=y_real,
        mode='lines',
        name=f'real dist. of {column_name}-th column' if bw_list is not None else names[0],
        fill='tozeroy' if shade else None  # Fill area under the curve
    ))
    
    # Loop over each synthetic dataset and plot its KDE
    for i in range(len(list_synt_data)):
        if isinstance(list_synt_data[i], pd.DataFrame):
            if column_name not in list_synt_data[i].columns:
                raise KeyError(f"Column '{column_name}' does not exist in the list_synt_data[{i}] DataFrame.")
    
            # Get the column as a NumPy array
            synth = list_synt_data[i][column_name].to_numpy().reshape(-1)
        else:
            synth = list_synt_data[i][:, column_name].reshape(-1)
        
        kde_synth = gaussian_kde(synth)
     
        x_synth = np.linspace(min(synth), max(synth), 1000)
        y_synth = kde_synth(x_synth)
        
        fig.add_trace(go.Scatter(
            x=x_synth,
            y=y_synth,
            mode='lines',
            name=f'synth. dist. of {column_name}-th column with bw = {bw_list[i]}' if bw_list is not None else names[i+1],
            fill='tozeroy' if shade else None  # Fill area under the curve
        ))
    
    # Update layout
    fig.update_layout(
        title=f"Density Comparison Plot of column {column_name}",
        xaxis_title="value",
        yaxis_title="Density",
        legend_title="Dataset and bw values" if bw_list is not None else "Densities:",
        # width= 1000,  # Set the width of the figure in pixels
        height=600  # Set the height of the figure in pixels
    )
    
    return fig


# def fit_mix_gaussian_to_(data, partition_portion=0.5, n=50, mode=None, 
#                          max_number_components=10, saved_path_for_plot=None):

#     if partition_portion == 'leave_one_out':
#         X = compute_min_distances(data, data, k=2)
#     else:    
#         m = int(len(data) * partition_portion)
#         X = []
#         for _ in range(n):
#             shuffled_data = np.random.permutation(data)
#             train = shuffled_data[:m]
#             held_out = shuffled_data[m:]
#             d_held_out_to_train = compute_min_distances(held_out, train, k=1)
            
#             if mode == 'mean':
#                 d = np.mean(d_held_out_to_train)
#                 X.append(d)
#             elif mode == 'median':
#                 d = np.median(data)
#                 X.append(d)
#             else: 
#                 X.extend(d_held_out_to_train)

#     X = np.array(X).reshape(-1, 1)
#     n_components_range = range(1, max_number_components + 1)

#     bic_scores = []
#     aic_scores = []
#     model_records = []

#     for n in n_components_range:

#         gmm = GaussianMixture(n_components=n)
#         # cprint('GaussianMixture model has been initaited')
#         # cprint(f'the shape of data to fit is {X.shape}')
#         gmm.fit(X)
#         # print('gmm model has been fitted')
#         model_records.append(gmm)
#         bic_scores.append(gmm.bic(X))
#         aic_scores.append(gmm.aic(X))

#     # Plotting
#     plt.figure(figsize=(10, 5))
#     plt.plot(n_components_range, bic_scores, label='BIC', marker='o')
#     plt.plot(n_components_range, aic_scores, label='AIC', marker='x')
#     plt.xlabel('Number of Components')
#     plt.ylabel('BIC / AIC Score')
#     plt.legend()
#     plt.title('BIC and AIC Scores for Different Number of GMM Components')

#     if saved_path_for_plot:
#         plt.savefig(saved_path_for_plot, bbox_inches='tight')
#         plt.close()
#     else:
#         plt.show()

#     optimal_components_bic = n_components_range[np.argmin(bic_scores)]
#     optimal_components_aic = n_components_range[np.argmin(aic_scores)]

#     print(f"Optimal number of components according to BIC: {optimal_components_bic}")
#     print(f"Optimal number of components according to AIC: {optimal_components_aic}")
#     cprint('Respectively corresponding models have been returned', color='blue', bold=True)

#     return model_records[optimal_components_bic - 1], model_records[optimal_components_aic - 1], X.reshape(-1)



def fit_mix_gaussian_to_(data, 
                         partition_portion = 0.5, 
                         n = 100, 
                         mode = None, 
                         max_number_components = 10,
                         indexing = 'KDTree',
                         percentile = None,
                         random_selection_size = None,
                         down_sampling = None,
                         saved_path_for_plot = None
                         ):

    if partition_portion == 'leave_one_out':
        X = compute_min_distances(data, data, k = 2)[:,  -1]
    else:    
        m = int(len(data)*partition_portion)
        X = []
        for _ in range(n):
            
            # shuffled_data = np.random.permutation(data[:2*random_selection_size] if random_selection_size is not None else data)
            
            if random_selection_size is not None:
                indices = np.random.choice(len(data), size=2 * random_selection_size, replace=False)
                shuffled_data = data.iloc[indices].copy()
                shuffled_data = shuffled_data.sample(frac=1).reset_index(drop=True)  # shuffle rows
            else:
                shuffled_data = data.sample(frac=1).reset_index(drop=True)  # full shuffle

            train = shuffled_data[:m] if random_selection_size is None else shuffled_data[:random_selection_size]
            held_out = shuffled_data[m:] if random_selection_size is None else shuffled_data[random_selection_size:]
            
            if mode == 'percentile':
                if percentile is None:
                    raise ValueError(f"Invalid value provided.\n For mood = percentile, the value of percentile should not be {percentile}")
                p = (percentile * m)//100
                d_held_out_to_train = compute_min_distances(held_out, train, k = p)
                X.append(d_held_out_to_train[:, -1])
            elif mode == '<=percentile':
                if percentile is None:
                    raise ValueError(f"Invalid value provided.\n For mood = percentile, the value of percentile should not be {percentile}")
                percentile_value = np.percentile(d_held_out_to_train, percentile)
                X.extend([x for x in d_held_out_to_train if x <= percentile_value])
            else: 
                if _ == 0:
                    pass
                    #print(f'compute_min_distances with indexing = {indexing}')
                d_held_out_to_train = compute_min_distances(held_out, train, k = 1)
                X.extend(d_held_out_to_train)
        
    X_whole = np.array(X).reshape(-1, 1)
    if down_sampling is not None:
        # Sample indices without replacement
        indices = np.random.choice(X_whole.shape[0], down_sampling, replace=False)

        # Select corresponding rows
        X = X_whole[indices]
    else:
        X = X_whole
     
    #print(f'Fitting a mixture of Gaussian over X which is of size {len(X)}')
    n_components_range = range(1, max_number_components+1)

    # Store BIC scores
    bic_scores = []
    aic_scores = []
    model_records = []
 
    
    # Fit models with different numbers of components
    for n in n_components_range:
        gmm = GaussianMixture(n_components=n)
        gmm.fit(X)
        model_records.append(gmm)
        bic_scores.append(gmm.bic(X))  # Append BIC for current model
        aic_scores.append(gmm.aic(X))  # Append AIC for current model
    
    # Plot BIC and AIC scores
    plt.figure(figsize=(10, 5))
    plt.plot(n_components_range, bic_scores, label='BIC', marker='o')
    plt.plot(n_components_range, aic_scores, label='AIC', marker='x')
    plt.xlabel('Number of Components')
    plt.ylabel('BIC / AIC Score')
    plt.legend()
    plt.title('BIC and AIC Scores for Different Number of GMM Components')

    if saved_path_for_plot:
        plt.savefig(saved_path_for_plot, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

    
    # Choose the best number of components based on the lowest BIC or AIC
    optimal_components_bic = n_components_range[np.argmin(bic_scores)]
    optimal_components_aic = n_components_range[np.argmin(aic_scores)]
    
    #print(f"Optimal number of components according to BIC: {optimal_components_bic}")
    #print(f"Optimal number of components according to AIC: {optimal_components_aic}")
    # print(optimal_components_bic, optimal_components_aic, model_records)
    #cprint('Respectively corresponding models have been returned', color = 'blue', bold = True)
    return model_records[optimal_components_bic-1], model_records[optimal_components_aic-1], X_whole.reshape(-1)



    

def DCP_dist_comparison(real_data, list_synt_data, evaluating_metric, names = None):
    # Estimate KDE for real data
    kde_real = gaussian_kde(real_data)
 
    x_real = np.linspace(min(real_data), max(real_data), 1000)
    y_real = kde_real(x_real)

    
    fill_colors = [
        'rgba(0, 0, 255, 0.2)', 
        'rgba(0, 255, 0, 0.2)', 
        'rgba(255, 165, 0, 0.2)', 
        'rgba(128, 0, 128, 0.2)',
        'rgba(255, 0, 0, 0.2)', 
        'rgba(0, 255, 255, 0.2)', 
        'rgba(255, 255, 0, 0.2)'
    ]
    
    line_colors = [
        'blue', 
        'green', 
        'orange', 
        'purple', 
        'red', 
        'cyan', 
        'yellow'
    ]

    
    # Create a figure
    fig = go.Figure()
        
    # Add shaded area for the real data
    fig.add_trace(go.Scatter(
        x=x_real,
        y=y_real,
        mode='lines',
        name='Real DCP Dist.' if evaluating_metric is not None else names[0], 
        fill='tozeroy',  # Fill area under the curve
        fillcolor='rgba(0, 0, 255, 0.2)',  # Semi-transparent blue shade
        line=dict(color='blue')
    ))
    
    # Loop over each synthetic dataset and plot its KDE with different colors
    for i, synth_data in enumerate(list_synt_data):
        if evaluating_metric is not None:    
            metric = evaluating_metric[i]
        elif names is not None:
            name = names[i+1]
        synth = synth_data.reshape(-1)
        kde_synth = gaussian_kde(synth)
     
        x_synth = np.linspace(min(synth), max(synth), 1000)
        y_synth = kde_synth(x_synth)
        
        fill_color = fill_colors[i+1]
        line_color = line_colors[i+1]
        
        fig.add_trace(go.Scatter(
            x=x_synth,
            y=y_synth,
            mode='lines',
            name= f'Best Fitted Mixture based on {metric} Score' if evaluating_metric is not None else name,
            fill='tozeroy',  # Fill area under the curve
            fillcolor=fill_color,
            line=dict(color = line_color)
        ))
    
    # Update layout
    fig.update_layout(
        title="Density Comparison",
        xaxis_title="Value",
        yaxis_title="Density",
        legend_title="All Distributions",
        height=600  # Set the height of the figure in pixels
    )
    
    return fig

def sample_points_via_DCP_dist(X, n_sample, model, noise_std = 1):

    m, d = X.shape 

    # Choose n random points from X
    random_indices = np.random.randint(0, m, size= n_sample)
    chosen_points = X[random_indices]  # Shape: n x d

    # Choose n uniform random directions from d-dimensional standard Gaussian and normalize them
    directions = np.random.normal(size=(n_sample, d))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)

    # Choose n random values from model (model is GaussianMixture from sklearn.mixture)
    distances = model.sample(n_sample)[0]

    # Choose n random noise vectors from d-dimensional Gaussian with variance sigma
    noise = np.random.normal(scale = noise_std, size=(n_sample, d))

    # Calculate the sampled points matrix (n, d)
    samples = chosen_points + (distances * directions) + noise

    return samples

def measure_of_independence(X, Y, n_samples = 1000):
    n_x = len(X)
    n_y = len(Y)
    random_indices_x = np.random.randint(0, n_x, size= n_samples)
    random_indices_y = np.random.randint(0, n_y, size= n_samples)
    X_selected = X[random_indices_x]
    Y_selected = Y[random_indices_y]
    EXY = np.sum(X_selected * Y_selected)/n_samples
    EX_EY = np.dot(np.mean(X_selected, axis = 0), np.mean(Y_selected, axis = 0))
    return EXY - EX_EY




def list_files(folder_path):
    """
    Lists all files in the given folder and its subfolders.

    Args:
        folder_path (str): Path to the folder to scan.

    Returns:
        List[str]: A list of file paths.
    """
    all_files = []
    for root, dirs, files in os.walk(folder_path):
        # for file in files:
        all_files.append(dirs)

    return all_files

