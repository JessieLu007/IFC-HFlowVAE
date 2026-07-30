from sklearn.preprocessing import StandardScaler
from scipy.spatial import KDTree

from sklearn.metrics import mutual_info_score
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import os
import json



def compute_category_map(df, column_name, v):
    # Step 1: Get unique categories and indices
    unique_categories = df[column_name].unique()

    # Step 2: Compute indices for each category
    category_indices = {c: np.where(df[column_name] == c)[0] for c in unique_categories}

    # Step 3: Compute the mean for each category
    category_means = {c: np.mean(v[indices]) for c, indices in category_indices.items()}

    # Step 4: Sort categories by their means and assign ranks
    sorted_categories = sorted(category_means.items(), key=lambda item: item[1])
    category_map = {c: rank+1 for rank, (c, _) in enumerate(sorted_categories)}

    return category_map


class DataProcessor:
    def __init__(self):
        self.scaler = None
        self.encoding_map = {}
        self.target_column = None
        self.decorrelation = None
        self.U = None
        self.eigens = None
        self.json_path = None 
        self.columns = None
        self.cat_columns = None
        self.num_columns = None
        self.v = None

    def fit(self, df_train, 
            target_column = None, 
            normalize=True, 
            decorrelation = False, 
            json_path = None,
            cat_encoding = None
           ):
     
        self.decorrelation = decorrelation
        df_train_encoded = df_train.copy()
        self.json_path = json_path
        data = {}
     
        # Check if JSON file exists
        if json_path is not None and os.path.exists(self.json_path):
            # Load the JSON file
            with open(self.json_path, 'r') as file:
                data = json.load(file)

            # Extract categorical column indices and names
            if "cat_col_idx" in data and "column_names" in data:
                self.columns = data["column_names"]
                self.cat_columns = [data["column_names"][idx] for idx in data["cat_col_idx"]]
            
            # Extract num column indices and names
            if "num_col_idx" in data and "column_names" in data:
                self.num_columns = [data["column_names"][idx] for idx in data["num_col_idx"]]
            
            if data["task_type"] != "regression":
                self.cat_columns.extend([data["column_names"][idx] for idx in data["target_col_idx"]])
            elif data["task_type"] == "regression":
                self.num_columns.extend([data["column_names"][idx] for idx in data["target_col_idx"]])
            else: 
                ttt = data["task_type"]
                raise ValueError(f"task_type = {ttt} is unknown")
                
        else:
            self.columns = df_train.columns
            self.cat_columns = df_train.select_dtypes(include=['object']).columns
            self.num_columns = df_train.select_dtypes(include=['number']).columns

        if cat_encoding == 'ours':
            self.scaler_1 = StandardScaler()
            standardized_data = self.scaler_1.fit_transform(df_train[self.num_columns])
        
            # Step 2: Apply PCA
            self.pca_1 = PCA(n_components=1)  # Keep only the most principal component
            self.v = self.pca_1.fit_transform(standardized_data)
     
        for col in self.cat_columns:
            if self.v is None:
                self.encoding_map[col] = {cat: code for code, cat in enumerate(df_train[col].astype('category').cat.categories, start=1)}
            else:
                self.encoding_map[col] = compute_category_map(df_train, col, self.v)
            df_train_encoded[col] = df_train[col].map(self.encoding_map[col])
            
        if self.target_column is not None:
            train_target = df_train_encoded.pop(target_column).to_numpy().astype(int)
        else: 
            train_target = None
            
        df_train_encoded = df_train_encoded.astype(float)

        if normalize:
            self.scaler = StandardScaler()
            df_train_encoded[:] = self.scaler.fit_transform(df_train_encoded)

        if self.decorrelation:
            pca = PCA(n_components = df_train_encoded.shape[1])
            pca.fit(df_train_encoded)
            self.U = pca.components_
            self.eigens = pca.explained_variance_
            df_train_encoded[:] = df_train_encoded@self.U.T
            
        return df_train_encoded, train_target
        
    def transform(self, df_test):
        df_test_encoded = df_test.copy()
        for col in self.cat_columns:
            df_test_encoded[col] = df_test[col].map(self.encoding_map[col])
            
        if self.target_column is not None:
            test_target = df_test_encoded.pop(target_column).to_numpy().astype(int)
        else: 
            test_target = None
            
        df_test_encoded = df_test_encoded.astype(float)

        if self.scaler is not None:
            df_test_encoded[:] = self.scaler.transform(df_test_encoded)
        if self.decorrelation:
            df_test_encoded[:] = df_test_encoded@self.U.T
         
        return df_test_encoded, test_target

    def decode(self, num_data):
        """
        Reverts the scaling and encoding back to the original data settings.
        """
     
        df_decoded = num_data.copy()
        if self.decorrelation:
            print('decorrelation is ON')
            df_decoded[:] = df_decoded@self.U
        
        if self.scaler is not None:
            df_decoded[:] = self.scaler.inverse_transform(df_decoded)

        for col, mapping in self.encoding_map.items():
            reverse_mapping = {code: cat for cat, code in mapping.items()}
            min_key = min(reverse_mapping.keys())
            max_key = max(reverse_mapping.keys())
            df_decoded[col] = df_decoded[col].round().astype(int).apply(
                lambda x: reverse_mapping.get(
                    min(max(x, min_key), max_key), reverse_mapping[min_key]
                )
            )

        return df_decoded
    def get_columns(self):
         """
         return columns, cat columns, and num columns 
         """
         return self.columns, self.cat_columns, self.num_columns



class EmpiricalTransformer:
    def __init__(self, df):
        """
        Initialize the EmpiricalTransformer with a DataFrame.
        
        Parameters:
        - df: A pandas DataFrame with k columns.
        """
        self.df = df
        self.df_sorted = None
        self.df_ranks = None
        self.fit()
    
    def fit(self, method = 'min'):
        """
        method: it determines who to treat with ties in ranking (min, max, average, dense)
        Fit the model on the given DataFrame:
        1. Sort each column and create df_sorted.
        2. Generate a dataframe of ranks for each entry in the original df.
        """
        self.df_sorted = self.df.apply(np.sort, axis=0)
        self.df_ranks = self.df.rank(method= method).astype(int)/self.df.shape[0] 
        #method = 'dense' : Assigns the minimum rank to all tied values and ranks are consecutive (no gaps between ranks)
        return self.df_ranks
    
    def inverse_empirical(self, u, sorted_col):
        """
        Transform a uniform random value using the inverse empirical distribution of a column.
        
        Parameters:
        - u: A uniform random value in [0, 1].
        - sorted_col: A sorted numpy array or pandas Series.
        
        Returns:
        - A value sampled from the empirical distribution.
        """
        n = len(sorted_col)
        ecdf = np.arange(1, n + 1) / n  # ECDF values
        return np.interp(u, ecdf, sorted_col)
    
    def convert(self, u_vectors):
        """
        Transform random vectors using the inverse empirical distribution of each column.
        
        Parameters:
        - u_vectors: A 2D numpy array where each row is a random vector (u_1, ..., u_k).
        
        Returns:
        - transformed_vectors_df: A DataFrame of transformed vectors with the same column names as df.
        """
        transformed_vectors = []
        for u_vec in u_vectors:
            transformed_vec = [
                self.inverse_empirical(u, self.df_sorted.iloc[:, i].values)
                for i, u in enumerate(u_vec)
            ]
            transformed_vectors.append(transformed_vec)
        
        transformed_vectors = np.array(transformed_vectors)
        transformed_vectors_df = pd.DataFrame(transformed_vectors, columns=self.df.columns)
        return transformed_vectors_df









