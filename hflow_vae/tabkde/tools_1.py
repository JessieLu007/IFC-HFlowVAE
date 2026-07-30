
from scipy.spatial import KDTree


import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
import os
import json


#----------------------------------------------------------------------------------

from scipy.stats import chi2_contingency
import seaborn as sns

from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, mutual_info_score
from sklearn.decomposition import PCA
from scipy.stats import ks_2samp, pearsonr
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import KBinsDiscretizer


from scipy.spatial.distance import jensenshannon


import argparse
import sys
import os

from sklearn.tree import DecisionTreeClassifier
minlength = 100


from collections import defaultdict




class CustomClassifier:
    def __init__(self, model='random'):
        self.model = model
        self.df = None
        self.C = None
        self.target = None
        self.depth = None
        self.n_estimators = None
        self.probabilities = defaultdict(dict)
        self.classifier = None
    
    def fit(self, df, C, target, depth=None, n_estimators=10, random_state = 42):
        """Fits the classifier based on the selected model."""
        self.df = df
        self.C = C
        self.target = target
        self.depth = depth
        self.n_estimators = n_estimators
        self.random_state = random_state
        
        if self.model == 'random':
            self._fit_random_model()
        elif self.model == 'decision_tree':
            self._fit_decision_tree()
        elif self.model == 'random_forest':
            self._fit_random_forest()
        elif self.model == 'adaboost':
            self._fit_adaboost()
        elif self.model == 'gradient_boosting':
            self._fit_gradient_boosting()
        elif self.model == 'xgboost':
            self._fit_xgboost()
        else:
            raise ValueError("Supported models: 'random', 'decision_tree', 'random_forest', 'adaboost', 'gradient_boosting', 'xgboost'")
    
    def _fit_random_model(self):
        """Fits a multinomial distribution for each unique combination of C."""
        grouped = self.df.groupby(self.C)[self.target].value_counts(normalize=True)
        
        for index, prob in grouped.items():
            if isinstance(index, tuple):  # If it's a multi-index
                features, target = index[:-1], index[-1]  # Split last element as target
            else:  # If only one feature is used
                features, target = (index,), grouped.index[-1]
            
            key = features if isinstance(features, tuple) else (features,)
            if key not in self.probabilities:
                self.probabilities[key] = {}
            self.probabilities[key][target] = prob



    def _fit_decision_tree(self):
        """Fits a Decision Tree classifier."""
        self.classifier = DecisionTreeClassifier(max_depth=self.depth, random_state=self.random_state)
        self.classifier.fit(self.df[self.C], self.df[self.target])
    
    def _fit_random_forest(self):
        """Fits a Random Forest classifier."""
        self.classifier = RandomForestClassifier(n_estimators=self.n_estimators, max_depth=self.depth, random_state=self.random_state)
        self.classifier.fit(self.df[self.C], self.df[self.target])
    
    def _fit_adaboost(self):
        """Fits an AdaBoost classifier."""
        self.classifier = AdaBoostClassifier(n_estimators=self.n_estimators, random_state=self.random_state)
        self.classifier.fit(self.df[self.C], self.df[self.target])
    
    def _fit_gradient_boosting(self):
        """Fits a Gradient Boosting classifier."""
        self.classifier = GradientBoostingClassifier(n_estimators=self.n_estimators, max_depth=self.depth, random_state=self.random_state)
        self.classifier.fit(self.df[self.C], self.df[self.target])
    
    # def _fit_xgboost(self):
    #     """Fits an XGBoost classifier."""
    #     self.classifier = xgb.XGBClassifier(n_estimators=self.n_estimators, max_depth=self.depth, use_label_encoder=False, 
    #                                         eval_metric='logloss', random_state=self.random_state)
    #     self.classifier.fit(self.df[self.C], self.df[self.target])

    def _fit_xgboost(self):
        """Fits an XGBoost classifier while ensuring class labels are numeric and consistent."""
        self.classifier = xgb.XGBClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.depth,
            eval_metric='logloss',  # Keep eval_metric for classification
            random_state=self.random_state
        )
    
        # Convert categorical target variable into numeric values
        self.label_encoder = LabelEncoder()
        y_encoded = self.label_encoder.fit_transform(self.df[self.target])
    
        # Train the XGBoost classifier
        self.classifier.fit(self.df[self.C], y_encoded)


    def predict(self, X):
        cprint(f'predicting for {self.target} using feature in {self.C}', color= 'blue', bold = True)
        """Predicts target values and samples based on predicted probabilities if applicable."""
        if not set(self.C).issubset(X.columns):
            raise ValueError("Input DataFrame does not contain the required feature columns.")
        
        if self.model == 'random':
            predictions = []
            for _, row in X.iterrows():
                key = tuple(row[col] for col in self.C) if len(self.C) > 1 else (row[self.C[0]],)
                
                if key in self.probabilities:
                    target_classes, probs = zip(*self.probabilities[key].items())
                    prediction = np.random.choice(target_classes, p=probs)  # Random sampling based on probability
                else:
                    prediction = np.random.choice(self.df[self.target].unique())  # Random pick if unseen
                predictions.append(prediction)
        
        elif self.model in ['decision_tree', 'random_forest', 'adaboost', 'gradient_boosting', 'xgboost']:
            probs = self.classifier.predict_proba(X[self.C])  # Get probability distribution
            classes = self.classifier.classes_  # Get class labels
            
            predictions = [np.random.choice(classes, p=p) for p in probs]  # Sample based on probabilities
            
            # Convert back to original labels if LabelEncoder was used
            if self.model == 'xgboost' and hasattr(self, "label_encoder"):
                predictions = self.label_encoder.inverse_transform(predictions)
        
        else:
            raise ValueError("Unsupported model type for prediction")
        
        # If the target column exists in X, compute accuracy
        if self.target in X.columns:
            accuracy = accuracy_score(X[self.target], predictions)
            return predictions, accuracy

        return predictions

    
    def fit_and_update_df(self, X):
        """Predicts and updates a new DataFrame with predictions as the target column."""
        if not set(self.C).issubset(X.columns):
            raise ValueError("Input DataFrame does not contain the required feature columns.")
        
        X = X.copy()
        X[self.target] = self.predict(X)
        return X



# class CustomClassifier:
#     def __init__(self, model='random'):
#         self.model = model
#         self.df = None
#         self.C = None
#         self.target = None
#         self.depth = None
#         self.n_estimators = None
#         self.probabilities = defaultdict(dict)
#         self.classifier = None
    
#     def fit(self, df, C, target, depth=None, n_estimators=10, random_state = 42):
#         """Fits the classifier based on the selected model."""
#         self.df = df
#         self.C = C
#         self.target = target
#         self.depth = depth
#         self.n_estimators = n_estimators
#         self.random_state = random_state
        
#         if self.model == 'random':
#             self._fit_random_model()
#         elif self.model == 'decision_tree':
#             self._fit_decision_tree()
#         elif self.model == 'random_forest':
#             self._fit_random_forest()
#         elif self.model == 'adaboost':
#             self._fit_adaboost()
#         elif self.model == 'gradient_boosting':
#             self._fit_gradient_boosting()
#         elif self.model == 'xgboost':
#             self._fit_xgboost()
#         else:
#             raise ValueError("Supported models: 'random', 'decision_tree', 'random_forest', 'adaboost', 'gradient_boosting', 'xgboost'")
    
#     def _fit_random_model(self):
#         """Fits a multinomial distribution for each unique combination of C."""
#         grouped = self.df.groupby(self.C)[self.target].value_counts(normalize=True)
        
#         for index, prob in grouped.items():
#             if isinstance(index, tuple):  # If it's a multi-index
#                 features, target = index[:-1], index[-1]  # Split last element as target
#             else:  # If only one feature is used
#                 features, target = (index,), grouped.index[-1]
            
#             key = features if isinstance(features, tuple) else (features,)
#             if key not in self.probabilities:
#                 self.probabilities[key] = {}
#             self.probabilities[key][target] = prob



#     def _fit_decision_tree(self):
#         """Fits a Decision Tree classifier."""
#         self.classifier = DecisionTreeClassifier(max_depth=self.depth, random_state=self.random_state)
#         self.classifier.fit(self.df[self.C], self.df[self.target])
    
#     def _fit_random_forest(self):
#         """Fits a Random Forest classifier."""
#         self.classifier = RandomForestClassifier(n_estimators=self.n_estimators, max_depth=self.depth, random_state=self.random_state)
#         self.classifier.fit(self.df[self.C], self.df[self.target])
    
#     def _fit_adaboost(self):
#         """Fits an AdaBoost classifier."""
#         self.classifier = AdaBoostClassifier(n_estimators=self.n_estimators, random_state=self.random_state)
#         self.classifier.fit(self.df[self.C], self.df[self.target])
    
#     def _fit_gradient_boosting(self):
#         """Fits a Gradient Boosting classifier."""
#         self.classifier = GradientBoostingClassifier(n_estimators=self.n_estimators, max_depth=self.depth, random_state=self.random_state)
#         self.classifier.fit(self.df[self.C], self.df[self.target])
    
#     # def _fit_xgboost(self):
#     #     """Fits an XGBoost classifier."""
#     #     self.classifier = xgb.XGBClassifier(n_estimators=self.n_estimators, max_depth=self.depth, use_label_encoder=False, 
#     #                                         eval_metric='logloss', random_state=self.random_state)
#     #     self.classifier.fit(self.df[self.C], self.df[self.target])

#     def _fit_xgboost(self):
#         """Fits an XGBoost classifier while ensuring class labels are numeric and consistent."""
#         self.classifier = xgb.XGBClassifier(
#             n_estimators=self.n_estimators,
#             max_depth=self.depth,
#             eval_metric='logloss',  # Keep eval_metric for classification
#             random_state=self.random_state
#         )
    
#         # Convert categorical target variable into numeric values
#         self.label_encoder = LabelEncoder()
#         y_encoded = self.label_encoder.fit_transform(self.df[self.target])
    
#         # Train the XGBoost classifier
#         self.classifier.fit(self.df[self.C], y_encoded)


#     def predict(self, X):
#         """Predicts target values and samples based on predicted probabilities if applicable."""
#         if not set(self.C).issubset(X.columns):
#             raise ValueError("Input DataFrame does not contain the required feature columns.")
        
#         if self.model == 'random':
#             predictions = []
#             for _, row in X.iterrows():
#                 key = tuple(row[col] for col in self.C) if len(self.C) > 1 else (row[self.C[0]],)
                
#                 if key in self.probabilities:
#                     target_classes, probs = zip(*self.probabilities[key].items())
#                     prediction = np.random.choice(target_classes, p=probs)  # Random sampling based on probability
#                 else:
#                     prediction = np.random.choice(self.df[self.target].unique())  # Random pick if unseen
#                 predictions.append(prediction)
        
#         elif self.model in ['decision_tree', 'random_forest', 'adaboost', 'gradient_boosting', 'xgboost']:
#             probs = self.classifier.predict_proba(X[self.C])  # Get probability distribution
#             classes = self.classifier.classes_  # Get class labels
            
#             predictions = [np.random.choice(classes, p=p) for p in probs]  # Sample based on probabilities
            
#             # Convert back to original labels if LabelEncoder was used
#             if self.model == 'xgboost' and hasattr(self, "label_encoder"):
#                 predictions = self.label_encoder.inverse_transform(predictions)
        
#         else:
#             raise ValueError("Unsupported model type for prediction")
        
#         # If the target column exists in X, compute accuracy
#         if self.target in X.columns:
#             accuracy = accuracy_score(X[self.target], predictions)
#             return predictions, accuracy
        
#         return predictions

    
#     def fit_and_update_df(self, X):
#         """Predicts and updates a new DataFrame with predictions as the target column."""
#         if not set(self.C).issubset(X.columns):
#             raise ValueError("Input DataFrame does not contain the required feature columns.")
        
#         X = X.copy()
#         X[self.target] = self.predict(X)
#         return X




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
            #print(colored_text)
        elif color in colors:
            colored_text = f"{bold_code}{colors[color]}{text}{reset}"
            #print(colored_text)
        else:
            pass
            #cprint(f"Invalid color '{color}'. Available colors: {', '.join(colors.keys())}, lets use red color instead", color = 'yellow', bold = True)
            #cprint(text, color = 'red', bold = True)



import random
import bisect

#latest Update

from scipy.stats import pointbiserialr, f_oneway

# Function to compute Eta-Squared (ANOVA-based)
def eta_squared(anova_result, total_ss):
    return anova_result.statistic * (anova_result.df_between / (anova_result.df_between + anova_result.df_within))


def max_correlation(data_frame, cat_column, numerical_columns, method='point_biserial'):
    print(f'lets find the most correlated numerical column for {cat_column}')
    correlation_scores = {}

    # Point-Biserial Correlation (for binary categorical variables)
    if method == 'point_biserial':
        if data_frame[cat_column].nunique() != 2:
            raise ValueError("Point-Biserial requires a binary categorical variable.")

        for num_col in numerical_columns:
            corr, _ = pointbiserialr(data_frame[cat_column], data_frame[num_col] - data_frame[num_col].mean())
            correlation_scores[num_col] = abs(corr)  # Use absolute value for comparison

    # ANOVA F-Test (for multi-class categorical variables)
    elif method == 'anova':
        for num_col in numerical_columns:
            groups = [group[num_col].dropna().values for _, group in (data_frame - data_frame.mean()).groupby(cat_column)]
            f_stat, _ = f_oneway(*groups)
            correlation_scores[num_col] = f_stat

    # Eta-Squared (Proportion of variance explained)
    elif method == 'eta_squared':
        for num_col in numerical_columns:
            groups = [group[num_col].dropna().values for _, group in (data_frame - data_frame.mean()).groupby(cat_column)]
            anova_result = f_oneway(*groups)
            total_ss = np.var(data_frame[num_col], ddof=1) * (len(data_frame[num_col]) - 1)
            eta_sq = (anova_result.statistic * (len(groups) - 1)) / (anova_result.statistic * (len(groups) - 1) + len(data_frame) - len(groups))
            correlation_scores[num_col] = eta_sq

    else:
        raise ValueError("Invalid method. Choose from 'point_biserial', 'anova', or 'eta_squared'.")

    # Return the numerical column with the maximum correlation
    max_corr_column = max(correlation_scores, key=correlation_scores.get)
    max_corr_value = correlation_scores[max_corr_column]
    #print(f'for column {cat_column} the most correlated numericl column is {max_corr_column}')
    return max_corr_column, max_corr_value






def compute_category_map(df, column_name,  num_columns, v, rank_encoding):
    # Step 0: find the most correlated numerical feature for encoding 
    if isinstance(v, str) and v == 'max_correlation':
        max_corr_column, max_corr_value = max_correlation(df, column_name, num_columns, method='eta_squared')
        #print(f'compute_category_map is done')
        v = np.array(df[max_corr_column])
     
    # Step 1: Get unique categories and indices
    unique_categories = df[column_name].unique()

    # Step 2: Compute indices for each category
    category_indices = {c: np.where(df[column_name] == c)[0] for c in unique_categories}

    # Step 3: Compute the mean for each category
    # category_means = {c: np.mean(v[indices]) for c, indices in category_indices.items()}
    category_means = {c: np.sum(v[indices]) for c, indices in category_indices.items()}

    if rank_encoding:
        # Step 4: Sort categories by their means and assign ranks
        sorted_categories = sorted(category_means.items(), key=lambda item: item[1])
        category_map = {c: rank+1 for rank, (c, _) in enumerate(sorted_categories)}
        return category_map
    else: 
        return category_means
  
def hierarchical_encoding(df, col1, col2):
    encoding_dict = {}
    start_value = 1
    
    for key, group in sorted(df[[col1, col2]].drop_duplicates().groupby(col1), key=lambda x: x[0]):
        unique_vals = sorted(group[col2].unique())
        group_encoding = {val: start_value + i for i, val in enumerate(unique_vals)}
        encoding_dict.update(group_encoding)
        start_value += len(unique_vals)
    return encoding_dict
    





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

def perturb_vector_with_partial_resampling_with_DCP(X, 
                                                    model, 
                                                    num_samples,
                                                    sigma = 'global', 
                                                    L_limits=None, 
                                                    U_limits=None, 
                                                    l = 2, 
                                                    allowed_outside_boundary = False):
    """
    Perturb a given vector x within a d-dimensional hypercube [0, 1]^d.

    This function modifies the input vector x by:
    1. Sampling a random vector u = (u_1, ..., u_d) from a normal distribution.
    2. Adding u to x to compute x'.
    3. Resampling only the out-of-bound coordinates of x' until all coordinates are within [0,1].

    Parameters:
    - x: np.ndarray of shape (n, d), the input vectors to be perturbed.

    Returns:
    - x_prime: np.ndarray of shape (n, d), the perturbed vectors within [0,1]^d.
    """
    #cprint(f'Imprortant: given sigma is {sigma}', color = 'red', bold = True)
    #cprint(f'Imprortant: allowed_outside_boundary is {allowed_outside_boundary}', color = 'blue', bold = True)
    n,d = X.shape
    if L_limits is None:
        L_limits = np.zeros(d)
    if U_limits is None:
        U_limits = np.ones(d)
    if isinstance(sigma, (int, float, np.float64, np.int32, np.int64)):  # Check if sigma is a real value
        Sigma = sigma * np.identity(d)
    elif isinstance(sigma, np.ndarray):  # Check if sigma is already a matrix
        Sigma = sigma
    elif sigma == 'local':
        tree = KDTree(X)
    elif sigma == 'global':
        Sigma = np.cov(X.T)
    
    else:
        raise TypeError("Sigma must be a real number or a numpy matrix/array.")
        
    synth = []
    counter = 0
    while counter < num_samples:
        x = X[counter % n]
        if isinstance(sigma, str) and sigma == 'local':
            _, I= tree.query(x, k = l*d)
            NN = X[I].T
            Sigma = np.cov(NN)
        x_prime = gen_one_point_at(x, model, Sigma,  L_limits, U_limits, allowed_outside_boundary)
        if x_prime is not None:
            synth.append(x_prime)
            counter += 1
    return np.array(synth)
    
    
def gen_one_point_at(x, model, Sigma,  L_limits, U_limits, allowed_outside_boundary):
    d = x.shape[0]
    if allowed_outside_boundary:
        r = -1
        while r < 0:
            r = model.sample(1)[0][0][0]
        u = np.random.multivariate_normal(np.zeros(d), Sigma)
        u /= np.linalg.norm(u)
        return x + r*u
    
    x_prime = np.zeros(x.shape)
    counter = 0
    outside_bounds = np.ones(x_prime.shape, dtype=bool)
    r = -1
    while r < 0:
        r = model.sample(1)[0][0][0]
    normilizer = 1
    flag = False
    while counter < 10 * d:
        d_new = sum(outside_bounds)
        if flag:
            return None 
        if d_new == 1:
            flag = True
            u *= -1
        else:
            u = np.random.multivariate_normal(np.zeros(d), Sigma)
            u /= np.linalg.norm(u)
        x_prime[outside_bounds] = x[outside_bounds] + normilizer*r*u[outside_bounds]
        outside_bounds = (x_prime < L_limits) | (x_prime > U_limits)
        normilizer = np.linalg.norm(u[outside_bounds])
        if not np.any(outside_bounds):
            return x_prime
    return None  



def compute_correlation_divergence(real_data, synthetic_data, show_numbers=True, title = None, vmin = 0, vmax = 1):
    """
    Compute the absolute divergence heatmap between real and synthetic data correlations.

    Parameters:
    real_data (pd.DataFrame): Real dataset with mixed features.
    synthetic_data (pd.DataFrame): Synthetic dataset with mixed features.
    show_numbers (bool): Whether to display numbers in heatmap squares.

    Returns:
    None: Displays a heatmap of the divergence.
    """
    def mixed_correlation(df):
        """Compute pairwise correlations for mixed-type data."""
        columns = df.columns
        n = len(columns)
        corr_matrix = pd.DataFrame(np.zeros((n, n)), index=columns, columns=columns)

        for i, col1 in enumerate(columns):
            for j, col2 in enumerate(columns):
                if i <= j:  # Compute only once for each pair
                    if pd.api.types.is_numeric_dtype(df[col1]) and pd.api.types.is_numeric_dtype(df[col2]):
                        # Real-to-Real: Pearson correlation
                        corr = df[col1].corr(df[col2])
                    elif isinstance(df[col1].dtype, pd.CategoricalDtype) and isinstance(df[col2].dtype, pd.CategoricalDtype):
                        # Categorical-to-Categorical: Mutual Information
                        corr = mutual_info_score(df[col1].astype(str), df[col2].astype(str))
                    else:
                        # Real-to-Categorical: Bin the numerical column and compute Mutual Information
                        if pd.api.types.is_numeric_dtype(df[col1]):
                            try:
                                binned = pd.cut(df[col1].astype(float), bins=10, labels=False)
                                corr = mutual_info_score(binned, df[col2].astype(str))
                            except ValueError:
                                corr = 0  # Handle empty or incompatible data
                        else:
                            try:
                                binned = pd.cut(df[col2].astype(float), bins=10, labels=False)
                                corr = mutual_info_score(binned, df[col1].astype(str))
                            except ValueError:
                                corr = 0  # Handle empty or incompatible data
                    corr_matrix.iloc[i, j] = corr
                    corr_matrix.iloc[j, i] = corr
        return corr_matrix

    # Ensure all categorical columns in synthetic and real data are categorical
    def ensure_categorical(df, reference_df, vmin = 0, vmax = 1):
        for col in reference_df.select_dtypes(include=['category']).columns:
            df[col] = df[col].astype('category')

    ensure_categorical(synthetic_data, real_data)

    # Compute correlation matrices for real and synthetic data
    real_corr = mixed_correlation(real_data)
    synthetic_corr = mixed_correlation(synthetic_data)

    # Compute absolute divergence
    divergence = np.abs(real_corr - synthetic_corr)

    # Plot heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(divergence, annot=show_numbers, cmap="Greens", cbar=True, 
                fmt=".2f" if show_numbers else None, linewidths=0.5, linecolor='white', 
                vmin=vmin,  # Set minimum value for the colour scale
                vmax=vmax   # Set maximum value for the colour scale
               )
    plt.title(f"Absolute Divergence in Pairwise Correlations\n({title})")
    plt.show()


def kolmogorov_smirnov_test(real_col, synthetic_col):
    return ks_2samp(real_col, synthetic_col).statistic

def total_variation_distance(real_col, synthetic_col):
    minlength = max(len(np.unique(real_col)), len(np.unique(synthetic_col)))
    real_dist = np.bincount(real_col, minlength = minlength) / len(real_col)
    synthetic_dist = np.bincount(synthetic_col, minlength = minlength) / len(synthetic_col)
    return np.sum(np.abs(real_dist - synthetic_dist)) / 2

def pearson_correlation(real_col1, real_col2, synthetic_col1, synthetic_col2):
    real_corr = pearsonr(real_col1, real_col2)[0]
    synthetic_corr = pearsonr(synthetic_col1, synthetic_col2)[0]
    return abs(real_corr - synthetic_corr)


def contingency_similarity(real_cat1, real_cat2, synthetic_cat1, synthetic_cat2):
    real_contingency = pd.crosstab(real_cat1, real_cat2)
    synthetic_contingency = pd.crosstab(synthetic_cat1, synthetic_cat2)
    
    # Align shapes by reindexing
    real_contingency = real_contingency.reindex(index=synthetic_contingency.index, columns=synthetic_contingency.columns, fill_value=0)
    synthetic_contingency = synthetic_contingency.reindex(index=real_contingency.index, columns=real_contingency.columns, fill_value=0)
    
    return jensenshannon(real_contingency.values.flatten(), synthetic_contingency.values.flatten())


def bucketize_and_contingency(real_num, real_cat, synthetic_num, synthetic_cat, n_bins=10):
    discretizer = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='uniform')
    real_num_binned = discretizer.fit_transform(np.array(real_num).reshape(-1, 1)).astype(int).flatten()
    synthetic_num_binned = discretizer.fit_transform(np.array(synthetic_num).reshape(-1, 1)).astype(int).flatten()
    return contingency_similarity(real_num_binned, real_cat, synthetic_num_binned, synthetic_cat)

def evaluate_column_wise(real_data, synthetic_data, Columns, num_columns):
    results = {}
    for col in Columns:
        if col in num_columns:
            results[col] = kolmogorov_smirnov_test(real_data[col], synthetic_data[col])
        else:
            results[col] = total_variation_distance(real_data[col].astype('category').cat.codes,
                                                    synthetic_data[col].astype('category').cat.codes)
    return results

def evaluate_pair_wise(real_data, synthetic_data, columns, num_columns, cat_columns):
    results = {}
    for col1 in columns:
        for col2 in columns:
            if col1 == col2:
                continue
            if col1 in num_columns and col2 in num_columns:
                # print('num, num', col1, col2)
                results[(col1, col2)] = pearson_correlation(real_data[col1], real_data[col2],
                                                            synthetic_data[col1], synthetic_data[col2])
            elif col1 in cat_columns and col2 in cat_columns:
                # print('cat, cat', col1, col2)
                results[(col1, col2)] = contingency_similarity(real_data[col1], real_data[col2],
                                                               synthetic_data[col1], synthetic_data[col2])
            else:
                # print('cat, num', col1, col2)
                num_col = col1 if col1 in num_columns else col2
                cat_col = col2 if col2 in cat_columns else col1
                results[(col1, col2)] = bucketize_and_contingency(real_data[num_col], real_data[cat_col],
                                                                  synthetic_data[num_col], synthetic_data[cat_col])
    return convert_to_dataframe(results, columns, num_columns)


def convert_to_dataframe(results, columns, num_columns):
    # Sort columns with numerical columns first
    sorted_columns = sorted(columns, key=lambda x: (x not in num_columns, x))
    
    # Create a DataFrame indexed and columned by sorted columns
    df = pd.DataFrame(index=sorted_columns, columns=sorted_columns)
    
    # Fill the DataFrame with the results
    for (col1, col2), value in results.items():
        df.loc[col1, col2] = value

    return df

def plot_column_wise_heatmap(column_wise_results, title = None):
    columns = list(column_wise_results.keys())
    values = list(column_wise_results.values())
   
    df = pd.DataFrame(values, index=columns, columns=['KST/TVD'])
   
    plt.figure(figsize=(10, len(columns) * 0.5))
    sns.heatmap(df, annot=True, cmap='coolwarm', cbar_kws={'label': 'Difference'},
        vmin=0,   # Set minimum colour range
        vmax=1    # Set maximum colour range
)
    plt.title('Column-Wise Results (KST for numerical, TVD for categorical)' if title is None else f'Column-Wise Results (KST for numerical, TVD for categorical)\n {title}' )
    plt.xlabel('Metric')
    plt.ylabel('Columns')
    plt.show()


def load_json_as_dict(json_file_path):
    """
    Loads a JSON file and returns its contents as a dictionary.

    Parameters:
        json_file_path (str): The path to the JSON file.

    Returns:
        dict: The contents of the JSON file as a dictionary.
    """
    try:
        with open(json_file_path, 'r') as file:
            data = json.load(file)
        return data
    except FileNotFoundError:
        print(f"Error: The file at {json_file_path} was not found.")
    except json.JSONDecodeError as e:
        print(f"Error: Failed to decode JSON. {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

def pretty_print_dict(dictionary):
   """
   Pretty prints a dictionary in a readable JSON format.
   """
   print(json.dumps(dictionary, indent=4, sort_keys=True))


def load_data_files(quality_file: str, shape_file: str, trend_file: str, coverage_file: str):
    """
    Load data from quality.txt, shape.csv, trend.csv, and coverage.csv.

    Args:
        quality_file (str): Path to the quality.txt file.
        shape_file (str): Path to the shape.csv file.
        trend_file (str): Path to the trend.csv file.
        coverage_file (str): Path to the coverage.csv file.

    Returns:
        tuple: A tuple containing:
            - quality_data (list of str): Lines from quality.txt.
            - shape_data (pd.DataFrame): DataFrame from shape.csv.
            - trend_data (pd.DataFrame): DataFrame from trend.csv.
            - coverage_data (pd.DataFrame): DataFrame from coverage.csv.
    """
    # Load quality.txt
    try:
        with open(quality_file, 'r') as file:
            quality_data = [line.strip() for line in file.readlines()]
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {quality_file}")

    # Load shape.csv
    try:
        shape_data = pd.read_csv(shape_file)
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {shape_file}")

    # Load trend.csv
    try:
        trend_data = pd.read_csv(trend_file)
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {trend_file}")

    # Load coverage.csv
    try:
        coverage_data = pd.read_csv(coverage_file)
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {coverage_file}")

    return quality_data, shape_data, trend_data, coverage_data


def create_dataframe_from_nested_dict(nested_dict):
    """
    Creates a DataFrame where:
    - Columns are indexed by keys of the main dictionary.
    - Rows are indexed by the keys of the inner 'XGBClassifier' dictionary.

    Parameters:
        nested_dict (dict): The input nested dictionary.

    Returns:
        pd.DataFrame: The transformed DataFrame.
    """
    # Extract column keys (outer dictionary keys)
    columns = nested_dict.keys()
    index_name = list(nested_dict[list(columns)[0]].keys())[0]

    # Extract row keys from 'XGBClassifier' of the first column
    # (assuming all 'XGBClassifier' keys are the same across columns)
    first_key = next(iter(columns))
    row_keys = nested_dict[first_key][index_name].keys()

    # Create a DataFrame with row keys as the index
    df = pd.DataFrame(index=row_keys)

    # Populate the DataFrame with values from the nested dictionary
    for col in columns:
        df[col] = pd.Series(nested_dict[col][index_name])
     
    # Set the name for the row index
    df.index.name = index_name

    return df


def load_npy_file(file_path):
    try:
        data = np.load(file_path)
        print("Shape of the data:", data.shape)
        print("Data type:", data.dtype)
        
        return data
    except Exception as e:
        print("An error occurred:", e)





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
        self.ordinal_columns = None
        self.num_columns = None
        self.v = None
        self.rank_encoding = None
        self.cat_cols_like_num = None
        self.hierarchical_columns = None
        self.forced_to_cat_columns = None
        self.selected_columns = None # if given the other columns will be ignored!
        self.classifier_models = None # used only for IBM data
        self.original_columns = None
        

    def fit(self, df_train, 
            target_column = None, 
            normalize=True, 
            decorrelation = False, 
            json_path = None,
            cat_encoding = None,
            rank_encoding = True,
            v = 'principalcomponent',
            ordinal_columns = None,
            encoding_base_vector = {},
            hierarchical_columns = {},
            selected_columns = None,
            forced_to_cat_columns = [] #
           ):
        # cprint('The new one', color = 'purple')
        self.ordinal_columns = ordinal_columns
        self.decorrelation = decorrelation
        self.json_path = json_path
        self.rank_encoding = rank_encoding
        #cprint(f'Attention: Rank encoding is {rank_encoding}', color = 'red', bold = True)
        data = {}
        self.encoding_base_vector = encoding_base_vector
        self.hierarchical_columns = hierarchical_columns
        
     
        # Check if JSON file exists
        if json_path is not None and os.path.exists(self.json_path):
            # Load the JSON file
            with open(self.json_path, 'r') as file:
                data = json.load(file)
            data_name_ = data['name']
            #cprint(f'The name of data = {data_name_}', color = 'purple', bold = True)
            self.original_columns = data["column_names"]
            #cprint(f'Order of the columns is {self.original_columns}', color = 'purple', bold = True)
            if data_name_ == 'ibm_func':
                self.selected_columns = ['User','Zip',  'MCC', 'Card', 'Amount', 'Year', 'Month', 'Day', 'Time',  'Use Chip', 'Is Fraud?']
                self.ordinal_columns = ['Year', 'Month', 'Day', 'Time', 'Zip']
            # if data_name_ in ['default', 'default_equal']:
            #     self.ordinal_columns = [
            #                             'BILL_AMT1', 
            #                             'BILL_AMT2', 
            #                             'BILL_AMT3',
            #                             'BILL_AMT4', 
            #                             'BILL_AMT5', 
            #                             'BILL_AMT6'
            #                            ]
            #    cprint(f'ordinal_columns are {self.ordinal_columns}', color = 'red', bold = True)

            # Extract categorical column indices and names
            if "cat_col_idx" in data and "column_names" in data:
                self.columns = data["column_names"] if self.selected_columns is None else self.selected_columns
                self.cat_columns = [data["column_names"][idx] for idx in data["cat_col_idx"] if data["column_names"][idx] in self.columns]
                
            
            # Extract num column indices and names
            if "num_col_idx" in data and "column_names" in data:
                self.num_columns = [data["column_names"][idx] for idx in data["num_col_idx"] if data["column_names"][idx] in self.columns]


            
            '''if data["task_type"] != "regression":
                self.cat_columns.extend([data["column_names"][idx] for idx in data["target_col_idx"] if data["column_names"][idx] in self.columns])
            elif data["task_type"] == "regression":
                self.num_columns.extend([data["column_names"][idx] for idx in data["target_col_idx"] if data["column_names"][idx] in self.columns])
            else: 
                ttt = data["task_type"]
                raise ValueError(f"task_type = {ttt} is unknown")'''
                
        else:
            data_name_ = None
            self.columns = df_train.columns
            self.cat_columns = df_train.select_dtypes(include=['object']).columns
            self.num_columns = df_train.select_dtypes(include=['number']).columns



        if data_name_ and data_name_ in ['shoppers', 'shoppers_equal']:
            forced_to_cat_columns = ['SpecialDay', 'ProductRelated', 'Informational']
        elif data_name_ and data_name_ in ['beijing', 'beijing_equal']:
            forced_to_cat_columns = ['Is', 'Ir', 'Iws']
            #cprint(f'Data is Default and columns in {forced_to_cat_columns} are forced to be categorical', bold = True, color = 'red')
        # elif data_name_ and data_name_ in ['default']:
        #     forced_to_cat_columns = [
        #                              'BILL_AMT1', 
        #                              'BILL_AMT2', 
        #                              'BILL_AMT3', 
        #                              'BILL_AMT4', 
        #                              'BILL_AMT5', 
        #                              'BILL_AMT6'
        #                             ]
        #   cprint(f'Data is Default and columns in {forced_to_cat_columns} are forced to be categorical', bold = True, color = 'red')
        elif data_name_ and data_name_ in ['ibm_func']:
           #  selected_columns = ['User','Zip',  'MCC', 'Card', 'Amount', 'Year', 'Month', 'Day', 'Time',  'Use Chip', 
           # 'Is Fraud?']
            #cprint('Model trick is activated for IBM!', color = 'green', bold = True)
            #cprint('If it Needs to be change,', color = 'green', bold = True)
            #cprint('go to tabsyn.tools_1.py', color = 'blue', bold = True)

            # 1
            self.classifier_models = []
            #cprint('fitting the classifier helper', bold = True)
            model_zip_state = CustomClassifier(model='random')
            model_zip_state.fit(df_train, C = ['Zip'], target = 'Merchant State')
            self.classifier_models.append(model_zip_state) 
            #cprint('classifier: Zip ---> Merchant State', bold = True)
         
            # 2         
            model_zip_city = CustomClassifier(model='random')
            model_zip_city.fit(df_train, C = ['Zip'], target = 'Merchant City')
            self.classifier_models.append(model_zip_city) 
            #cprint('classifier: Zip ---> Merchant City', bold = True)
         
            # 3
            model_MCC_name = CustomClassifier(model='random')
            model_MCC_name.fit(df_train, C = ['MCC'], target = 'Merchant Name')
            self.classifier_models.append(model_MCC_name) 
            #cprint('classifier: MCC ---> Merchant Name', bold = True)
        else: 
            self.selected_columns = selected_columns 

        self.selected_columns = selected_columns
        self.forced_to_cat_columns = forced_to_cat_columns if not selected_columns else [c for c in forced_to_cat_columns if c in selected_columns]
        if self.forced_to_cat_columns:
            cprint(f'Warning: The num columns in {forced_to_cat_columns} are treated as cat columns')


            
        if cat_encoding == 'ours':
            #print(f'encoding is {cat_encoding}')
            scaler_1 = StandardScaler()
            standardized_data = scaler_1.fit_transform(df_train[self.num_columns])
        
            if v == 'principalcomponent':
                pca_1 = PCA(n_components=1)  # Keep only the most principal component
                self.v = pca_1.fit_transform(standardized_data)
            elif v == 'Cauchy_sampling':
                #print('Encoding is based on Cauchy sampling')
                n_samples = standardized_data.shape[0]
                self.v = np.random.standard_cauchy(n_samples)
            elif v == 'max_correlation':
                 #print('Encoding is based on max_correlation!!')
                 self.v = 'max_correlation'
            else:
                self.v = None

     
        cat_columns_ = self.num_columns.copy()
        self.cat_columns_to_work = self.cat_columns + self.forced_to_cat_columns
        df_train_encoded = df_train[self.columns].copy()

     
        for col in self.cat_columns_to_work: #
            if self.ordinal_columns is not None and col in self.ordinal_columns:
                self.encoding_map[col] = {cat: int(cat) for cat in df_train[col].unique() if str(cat).isdigit()}
            elif col in self.encoding_base_vector:
                base_col = self.encoding_base_vector[col]
                v_base = df_train_encoded[base_col].to_numpy()
                self.encoding_map[col] = compute_category_map(df_train_encoded, col, cat_columns_, v_base, self.rank_encoding)
            elif self.v is None:
                self.encoding_map[col] = {cat: code for code, cat in enumerate(df_train[col].astype('category').cat.categories, start=1)}
            else:
                self.encoding_map[col] = compute_category_map(df_train_encoded, col, cat_columns_, self.v, self.rank_encoding)
             
            df_train_encoded[col] = df_train[col].map(self.encoding_map[col]).astype(float)
            if col not in self.forced_to_cat_columns:
                cat_columns_.append(col)
            
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
        df_test_encoded = df_test[self.columns].copy()
        for col in self.cat_columns_to_work:
            df_test_encoded[col] = df_test[col].map(self.encoding_map[col])
            
            # Handle unseen categories by assigning 2*max - mean + 1
            max_encoding_value = max(self.encoding_map[col].values(), default=0)
            mean_encoding_value = sum(self.encoding_map[col].values()) / len(self.encoding_map[col].values()) if self.encoding_map[col] else 0
            unseen_category_value = 2 * max_encoding_value - mean_encoding_value + 1
            # df_test_encoded[col].fillna(unseen_category_value, inplace=True)
            df_test_encoded[col] = df_test_encoded[col].fillna(unseen_category_value)

            
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
            #print('decorrelation is ON')
            df_decoded[:] = df_decoded @ self.U
        
        if self.scaler is not None:
            df_decoded[:] = self.inverse_transform_subset(df_decoded)
     
        
        for col, mapping in self.encoding_map.items():
         
            # it is here to support the case that we only passed a subset of all columns
            if col not in num_data.columns:
                continue 
             
            reverse_mapping = {code: cat for cat, code in mapping.items()}
            
            if self.rank_encoding:  # Rank encoding case
                min_key = min(reverse_mapping.keys())
                max_key = max(reverse_mapping.keys())
                df_decoded[col] = df_decoded[col].round().astype(int).apply(
                    lambda x: reverse_mapping.get(
                        min(max(x, min_key), max_key), reverse_mapping[min_key]
                    )
                )
            else:  # Mean encoding case
                sorted_items = sorted(mapping.items(), key=lambda item: item[1])
                category_keys, category_values = zip(*sorted_items)

                def interpolate_category(value):
                    if value <= category_values[0]:
                        return category_keys[0]
                    if value >= category_values[-1]:
                        return category_keys[-1]
                    
                    idx = bisect.bisect_left(category_values, value)
                    lower_value, upper_value = category_values[idx - 1], category_values[idx]
                    lower_key, upper_key = category_keys[idx - 1], category_keys[idx]
                    
                    proportion = (value - lower_value) / (upper_value - lower_value)
                    return random.choices([lower_key, upper_key], weights=[1 - proportion, proportion])[0]
                
                
                df_decoded[col] = df_decoded[col].apply(interpolate_category)
        if self.classifier_models:
            #cprint('fit and update by classifier')
            for model in self.classifier_models:
                print(df_decoded.columns)
                df_decoded = model.fit_and_update_df(df_decoded)
        #cprint(f'columns of the raw decoded data: {df_decoded.columns}', color = 'purple', bold = True)
        R = df_decoded[self.original_columns] if self.original_columns else df_decoded
        #cprint(f'Final columns: {R.columns}', color = 'purple', bold = True)
        return R
     
    def get_columns(self):
         if self.selected_columns:
             pass
             #print('Attention: selected_columns is active')
         """
         return columns, cat columns, and num columns 
         """
         return self.columns, self.cat_columns, self.num_columns


    
    def inverse_transform_subset(self, df_decoded):
        """
        Inverse transform only a subset of columns while keeping the others unchanged.
        """
        all_columns = self.columns  # The original column order used in fitting StandardScaler
        cols = list(df_decoded.columns)
        
        # Create a DataFrame with the same column order, filling missing columns with zeros
        df_padded = pd.DataFrame(np.zeros((df_decoded.shape[0], len(all_columns))), columns=all_columns)
        
        # Fill in the available columns from df_decoded
        for col in cols:
            df_padded[col] = df_decoded[col].values  # Keep only the required subset
        
        # Apply inverse transformation
        df_padded = self.scaler.inverse_transform(df_padded)
    
        # Convert back to DataFrame and extract only the requested subset of columns
        df_decoded_transformed = pd.DataFrame(df_padded, columns=all_columns)
    
        return df_decoded_transformed[cols]  # Return only the requested columns