#evaluation/classifier.py
import numpy as np
import pandas as pd

from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import recall_score, f1_score
from xgboost import XGBClassifier
from imblearn.metrics import geometric_mean_score


class ClassificationEvaluator:
    """
    使用固定的分类器配置评估分类性能
    Parameters
    ----------
    random_state: int 随机种子
    """
     
    def __init__(self):
        pass

    def _build_classifiers(self, random_state):
        """构建分类器"""

        classifiers = {
            'SVC': LinearSVC(dual=False, class_weight='balanced', C=0.05, 
                             max_iter=1000, random_state=random_state, tol=1e-4),
        
            'MLP': MLPClassifier(hidden_layer_sizes=(64, 32), 
                             max_iter=1000, 
                             alpha=0.01, 
                             early_stopping=True,
                             validation_fraction=0.15,
                             n_iter_no_change=20,
                             random_state=random_state),
        
            'XGB': XGBClassifier(n_estimators=100, 
                             max_depth=2, 
                             learning_rate=0.05,
                             subsample=0.8,
                             colsample_bytree=0.8,
                             eval_metric='logloss',
                             random_state=random_state) 
        }

        """classifiers = {
            "LR": LogisticRegression(
                max_iter=1000,
                solver="liblinear",
                random_state=random_state
            ),

            "SVC": SVC(
                kernel="rbf",
                C=1.0,
                gamma="scale",
                random_state=random_state
            ),

            "MLP": MLPClassifier(
                hidden_layer_sizes=(64, 32),
                activation="relu",
                solver="adam",
                alpha=1e-3,
                max_iter=1000,
                early_stopping=True,
                validation_fraction=0.15,
                random_state=random_state
            ),

            "XGB": XGBClassifier(
                n_estimators=100,
                max_depth=2,
                learning_rate=0.05,
                subsample=1.0,
                colsample_bytree=1.0,
                eval_metric="logloss",
                random_state=random_state
                )
        }"""
        return classifiers
    
    def _calculate_metrics(self, y_true, y_pred):
        """计算分类指标"""
        return {
            "Recall": recall_score(y_true, y_pred, zero_division=0),
            "F1_score": f1_score(y_true, y_pred, zero_division=0),
            "G_mean": geometric_mean_score(y_true, y_pred)
        }

    def evaluate(self, X_train, y_train, X_test, y_test, random_state):
        """
        分类训练和评估
        """
        y_train = np.asarray(y_train).ravel()
        y_test = np.asarray(y_test).ravel()

        classifiers = self._build_classifiers(
            random_state=random_state
        )

        results = []

        for clf_name, clf in classifiers.items():
            
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            metrics = self._calculate_metrics(y_test, y_pred)

            results.append({
                "Classifier": clf_name,
                **metrics
            })

        return pd.DataFrame(results)