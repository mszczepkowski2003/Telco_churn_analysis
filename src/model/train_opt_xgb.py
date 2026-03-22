from xgboost import XGBClassifier
from sklearn.metrics import ConfusionMatrixDisplay, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
import xgboost as xgb
import optuna
import numpy as np
import os
import sys
import gc
import pandas as pd
from typing import Any

root_path = os.path.abspath(os.path.join(os.path.dirname(__file__),'..', '..'))
if root_path not in sys.path:
     sys.path.append(root_path)
                        
from src.data.load_data import train_dev_test_split, load_db, structurize,load_prepare_csv
from src.pipeline.pipe_config import preprocessor

RANDOM_STATE = 42

def objective(trial: optuna.trial.Trial, X_train: pd.DataFrame, y_train: pd.Series, X_dev: pd.DataFrame, y_dev: pd.Series) -> float:
    """Funkcja celu podlegająca maksymalizacji podczas optymalizacji modelu XGBoost przez bibliotekę Optuna.
    
    Generuje i bada spersonalizowany model XGBClassifier stosując cross-walidację. W trakcie wykonywania prób wykorzystuje mechanizm `XGBoostPruningCallback` do odrzucania mało obiecujących przestrzeni hiperparametrycznych i zwalnia zasoby po każdej obróbce algorytmem z GC.
    
    Args:
        trial (optuna.trial.Trial): Obiekt reprezentujący pojedynczą próbę poszukiwania parametrów w badaniu Optuna.
        X_train (pd.DataFrame): Pula danych wykorzystywana do treningu w iteracjach K-Fold.
        y_train (pd.Series): Etykiety zbioru treningowego potrzebne w weryfikacji estymatora.
        X_dev (pd.DataFrame): Zbiór walidacyjny danych do pruningu (odrzucania).
        y_dev (pd.Series): Zbiór testowy do estymatora i pruningu modelu.

    Returns:
        float: Zwraca uśredniony pomiar ewaluacyjny (AUC score) wyciągnięty na cross-walidowanych folderach zbioru. 
    """
    # We use log=True for learning_rate because order of magnitude matters more than absolute value
    param = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 5000),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 9),
        
        # Stochastic parameters
        'subsample': trial.suggest_float('subsample', 0.5, 1.0), # Fraction of rows used by each tree
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),# Fraction of features used by each tree
        
        # Regularization
        # The minimum loss reduction required to make a further split. Higher values make the model more conservative.
        'gamma': trial.suggest_float('gamma', 1e-8, 1.0, log=True),
        
        # Centering weight search around your calculated ratio (approx 2.5)
        'scale_pos_weight': trial.suggest_float('scale_pos_weight', 2.0, 3.0),
        'eval_metric': 'auc'
    }

    pruning_callback = optuna.integration.XGBoostPruningCallback(trial, "validation-auc")
    clf = XGBClassifier(**param)
    pipe = Pipeline([('preprocessor', preprocessor), 
                     ('model', clf)])
    
   

    cv_score = cross_val_score(pipe, X_train, y_train, cv=5, scoring='roc_auc', n_jobs=-1)
    gc.collect()
    # Return the mean score across all 5 folds
    return cv_score.mean()

    # Your pipeline and fitting logic here...


if __name__ == '__main__': 
    print("Ładownanie danych...")
    df_ori = load_db("SELECT * FROM v_telcom_full_data")
    # Strukturyzacja
    df_structurized= structurize(df_ori)
    # Przygotowanie danych do podziału
    df_prepared_for_split = load_prepare_csv(data_frame=df_structurized,
                    cols_to_drop=['customer_id','churn_reason','churn_category', 'age', 'mean_income','zip_code','gender'])
    # Split na 3 zbiory, wykorzystując autorską funkcję
    X_train, X_dev, _, y_train, y_dev, _ = train_dev_test_split(df_prepared_for_split, RANDOM_STATE)
    preprocessor = preprocessor()
    # Dopasowanie i przekształcenie zbioru treningowego



    db_path = os.path.join(os.path.dirname(__file__), "optuna_xgb_results.db")
    storage_url = f'sqlite:///{db_path}'
    study = optuna.create_study(
        study_name="churn_opt_xgb_v1",
        storage = storage_url,
        direction='maximize',
        load_if_exists=True,
        pruner = optuna.pruners.MedianPruner()
    )

    func = lambda trial: objective(trial, X_train, y_train, X_dev, y_dev)
   
    print('Uruchamiam optymalizację...')
    study.optimize(func, n_trials = 50, n_jobs=1)

    print(f"Najlepsze AUC: {study.best_value}")
    print(f"Najlepsze parametry: {study.best_params}")