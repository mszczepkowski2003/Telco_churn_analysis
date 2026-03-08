from xgboost import XGBClassifier
import optuna 
import os 
import joblib
import pandas as pd 
from sklearn.pipeline import Pipeline
from typing import Optional, Union
from src.pipeline.pipe_config import preprocessor
from src.data.load_data import load_db, train_dev_test_split, structurize,load_prepare_csv

RANDOM_STATE = 42


def train_and_save_model(return_model: bool = False, save_model: bool = False) -> Optional[Pipeline]: 
    """Trenuje zoptymalizowany model XGBoost i zapisuje go na dysku.

    Funkcja kompleksowo przeprowadza proces przygotowania danych (pobranie,
    strukturyzacja, podział), ładuje najlepsze hiperparametry bazując 
    na dotychczasowych wynikach z Optuna, po czym buduje cały potok (Pipeline). 
    Gotowy model klasyfikacyjny zapisywany jest z użyciem biblioteki joblib 
    w głównym folderze projektu pod nazwą `final_model.joblib`.

    Args:
        return_model (bool, optional): Czy funkcja ma zwrócić zbudowany Pipeline?. Domyślnie False.
        save_model (bool, optional): Czy funkcja ma zapisać model na dysku?. Domyślnie False.

    Returns:
        Optional[Pipeline]: Potok uczący Scikit-learn wraz z ostatecznym estymatorem XGBClassifier lub `None`.
    """
    # Wczytanie danych z bazy 
    df_ori = load_db("SELECT * FROM v_telcom_full_data")
    # Strukturyzacja
    df_structurized= structurize(df_ori)
    # Przygotowanie danych do podziału
    df_prepared_for_split = load_prepare_csv(data_frame=df_structurized,
                    cols_to_drop=['customer_id','churn_reason','churn_category', 'age', 'mean_income','zip_code','gender'])
    # Split na 3 zbiory, wykorzystując autorską funkcję
    X_train, X_dev, _, y_train, y_dev, _ = train_dev_test_split(df_prepared_for_split, RANDOM_STATE)
    full_X  = pd.concat([X_train, X_dev])
    full_y = pd.concat([y_train, y_dev])

    base_dir = os.path.abspath(os.path.join(os.getcwd(), '..'))
    db_path = os.path.join(base_dir, 'src', 'model', 'optuna_xgb_results.db')
    storage_url = f"sqlite:///{db_path}"
    study = optuna.load_study(study_name='churn_opt_xgb_v1', storage=storage_url)
    print(f" Załadowano study! Najlepszy wynik: {study.best_value}")

    best_xgb_study = study.best_params
    fin_xgb = XGBClassifier(**best_xgb_study)

    fin_xgb = Pipeline([
        ('preprocessor', preprocessor()),
        ('classifier', fin_xgb)
    ])

    fin_xgb.fit(full_X, full_y)

    project_root = os.path.abspath(os.path.join(os.getcwd(), '..'))
    model_path = os.path.join(project_root, 'app/final_model.joblib')
    if save_model == True:
        joblib.dump(fin_xgb, model_path)
        print(f"Zakończono i zapisano model w {model_path}")
    out = None if not return_model else fin_xgb
    return out 
