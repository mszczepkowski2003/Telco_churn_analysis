import tensorflow as tf
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense, Dropout, Input, BatchNormalization, LeakyReLU
from tensorflow.keras.callbacks import EarlyStopping,ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import optuna
import pandas as pd
import numpy as np
import os
import sys
import gc
from typing import List, Tuple, Dict, Any

root_path = os.path.abspath(os.path.join(os.path.dirname(__file__),'..', '..'))
if root_path not in sys.path:
     sys.path.append(root_path)
                        
from src.data.load_data import train_dev_test_split, load_db, structurize,load_prepare_csv
from src.pipeline.pipe_config import preprocessor

EPOCHS = 150
TRIALS = 300
BATCH_SIZE = 64
RANDOM_STATE = 42

def get_metrics() -> List[tf.keras.metrics.Metric]:
    """Inicjalizuje i zwraca listę metryk Keras do śledzenia podczas treningu.
    
    Obejmuje klasyczne metryki binarne, takie jak pole pod krzywą (AUC), 
    dokładność, precyzja, czułość oraz FBetaScore.

    Returns:
        List[tf.keras.metrics.Metric]: Lista metryk oceniających model klasyfikacji binarnej.
    """
    metrics = [
        tf.keras.metrics.BinaryCrossentropy(name='cross entropy'),  # same as model's loss
        tf.keras.metrics.MeanSquaredError(name='Brier score'),
        tf.keras.metrics.TruePositives(name='tp'),
        tf.keras.metrics.FalsePositives(name='fp'),
        tf.keras.metrics.TrueNegatives(name='tn'),
        tf.keras.metrics.FalseNegatives(name='fn'),
        tf.keras.metrics.BinaryAccuracy(name='accuracy'),
        tf.keras.metrics.Precision(name='precision'), # Percentage of predicted positives that were correctly classified
        tf.keras.metrics.Recall(name='recall'),       # Percentage of actual positives that were correctly classified
        tf.keras.metrics.AUC(name='auc'),
        tf.keras.metrics.AUC(name='prc', curve='PR'), # precision-recall curve
        tf.keras.metrics.FBetaScore(beta=1.0, threshold=0.5, name='f1_score')
    ]
    return metrics

def get_params(X_train: np.ndarray, y_train: pd.Series) -> Tuple[float, Dict[int, float]]: 
    """Oblicza początkową wartość biasu warstwy wyjściowej oraz wagi klas dla zbalansowania treningu.

    Args:
        X_train (np.ndarray): Pula danych treningowych (cechy).
        y_train (pd.Series): Etykiety zbioru treningowego.

    Returns:
        Tuple[float, Dict[int, float]]: Krotka zawierająca:
            - początkową wartość odchylenia (initial_bias).
            - słownik wag klasowych do parametryzacji procedury uczącej.
    """
    neg, pos = np.bincount(y_train)
    initial_bias = np.log([pos/neg]) 

    total = len(y_train)

    weight_for_0 = (1 / neg) * (total / 2.0)
    weight_for_1 = (1 / pos) * (total / 2.0)
    class_weight = {
        0: weight_for_0,
        1: weight_for_1
    }
    return initial_bias, class_weight

def create_model_opt(trial: optuna.trial.Trial, n_features: int, initial_bias: float) -> Sequential: 
    """Generuje dynamiczny model sieci neuronowej dla określonej konfiguracji zadanej przez Optuna.
    
    Funkcja buduje architekturę na podstawie przestrzeni prawdopodobnych parametrów zaproponowanych przez `trial`.

    Args:
        trial (optuna.trial.Trial): Obiekt reprezentujący pojedynczą próbę poszukiwania parametrów w badaniu Optuna.
        n_features (int): Wymiar wektora pojedynczej wejściowej próbki danych.
        initial_bias (float): Wyliczona wartość bazowa biasu zredukowania wpływu asymetrycznej dystrybucji zmiennej docelowej w pierwszej epoce.

    Returns:
        Sequential: Samodzielny, skompilowany model dla poszczególnej próby optymalizacyjnej Optuna.
    """
    he_init = tf.keras.initializers.HeNormal()
    n_layers = trial.suggest_int("n_layers", 2, 5)
    
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-1, log=True)
    gradient_clipping = trial.suggest_float('clip_norm', 0.1, 2.0)
    
    # Dodatkowa regularyzacja: 
    l2_reg = trial.suggest_float('l2_reg', 1e-5, 1e-2, log=True)
    model = Sequential()
    model.add(Input(shape=(n_features,)))

    for i in range(n_layers):
        n_nodes = trial.suggest_int(f"nodes_l{i}", 16,512, log=True)
        model.add(Dense(n_nodes, kernel_initializer=he_init,kernel_regularizer = tf.keras.regularizers.l2(l2_reg)))
        model.add(BatchNormalization())
        model.add(LeakyReLU(negative_slope=0.01))

        n_dropout = trial.suggest_float(f"dropout_l{i}", 0.1, 0.5)
        model.add(Dropout(n_dropout))

    
    model.add(Dense(1, activation='sigmoid', bias_initializer= tf.keras.initializers.Constant(initial_bias)))
    model.compile(
          optimizer = Adam(learning_rate = learning_rate, clipnorm= gradient_clipping),
          loss = 'binary_crossentropy',
          metrics = get_metrics()
    )
    
    return model

def early_stopping(patience: int) -> EarlyStopping:
    """Implementuje callback dla wczesnego zatrzymywania procedury uczenia.

    Args:
        patience (int): Epoki do przeczekania w przypadku stagnacji na wartości bazowej bez widzialnej poprawy uczenia.

    Returns:
        EarlyStopping: Skonfigurowany objekt wczesnego zakończenia obserwujący `val_auc`.
    """
    return EarlyStopping(
        monitor='val_auc', 
        patience=patience, 
        restore_best_weights=True,
        verbose=1
    )

def lr_decay() -> ReduceLROnPlateau:
    """Implementuje callback adaptacyjnego pomniejszania wielkości kroku uczenia w momentach stagnacji nauki z obserwacji `val_loss`.

    Returns:
        ReduceLROnPlateau: Konstrukt dla optymalizatora keras pomniejszania parametrów learning_rate.
    """
    return tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', 
        factor=0.2, 
        patience=5, 
        min_lr=1e-6,
        verbose=1
    )

def objective(trial: optuna.trial.Trial, 
              X_train: np.ndarray, y_train: pd.Series, 
              X_dev: np.ndarray, y_dev: pd.Series, 
              n_features: int, 
              class_weight: Dict[int, float], 
              initial_bias: float) -> float: 
    """Funkcja celu podlegająca maksymalizacji podczas pracy biblioteki Optuna.
    
    Inicjalizuje nową eksperymentalną iterację na zadanym modelu (uzyskiwanego funkcją 
    `create_model_opt`), rozpoczyna jego szkolenie, dodając specjalny `pruning_callback` 
    do przerywania beznadziejnych badań w środku egzekucji i zwraca walidacyjne maksymalne
    osiągnięcie (najczęściej odczytywanego jako ostateczne) na wymogu miary `val_auc`.

    Args:
        trial (optuna.trial.Trial): Parametrowana iteracja optymalizacyjna.
        X_train (np.ndarray): Pula ucząca danych.
        y_train (pd.Series): Etykiety zbioru trenignowego danych.
        X_dev (np.ndarray): Dane walidacyjne zbioru wykorzystywowana przy weryfikacji po-epocznej.
        y_dev (pd.Series): Zbiór wytycznych wyników zbioru z danymi w ewaluacji.
        n_features (int): Ilość atrybutów modelu.
        class_weight (Dict[int, float]): Definiowana waga parametrów zbalansowania położeń odchył z rozkładu proporcji.
        initial_bias (float): Wyliczona wartość bazowa odchylenia.

    Returns:
        float: Maksymalny uzsykany `val_auc` modelu w czasie szkolenia.
    """
    model_opt = create_model_opt(trial, n_features, initial_bias)

    pruning_callback = optuna.integration.TFKerasPruningCallback(trial, 'val_auc')
    callbacks = [EarlyStopping(monitor='val_auc', patience=12, restore_best_weights=True, verbose=0),
                 ReduceLROnPlateau(monitor = 'val_auc', factor = 0.2, patience=5, min_lr = 1e-6, verbose=0), 
                 pruning_callback]
   
    history = model_opt.fit(X_train,
                  y_train, 
                  epochs = EPOCHS,
                  batch_size = BATCH_SIZE, 
                  shuffle = True, # Zapobiega uczenia sie kolejnosci miesza obserwacje treningowe przed każdą epoch
                  validation_data = (X_dev, y_dev),
                  callbacks=callbacks,
                  class_weight = class_weight,
                  verbose =0
                 )
    scores = model_opt.evaluate(X_dev, y_dev, verbose = 0)
    val_auc = max(history.history['val_auc'])
    tf.keras.backend.clear_session()
    gc.collect()
    return val_auc
    




if __name__ == '__main__':

    print("Ładownanie danych...")
    df_ori = load_db("SELECT * FROM v_telcom_full_data")
    # Strukturyzacja
    df_structurized= structurize(df_ori)
    # Przygotowanie danych do podziału
    df_prepared_for_split = load_prepare_csv(data_frame=df_structurized,
                    cols_to_drop=['customer_id','churn_reason','churn_category', 'age', 'mean_income','zip_code','gender'])
    # Split na 3 zbiory, wykorzystując autorską funkcję
    X_train, X_dev, X_test, y_train, y_dev, y_test = train_dev_test_split(df_prepared_for_split, RANDOM_STATE)
    preprocessor = preprocessor()
    # Dopasowanie i przekształcenie zbioru treningowego
    X_train_proc = preprocessor.fit_transform(X_train,y_train)
    X_dev_proc = preprocessor.transform(X_dev)
    X_test_proc = preprocessor.transform(X_test)

    initial_bias, class_weight = get_params(X_train_proc, y_train)
    n_features = X_train_proc.shape[1]


    db_path = os.path.join(os.path.dirname(__file__), "optuna_results.db")
    storage_url = f'sqlite:///{db_path}'
    study = optuna.create_study(
                study_name = "churn_opt_v1",
                storage = storage_url,
                direction="maximize",
                load_if_exists=True,
                pruner = optuna.pruners.MedianPruner()) #Odbiornik informacji o słabych trialach z callbacka z objective
    func = lambda trial: objective(trial, X_train_proc, y_train, X_dev_proc, y_dev, n_features, class_weight, initial_bias)
   
    print('Uruchamiam optymalizację...')
    study.optimize(func, n_trials = TRIALS, n_jobs=1)

    print(f"Najlepsze AUC: {study.best_value}")
    print(f"Najlepsze parametry: {study.best_params}")