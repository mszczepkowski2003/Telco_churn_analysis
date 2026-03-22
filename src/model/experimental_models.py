import pandas as pd 
import os 
import sys
from dotenv import load_dotenv
from pathlib import Path 
import re 
import numpy as np 
import plotly.express as px 
import plotly.io as pio
pio.renderers.default = "notebook_connected" 
import matplotlib as mpl
import matplotlib.pyplot as plt 
import seaborn as sns 
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf 
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense, Dropout, Input, LeakyReLU, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping 
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Input
from sklearn.metrics import confusion_matrix, roc_curve,precision_recall_curve
import tempfile
import math
import shap 
from typing import Optional, Tuple, Dict, Any

from src.model.train_opt import get_metrics, early_stopping, lr_decay



def build_model_1(input_shape: int, learning_rate: float = 0.0001, output_bias: Optional[float] = None) -> Sequential:
    """Tworzy pierwszy eksperymentalny model sieci neuronowej.
    
    Prosty dwuwarstwowy model służący jako punkt wyjścia do analiz.
    Opcjonalnie przyjmuje początkową wartość biasu (output_bias) 
    dla warstwy wyjściowej, co pomaga w uczeniu się niezbalansowanych zbiorów 
    danych we wczesnych fazach treningu.

    Args:
        input_shape (int): Liczba cech wejściowych (wymiar wejścia).
        learning_rate (float, optional): Szybkość uczenia (learning rate) dla optymalizatora Adam. Domyślnie 0.0001.
        output_bias (Optional[float], optional): Inicjalizator biasu dla warstwy wyjściowej. Domyślnie None.

    Returns:
        Sequential: Skompilowany model sieci neuronowej Keras.
    """
    if output_bias is not None:
        output_bias = tf.keras.initializers.Constant(output_bias)
    model = Sequential()
    model.add(Input(shape=(input_shape,)))
    model.add(Dense(units=64,activation='relu'))
    model.add(Dense(units=32,activation='relu'))
    model.add(Dense(units=1, activation='sigmoid',
                    bias_initializer = output_bias)) #Ułatwia uczenie we wczesnych fazach treningu
    

    model.compile(optimizer=Adam(learning_rate=learning_rate),
                loss='binary_crossentropy',
                metrics=get_metrics())
                  
    return model 

def build_model_2(input_shape: int, units: Tuple[int, int, int], learning_rate: float = 0.0001, output_bias: Optional[float] = None) -> Sequential:
    """Tworzy drugi eksperymentalny model sieci neuronowej - głębszy i szerszy.
    
    Model składa się z trzech warstw ukrytych opartych na funkcji aktywacji 
    LeakyReLU (wspomagającej problem "martwych neuronów") oraz inicjalizacji 
    HeNormal poprawiającej zbieżność treningu.

    Args:
        input_shape (int): Liczba cech wejściowych.
        units (Tuple[int, int, int]): Krotka z trzema wartościami określającymi liczbę neuronów w kolejnych trzech warstwach układu (np. (64, 128, 64)).
        learning_rate (float, optional): Szybkość uczenia optymalizatora Adam. Domyślnie 0.0001.
        output_bias (Optional[float], optional): Inicjalny bias dla ostatniej warstwy. Domyślnie None.

    Returns:
        Sequential: Skompilowany model Keras z uwzględnionym zabezpieczeniem (clipnorm).
    """
    he_init = tf.keras.initializers.HeNormal()
    un_1, un_2, un_3 = units
    if output_bias is not None:
        output_bias_init = tf.keras.initializers.Constant(output_bias)
    else:
        output_bias_init = "zeros" # Default behavior

    model = Sequential()
    model.add(Input(shape=(input_shape,)))
    
    # Layer 1
    model.add(Dense(units=un_1, kernel_initializer=he_init))
    model.add(LeakyReLU(negative_slope=0.01)) # Prevents "Dead Neurons"
 
    # Layer 2
    model.add(Dense(units=un_2, kernel_initializer=he_init)) # Keep it wide longer
    model.add(LeakyReLU(negative_slope=0.01))

    # Layer 3
    model.add(Dense(units=un_3, kernel_initializer=he_init))
    model.add(LeakyReLU(negative_slope=0.01))
    
    # Output Layer
    model.add(Dense(units=1, activation='sigmoid',
                    bias_initializer=output_bias_init))
    
    # ADDED: clipnorm=1.0 to handle weighted gradient spikes
    model.compile(
        optimizer=Adam(learning_rate=learning_rate, clipnorm=1.0),
        loss='binary_crossentropy',
        metrics=get_metrics()# or your get_metrics()
    )
    return model 



def build_best_opt_model(best_params: Dict[str, Any], initial_bias: float, n_features: int) -> Sequential:
    """Buduje optymalny model sieci neuronowej bazując na zestawie najlepszych znalezionych hiperparametrów (Optuna).
    
    Implementacja parametrycznego generowania sieci: wielkość i liczba warstw (nodes/layers) uwarunkowane 
    są słownikiem wejściowym best_params. Funkcja dołącza również warstwy BatchNormalization oraz 
    Dropout zapobiegające przetrenowaniu (overfitting).

    Args:
        best_params (Dict[str, Any]): Słownik zawierający architekturę i hiperparametry odnalezione 
                                      podczas optymalizacji (np. n_layers, learning_rate, clip_norm, 
                                      i indywidualne dla każdej warstwy wartość dropout oraz liczbę węzłów).
        initial_bias (float): Wartość bazowa wyjścia obliczona w celu wspomagania niezbalansowanych klas.
        n_features (int): Ilość wejściowych cech (kolumn).

    Returns:
        Sequential: Skompilowany i zoptymalizowany pod dany wektor konfiguracyjny model sztucznej sieci neuronowej.
    """
    he_init = tf.keras.initializers.HeNormal()
    model = Sequential()
    model.add(Input(shape=(n_features,)))

    for i in range(best_params['n_layers']):
        model.add(Dense(best_params[f'nodes_l{i}'], kernel_initializer=he_init))
        model.add(BatchNormalization())
        model.add(LeakyReLU(negative_slope=0.01))
        model.add(Dropout(best_params[f'dropout_l{i}']))

    model.add(Dense(1, activation='sigmoid', bias_initializer=tf.keras.initializers.Constant(initial_bias)))
    model.compile(
        optimizer = Adam(learning_rate = best_params['learning_rate']
        , clipnorm= best_params['clip_norm'])
        , loss = 'binary_crossentropy'
        , metrics = get_metrics()
            )
    return model



