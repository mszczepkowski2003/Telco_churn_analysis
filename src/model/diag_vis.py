import matplotlib.pyplot as plt 
import matplotlib as mpl 
import seaborn as sns 
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve
import numpy as np 
from typing import List, Tuple, Any, Optional
import os 
from dotenv import load_dotenv

root_path = os.path.abspath(os.path.join(os.path.dirname(__file__),'..','..'))
env_path = os.path.join(root_path, '.env')


# This loads the variables from .env file
load_dotenv(dotenv_path=env_path)
COLORS = plt.rcParams['axes.prop_cycle'].by_key()['color']
BATCH_SIZE = os.getenv('BATCH_SIZE')

def plot_loss(history: Any, label: str, n: int) -> None: 
    """Rysuje wykres funkcji straty na podstawie historii uczenia modelu.

    Args:
        history (Any): Obiekt historii Keras zwracany przez metodę fit().
        label (str): Etykieta określająca zbiór (np. 'Base', 'Optuna').
        n (int): Indeks używany do wyboru koloru z predefiniowanej palety.
    """
    plt.semilogy(history.epoch, history.history['loss'],
                 color = COLORS[n], label = 'Train ' + label)
    plt.semilogy(history.epoch, history.history['val_loss'],
               color=COLORS[n], label='Val ' + label,
               linestyle="--")
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

def plot_metrics(history: Any, metrics: List[str] = ['loss', 'prc', 'precision', 'recall']) -> None:
    """Tworzy zbiór wykresów dla zadanych metryk z historii uczenia.

    Args:
        history (Any): Obiekt historii Keras zwracany przez metodę fit().
        metrics (List[str], optional): Lista nazw metryk z obiektu history. Domyślnie ['loss', 'prc', 'precision', 'recall'].
    """
    n_metrics = len(metrics)
    rows = n_metrics if n_metrics % 2 == 0 else n_metrics+1
    cols = 2
    plt.figure(figsize=(10,12), dpi = 150)
    for n, metric in enumerate(metrics):
        name = metric.replace("_"," ").capitalize()
        plt.subplot(rows,cols,n+1)
        plt.plot(history.epoch, history.history[metric], color=COLORS[0], label='Train')
        plt.plot(history.epoch, history.history['val_'+metric],
             label='Val', color = COLORS[1])
        plt.xlabel('Epoch')
        plt.ylabel(name)
        plt.legend()
     

    plt.tight_layout()



def plot_cm(labels: Tuple[np.ndarray, ...], 
            predictions: Tuple[np.ndarray, ...], 
            threshold: float = 0.5) -> None:
    """Rysuje macierz błędów (confusion matrix) dla jednego lub wielu zbiorów predykcji.

    Args:
        labels (Tuple[np.ndarray, ...]): Krotka rzeczywistych wartości zmiennej docelowej (np. dane treningowe i walidacyjne).
        predictions (Tuple[np.ndarray, ...]): Krotka predykcji z modelu w postaci prawdopodobieństw.
        threshold (float, optional): Próg odcięcia prawdopodobieństwa dla klasyfikacji. Domyślnie 0.5.
    """
    
    i = 1
    plt.figure(figsize=(22,10), dpi=150)
    
    if len(labels) > 1:
        plt.suptitle("Confusion Matrix dev vs train")
    for label, prediction in zip(labels, predictions):
        cm = confusion_matrix(label, prediction > threshold)
        plt.subplot(1,2,i)
        sns.heatmap(cm, annot=True, fmt="d")
        plt.title('Confusion matrix @{:.2f};'.format(threshold))
        plt.ylabel('Actual label')
        plt.xlabel('Predicted label')

        print('True Negatives: ', cm[0][0])
        print('False Positives: ', cm[0][1])
        print('False Negatives: ', cm[1][0])
        print('True Positives: ', cm[1][1])
        print('Total Customers that churned: ', np.sum(cm[1]))
        i+=1
    plt.tight_layout()



def plot_roc(name: str, labels: np.ndarray, predictions: np.ndarray, **kwargs: Any) -> None:
    """Rysuje krzywą wskaźnika skuteczności ROC (Receiver Operating Characteristic).

    Args:
        name (str): Nazwa do wyświetlenia w legendzie dla przekazanej krzywej.
        labels (np.ndarray): Prawdziwe wartości zmiennej docelowej.
        predictions (np.ndarray): Przydzielone przez model prawdopodobieństwa.
        **kwargs: Opcjonalne argumenty słownikowe przekazywane do matplotlib.pyplot.plot.
    """
    fp, tp, _ = roc_curve(labels, predictions)

    plt.plot(100*fp, 100*tp, label=name, linewidth=2, **kwargs)
    plt.xlabel('False positives [%]')
    plt.ylabel('True positives [%]')
    # plt.xlim([-0.5,60])
    # plt.ylim([60,100.5])
    plt.grid(True)
    ax = plt.gca()
    ax.set_aspect('equal')

def plot_prc(name: str, labels: np.ndarray, predictions: np.ndarray, **kwargs: Any) -> None:
    """Rysuje krzywą precyzji-czułości PRC (Precision-Recall Curve).

    Args:
        name (str): Nazwa do wyświetlenia w legendzie dla wizualizowanej krzywej.
        labels (np.ndarray): Prawdziwe wartości zmiennej objaśnianej.
        predictions (np.ndarray): Zwrócone prawdopodobieństwa (wynik modelu).
        **kwargs: Opcjonalne dodatkowe parametry estetyczne wykresu z matplotlib.
    """
    precision, recall, _ = precision_recall_curve(labels, predictions)

    plt.plot(precision, recall, label=name, linewidth=2, **kwargs)
    plt.xlabel('Precision')
    plt.ylabel('Recall')
    plt.grid(True)
    ax = plt.gca()
    ax.set_aspect('equal')


def get_model_summary(model: Any,
                      model_history: Any,
                      data: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray], 
                      b_size: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]: 
    """ Tworzy podsumowanie parametryczne i graficzne oceniające model.
    
    Generuje wykresy wybranych metryk poprzez funkcję `plot_metrics`, tworzy macierz 
    błędów oraz wizualizacje ROC i PRC wyświeltając pozycje na predykcjach zestawów w zbiorze.

    Args:
        model (Any): Wytrenowany model uczenia maszynowego (np. z Keras/Tensorflow).
        model_history (Any): Zwrócona wcześniej z modelu historia szkolenia po wywołaniu `fit()`.
        data (Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]): Próbki danych w proporcjach przeznaczonych do ewaluacji: (X_tr, y_tr, X_dev, y_dev).
        b_size (Optional[int], optional): Rozmiar partii przetwarzanych danych. Domyślnie z `BATCH_SIZE` w środowisku.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Tuple dwu zestawów predykcji wykonanych na zestawie danych (na podzbiorach odpowiednio Train oraz Dev).
    """
    if b_size is None:
        b_size = int(BATCH_SIZE) if BATCH_SIZE else 32

    X_tr, y_tr, X_dev, y_dev = data
    plot_metrics(model_history)
    train_predictions_baseline = model.predict(X_tr, batch_size=b_size, verbose=0)
    test_predictions_baseline = model.predict(X_dev, batch_size=b_size, verbose=0)

    baseline_results = model.evaluate(X_dev, y_dev,
                                  batch_size=b_size, verbose=0,return_dict = True)
    for name, value in baseline_results.items():
        print(name, ': ', np.round(value,5))
    
    plot_cm((y_dev,y_tr),
        predictions= (test_predictions_baseline,train_predictions_baseline))
    
    plt.figure(figsize=(22,10), dpi = 150)
    plt.subplot(1,2,1)
    plot_roc("Train baseline", y_tr, train_predictions_baseline, color =COLORS[0])
    plot_roc("dev baseline", y_dev, test_predictions_baseline, color =COLORS[1], linestyle='--')
    plt.legend()
    plt.subplot(1,2,2)
    plot_prc("Train baseline", y_tr, train_predictions_baseline, color =COLORS[0])
    plot_prc("dev baseline", y_dev, test_predictions_baseline, color =COLORS[1], linestyle='--')
    plt.legend()

    return train_predictions_baseline, test_predictions_baseline

