import pandas as pd 
import numpy as np 
from sklearn.impute import KNNImputer

from sklearn.neighbors import BallTree
from sklearn import set_config
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import IsolationForest

from typing import Optional, List, Any, Union, Tuple
from src.data.load_data import train_dev_test_split, load_prepare_csv

RANDOM_STATE = 42 # GLOBALNA ZMIENNA ZIARNA GENERATORA DLA POWTARZALNOSCI WYNIKOW
set_config(transform_output = 'pandas')

class Winsorizer(BaseEstimator, TransformerMixin):
    """Zastępuje wartości mniejszościowe parametru podaną flagą 'OTHER' lub dla zmiennych numerycznych maksymalną spotykaną wielkością dla częstości.

    Args:
        variable (str): Nazwa kolumny, w której zliczane będą wycinkowe obserwacje.
        treshold (int): Progowa ilość wystąpień powyej której wartość nie ulega zmianom (zachowuje natywność).
    """

    def __init__(self, variable: str, treshold: int) -> None:
        self.common_appereances: Optional[List[Any]] = None
        self.variable = variable
        self.treshold = treshold

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'Winsorizer':
        """Analizuje kolumnę i identyfikuje wartości spełniające warunek częstotliwości wystąpień."""
        X = X.copy()
        tabl = X[self.variable].value_counts()
        self.common_appereances = list(tabl[tabl > self.treshold].index)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Zastępuje wartości poniżej progu na podstawie danych analitycznych (z metody fit)."""
        X = X.copy() # ENSURES THAT We won't overwrite original data frame
        
        if X[self.variable].dtype in ['float64', 'int64']:
            fillin = np.max(self.common_appereances)
            X[self.variable] = np.minimum(X[self.variable], fillin)
        if X[self.variable].dtype in ['object', 'category']:
             X[self.variable] = X[self.variable].where(X[self.variable].isin(self.common_appereances), 'OTHER')
        return X


class SpatialNeighborTransformer(BaseEstimator, TransformerMixin):
    """Transformer inżynierii cech zliczający bliskich sąsiadów geolokacyjnych punktu w podanym promieniu.

    Używa promienia Ziemi oraz transformacji haversine aby szybko zidentyfikować liczbę lokalizacji 
    w pobliżu klienta korzystając z instancji algorytmu BallTree na koordynatach z zestawu treningowego.

    Args:
        radius_km (int): Wyznacza domyślny promień wyszukiwań wyrażony w kilometrach.
    """

    def __init__(self, radius_km: int = 10) -> None:
        self.radius_km = radius_km
        self.earth_radius_km = 6371.0
        self.tree_: Optional[BallTree] = None
        self.train_coords_: Optional[np.ndarray] = None

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'SpatialNeighborTransformer':
        """Generuje główne drzewo wektorów w zdefiniowanej przestrzeni punktów bazowych treningowych."""
        # Convert degrees to radians for Haversine metric
        self.train_coords_ = np.radians(X[['latitude', 'longitude']])
        # Build the tree using only training data
        self.tree_ = BallTree(self.train_coords_, metric='haversine')
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Przypisuje zmienną 'neighbors_within_10km' zliczającą punkty treningowe z pobliskiego promienia na wejściowym requeście."""
        X = X.copy()
        X_coords = np.radians(X[['latitude', 'longitude']])
        
        # Query the tree (which contains ONLY training points)
        # r = radius / earth_radius
        counts = self.tree_.query_radius(
            X_coords, 
            r=self.radius_km / self.earth_radius_km, 
            count_only=True
        )

        # Check if we are transforming the training set (to avoid counting self)
        # We do this by checking if the input is the exact same object as fitted
        if X_coords is self.train_coords_:
            counts = counts - 1
            
        # Return as a 2D array or DataFrame for the pipeline
        X['neighbors_within_10km'] = pd.Series(counts, index=X.index)
        return X#count
    

class FeatureEngineerOne(BaseEstimator, TransformerMixin):
    """Obejmuje pierwszy proces ręcznej inżynierii cech, mapując konkretne dane operacyjne klientów na użyteczne metryki.

    Generuje binarne wartości klasyfikacyjne w kwestiach posiadania przez klienta zwrotów środków z salda 
    oraz oznacza ilość odnotowanych usług ekskluzywnych. Rekorduje również czy w zdefiniowanych zmiennych 
    wystąpiły wartości puste.

    Args:
        missing_cols (Optional[List[str]]): Lista wejściowa kolumn używanych do flagowania luk informatycznych.
        check_refund (bool): Generuje kolumnę z informacją boolean o braku zarejestrowanego zwrotu całkowiego środków.
        premium_columns (Optional[List[str]]): Szuka podanej listy usług by móc sumarycznie sparametryzować aktywność 'premium_services'.
    """

    def __init__(self, missing_cols: Optional[List[str]] = None, check_refund: bool = False, premium_columns: Optional[List[str]] = None) -> None:
        self.missing_cols = missing_cols if missing_cols else []
        self.check_refund = check_refund
        self.premium_columns = premium_columns if premium_columns else []

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'FeatureEngineerOne':
        """Brak operacji analitycznej - instancja Stateless."""
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Powołuje nową konfigurację i dołącza kolumny wygenerowane z instrukcji potoku na przekazanym rejestrze."""
        X = X.copy()

        for col in self.missing_cols:
            X[f'missing_{col}'] = (np.isnan(X[col])).astype(int)

        if self.check_refund:
            X['refund_present'] = (X['total_refunds'] == 0).astype(int)

        X['premium_services'] = ((X[self.premium_columns] == 'Yes').sum(axis=1) >= 3).astype(int)
        return X
        
class FeatureEngingeerTwo(BaseEstimator, TransformerMixin):
    """Zarządza drugim zestawem obróbki cech.

    Klasa wywołuje metodę kategoryzacji zmiennej ciągłej ('age_NA') na równomierne ilościowo koszyki rozkładu.

    Args:
        bins (Optional[Any]): Miejsca styku/limitów kubełkowania (zakres). Często domniemany wynik na zbiorze z `pd.qcut`.
        q (int): Docelowa ilość generowanych, wyodrębnionych klas podczas kategoryzacji cechy wiekowej. Domyślnie 6.
    """

    def __init__(self, bins: Optional[Any] = None, q: int = 6) -> None:
        self.bins = bins
        self.q = q

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'FeatureEngingeerTwo':
        """Generuje rozkład przedziałów przypisując wartości optymalnego progowania kubełków."""
        _, self.bins = pd.qcut(X['age_NA'], q=self.q, retbins=True)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Kategoryzuje próbki ciągłe wieku na uprzednio zdefiniowane parametry przedziałowe kwartyli."""
        X = X.copy()
        X['age_NA'] = pd.cut(X['age_NA'], bins=self.bins, include_lowest=True)
        # X['age_NA'] = X['age_NA'].factorize()
        return X
    
class IsolationForestTransformer(TransformerMixin, BaseEstimator):
    """Transformer wykorzystywany w celach bez-nadzorowego klasyfikowania obserwacji ekstremalnych/nietypowych.
    
    Model uczy się na podzbiorze ewaluowanych danych identyfikując próbki odosobnione po uwzględnieniu 
    parametru określającego oczekiwany procent danych nietypowych (`contamination`).
    
    Args:
        n_estimators (int): Ilość budowanych drzew u podstaw algorytmu ensemble.
        contamination (Union[str, float]): Szacowany stopień ustrukturyzowanych nietypowości w zbiorze.
        max_samples (Union[str, int, float]): Maksymalna dozwolona podpróba wyodrębniona w celu zbadania odizolowanego wariantu dla jednego elementu drzewa.
        max_features (Union[int, float]): Limit cech uwzględnianych w szkoleniu poszczególnych drzew.
        random_state (Optional[int]): Gniazdo losowości generowania algorytmu izolacji statystycznej.
    """

    def __init__(self, n_estimators: int = 100, contamination: Union[str, float] = 'auto', 
                 max_samples: Union[str, int, float] = 'auto', max_features: Union[int, float] = 1.0, 
                 random_state: Optional[int] = None) -> None:
        self.n_estimators = n_estimators
        self.contamination = contamination
        self.max_samples = max_samples
        self.max_features = max_features
        self.random_state = random_state
        self.isolation_forest: Optional[IsolationForest] = None

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'IsolationForestTransformer':
        """Buduje architekturę izolacji wyłapując krawędziowe nieregularności."""
        # TU UCZYMY MODEL
        self.isolation_forest = IsolationForest(
            n_estimators=self.n_estimators, 
            contamination=self.contamination,
            max_samples=self.max_samples, 
            random_state=self.random_state
        )
        self.isolation_forest.fit(X)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Kategoryzuje rzędowe obserwacje klasyfikując outliery flagą 0, natomiast próbki znormalizowane 1."""
        X = X.copy()
        X['outlier_label'] = self.isolation_forest.predict(X)
        X['outlier_label'] = X['outlier_label'].map({-1: 1, 1:0})
        return X

                
        