from sqlalchemy import create_engine 
import pandas as pd 
import os 
import re 
from dotenv import load_dotenv
from pathlib import Path 
import sys
from sklearn.model_selection import train_test_split
import numpy as np
from typing import List, Optional, Tuple, Any
from fastapi import HTTPException

root_path = os.path.abspath(os.path.join(os.path.dirname(__file__),'..','..'))
env_path = os.path.join(root_path, '.env')

# Ładowanie zmiennych środowiskowych
load_dotenv(dotenv_path=env_path)


def load_db(query: str) -> pd.DataFrame:
    """Pobiera dane z bazy danych PostgreSQL do obiektu DataFrame.
    
    Funkcja wykorzystuje zmienne środowiskowe (DB_USER, DB_PASSWORD, itd.)
    do nawiązania połączenia za pomocą SQLAlchemy i zwraca wynik
    podanego zapytania SQL.

    Args:
        query (str): Ciąg znaków zawierający zapytanie SQL do wykonania.

    Returns:
        pd.DataFrame: DataFrame zawierający wyniki zapytania.
    """
    engine = create_engine(
    f"postgresql+psycopg2://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}@"
    f"{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
    )

    return pd.read_sql(query, engine)



def train_dev_test_split(df: pd.DataFrame, rs: int) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """Dzieli dane na trzy zbiory (treningowy, walidacyjny, testowy)
    z zachowaniem proporcji klas (stratify) dla zmiennej docelowej.

    Funkcja wykorzystuje moduł train_test_split z biblioteki sklearn
    do utworzenia zbiorów treningowego, testowego i walidacyjnego. 
    Stosuje stratyfikację ze względu na zmienną docelową, aby uniknąć 
    problemów z niezbalansowanymi klasami.
    
    Args:
        df (pd.DataFrame): DataFrame z danymi do podziału.
        rs (int): Ziarno losowości (random state) dla powtarzalności wyników.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]: 
            Krotka zawierająca odpowiednio (X_train, X_dev, X_test, y_train, y_dev, y_test).
    """
    X = df.drop(['target','churn_label'], axis = 1)
    y = df['target']
    X_train, X_s, y_train, y_s = train_test_split(X, y, test_size=0.2, random_state=rs, stratify=y)
    X_dev, X_test, y_dev, y_test = train_test_split(X_s, y_s,test_size=0.25, random_state=rs,stratify=y_s)
    return X_train, X_dev, X_test, y_train, y_dev, y_test


def structurize(df: pd.DataFrame) -> pd.DataFrame:
    """Przeprowadza pełen cykl strukturyzacji na zbiorze danych telekomunikacyjnych.

    Args:
        df (pd.DataFrame): DataFrame z surowymi danymi (v_telco_full_data).

    Returns:
        pd.DataFrame: Ustrukturyzowany DataFrame gotowy do dalszego przetwarzania.
    """
    
    df.loc[df['offer']=='None', 'offer'] = "Offer wasn't made"
    df.loc[df['interntet_service'] == 'No', 'internet_type'] = 'No internet'

    df['churn_category'] = df['churn_category'].apply(lambda x: 'did not churn' if re.match(r'^\s*$',x) else x)
    df['churn_reason'] = df['churn_reason'].apply(lambda x: 'did not churn' if re.match(r'^\s*$',x) else x)

    df['longitude'] = round(df['longitude'].astype('float64'),5)
    df['latitude'] = round(df['latitude'].astype('float64'),5)

    
    drop_cols = ['country', 'state', 'quarter', 'location_id', 'service_id', 'status_id','churn_score', 'customer_status']

    np.random.seed(42) # Ziarno generatora dla powtarzalnych wynikow losowania 
    idx_to_rep = np.random.randint(0, len(df)-1,round((len(df)-1)*0.2))
    df['age_NA'] = df['age']
    df.loc[idx_to_rep, 'age_NA'] = np.nan

 
    df = df[df['customer_status'] != 'Joined']
    df = df.drop(drop_cols, axis=1)
    
    print(f"Liczba rekordów po usunięciu nowych klientów: {len(df)}")

    return df 


def load_prepare_csv(cols_to_drop: List[str], data_frame: Optional[pd.DataFrame] = None, filepath: Optional[str] = None) -> pd.DataFrame:
    """Ładuje ustrukturyzowany plik CSV lub przyjmuje gotowy DataFrame i przygotowuje do podziału.
    
    Funkcja wykorzystuje bibliotekę pandas do usunięcia zbędnych 
    kolumn i utworzenia numerycznej zmiennej docelowej (target).
    
    Args: 
        cols_to_drop (List[str]): Lista nazw kolumn do usunięcia (zbędnych w potoku).
        data_frame (Optional[pd.DataFrame], optional): Wejściowy DataFrame. Domyślnie None.
        filepath (Optional[str], optional): Ścieżka do pliku CSV. Domyślnie None.

    Returns: 
        pd.DataFrame: Przetworzony DataFrame z dodaną kolumną zmiennej docelowej 'target'.
    """
    if filepath is not None:
        df = pd.read_csv(filepath)
    elif data_frame is not None:
        df = data_frame
    else:
        raise(ValueError('No input data'))
    
        
    df.drop(cols_to_drop, axis=1,inplace=True, errors='ignore')

    df['target'] = (df['churn_label'] == 'Yes').astype(int)


   

    return df


def prepare_data(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Kompleksowo przygotowuje surowe dane do działania modelu predykcyjnego.
    
    Funkcja łączy proces strukturyzacji z usuwaniem zbędnych kolumn. 
    W przypadku niezgodności struktury danych zwraca wyjątek HTTPException (błąd 400).

    Args:
        raw_df (pd.DataFrame): DataFrame zawierający surowe dane wejściowe.

    Returns:
        pd.DataFrame: DataFrame (X) zawierający wyłącznie cechy wejściowe dla modelu.

    Raises:
        HTTPException: W przypadku błędnej struktury danych (zła nazwa kolumny, typ itp.).
    """
    try:
        df= structurize(raw_df)
        # Przygotowanie danych do podziału
        df = load_prepare_csv(data_frame=df,
                        cols_to_drop=['customer_id','churn_reason','churn_category', 'age', 'mean_income','zip_code','gender'])
        X = df.drop(['target','churn_label'], axis = 1,errors='ignore') #
        return X
    except (ValueError, KeyError) as e:
        raise HTTPException(
            status_code = 400,
            detail = f'Błędna struktura danych wejściowych. Błędne nazwy kolumn lub zły typ: {str(e)}'
        )
     

  