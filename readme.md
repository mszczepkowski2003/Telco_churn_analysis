# Przewidywanie Odejść Klientów (Customer Churn) & Optymalizacja Retencji

![Python](https://img.shields.io/badge/Python-3.10-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.135-009688.svg)
![XGBoost](https://img.shields.io/badge/XGBoost-3.1.3-orange.svg)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.7-F7931E.svg)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-Data%20Source-336791.svg)
![Optuna](https://img.shields.io/badge/Optuna-Hyperparameter%20Tuning-blueviolet.svg)
![Docker](https://img.shields.io/badge/Docker-19.03-4662D1.svg)
[**IBM Telco Customer Churn Dataset**](https://accelerator.ca.analytics.ibm.com/bi/?perspective=authoring&pathRef=.public_folders%2FIBM%2BAccelerator%2BCatalog%2FContent%2FDAT00148)

> **Ważna uwaga dla rekruterów:** 
> **[Kliknij tutaj, aby zobaczyć Podsumowanie & Raport EDA (Quarto HTML)](https://churn-eda-report.netlify.app/)** 🌟  
> *Ten szczegółowy raport Quarto przeprowadza przez wszystkie etapy procesu eksploracji i oraz czyszczenia danych*

---

## Przegląd Projektu
Odejścia klientów to kluczowa metryka dla firm opartych na subskrypcjach. Ten projekt dostarcza kompleksowy pipeline Machine Learning do identyfikacji klientów zagrożonych rezygnacją z usług, co pozwala zespołom ds. retencji na proaktywne interwencje i efektywną alokację budżetów marketingowych.

Do projektu podszedłem z **perspektywy zorientowanej na biznes**, skupiając się nie tylko na samej dokładności modelu, ale również na zrozumieniu zależności biznesowych, które związane są z odejściem klientów

### Kluczowe Cechy:
1. **End-to-End Pipeline**: Od surowych zapytań SQL do zserializowanego potoku przetwarzania wstępnego w Scikit-learn.
2. **Zaawansowane Modelowanie**: Porównanie modeli głębokiego uczenia Keras/TensorFlow (MLP) z modelem XGBoost.
3. **Optymalizacja Hiperparametrów**: Zintegrowano **Optuna** z buforowaniem w SQLite do śledzenia eksperymentów i znajdowania optymalnych parametrów dla modelu XGBoost.
4. **API Webowe gotowe do Produkcji**: Finalny model opakowano w usługę **FastAPI** (`app/server.py`), która przyjmuje ścieżki do plików CSV i zwraca prawdopodobieństwa predykcji oraz predykcję klasy na ustalonym wcześniej progu.

---

## Stos Technologiczny
* **Język:** Python 3.10.19
* **Przetwarzanie Danych:** Pandas, NumPy, Scikit-learn (Pipelines, ColumnTransformers, Custom BaseEstimators)
* **Modelowanie:** XGBoost, TensorFlow/Keras
* **Tuning:** Optuna
* **Baza Danych:** PostgreSQL (z `psycopg2` & `SQLAlchemy`)
* **Budowa API:** FastAPI, Uvicorn, Pydantic
* **Konteneryzacja:** Docker
* **Dokumentacja i Raportowanie:** Quarto, Jupyter Notebooks, Plotly, Seaborn

---

## Modelowanie i Architektura
Aby sprostać niezbalansowaniu klas, które jest nieodłącznym elementem przewidywania Churnu:
- Zaprojektowano Niestandardowe Transformatory Scikit-learn do fizycznej imputacji brakujących danych (KNN i Random Forest), mapowania przestrzennego opartego na odległościach Haversine'a (`BallTree`) oraz detekcji anomalii (`IsolationForest`).
- Przetestowano różne architektury modeli, inicjalizując na wstępie niestandardowe wagi Output Biases ($b_0 = \log(pos/neg)$), aby zapobiec ślepemu przewidywaniu klasy większościowej przez sieci neuronowe podczas wczesnych epok.
- Wdrożono Optunę do rygorystycznego przeszukiwania z kroswalidacją (K-Fold CV) w celu optymalizacji hiperparametrów XGBoost.
- Całość bezpiecznie spakowano w niezmienny potok (Pipeline) zserializowany jako `.joblib`, co całkowicie zapobiega wyciekom danych w fazie predykcji.
- Aby zapewnić powtarzalność wyników i łatwość wdrożenia, projekt został w pełni skonteneryzowany i wystawiony jako usługa webowa:
 - Zaimplementowano serwer REST API, który obsługuje predykcje w czasie rzeczywistym. Dzięki automatycznej dokumentacji Swagger UI, integracja z frontendem jest natychmiastowa.
 - Dockerization: Całe środowisko zostało zamknięte w lekkim obrazie Dockera.


---

## 📁 Struktura Repozytorium
```text
.
├── app/
│   ├── final_model.joblib        # Seryjalizowany pipeline i model gotowy na produkcję
│   └── server.py                 # Aplikacja FastAPI udostępniająca endpoint /predict
├── data/                         # plik z subsetem danych testowych (Lokalnie pliki z surowymi danymi)
├── notebooks/                    # EDA & Eksperymenty ML
│   ├── exp_extra_data_eda.ipynb  # Strukturyzowanie danych dowiązywanych z innego źródła
│   ├── model_development.ipynb   # Porównywanie modeli, trening i tuning Optuną
│   └── report.ipynb              # Notatnik źródłowy dla raportu HTML Quarto
├── sql/                          # Skrypty SQL do migracji danych i tworzenia widoków
│   ├── 01_schema.sql
│   ├── 02_cleaning.sql
│   └── 03_views.sql
├── src/                          # Modułowy kod źródłowy
│   ├── data/                     # Skrypty do ładowania i strukturyzowania danych
│   ├── model/                    # Skrypty treningowe, logiki Optuny, architektury DL
│   └── pipeline/                 # Customowe Transformatory Sklearn & konfiguracja pipeline-u
├── .gitignore                    # Pliki do zignorowania przez git
├── Dockerfile                    # Dockerfile do budowania obrazu
├── .env                          # Plik z zmiennymi środowiskowymi
├── .dockerignore                 # Pliki do zignorowania przez docker
├── requirements.txt              # Zależności bibliotek w systemie Python
└── readme.md                     # Dokumentacja Projektu
```

---

## Jak uruchomić lokalnie (Docker)

Projekt jest w pełni skonteneryzowany. Aby go uruchomić, upewnij się, że masz zainstalowanego Dockera.

### 1. Budowa Obrazu
W głównym katalogu projektu uruchom następujące polecenie, aby zbudować obraz Dockera:
```bash
docker build -t churn-api-img .
```

### 2. Uruchomienie Kontenera
Po pomyślnym zbudowaniu obrazu, wystartuj kontener:
```bash
docker run --name churn-api 
-p 8000:8000 \
-v "filepath/21_12_25_customer_churn/data:/data" \ <-- Przekazywanie danych do kontenera
churn-api-img

```

### 3. Testowanie API
Po uruchomieniu wejdź pod adres [http://localhost:8000/docs](http://localhost:8000/docs), aby uzyskać dostęp do interaktywnego Swagger UI i przetestować endpoint `/predict`.

---
*Autor: [Michał Szczepkowski]*