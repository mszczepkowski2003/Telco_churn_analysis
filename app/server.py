import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.data.load_data import  prepare_data
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
model_path = BASE_DIR / 'final_model.joblib'

app = FastAPI(title="Churn Prediction API", version="1.0")
# 1. Ładowanie "mózgu" (Pipeline: Preprocessing + XGBoost)
try:
    model = joblib.load(model_path)
    print("Załadowano model")
except Exception as e:
    print(f"Błąd ładowania modelu: {e}")

# Definicja struktury zapytania
class FilePath(BaseModel):
    path: str

@app.post('/predict')
def predict(data: FilePath):
    from sklearn import set_config
    set_config(transform_output="pandas")
  
    try: 
        df = pd.read_csv(data.path)
    except FileNotFoundError:
        raise HTTPException(
            status_code=404,
            detail=f'Nie znaleziono pliku w podanej lokalizacji{data.path}')
    # Strukturyzacja
    X = prepare_data(df)
    proba_preds = model.predict_proba(X)
    preds = model.predict(X)

    response = {
        'probabilities': proba_preds[:,1].tolist(),
        'predictions': preds.tolist()
    }

    return response

if __name__ == '__main__': 
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8000)
    