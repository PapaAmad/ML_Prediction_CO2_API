from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
from pathlib import Path
import joblib
from sklearn.linear_model import LinearRegression
from datetime import datetime
import json
from fastapi.middleware.cors import CORSMiddleware


# ----------------------
# Chemins vers dossiers
# ----------------------
model_path = Path("models") / "co2_model.joblib"
data_folder = Path("data")
predictions_file = data_folder / "predictions.json"

# ----------------------
# Charger ou créer modèle
# ----------------------
if model_path.exists():
    model = joblib.load(model_path)
else:
    model = LinearRegression()

# ----------------------
# FastAPI
# ----------------------
app = FastAPI(title="CO2 Emissions Prediction API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins = ["*"],  # pour dev
    allow_credentials = True,
    allow_methods = ["*"],
    allow_headers = ["*"],
)

# ----------------------
# Pydantic pour validation des entrées
# ----------------------
class CO2Input(BaseModel):
    YearBuilt: int
    NumberofBuildings: int
    NumberofFloors: int
    PropertyGFATotal: float
    # ajoute d'autres colonnes si nécessaire

# ----------------------
# Fonction pour sauvegarder une prédiction dans JSON
# ----------------------
def save_prediction_json(data_dict, predicted_CO2):
    data_folder.mkdir(parents=True, exist_ok=True)
    row = data_dict.copy()
    row["predicted_CO2"] = predicted_CO2
    row["timestamp"] = datetime.utcnow().isoformat()

    if predictions_file.exists():
        with open(predictions_file, 'r') as f:
            all_preds = json.load(f)
    else:
        all_preds = []

    all_preds.append(row)

    with open(predictions_file, 'w') as f:
        json.dump(all_preds, f, indent=4)

# ----------------------
# Endpoints
# ----------------------
@app.get("/")
def home():
    return {"message": "API CO2 is running 🚀"}

@app.post("/predict")
def predict(input_data: CO2Input):
    df = pd.DataFrame([input_data.dict()])
    try:
        prediction = model.predict(df)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur prédiction: {str(e)}")

    pred_value = float(prediction[0])
    save_prediction_json(input_data.dict(), pred_value)

    return {"prediction_CO2": pred_value}

@app.get("/data")
def get_data(limit: int = 100):
    if not predictions_file.exists():
        return {"predictions": []}
    with open(predictions_file, 'r') as f:
        all_preds = json.load(f)
    # Retourner les dernières prédictions
    return {"predictions": all_preds[-limit:]}

@app.get("/metrics")
def metrics():
    if not predictions_file.exists():
        return {"total_predictions": 0, "average_CO2": None}
    with open(predictions_file, 'r') as f:
        all_preds = json.load(f)
    total = len(all_preds)
    avg = sum(pred["predicted_CO2"] for pred in all_preds) / total
    return {"total_predictions": total, "average_CO2": avg}

@app.get("/health")
def health():
    try:
        model_status = model is not None
        return {"status": "ok" if model_status else "error", "model_loaded": model_status}
    except:
        return {"status": "error", "model_loaded": False}

@app.post("/retrain")
def retrain():
    try:
        # Lire tous les CSV du dossier data
        all_files = list(data_folder.glob("*.csv"))
        if not all_files:
            raise HTTPException(status_code=400, detail="Aucun fichier CSV trouvé dans data/")

        df_list = [pd.read_csv(f) for f in all_files]
        data = pd.concat(df_list, ignore_index=True)

        # Colonnes d'entrée / sortie
        X_cols = ["YearBuilt", "NumberofBuildings", "NumberofFloors", "PropertyGFATotal"]
        y_col = "TotalCO2"  # Remplace par le nom exact de ta colonne cible dans tes CSV

        X = data[X_cols]
        y = data[y_col]

        global model
        model = LinearRegression()
        model.fit(X, y)

        # Sauvegarde du modèle
        joblib.dump(model, model_path)

        return {"status": "ok", "message": "Modèle réentraîné avec succès à partir des CSV"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur retrain: {str(e)}")


@app.get("/data-raw")
def get_raw_data():
    df = pd.read_csv(data_folder / "data_cleaned.csv")
    
    # Remplacer les NaN par une valeur acceptable pour JSON
    df = df.fillna("")  # ici on met une chaîne vide, tu peux aussi mettre 0 si c’est numérique

    # Convertir tous les types en Python natif pour éviter les float64 etc.
    data_dict = df.astype(object).to_dict(orient="records")
    return {"raw_data": data_dict}
