"""
app.py

API Flask pour les prédicitions du modèle final.

Fonctionnalités :

- /health : vérifie que l'API répond
- /predict :
    - prédire par ID client (SK_ID_CURR) à partir des jeux finaux exportés
    - ou prédire à partir d'un dictionnaire de features

Le modèle est chargé depuis MLflow via api/model_loader.py
et l'identifiant du run est fourni par variable d'environnement.
"""


import os
import pickle
from pathlib import Path

from dotenv import load_dotenv

import numpy as np
import pandas as pd
from flask import Flask, jsonify, request
from flask_cors import CORS

from .model_loader import load_model

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Charge les variables d'environnement depuis .env si le fichier existe.
load_dotenv(dotenv_path=PROJECT_ROOT / ".env", override=False)

# Cette partie sert à créer l'application web (objet principal qui va gérer les routes, les requêtes HTTP, les réponses, etc..)
app = Flask(__name__)
CORS(app)

# Configuration
THRESHOLD = float(os.getenv("THRESHOLD", "0.51"))  # seuil métier (on utilise le seuil optimal trouvé pendant la modèlisation)
ID_COL = os.getenv("ID_COL", "SK_ID_CURR")

DATA_FINAL_DIR = PROJECT_ROOT / "data" / "final_data"
CLIENTS_PKL = os.getenv("CLIENTS_PKL", "clients_production_sample.pkl")
SHAP_PKL = os.getenv("SHAP_PKL", "shap_top10.pkl")


# Chargements au démarrage
model = load_model()

clients_path = DATA_FINAL_DIR / CLIENTS_PKL
if not clients_path.exists():
    raise FileNotFoundError(
        f"Clients dataset not found: {clients_path}. "
    )

with open(clients_path, "rb") as f:
    df_clients = pickle.load(f)

# Chargement SHAP
shap_path = DATA_FINAL_DIR / SHAP_PKL

df_shap = None
if shap_path.exists():
    with open(shap_path, "rb") as f:
        df_shap = pickle.load(f)
    print(f"Loaded SHAP top10: {shap_path}")
else:
    print(f"SHAP file not found (optional): {shap_path}")


def get_features_by_id(sk_id: int) -> pd.DataFrame:

    """
    Récupère 1 ligne de features (DataFrame) pour un client à l'aide d'un l'ID.
    On cherche d'abord dans train (sans TARGET), puis dans test.
    """

    client = df_clients[df_clients[ID_COL] == sk_id]
    if not client.empty:
        return client

    raise KeyError(f"{ID_COL}={sk_id} introuvable dans le dataset production.")

# --------------------------------------------------------------------------------------------------------------------

def predict_proba_default(X: pd.DataFrame) -> float:

    """
    Retourne la probabilité de défaut (classe 1).

    Le modèle est chargé via mlflow.sklearn.load_model,
    on utilise donc predict_proba.
    """
    proba = model.predict_proba(X)[:, 1]
    return float(proba[0])


@app.get("/health") # Décorateur qui sert à relier une URL à une fonction
def health():
    """Endpoint de santé."""
    return jsonify(status="ok")


@app.post("/predict")
def predict():
    
    """
    Endpoint de prédiction du score de défaut.

    1) Prédiction par identifiant client (SK_ID_CURR)
    - Mode principal utilisé par le dashboard
    - Les features sont récupérées à partir des datasets finaux

    2) Prédiction par features (secondaire)
    - permettre des simulations de nouveaux clients
    - faciliter les tests unitaires
    """

    payload = request.get_json(force=True)

    # Méthode principale : prédire par ID
    if "sk_id_curr" in payload:
        sk_id = int(payload["sk_id_curr"])
        try:
            X = get_features_by_id(sk_id)
        except KeyError as e:
            return jsonify(error=str(e)), 404

    # Méthode secondaire : prédire par features (Utile pour les tests et les cas de nouveaux clients non présents sans id)
    elif "features" in payload:
        X = pd.DataFrame([payload["features"]])
    else:
        return jsonify(error="Le payload doit contenir 'sk_id_curr' ou 'features'."), 400

    X = X.drop(columns=[ID_COL], errors="ignore")
    proba = predict_proba_default(X)

    # TARGET : 1 = défaut donc si proba >= seuil alors refus
    decision = int(proba >= THRESHOLD)  # 1 = REFUSE, 0 = ACCEPTE
    decision_label = "REFUSE" if decision == 1 else "ACCEPTE"

    shap_top = None
    if df_shap is not None and "sk_id_curr" in payload:
        shap_top = df_shap[df_shap["SK_ID_CURR"] == sk_id].to_dict("records")

    return jsonify(
        proba_default=proba,
        threshold=THRESHOLD,
        decision=decision,
        decision_label=decision_label,
        shap_top_features=shap_top,
    )

# Ce bloc permet de lancer le serveur uniquement lorsque le fichier est exécuté directement, et pas lorsqu’il est importé.
if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)