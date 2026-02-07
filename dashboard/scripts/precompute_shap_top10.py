"""
Pré-calcul des explications SHAP locales (Top 10) pour le modèle de scoring crédit.

Ce script est exécuté hors ligne et permet de générer, pour chaque client
du jeu de données de production (échantillon), les 10 variables les plus influentes
dans la décision du modèle (SHAP local).

Objectifs :

- Fournir au dashboard interactif des explications locales claires et compréhensibles
  pour chaque client.

Ce fichier est ensuite chargé par l’API de prédiction afin d’exposer les explications
locales au dashboard sans surcoût de calcul.
"""



import pickle
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PIPELINE_PATH = PROJECT_ROOT / "api" / "model_prod" / "model.joblib"
DATA_FINAL_DIR = PROJECT_ROOT / "data" / "final_data"


CLIENTS_PATH = DATA_FINAL_DIR / "clients_production_sample.pkl"
OUT_PATH = DATA_FINAL_DIR / "shap_top10.pkl"

ID_COL = "SK_ID_CURR"
TOPK = 10

# Récupéreration preprocessor + modèle depuis le pipeline
pipeline = joblib.load(PIPELINE_PATH)
preprocessor = pipeline.named_steps["preprocessing"]
model = pipeline.named_steps["classifier"]

# Données pour le dashboard
with open(CLIENTS_PATH, "rb") as f:
    df = pickle.load(f)

ids = df[ID_COL].astype(int).values
X = df.drop(columns=[ID_COL, "TARGET"], errors="ignore")

# Transformation
X_enc = preprocessor.transform(X)
X_enc = X_enc.toarray() if hasattr(X_enc, "toarray") else np.asarray(X_enc)

feature_names = preprocessor.get_feature_names_out()

# SHAP
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_enc)

# Selon versions: shap_values peut être list (2 classes) ou array
if isinstance(shap_values, list):
    # on prend la classe positive (1)
    shap_values_pos = shap_values[1]
else:
    shap_values_pos = shap_values

# Top10 par client
rows = []
for row_idx, client_id in enumerate(ids):

    # SHAP values pour ce client
    shap_values_client = shap_values_pos[row_idx]

    # Valeurs des features encodées pour ce client
    encoded_features_client = X_enc[row_idx]

    # Indices des TOP features par importance absolue
    top_feature_indices = np.argsort(
        np.abs(shap_values_client)
    )[-TOPK:][::-1]

    for feature_idx in top_feature_indices:
        rows.append({
            "SK_ID_CURR": int(client_id),
            "feature": str(feature_names[feature_idx]),
            "shap_value": float(shap_values_client[feature_idx]),
            "feature_value": float(encoded_features_client[feature_idx])
        })

df_shap = pd.DataFrame(rows)

with open(OUT_PATH, "wb") as f:
    pickle.dump(df_shap, f)

print("Saved:", OUT_PATH, df_shap.shape)