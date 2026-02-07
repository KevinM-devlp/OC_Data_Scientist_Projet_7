"""
Tests unitaires de l'API de scoring crédit.

Objectifs :

- Vérifier que l'API est accessible (endpoint /health)
- Vérifier qu'une prédiction peut être réalisée à partir d'un ID client
- S'assurer que les probabilités retournées sont valides

Ces tests sont utilisés dans le cadre d'une démarche MLOps
pour garantir la stabilité du service de prédiction.
"""


import os
import pickle
import pytest
from pathlib import Path

try:
    from dotenv import load_dotenv
except Exception:
    load_dotenv = None

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Si .env est dispo, on utilise load_dotenv
if load_dotenv is not None:
    load_dotenv(dotenv_path=PROJECT_ROOT / ".env", override=False)

@pytest.fixture(scope="session", autouse=True)
def set_env():
    
    """
    Initialise les variables d'environnement nécessaires
    au chargement du modèle MLflow avant l'import de l'API.
    """

    # Pour MLflow en local si un run est fourni
    if os.getenv("MLFLOW_RUN_ID"):
        mlruns_path = PROJECT_ROOT / "mlruns"
        os.environ.setdefault("MLFLOW_TRACKING_URI", f"file:///{mlruns_path.as_posix()}")


    os.environ.setdefault("MLFLOW_MODEL_ARTIFACT", "model")
    os.environ.setdefault("THRESHOLD", "0.51")
    os.environ.setdefault("CLIENTS_PKL", "clients_production_sample.pkl")
    os.environ.setdefault("ID_COL", "SK_ID_CURR")


def test_health():

    """
    Test de disponibilité de l'API.
    Vérifie que l'endpoint /health répond correctement.
    """

    from api.app import app

    client = app.test_client()
    response = client.get("/health")

    assert response.status_code == 200
    assert response.get_json()["status"] == "ok"


def test_predict_by_id():

    """
    Test de prédiction à partir d'un identifiant client existant.
    Vérifie la présence et la cohérence des champs retournés.
    """

    from api.app import app

    # On lit le fichier pour récupérer un SK_ID_CURR existant
    data_final_dir = PROJECT_ROOT / "data" / "final_data"
    clients_pkl = os.getenv("CLIENTS_PKL", "clients_production_sample.pkl")

    with open(data_final_dir / clients_pkl, "rb") as f: 
        df_clients = pickle.load(f)

    id_col = os.getenv("ID_COL", "SK_ID_CURR")
    sk_id = int(df_clients[id_col].iloc[0])

    client = app.test_client()
    response = client.post("/predict", json={"sk_id_curr": sk_id}  )

    assert response.status_code == 200

    data = response.get_json()

    assert "proba_default" in data
    assert 0.0 <= float(data["proba_default"]) <= 1.0

    assert "decision" in data
    assert data["decision"] in [0, 1]

    assert "threshold" in data

