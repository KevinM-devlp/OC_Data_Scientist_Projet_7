"""
model_loader.py

Chargement du modèle final.

Ce module comprend 2 modes :

1) Utilisation de MLFlow
   - Charge un modèle enregistré par MLflow dans un run local (dossier mlruns).
   - Utilise MLFLOW_TRACKING_URI,MLFLOW_RUN_ID et MLFLOW_MODEL_ARTIFACT.

2) Utilisation de joblib
   - Charge un modèle figé exporté au format joblib.
   - Utilise un chemin par défaut : api/model_prod/model.joblib

Cela permet :
- de garder un mode "expérimentation" via MLflow en local
- d'avoir un chargement simple et stable en CI/prod sans dépendre de mlruns 
  (car tentative de push trop lourde pour le dossier mlruns)
"""

from pathlib import Path
import os
import joblib

import mlflow
import mlflow.sklearn



def load_model():

    """
    Charge le modèle de scoring avec deux méthodes possibles 
    soit MLflow si MLFLOW_RUN_ID est défini (usage local / dev) ou 
    Joblib (usage CI / prod), via api/model_prod/model.joblib.

    Retourne :

        Modèle prêt à être utilisé pour la prédiction.
    """

    # Tentative MLflow (en local)
    run_id = os.getenv("MLFLOW_RUN_ID")

    if run_id:
        tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)

        artifact_path = os.getenv("MLFLOW_MODEL_ARTIFACT", "model")
        model_uri = f"runs:/{run_id}/{artifact_path}"

        try:
            print("Loaded model via MLflow:", model_uri)
            return mlflow.sklearn.load_model(model_uri)

        except Exception:
            pass  

    # Joblib pour CI/prod
    project_root = Path(__file__).resolve().parents[1]
    joblib_path = project_root / "api" / "model_prod" / "model.joblib"

    if joblib_path.exists():
        print("Loaded model via joblib:", joblib_path)
        return joblib.load(joblib_path)

    # Léve une erreur si mlflow ou joblib ne fonctionnent pas.
    raise RuntimeError(
        "No model could be loaded (MLflow and Joblib model not found).\n"
        f"Expected joblib path: {joblib_path}"
    )