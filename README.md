# OC_Data_Scientist_Projet_7

## Intitulé du projet :

Vous êtes Data Scientist au sein d'une société financière, nommée "Prêt à dépenser", qui propose des crédits à la consommation pour des personnes ayant peu ou pas du tout d'historique de prêt.

L’entreprise souhaite mettre en œuvre un outil de “scoring crédit” pour calculer la probabilité qu’un client rembourse son crédit, puis classifie la demande en crédit accordé ou refusé. Elle souhaite donc développer un algorithme de classification en s’appuyant sur des sources de données variées (données comportementales, données provenant d'autres institutions financières, etc.).

De plus, les chargés de relation client ont fait remonter le fait que les clients sont de plus en plus demandeurs de transparence vis-à-vis des décisions d’octroi de crédit. Cette demande de transparence des clients va tout à fait dans le sens des valeurs que l’entreprise veut incarner.

Prêt à dépenser décide donc de développer un dashboard interactif pour que les chargés de relation client puissent à la fois expliquer de façon la plus transparente possible les décisions d’octroi de crédit, mais également permettre à leurs clients de disposer de leurs informations personnelles et de les explorer facilement. 

---

## Objectifs du projet

Construire un modèle de scoring qui donnera une prédiction sur la probabilité de faillite d'un client de façon automatique.

Construire un dashboard interactif à destination des gestionnaires de la relation client permettant d'interpréter les prédictions faites par le modèle, et d’améliorer la connaissance client des chargés de relation client.

Mettre en production le modèle de scoring de prédiction à l’aide d’une API, ainsi que le dashboard interactif qui appelle l’API pour les prédictions.

---

## Source des données


Lien vers les données : https://www.kaggle.com/c/home-credit-default-risk/data

---

## Architecture générale

Utilisateur -> Dashboard Streamlit (Cloud) -> requêtes HTTP -> API Flask (Render) -> Modèle(LightGBM)

- Le **dashboard** consomme l’API via des requêtes REST
- L’API est **indépendante** du dashboard
- Les prédictions sont effectués **dans le cloud**

## Structure du projet

```text
api/
├── app.py                # Endpoints /health et /predict
├── model_loader.py       # Chargement du modèle (MLflow ou joblib)
├── model_prod/
│   └── model.joblib      # Modèle final
└── requirements.txt      # Dépendances API

dashboard/
├── app.py                # Interface utilisateur Streamlit
└── requirements.txt      # Dépendances dashboard

data/
└── final_data/           # Jeux de données finaux (clients, SHAP, etc.)

notebooks/
└── requirements.txt      # Dépendances notebooks

requirements-dev.txt      # Dépendances développement (tests)
.github/workflows/ci.yml  # Pipeline CI (GitHub Actions)
README.md                 # Documentation du projet
```
---

## API de prédiction (Flask)

### Endpoints disponibles

- `GET /health`  
  Vérifie que l’API est opérationnelle.

- `POST /predict`  
  Retourne la probabilité de défaut, la décision associée et les variables SHAP.

L’API est déployée sur Render.

---

## Dashboard (Streamlit)

Le dashboard permet :

 - d’afficher le score de défaut et la décision

 - d’expliquer la prédiction via SHAP

 - de comparer un client à la population globale, à un groupe similaire (âge, genre, revenus) ou aux clients prédits solvables / défaillants

Le dashboard est déployé sur Streamlit Cloud et consomme l’API distante.

## Déploiement cloud

API : Render (Flask + Gunicorn)

Dashboard : Streamlit Cloud

CI/CD : GitHub Actions (tests automatisés)

## Tests et qualité

Tests automatisés avec pytest

Pipeline CI exécuté à chaque push

Chargement du modèle sécurisé (fallback MLflow → joblib)

## Auteur

Projet réalisé par Kevin M.
Dans le cadre du parcours Data Scientist – OpenClassrooms
