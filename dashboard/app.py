import os
import pickle
import joblib
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go


# Titre de page

st.set_page_config(
    page_title="Prêt à dépenser — Dashboard de scoring",
    layout="wide",
)


# Config projet

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_FINAL_DIR = PROJECT_ROOT / "data" / "final_data"
PIPELINE_PATH = PROJECT_ROOT / "api" / "model_prod" / "model.joblib"

API_URL = os.getenv("API_URL", "http://127.0.0.1:5000").rstrip("/")
CLIENTS_PKL = os.getenv("CLIENTS_PKL", "clients_production_sample.pkl")
ID_COL = os.getenv("ID_COL", "SK_ID_CURR")


# Variables (profil + comparaison)

PROFILE_VARS = [
    "CODE_GENDER", "AGE_YEARS", "CNT_CHILDREN", "CNT_FAM_MEMBERS",
    "FLAG_OWN_CAR", "FLAG_OWN_REALTY", "OCCUPATION_TYPE", "ORGANIZATION_TYPE",
    "AMT_INCOME_TOTAL", "AMT_ANNUITY", "AMT_GOODS_PRICE",
    "ANNUITY_INCOME_RATIO", "EXT_SOURCE_MEAN",
]

COMPARE_VARS = [
    "AGE_YEARS", "AMT_INCOME_TOTAL", "AMT_ANNUITY", "AMT_GOODS_PRICE",
    "ANNUITY_INCOME_RATIO", "EXT_SOURCE_MEAN", "CNT_CHILDREN", "CNT_FAM_MEMBERS",
]

LABELS = {
    "CODE_GENDER": "Genre",
    "AGE_YEARS": "Âge (années)",
    "CNT_CHILDREN": "Nombre d'enfants",
    "CNT_FAM_MEMBERS": "Taille du foyer",
    "FLAG_OWN_CAR": "Possède une voiture (0/1)",
    "FLAG_OWN_REALTY": "Propriétaire immobilier (0/1)",
    "OCCUPATION_TYPE": "Profession",
    "ORGANIZATION_TYPE": "Secteur d'activité",
    "AMT_INCOME_TOTAL": "Revenu annuel (€)",
    "AMT_ANNUITY": "Annuité (€)",
    "AMT_GOODS_PRICE": "Montant du bien (€)",
    "ANNUITY_INCOME_RATIO": "Taux d'effort (mensualité / revenu)",
    "EXT_SOURCE_MEAN": "Score externe moyen",
}

GENDER_MAP = {0: "Femme", 1: "Homme"}
BOOL_MAP = {0: "Non", 1: "Oui"}

# Helpers

@st.cache_data(show_spinner=False)
def load_clients() -> pd.DataFrame:
    p = DATA_FINAL_DIR / CLIENTS_PKL
    with open(p, "rb") as f:
        df = pickle.load(f)
    df = df.copy()
    if ID_COL in df.columns:
        df[ID_COL] = df[ID_COL].astype(int)
    return df

@st.cache_resource(show_spinner=False)
def load_pipeline():
    pipeline = joblib.load(PIPELINE_PATH)
    return pipeline

@st.cache_data(show_spinner=False)
def add_pred_labels(df: pd.DataFrame, threshold: float, id_col: str) -> pd.DataFrame:
    """
    Retourne une copie du df avec :
      - PROBA_DEFAULT_ALL : proba de défaut
      - PRED_DEFAULT_ALL  : 1 si proba >= threshold else 0
    """

    pipeline = load_pipeline()

    df_proba = df.copy()


    X_all = df_proba.drop(columns=[id_col], errors="ignore")
    proba_all = pipeline.predict_proba(X_all)[:, 1]
    df_proba["PROBA_DEFAULT"] = proba_all
    df_proba["PRED_DEFAULT"] = (df_proba["PROBA_DEFAULT"] >= float(threshold)).astype(int)

    return df_proba

@st.cache_data(show_spinner=False)
def api_predict(sk_id: int) -> dict:
    r = requests.post(f"{API_URL}/predict", json={"sk_id_curr": int(sk_id)}, timeout=30)
    r.raise_for_status()
    return r.json()

def format_value(col, val):
    if pd.isna(val):
        return "—"

    # Genre (0/1 -> libellé)
    if col == "CODE_GENDER":
        try:
            v = int(float(val))
            return GENDER_MAP.get(v, str(val))
        except Exception:
            return str(val)

    # Booléens (0/1 -> Oui/Non)
    if col in ["FLAG_OWN_CAR", "FLAG_OWN_REALTY"]:
        try:
            v = int(float(val))
            return BOOL_MAP.get(v, str(val))
        except Exception:
            return str(val)

    # Formats numériques
    if col in ["AMT_INCOME_TOTAL", "AMT_ANNUITY", "AMT_GOODS_PRICE"]:
        return f"{float(val):,.0f}".replace(",", " ")

    if col in ["AGE_YEARS", "CNT_CHILDREN", "CNT_FAM_MEMBERS"]:
        return f"{float(val):.0f}"

    if col in ["ANNUITY_INCOME_RATIO", "EXT_SOURCE_MEAN"]:
        return f"{float(val):.3f}"

    return str(val)

def build_ref_group(df, gender=None, age_range=None, income_range=None):
    ref = df.copy()

    # Genre
    if gender is not None and gender != "Tous" and "CODE_GENDER" in ref.columns:
        g = pd.to_numeric(gender, errors="coerce")  # "0"/"1" -> 0/1
        ref_gender = pd.to_numeric(ref["CODE_GENDER"], errors="coerce")
        ref = ref[ref_gender == g]

    # Age
    if age_range and "AGE_YEARS" in ref.columns:
        age = pd.to_numeric(ref["AGE_YEARS"], errors="coerce")
        ref = ref[(age >= age_range[0]) & (age <= age_range[1])]

    # Revenu
    if income_range and "AMT_INCOME_TOTAL" in ref.columns:
        inc = pd.to_numeric(ref["AMT_INCOME_TOTAL"], errors="coerce")
        ref = ref[(inc >= income_range[0]) & (inc <= income_range[1])]

    return ref


# Interface utilisateur (UI)

st.title("Dashboard de scoring crédit — Prêt à dépenser")
st.caption("Parcours : 1) score → 2) explication (SHAP) → 3) comparaison population / groupe similaire.")

df_clients = load_clients()

# Sidebar
with st.sidebar:
    st.header("Sélection client")
    client_id = st.selectbox("ID client", df_clients[ID_COL].sort_values().unique())
    st.divider()

    st.header("Groupe (prédiction)")
    pred_group = st.selectbox(
        "Comparer à",
        ["Tous", "Prédits solvables", "Prédits défaillants"],
        index=0
    )
    st.divider()

    st.header("Groupe similaire (filtres)")
    st.caption("Ces filtres servent à comparer le client à un sous-groupe.")

    gender_opt = "Tous"
    if "CODE_GENDER" in df_clients.columns:
        gender_opt = st.selectbox("Genre (CODE_GENDER)", ["Tous", "0", "1"], index=0)

    age_range = None
    if "AGE_YEARS" in df_clients.columns:
        amin = int(np.nanmin(df_clients["AGE_YEARS"]))
        amax = int(np.nanmax(df_clients["AGE_YEARS"]))
        age_range = st.slider("Âge", amin, amax, (amin, amax))

    income_range = None
    if "AMT_INCOME_TOTAL" in df_clients.columns:
        imin = float(np.nanmin(df_clients["AMT_INCOME_TOTAL"]))
        imax = float(np.nanmax(df_clients["AMT_INCOME_TOTAL"]))
        income_range = st.slider("Revenu annuel (€)", imin, imax, (imin, imax))

# Client
client_row = df_clients[df_clients[ID_COL] == int(client_id)].iloc[0]

# Appel de l'API
with st.spinner("Appel à l’API de scoring..."):
    pred = api_predict(int(client_id))

proba = float(pred["proba_default"])
threshold = float(pred["threshold"])
decision = int(pred["decision"])
decision_label = pred["decision_label"]
shap_top = pred.get("shap_top_features", [])

# Mise en page (1ére ligne)
row1_left, row1_right = st.columns([1.2, 1.0], gap="large")


# 1) Score + Profil
with row1_left:

    st.subheader("1) Score et décision")

    st.write(f"**Probabilité de défaut : {proba:.3f}** (seuil : {threshold:.2f})")
    fig_gauge = go.Figure(go.Indicator(
    mode="gauge+number",
    value=proba,
    number={"valueformat": ".3f"},
    gauge={
        "axis": {"range": [0, 1]},
        "threshold": {
            "line": {"width": 4},
            "thickness": 0.85,
            "value": threshold,
        },
        "bar": {"thickness": 0.35},
    },
    title={"text": f"Probabilité de défaut"}
    ))
    fig_gauge.update_layout(height=250, margin=dict(l=10, r=10, t=60, b=10))
    st.plotly_chart(fig_gauge, use_container_width=True)

    # WCAG: pas seulement la couleur -> texte + icône
    if decision == 1:
        st.error(f"Décision : **{decision_label}** (risque ≥ seuil)", icon="⛔")
    else:
        st.success(f"Décision : **{decision_label}** (risque < seuil)", icon="✅")

    st.caption("Le score est une probabilité. La décision est obtenue en comparant au seuil métier.")

with row1_right:
    st.subheader("2) Profil client (infos descriptives)")

    cols_ok = [v for v in PROFILE_VARS if v in df_clients.columns]
    profile_data = []
    for v in cols_ok:
        profile_data.append({
            "Information": LABELS.get(v, v),
            "Valeur": format_value(v, client_row[v]),
        })
    st.dataframe(pd.DataFrame(profile_data), use_container_width=True, hide_index=True)

st.divider()

# 2) SHAP + Comparaisons

# Mise en page (2ème ligne)
row2_left, row2_right = st.columns([1.2, 1.0], gap="large")

with row2_left:

    st.subheader("3) Explication locale (Top 10 facteurs)")

    if not shap_top:
        st.warning("Aucune donnée SHAP reçue pour ce client.")
    else:
        df_shap = pd.DataFrame(shap_top).copy()

        # Sécuriser les types
        df_shap["feature"] = df_shap["feature"].astype(str)
        df_shap["shap_value"] = pd.to_numeric(df_shap["shap_value"], errors="coerce")
        df_shap["feature_value"] = pd.to_numeric(df_shap["feature_value"], errors="coerce")
        df_shap = df_shap.dropna(subset=["shap_value", "feature"]).copy()

        # Nettoyage des noms
        def clean_feature_name(name: str) -> str:
            for prefix in ["num__", "cat__", "remainder__"]:
                if name.startswith(prefix):
                    return name[len(prefix):]
            return name

        df_shap["feature"] = df_shap["feature"].apply(clean_feature_name)

        # Colonnes utiles
        df_shap["abs_shap"] = df_shap["shap_value"].abs()
        df_shap["impact_sens"] = np.where(df_shap["shap_value"] >= 0, "Augmente le risque", "Réduit le risque")

        # Tri pour graphe (croissant => les plus importants en haut)
        df_shap_plot = df_shap.sort_values("abs_shap", ascending=True)

        x_vals = df_shap_plot["shap_value"].tolist()
        y_vals = df_shap_plot["feature"].tolist()

        max_abs = float(df_shap_plot["shap_value"].abs().max())
        x_lim = max(0.05, max_abs * 1.25)

        fig_shap = go.Figure(
            data=[
                go.Bar(
                    x=x_vals,
                    y=y_vals,
                    orientation="h",
                    text=[f"{v:.3f}" for v in x_vals],
                    textposition="outside",
                    cliponaxis=False,
                )
            ]
        )

        fig_shap.add_vline(x=0, line_width=2, line_dash="dash")

        fig_shap.update_layout(
            title="Top 10 variables expliquant la décision",
            xaxis_title="Impact SHAP (positif = + risque)",
            yaxis_title="Variable",
            height=420,
            margin=dict(l=10, r=10, t=60, b=10),
        )

        fig_shap.update_xaxes(range=[-x_lim, x_lim], zeroline=True, showgrid=True)

        st.plotly_chart(fig_shap, use_container_width=True, key="shap_plot")

        st.caption("Impact SHAP > 0 : augmente le risque. Impact SHAP < 0 : réduit le risque.")

with row2_right:

        st.subheader("Tableau SHAP (valeurs réelles + influence)")

        # Tableau (décroissant => top en premier)
        df_table = df_shap.sort_values("abs_shap", ascending=False)

        st.dataframe(
            df_table[["feature", "feature_value", "shap_value", "impact_sens"]],
            use_container_width=True,
            hide_index=True,
        )
st.divider()


st.subheader("4) Comparaison client vs population ")

df_clients_labeled = add_pred_labels(df_clients, threshold=threshold, id_col=ID_COL)
df_ref = build_ref_group(df_clients_labeled, gender=gender_opt, age_range=age_range, income_range=income_range)

# Filtre prédiction (solvable / défaillant)
if pred_group == "Prédits solvables":
    df_ref = df_ref[df_ref["PRED_DEFAULT"] == 0]
elif pred_group == "Prédits défaillants":
    df_ref = df_ref[df_ref["PRED_DEFAULT"] == 1]

st.caption(f"Groupe de comparaison : **{len(df_ref):,} clients**.".replace(",", " "))

# Choix de la variable pour l'histogramme
vars_ok = [v for v in COMPARE_VARS if v in df_clients.columns]
if not vars_ok:
    st.info("Aucune variable disponible pour la comparaison.")
else:
    var = st.selectbox("Variable à comparer (distribution)", vars_ok, index=0, format_func=lambda x: LABELS.get(x, x))

    ref_vals = pd.to_numeric(df_ref[var], errors="coerce").dropna()
    client_val = client_row[var]
    client_val = float(client_val) if pd.notna(client_val) else None

    if ref_vals.empty:
        st.warning("Pas de données valides pour afficher la distribution.")
    else:
        xmin, xmax = float(ref_vals.min()), float(ref_vals.max())
        pad = (xmax - xmin) * 0.05 if xmax > xmin else 1.0

        counts, edges = np.histogram(ref_vals.values, bins=30, range=(xmin, xmax))
        centers = (edges[:-1] + edges[1:]) / 2

        fig_hist = go.Figure()
        fig_hist.add_trace(go.Bar(
            x=centers.astype(float).tolist(),
            y=counts.astype(int).tolist(),
            name="Groupe"
        ))

        if client_val is not None:
            fig_hist.add_vline(x=client_val, line_width=3, line_dash="dash")
            fig_hist.add_annotation(x=client_val, y=1.0, yref="paper",
                                    text="Client", showarrow=False, xanchor="left")

        fig_hist.update_layout(
            title=f"Distribution — {LABELS.get(var, var)}",
            xaxis_title=LABELS.get(var, var),
            yaxis_title="Nombre de clients",
            height=420,
            margin=dict(l=10, r=10, t=60, b=10),
            bargap=0.02
        )
        fig_hist.update_xaxes(range=[xmin - pad, xmax + pad], showgrid=True)
        fig_hist.update_yaxes(range=[0, int(counts.max()) * 1.15], showgrid=True)

        st.plotly_chart(fig_hist, use_container_width=True, key=f"hist_{var}_{client_id}")


st.subheader("5) Comparaison sur une variable (client vs moyenne du groupe)")

vars_ok = [v for v in COMPARE_VARS if v in df_clients.columns]
if not vars_ok:
    st.info("Aucune variable disponible pour la comparaison.")
else:
    var2 = st.selectbox(
        "Variable à comparer (barres)",
        vars_ok,
        index=0,
        format_func=lambda x: LABELS.get(x, x),
        key="var_compare_bar"
    )

    client_v = pd.to_numeric(client_row[var2], errors="coerce")
    group_mean = pd.to_numeric(df_ref[var2], errors="coerce").mean()

    if pd.isna(client_v) or pd.isna(group_mean):
        st.warning("Données insuffisantes pour cette variable.")
    else:
        y_vals = [float(client_v), float(group_mean)]
        ymax = max(y_vals)
        pad_y = (abs(ymax) * 0.15) if ymax != 0 else 1.0

        fig_bar = go.Figure()
        fig_bar.add_trace(go.Bar(
            x=["Client", "Moyenne groupe"],
            y=y_vals,
            text=[format_value(var2, client_v), format_value(var2, group_mean)],
            textposition="outside",
            name="Valeur"
        ))

        fig_bar.update_layout(
            title=f"{LABELS.get(var2, var2)} — Client vs groupe",
            xaxis_title="",
            yaxis_title=LABELS.get(var2, var2),
            height=420,
            margin=dict(l=10, r=10, t=60, b=10),
        )
        fig_bar.update_yaxes(range=[min(0, min(y_vals)) - pad_y, ymax + pad_y])

        st.plotly_chart(fig_bar, use_container_width=True, key=f"bar_{var2}_{client_id}")

        # Ecart en pourcentage
        if group_mean != 0:
            delta_pct = (float(client_v) - float(group_mean)) / float(group_mean) * 100
            st.caption(f"Écart vs moyenne du groupe : **{delta_pct:+.1f}%**")

st.divider()
st.caption(
    "Accessibilité : titres structurés, texte explicatif (en plus de la couleur ou autre éléments visuels), graphiques lisibles et interactifs."
)