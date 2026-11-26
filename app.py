"""What-if explorer: edit a customer's contract, charges and tenure and watch the churn score.

Pick a customer from the validation split, change their features, and see the
model score, the targeting decision at the chosen threshold, and the SHAP
drivers update. Needs the models saved by analysis.ipynb.

Run: streamlit run app.py
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import shap
import streamlit as st

DATA = Path(__file__).resolve().parent / "data"
DOCS = DATA / "results"
MODELS_DIR = DATA / "models"
SEED = 42  # same seed as CONFIG in analysis.ipynb

NUMERIC = ["tenure", "MonthlyCharges", "TotalCharges"]
CATEGORICAL = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]
FEATURES = NUMERIC + CATEGORICAL


def shap_values_for(pipe, X: pd.DataFrame, background: pd.DataFrame, seed: int) -> tuple[np.ndarray, pd.DataFrame]:
    """Return SHAP values (positive class) and the transformed feature frame."""
    prep, model = pipe.named_steps["prep"], pipe.named_steps["model"]
    names = prep.get_feature_names_out()
    Xt = pd.DataFrame(prep.transform(X), columns=names, index=X.index)
    if hasattr(model, "coef_"):
        bg = pd.DataFrame(prep.transform(background), columns=names)
        sv = shap.LinearExplainer(model, bg).shap_values(Xt)
    else:
        sv = shap.TreeExplainer(model).shap_values(Xt)
    sv = np.asarray(sv)
    if sv.ndim == 3:  # (n, features, classes) for random forest
        sv = sv[:, :, 1]
    return sv, Xt


@dataclass
class Scorer:
    """The chosen model, its threshold and a SHAP background sample."""

    model: object
    explained: object  # uncalibrated pipeline of the same family, used for SHAP
    threshold: float
    name: str
    background: pd.DataFrame


def load_scorer() -> Scorer:
    """Load the model and threshold chosen in analysis.ipynb (Part 3)."""
    choice = json.loads((DOCS / "threshold.json").read_text())
    background = pd.read_csv(DATA / "train.csv")[FEATURES].sample(200, random_state=SEED)
    name = choice["chosen_model"]
    return Scorer(
        joblib.load(MODELS_DIR / f"{name}.joblib"),
        joblib.load(MODELS_DIR / f"{name.removesuffix('_calibrated')}.joblib"),
        choice["threshold"],
        name,
        background,
    )


def edit(
    row: pd.Series,
    contract: str | None = None,
    monthly: float | None = None,
    tenure: int | None = None,
    keep_total_consistent: bool = True,
    **other: str,
) -> pd.DataFrame:
    """Return a one-row frame with the requested feature changes.

    TotalCharges is reset to MonthlyCharges x tenure when either changes (if asked), so the
    edited customer stays internally consistent.
    """
    out = row[FEATURES].copy()
    if contract is not None:
        out["Contract"] = contract
    if monthly is not None:
        out["MonthlyCharges"] = float(monthly)
    if tenure is not None:
        out["tenure"] = int(tenure)
    for k, v in other.items():
        out[k] = v
    if keep_total_consistent and (monthly is not None or tenure is not None):
        out["TotalCharges"] = float(out["MonthlyCharges"]) * int(out["tenure"])
    return out.to_frame().T.astype({"tenure": int, "MonthlyCharges": float, "TotalCharges": float})


def score(scorer: Scorer, X: pd.DataFrame) -> float:
    """Model score (class-weighted random forest: a ranking score, not a calibrated probability)."""
    return float(scorer.model.predict_proba(X)[:, 1][0])


def drivers(scorer: Scorer, X: pd.DataFrame, top: int = 8) -> pd.DataFrame:
    """Top features pushing this customer's score up or down (SHAP, positive class)."""
    sv, Xt = shap_values_for(scorer.explained, X, scorer.background, seed=0)
    df = pd.DataFrame({"column": Xt.columns, "shap": sv[0]})
    df = df.reindex(df["shap"].abs().sort_values(ascending=False).index).head(top)
    df["feature"] = df["column"].str.replace(r"^(num|cat)__", "", regex=True)
    df["customer_value"] = [readable(X, c, Xt.iloc[0][c]) for c in df["column"]]
    return df[["feature", "customer_value", "shap"]].reset_index(drop=True)


def readable(X: pd.DataFrame, column: str, encoded: float) -> str:
    """The customer's raw value for a numeric column, or yes/no for a one-hot column."""
    name = column.removeprefix("num__")
    if name in NUMERIC:
        return f"{float(X.iloc[0][name]):,.2f}"
    return "yes" if encoded == 1 else "no"


st.set_page_config(page_title="Churn what-if", layout="wide")

CONTRACTS = ["Month-to-month", "One year", "Two year"]
INTERNET = ["DSL", "Fiber optic", "No"]
PAYMENT = ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]
YES_NO = ["No", "Yes"]


@st.cache_resource
def scorer():
    """Chosen model, threshold and SHAP background (loaded once)."""
    return load_scorer()


@st.cache_data
def customers():
    """Validation customers (the test split is never used here)."""
    return pd.read_csv(DATA / "val.csv").set_index("customerID")


try:
    S, val = scorer(), customers()
except FileNotFoundError:
    st.error("Models or data not found. Run analysis.ipynb first.")
    st.stop()

st.title("Churn what-if explorer")
st.caption(
    f"Model `{S.name}`, targeting threshold {S.threshold:.2f} (chosen on validation by expected value). "
    "Scores rank customers by risk; they are not calibrated probabilities. SHAP shows what the model "
    "relies on, not what would change a real customer's behaviour."
)

with st.sidebar:
    st.header("Customer")
    preset = st.radio("Start from", ["Highest-risk customers", "Any validation customer"], horizontal=False)
    base_scores = S.model.predict_proba(val.drop(columns="Churn"))[:, 1]
    order = val.index[base_scores.argsort()[::-1]]
    options = list(order[:50]) if preset == "Highest-risk customers" else list(val.index)
    cid = st.selectbox("Customer ID", options)
    row = val.loc[cid]
    st.caption(f"Actually churned: {'yes' if row['Churn'] == 1 else 'no'}")

    st.header("Change features")
    contract = st.selectbox("Contract", CONTRACTS, index=CONTRACTS.index(row["Contract"]))
    monthly = st.slider("Monthly charges ($)", 18.0, 120.0, float(row["MonthlyCharges"]), 0.5)
    tenure = st.slider("Tenure (months)", 0, 72, int(row["tenure"]))
    internet = st.selectbox("Internet service", INTERNET, index=INTERNET.index(row["InternetService"]))
    payment = st.selectbox("Payment method", PAYMENT, index=PAYMENT.index(row["PaymentMethod"]))
    support = st.selectbox(
        "Tech support",
        sorted(val["TechSupport"].unique()),
        index=sorted(val["TechSupport"].unique()).index(row["TechSupport"]),
    )
    consistent = st.checkbox("Set total charges = monthly x tenure", value=True)

original = edit(row, keep_total_consistent=False)
changes = dict(
    contract=contract,
    monthly=monthly,
    tenure=tenure,
    InternetService=internet,
    PaymentMethod=payment,
    TechSupport=support,
)
edited = edit(row, keep_total_consistent=consistent, **changes)
p0, p1 = score(S, original), score(S, edited)

c1, c2, c3 = st.columns(3)
c1.metric("Original score", f"{p0:.2f}", help="Before any edits")
c2.metric("Score after edits", f"{p1:.2f}", delta=f"{p1 - p0:+.2f}", delta_color="inverse")
c3.metric("Decision", "Send offer" if p1 >= S.threshold else "No offer", help=f"Offer if score >= {S.threshold:.2f}")

left, right = st.columns([3, 2])
d = drivers(S, edited, top=10)
d["direction"] = d["shap"].map(lambda v: "raises risk" if v > 0 else "lowers risk")
fig = px.bar(
    d.iloc[::-1],
    x="shap",
    y="feature",
    orientation="h",
    color="direction",
    color_discrete_map={"raises risk": "#c8553d", "lowers risk": "#2a6f97"},
    hover_data=["customer_value"],
    title="Top drivers of the edited customer's score (SHAP)",
    labels={"shap": "Contribution to score", "feature": ""},
)
left.plotly_chart(fig, width="stretch")
right.subheader("Edited customer")
shown = edited.T.rename(columns={edited.index[0]: "value"})
shown["original"] = original.T.iloc[:, 0]
right.dataframe(shown.astype(str), width="stretch")
