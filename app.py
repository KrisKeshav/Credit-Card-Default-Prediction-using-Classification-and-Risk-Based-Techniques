"""Streamlit demo: score a credit card customer.

Run the notebook first (it saves models/credit_risk_model.joblib), then:
    streamlit run app.py
"""
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import streamlit as st

from features import add_features, BILL_COLS, PAY_AMT_COLS, PAY_STATUS_COLS, ID_COL

MODEL_PATH = Path("models/credit_risk_model.joblib")
SCORING_DATA = Path("data/validate_dataset_final.csv")

# Readable names for the explanation chart
LABELS = {
    "pay_0": "Repayment status, latest month", "pay_2": "Repayment status, 2 months ago",
    "pay_3": "Repayment status, 3 months ago", "pay_4": "Repayment status, 4 months ago",
    "pay_5": "Repayment status, 5 months ago", "pay_6": "Repayment status, 6 months ago",
    "LIMIT_BAL": "Credit limit", "age": "Age", "AVG_Bill_amt": "Average bill",
    "PAY_TO_BILL_ratio": "Payments ÷ bills", "delinquency_count": "Months paid late",
    "severe_delinquency_count": "Months 3+ months late", "max_delay": "Worst delay (months)",
    "max_delinquency_streak": "Longest late-payment streak", "credit_utilization": "Credit utilization",
    **{f"Bill_amt{i}": f"Bill, {i} month(s) ago" for i in range(1, 7)},
    **{f"pay_amt{i}": f"Payment, {i} month(s) ago" for i in range(1, 7)},
}


@st.cache_resource
def load_model():
    bundle = joblib.load(MODEL_PATH)
    bundle["explainer"] = shap.TreeExplainer(bundle["model"])
    return bundle


@st.cache_data
def load_customers():
    return pd.read_csv(SCORING_DATA)


def score(bundle, customer: pd.DataFrame):
    """Return calibrated PD and SHAP contributions for a one-row DataFrame."""
    x = bundle["preprocessor"].transform(add_features(customer)[bundle["features"]])
    raw = bundle["model"].predict_proba(x)[:, 1]
    logit = np.log(np.clip(raw, 1e-6, 1 - 1e-6) / (1 - np.clip(raw, 1e-6, 1 - 1e-6)))
    pd_value = bundle["calibrator"].predict_proba(logit.reshape(-1, 1))[0, 1]

    contrib = bundle["explainer"].shap_values(x)
    if isinstance(contrib, list):  # older SHAP returns one array per class
        contrib = contrib[1]
    return pd_value, pd.Series(contrib[0], index=bundle["feature_names"])


def risk_band(bundle, pd_value):
    return bundle["band_labels"][int(np.digitize(pd_value, bundle["band_edges"]))]


# ---------------------------------------------------------------------------
st.set_page_config(page_title="Credit Risk Scoring", layout="wide")
st.title("Credit card default risk scoring")

if not MODEL_PATH.exists():
    st.error("Model not found. Run `credit_risk_model.ipynb` first to create models/credit_risk_model.joblib.")
    st.stop()

bundle = load_model()
customers = load_customers()

# --- Inputs ----------------------------------------------------------------
st.sidebar.header("Customer")
customer_id = st.sidebar.selectbox("Start from an existing customer", customers[ID_COL])
base = customers.loc[customers[ID_COL] == customer_id].iloc[0]

limit_bal = st.sidebar.number_input("Credit limit", min_value=0, value=int(base["LIMIT_BAL"]), step=10000)
age = st.sidebar.number_input("Age", min_value=18, max_value=100, value=int(base["age"]))
sex = st.sidebar.selectbox("Sex (coded)", [0, 1], index=int(base["sex"]))
education = st.sidebar.selectbox("Education (coded)", list(range(7)), index=int(base["education"]))
marriage = st.sidebar.selectbox("Marital status (coded)", list(range(4)), index=int(base["marriage"]))

st.subheader("Last 6 months")
st.caption("Repayment status: -2/-1/0 = paid on time or no use, 1–8 = number of months late. Month 1 = most recent.")
history = pd.DataFrame({
    "month": range(1, 7),
    "repayment_status": [int(base[c]) for c in PAY_STATUS_COLS],
    "bill_amount": [float(base[c]) for c in BILL_COLS],
    "payment_amount": [float(base[c]) for c in PAY_AMT_COLS],
})
edited = st.data_editor(history, hide_index=True, disabled=["month"], width="stretch")

# --- Build the model input -------------------------------------------------
customer = base.drop(ID_COL).to_frame().T
customer[["LIMIT_BAL", "age", "sex", "education", "marriage"]] = [limit_bal, age, sex, education, marriage]
customer[PAY_STATUS_COLS] = edited["repayment_status"].to_numpy()
customer[BILL_COLS] = edited["bill_amount"].to_numpy()
customer[PAY_AMT_COLS] = edited["payment_amount"].to_numpy()

# The dataset ships AVG_Bill_amt and PAY_TO_BILL_ratio; recompute them only if amounts were edited.
if not edited[["bill_amount", "payment_amount"]].equals(history[["bill_amount", "payment_amount"]]):
    bills, payments = customer[BILL_COLS].sum(axis=1), customer[PAY_AMT_COLS].sum(axis=1)
    customer["AVG_Bill_amt"] = customer[BILL_COLS].mean(axis=1)
    customer["PAY_TO_BILL_ratio"] = (payments / bills.replace(0, np.nan)).fillna(0).round(2)
customer = customer.astype(float).astype({"sex": int, "education": int, "marriage": int})

# --- Score -----------------------------------------------------------------
pd_value, contributions = score(bundle, customer)
band = risk_band(bundle, pd_value)
flagged = pd_value >= bundle["f2_threshold"]
ead = max(customer["Bill_amt1"].iloc[0], 0)
expected_loss = pd_value * bundle["lgd"] * ead

st.subheader("Result")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Probability of default", f"{pd_value:.1%}")
c2.metric("Risk band", band, help=f"Actual default rate in band {band} on the test set: "
                                    f"{bundle['band_default_rates'][band]:.0%}")
c3.metric("Decision", "Flag for review" if flagged else "No action",
          help=f"Flagged when PD ≥ {bundle['f2_threshold']:.2f} (F2-optimal threshold)")
c4.metric("Expected loss", f"{expected_loss:,.0f}", help=f"PD × LGD ({bundle['lgd']:.0%}) × latest bill ({ead:,.0f})")

# --- Explanation -----------------------------------------------------------
st.subheader("Top reasons")
top = contributions.reindex(contributions.abs().sort_values(ascending=False).index).head(6)[::-1]
fig, ax = plt.subplots(figsize=(7, 3.2))
ax.barh([LABELS.get(n, n) for n in top.index], top.values,
        color=["#C44E52" if v > 0 else "#55A868" for v in top.values])
ax.axvline(0, color="grey", lw=0.8)
ax.set_xlabel("← lowers risk        raises risk →")
st.pyplot(fig, width="content")
st.caption(f"Model: {bundle['model_name']}. Bar length = SHAP contribution to the model's risk score.")
