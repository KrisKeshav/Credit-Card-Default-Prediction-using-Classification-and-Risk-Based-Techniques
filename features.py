"""Feature engineering shared by the notebook and the Streamlit app.

Keeping this in one file guarantees that training and scoring compute
features in exactly the same way.
"""
import numpy as np
import pandas as pd

ID_COL = "Customer_ID"
TARGET = "next_month_default"

# Repayment status, most recent month first (the dataset has no pay_1).
# -2/-1/0 = paid on time or no use; 1..8 = number of months payment is late.
PAY_STATUS_COLS = ["pay_0", "pay_2", "pay_3", "pay_4", "pay_5", "pay_6"]
BILL_COLS = [f"Bill_amt{i}" for i in range(1, 7)]
PAY_AMT_COLS = [f"pay_amt{i}" for i in range(1, 7)]

CATEGORICAL_COLS = ["sex", "marriage", "education"]

ENGINEERED_COLS = [
    "delinquency_count",
    "severe_delinquency_count",
    "max_delay",
    "max_delinquency_streak",
    "credit_utilization",
]


def longest_delinquency_streak(pay_status: pd.DataFrame) -> np.ndarray:
    """Longest run of consecutive late months (status >= 1) per customer."""
    streak = np.zeros(len(pay_status), dtype=int)
    best = np.zeros(len(pay_status), dtype=int)
    for col in pay_status.columns:
        late = pay_status[col].to_numpy() >= 1
        streak = np.where(late, streak + 1, 0)
        best = np.maximum(best, streak)
    return best


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add behavioural (repayment-history) features to a copy of df."""
    df = df.copy()
    pay = df[PAY_STATUS_COLS]

    df["delinquency_count"] = (pay >= 1).sum(axis=1)          # months paid late
    df["severe_delinquency_count"] = (pay >= 3).sum(axis=1)   # months 3+ months late
    df["max_delay"] = pay.max(axis=1)                         # worst delay seen
    df["max_delinquency_streak"] = longest_delinquency_streak(pay)
    df["credit_utilization"] = df["AVG_Bill_amt"] / df["LIMIT_BAL"]  # avg bill / limit
    return df


def feature_columns(df: pd.DataFrame) -> list:
    """All model inputs: everything except the ID and the target."""
    return [c for c in df.columns if c not in (ID_COL, TARGET)]
