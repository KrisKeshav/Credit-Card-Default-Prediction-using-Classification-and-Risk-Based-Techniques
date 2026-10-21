# Credit Card Default Prediction & Risk Scoring

Predicts which credit card customers will default next month, then turns the prediction into risk
decisions: calibrated probability of default (PD), risk bands A–E, expected loss, and a cost-based
decision threshold. Includes a Streamlit app that scores a single customer and explains the result.

## Results (held-out test set, 5,050 customers, never used for any modelling decision)

| Metric | Value |
|---|---|
| Final model | LightGBM, trained on SMOTEENN-balanced data |
| F2 / recall / precision | 0.595 / 0.76 / 0.32 |
| ROC AUC / Gini / KS | 0.78 / 0.56 / 43 |
| Calibration | mean predicted PD 19.0% vs actual default rate 19.0% |
| Risk bands | default rate rises from 5% (band A) to 51% (band E); band E holds 53% of all defaulters |
| Expected loss (PD × LGD × EAD) | predicted 34.2M vs actual 33.6M |
| Cost-based threshold | 50% lower loss than flagging nobody, while flagging 34% of customers |

**Why F2?** Missing a defaulter costs far more than reviewing a good customer, so recall is weighted 2× precision.

## Approach

1. **Behavioural features** ([features.py](features.py)): months paid late, months 3+ late, worst delay,
   longest late-payment streak, credit utilization.
2. **Train / validation / test split (60/20/20).** Thresholds, tuning (Optuna), model choice and
   calibration use the validation set only. The test set is used once, for the final numbers.
3. **Preprocessing:** median imputation, scaling, one-hot encoding (fit on train only).
   **SMOTEENN** balances the training set only.
4. **Models compared:** Logistic Regression, XGBoost, LightGBM, MLP, Optuna-tuned LightGBM, soft-voting
   ensemble. LightGBM had the best validation F2; tuning and ensembling were tied with it, so the simpler
   model was kept.
5. **Risk layer:** Platt scaling turns scores into real PDs (SMOTEENN inflates raw scores), then KS/Gini,
   risk bands and expected loss are computed.
6. **Cost-based threshold:** flag a customer when `PD × LGD > (1 − PD) × profit margin`.
7. **Explainability:** SHAP, globally in the notebook and per customer in the app.

## Repository

| File | Contents |
|---|---|
| [credit_risk_model.ipynb](credit_risk_model.ipynb) | Main notebook: full pipeline, results and discussion |
| [features.py](features.py) | Feature engineering shared by the notebook and the app |
| [app.py](app.py) | Streamlit app: PD, risk band, decision, expected loss, top reasons |
| `models/credit_risk_model.joblib` | Saved preprocessor, model, calibrator, thresholds and band cut-offs |
| `outputs/predictions.csv` | Predictions for every customer in the scoring file |
| [experiments_original.ipynb](experiments_original.ipynb) | Original exploratory notebook (EDA, neural-net and stacking experiments) |
| [FC_Project_report.pdf](FC_Project_report.pdf) | Project report (revised edition) |
| `data/` | Training data (`train_dataset_final1.csv`) and scoring data (`validate_dataset_final.csv`) |

## How to run

```bash
pip install -r requirements.txt
# the dataset is included in data/
jupyter nbconvert --to notebook --execute --inplace credit_risk_model.ipynb   # ~2 minutes
streamlit run app.py
```

## Limitations

- LGD (75%) and profit margin (15%) are assumptions, not estimated from data.
- EAD uses the latest bill only; banks would also add part of the unused limit.
- A single data split: scores vary by about ±0.01 F2 between splits.
- `sex`, `marriage` and `age` are model inputs; a real lender would need a fair-lending review.
