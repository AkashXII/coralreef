"""
train_xgboost.py — XGBoost Environmental Model Training
Coral Bleaching Detection System

This script trains the tabular environmental model and saves xgboost_v2.pkl.
The bundle contains both the trained XGBoost model and the feature name list
so inference always uses the exact same feature set.

Run:
    python train_xgboost.py --data data.csv

Output:
    xgboost_v2.pkl  — model bundle {model, features}

Dataset expected columns:
    Temperature_Mean    (Kelvin — converted to Celsius internally)
    SSTA_DHW            (degree heating weeks)
    SSTA                (sea surface temp anomaly)
    Turbidity
    Exposure            (Sheltered / Exposed / Sometimes)
    TSA                 (thermal stress anomaly)
    Windspeed
    Bleaching_condition (target — text label)
"""

import argparse
import numpy as np
import pandas as pd
import joblib
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
import xgboost as xgb


# ── Feature engineering ───────────────────────────────────────────────────────

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Builds 10 features from the raw dataset columns.

    Why these features:
    ─────────────────────────────────────────────────────────────────────────
    temperature_c   Raw sea temperature in Celsius. The dataset stores this
                    in Kelvin (around 297-303K) so we subtract 273.15.
                    Coral bleaching begins above ~28°C.

    dhw_log         Log-transformed degree heating weeks. DHW is the NOAA
                    standard metric for cumulative heat stress — it counts
                    how many weeks the temperature exceeded the bleaching
                    threshold by 1°C or more. We log-transform it because
                    the relationship between DHW and bleaching probability
                    is non-linear: going from 0→4 DHW matters more than
                    going from 12→16 DHW. Log compression makes this
                    relationship more learnable for the tree model.

    ssta_pos        Sea surface temperature anomaly, positive component only.
                    SSTA = how much warmer the water is than the long-term
                    average for that location and time of year. Negative SSTA
                    (cooler than normal) is protective — zeroing it out
                    prevents the model from treating cooling as stressful.

    ssta_neg        Negative SSTA (anomalous cooling), kept as a separate
                    feature. A large negative SSTA is actually a signal of
                    reef health — the model can use this to reduce bleaching
                    probability in cooling conditions.

    turbidity       Raw turbidity value. Higher turbidity = murkier water.
                    This affects light availability and thermal stratification,
                    both of which influence coral stress.

    sheltered       Binary: 1 if the site is sheltered (lagoon, reef flat),
                    0 if exposed to open ocean. Sheltered sites have less
                    wave-driven mixing, meaning thermal stress accumulates
                    more easily — but they also have different baseline
                    bleaching rates, so the model learns this context.

    windspeed       Wind speed. Higher wind drives wave mixing which distributes
                    heat through the water column and reduces surface thermal
                    stress. Low windspeed during a heatwave is a risk factor.

    temp_stress     Degrees above 26°C (not 28°C). We use 26 as the threshold
                    rather than 28 because coral physiological stress begins
                    slightly below the visual bleaching threshold. This creates
                    a gradual feature rather than a sharp step at 28°C.

    dhw_x_ssta      Interaction term: log(1+DHW) × SSTA_positive.
                    This captures the combined effect — high DHW alone is
                    dangerous, high SSTA alone is concerning, but both
                    together is far more dangerous than either individually.
                    Tree models can find interactions but making them explicit
                    helps with small datasets (1,252 rows).

    sheltered_stress  Interaction: (1-sheltered) × SSTA_positive.
                    An exposed site with high positive SSTA is specifically
                    risky because there's no shelter to buffer the thermal
                    stress. This feature captures that interaction directly.
    ─────────────────────────────────────────────────────────────────────────
    """
    out = pd.DataFrame()

    # Temperature: Kelvin → Celsius
    df["Temperature_Mean"] = pd.to_numeric(df["Temperature_Mean"], errors="coerce")
    out["temperature_c"]   = df["Temperature_Mean"] - 273.15

    # Degree heating weeks — log transform for non-linear relationship
    df["SSTA_DHW"]  = pd.to_numeric(df["SSTA_DHW"], errors="coerce")
    out["dhw_log"]  = np.log1p(df["SSTA_DHW"])

    # SSTA split into positive (warming) and negative (cooling) components
    df["SSTA"]       = pd.to_numeric(df["SSTA"], errors="coerce")
    out["ssta_pos"]  = df["SSTA"].clip(lower=0)
    out["ssta_neg"]  = (-df["SSTA"]).clip(lower=0)

    # Turbidity — raw value
    out["turbidity"] = df["Turbidity"]

    # Site exposure: sheltered=1, exposed/sometimes=0
    out["sheltered"] = df["Exposure"].map(
        {"Sheltered": 1, "Exposed": 0, "Sometimes": 0}
    ).astype(float)

    # Windspeed
    df["Windspeed"]   = pd.to_numeric(df["Windspeed"], errors="coerce")
    out["windspeed"]  = df["Windspeed"]

    # Thermal stress above 26°C (gradual, not a hard step at 28°C)
    out["temp_stress"] = (out["temperature_c"] - 26).clip(lower=0)

    # Interaction: DHW × SSTA — combined thermal stress signal
    out["dhw_x_ssta"] = out["dhw_log"] * out["ssta_pos"]

    # Interaction: exposed site under positive anomaly — specifically risky
    out["sheltered_stress"] = (1.0 - out["sheltered"]) * out["ssta_pos"]

    return out


FEATURES = [
    "temperature_c",
    "dhw_log",
    "ssta_pos",
    "ssta_neg",
    "turbidity",
    "sheltered",
    "windspeed",
    "temp_stress",
    "dhw_x_ssta",
    "sheltered_stress",
]


# ── Target encoding ───────────────────────────────────────────────────────────

def encode_target(df: pd.DataFrame) -> pd.Series:
    """
    Converts text bleaching condition to binary label.

    Original dataset has 3 classes:
        Mild      (1-10% bleached)   → 0 (healthy)
        Moderate  (11-50% bleached)  → 1 (bleached)
        Severe    (>50% bleached)    → 1 (bleached)

    We treat mild as healthy because 1-10% bleaching is within normal
    variation and not actionable from a conservation standpoint.
    Moderate and severe are both conservation concerns.
    """
    return df["Bleaching_condition"].apply(
        lambda x: 0 if "Mild" in str(x) else 1
    )


# ── Physics-based prior ───────────────────────────────────────────────────────

def physics_prior(temperature: float, dhw: float, ssta: float) -> float:
    """
    A scientifically grounded bleaching risk score [0, 1] built directly
    from NOAA coral bleaching alert thresholds.

    WHY WE NEED THIS:
    ─────────────────────────────────────────────────────────────────────────
    The XGBoost model learns from 1,252 reef site observations. This is
    a reasonable dataset but it has sparse coverage at the extremes —
    very low stress conditions (DHW≈0, temp≈18°C) and very high stress
    conditions (DHW>20) are underrepresented. When a tree model sees
    inputs outside its training distribution it extrapolates poorly,
    often getting stuck at its prior class probability (~0.52 for this
    dataset which is 55% bleached).

    The physics prior is a deterministic, monotonic function that always
    produces the correct direction of prediction:
        - DHW=0, temp=24, SSTA=0   → prior≈0.00  (no stress)
        - DHW=8, temp=28, SSTA=1   → prior≈0.54  (bleaching threshold)
        - DHW=16, temp=30, SSTA=3  → prior≈1.00  (extreme)

    We blend: final_xgb = 0.60 × xgb_raw + 0.40 × physics_prior

    This means:
        - In the well-represented middle of the distribution, XGBoost
          dominates (60%) and the prior provides stability (40%)
        - At the extremes, the monotonic prior pulls the output to a
          physically sensible value even when XGBoost extrapolates badly

    THRESHOLDS USED (from NOAA CoralWatch):
        DHW / 16:      8 DHW = bleaching alert level 1
                       16 DHW = bleaching alert level 2 (mortality risk)
                       We normalise by 16 so 8 DHW → 0.5 prior contribution

        (temp-26) / 4: Thermal stress begins above 26°C on most reefs
                       28°C = the commonly cited bleaching threshold
                       30°C = severe stress (26+4)
                       Normalise so 28°C → 0.5 contribution

        ssta / 3:      SSTA > 1°C = anomalous warming
                       SSTA = 3°C = extreme event (rare, e.g. 1998 El Niño)
                       Normalise so 1.5°C → 0.5 contribution

    WEIGHTS:
        0.40 × DHW — DHW is the single strongest predictor of bleaching
        0.35 × temp — temperature is the direct physical driver
        0.25 × SSTA — SSTA is derivative of temperature but adds context
    ─────────────────────────────────────────────────────────────────────────
    """
    dhw_score  = np.clip(dhw / 16.0, 0.0, 1.0)
    temp_score = np.clip((temperature - 26.0) / 4.0, 0.0, 1.0)
    ssta_score = np.clip(ssta / 3.0, 0.0, 1.0)

    return float(0.40 * dhw_score + 0.35 * temp_score + 0.25 * ssta_score)


# ── Training ──────────────────────────────────────────────────────────────────

def train(data_path: str, output_path: str = "xgboost_v2.pkl"):
    print(f"\nLoading data from: {data_path}")
    df = pd.read_csv(data_path)

    X = engineer_features(df)
    y = encode_target(df)

    # Drop rows where any feature is missing
    mask = X.notna().all(axis=1)
    X, y = X[mask], y[mask]

    print(f"Dataset: {len(X)} rows | healthy={sum(y==0)} bleached={sum(y==1)}")
    print(f"Class balance: {y.mean():.1%} bleached\n")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # ── XGBoost config ──────────────────────────────────────────────────────
    # max_depth=3: shallower trees → smoother probability outputs
    #              (deeper trees overfit the noisy environmental data)
    # min_child_weight=10: requires at least 10 samples per leaf
    #                      → prevents learning from tiny noisy groups
    # reg_alpha + reg_lambda: L1 + L2 regularisation → reduces overfitting
    # scale_pos_weight: compensates for class imbalance (55% bleached)
    #                   without this the model is biased toward bleached
    # gamma=0.3: minimum loss reduction to make a split — prunes weak splits
    # ────────────────────────────────────────────────────────────────────────
    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=10,
        gamma=0.3,
        reg_alpha=1.0,
        reg_lambda=3.0,
        scale_pos_weight=sum(y_train == 0) / sum(y_train == 1),
        random_state=42,
        eval_metric="logloss",
        early_stopping_rounds=30,
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )

    # ── Evaluation ──────────────────────────────────────────────────────────
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    print(f"Test accuracy : {accuracy_score(y_test, y_pred):.4f}")
    print(f"ROC-AUC       : {roc_auc_score(y_test, y_proba):.4f}")
    print()
    print(classification_report(y_test, y_pred, target_names=["healthy", "bleached"]))

    # 5-fold cross-validation (uses model without early stopping)
    cv_model = xgb.XGBClassifier(
        n_estimators=model.best_iteration or 300,
        max_depth=3, learning_rate=0.05, subsample=0.8,
        colsample_bytree=0.8, min_child_weight=10, gamma=0.3,
        reg_alpha=1.0, reg_lambda=3.0,
        scale_pos_weight=sum(y_train == 0) / sum(y_train == 1),
        random_state=42, eval_metric="logloss",
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(cv_model, X, y, cv=cv, scoring="accuracy")
    print(f"5-fold CV accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # Feature importance
    print("\nFeature importances:")
    imp = pd.Series(model.feature_importances_, index=FEATURES).sort_values(ascending=False)
    for feat, val in imp.items():
        print(f"  {feat}: {val:.4f}")

    # ── Sanity checks ────────────────────────────────────────────────────────
    print("\nSanity checks (XGB raw + physics blend):")

    def predict_blended(t, dhw, ssta, turb, shelt, wind):
        ssta_p = max(0, ssta); ssta_n = max(0, -ssta)
        row = pd.DataFrame([{
            "temperature_c": t, "dhw_log": np.log1p(dhw),
            "ssta_pos": ssta_p, "ssta_neg": ssta_n,
            "turbidity": turb, "sheltered": float(shelt),
            "windspeed": wind,
            "temp_stress": max(0, t - 26),
            "dhw_x_ssta": np.log1p(dhw) * ssta_p,
            "sheltered_stress": (1 - shelt) * ssta_p,
        }])
        raw   = float(model.predict_proba(row[FEATURES])[0][1])
        prior = physics_prior(t, dhw, ssta)
        return float(0.60 * raw + 0.40 * prior)

    cases = [
        ("Low  stress: temp=26 dhw=0.5 ssta=0.2 turb=0.04 shelt=1 wind=10", 26, 0.5, 0.2, 0.04, 1, 10),
        ("Med  stress: temp=28 dhw=5 ssta=1.0 turb=0.1 shelt=1 wind=4",     28, 5,   1.0, 0.10, 1, 4),
        ("High stress: temp=30 dhw=12 ssta=2.5 turb=0.2 shelt=0 wind=2",    30, 12,  2.5, 0.20, 0, 2),
    ]
    for name, *args in cases:
        p    = predict_blended(*args)
        risk = "Low" if p < 0.35 else ("Moderate" if p < 0.65 else "High")
        print(f"  {name}")
        print(f"    blended={p:.4f} → {risk}")

    # ── Save bundle ──────────────────────────────────────────────────────────
    bundle = {"model": model, "features": FEATURES}
    joblib.dump(bundle, output_path)
    print(f"\nModel bundle saved to: {output_path}")
    print(f"Bundle keys: {list(bundle.keys())}")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train XGBoost coral bleaching environmental model"
    )
    parser.add_argument("--data",   default="data.csv",        help="Path to raw CSV dataset")
    parser.add_argument("--output", default="xgboost_v2.pkl",  help="Output pkl path")
    args = parser.parse_args()

    train(args.data, args.output)
