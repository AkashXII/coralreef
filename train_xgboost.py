import argparse
import numpy as np
import pandas as pd
import joblib
import warnings
warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
import xgboost as xgb
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame()
    df["Temperature_Mean"] = pd.to_numeric(df["Temperature_Mean"], errors="coerce")
    out["temperature_c"]   = df["Temperature_Mean"] - 273.15
    df["SSTA_DHW"]  = pd.to_numeric(df["SSTA_DHW"], errors="coerce")
    out["dhw_log"]  = np.log1p(df["SSTA_DHW"])
    df["SSTA"]       = pd.to_numeric(df["SSTA"], errors="coerce")
    out["ssta_pos"]  = df["SSTA"].clip(lower=0)
    out["ssta_neg"]  = (-df["SSTA"]).clip(lower=0)
    out["turbidity"] = df["Turbidity"]
    out["sheltered"] = df["Exposure"].map(
        {"Sheltered": 1, "Exposed": 0, "Sometimes": 0}
    ).astype(float)
    df["Windspeed"]   = pd.to_numeric(df["Windspeed"], errors="coerce")
    out["windspeed"]  = df["Windspeed"]
    out["temp_stress"] = (out["temperature_c"] - 26).clip(lower=0)
    out["dhw_x_ssta"] = out["dhw_log"] * out["ssta_pos"]
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
def encode_target(df: pd.DataFrame) -> pd.Series:
    return df["Bleaching_condition"].apply(
        lambda x: 0 if "Mild" in str(x) else 1
    )
def physics_prior(temperature: float, dhw: float, ssta: float) -> float: #need to balance

    dhw_score  = np.clip(dhw / 16.0, 0.0, 1.0)
    temp_score = np.clip((temperature - 26.0) / 4.0, 0.0, 1.0)
    ssta_score = np.clip(ssta / 3.0, 0.0, 1.0)

    return float(0.40 * dhw_score + 0.35 * temp_score + 0.25 * ssta_score)

def train(data_path: str, output_path: str = "xgboost_v2.pkl"):
    print(f"\nLoading data from: {data_path}")
    df = pd.read_csv(data_path)

    X = engineer_features(df)
    y = encode_target(df)

    mask = X.notna().all(axis=1)
    X, y = X[mask], y[mask]

    print(f"Dataset: {len(X)} rows | healthy={sum(y==0)} bleached={sum(y==1)}")
    print(f"Class balance: {y.mean():.1%} bleached\n")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
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

    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    print(f"Test accuracy : {accuracy_score(y_test, y_pred):.4f}")
    print(f"ROC-AUC       : {roc_auc_score(y_test, y_proba):.4f}")
    print()
    print(classification_report(y_test, y_pred, target_names=["healthy", "bleached"]))

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

    print("\nFeature importances:")
    imp = pd.Series(model.feature_importances_, index=FEATURES).sort_values(ascending=False)
    for feat, val in imp.items():
        print(f"  {feat}: {val:.4f}")
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
    bundle = {"model": model, "features": FEATURES}
    joblib.dump(bundle, output_path)
    print(f"\nModel bundle saved to: {output_path}")
    print(f"Bundle keys: {list(bundle.keys())}")
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train XGBoost coral bleaching environmental model"
    )
    parser.add_argument("--data",   default="data.csv",        help="Path to raw CSV dataset")
    parser.add_argument("--output", default="xgboost_v2.pkl",  help="Output pkl path")
    args = parser.parse_args()

    train(args.data, args.output)
