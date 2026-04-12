
from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware

from groq import Groq
from ultralytics import YOLO
from preprocessor import process_image

import numpy as np
import pandas as pd
import cv2
import tempfile
import os
import joblib

app = FastAPI(title="Coral Bleaching Detector")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# ── Load models once at startup ───────────────────────────────────────────────
cnn_model    = YOLO("models/best.pt")
xgb_bundle   = joblib.load("xgboost_v2.pkl")
xgb_model    = xgb_bundle["model"]
XGB_FEATURES = xgb_bundle["features"]
groq_client  = Groq(api_key=os.getenv("GROQ_API_KEY"))


# ── Physics-based prior ───────────────────────────────────────────────────────

def physics_prior(temperature: float, dhw: float, ssta: float) -> float:
    """
    Scientifically grounded bleaching risk based on NOAA thresholds.
    DHW > 8 = bleaching. Temp > 28C = thermal stress. SSTA > 1C = anomalous.
    Monotonic by construction — ensures correct behaviour at extreme inputs.
    """
    dhw_score  = np.clip(dhw / 16.0, 0.0, 1.0)
    temp_score = np.clip((temperature - 26.0) / 4.0, 0.0, 1.0)
    ssta_score = np.clip(ssta / 3.0, 0.0, 1.0)
    return float(0.40 * dhw_score + 0.35 * temp_score + 0.25 * ssta_score)


# ── XGB inference ─────────────────────────────────────────────────────────────

def get_xgb_prob(
    temperature: float,
    dhw: float,
    ssta: float,
    turbidity: float,
    sheltered: int,
    windspeed: float,
) -> float:
    """
    Blended environmental bleaching probability.
    60% XGBoost learned signal + 40% physics prior.
    Blend ensures monotonic and physically sensible outputs
    in input ranges underrepresented in training data.
    """
    ssta_p = max(0.0, ssta)
    ssta_n = max(0.0, -ssta)

    row = pd.DataFrame([{
        "temperature_c":    temperature,
        "dhw_log":          np.log1p(dhw),
        "ssta_pos":         ssta_p,
        "ssta_neg":         ssta_n,
        "turbidity":        turbidity,
        "sheltered":        float(sheltered),
        "windspeed":        windspeed,
        "temp_stress":      max(0.0, temperature - 26.0),
        "dhw_x_ssta":       np.log1p(dhw) * ssta_p,
        "sheltered_stress": (1.0 - sheltered) * ssta_p,
    }])

    xgb_raw = float(xgb_model.predict_proba(row[XGB_FEATURES])[0][1])
    prior   = physics_prior(temperature, dhw, ssta)
    return float(0.60 * xgb_raw + 0.40 * prior)


# ── CNN inference ─────────────────────────────────────────────────────────────

def get_cnn_prob(image_path: str) -> float:
    """Probability of bleaching from image. 0 = healthy, 1 = bleached."""
    result = cnn_model(image_path, verbose=False)

    if result[0].probs is None:
        return 0.5

    probs        = result[0].probs.data.cpu().numpy()
    names        = cnn_model.names
    bleached_idx = next(k for k, v in names.items() if v == "bleached")
    return float(probs[bleached_idx])


# ── Fusion ────────────────────────────────────────────────────────────────────

def fuse(cnn_prob: float, xgb_prob: float) -> float:
    """Fixed-weight fusion: CNN 65%, XGB 35%. CNN carries more weight
    because it has direct visual evidence of bleaching."""
    return 0.65 * cnn_prob + 0.35 * xgb_prob


# ── Risk level ────────────────────────────────────────────────────────────────

def get_risk(prob: float) -> str:
    if prob < 0.35:
        return "Low"
    elif prob < 0.65:
        return "Moderate"
    else:
        return "High"


# ── Groq explanation ──────────────────────────────────────────────────────────

def generate_explanation(
    prediction: str,
    risk: str,
    cnn_prob: float,
    xgb_prob: float | None,
    inputs: dict | None,
) -> str:
    """
    Builds a structured summary of the prediction signals and passes it
    to Groq LLM to generate a clean, concise natural language explanation.
    Falls back to a rule-based string if the API call fails.
    """

    # Image signal label
    if cnn_prob < 0.25:
        img_label = "visually healthy coral with normal pigmentation"
    elif cnn_prob < 0.45:
        img_label = "mostly healthy coral with minor signs of stress"
    elif cnn_prob < 0.65:
        img_label = "possible early-stage bleaching visible"
    else:
        img_label = "clear bleaching with significant pigment loss"

    # Environmental signal label
    if xgb_prob is None:
        env_label = "no environmental data provided"
    elif xgb_prob < 0.25:
        env_label = "well within safe environmental range"
    elif xgb_prob < 0.40:
        env_label = "low environmental stress"
    elif xgb_prob < 0.55:
        env_label = "mild environmental stress"
    elif xgb_prob < 0.70:
        env_label = "moderate thermal stress"
    else:
        env_label = "high thermal stress"

    # Stress drivers
    drivers = []
    if inputs and xgb_prob is not None and xgb_prob > 0.40:
        if inputs.get("dhw", 0) >= 8:
            drivers.append(f"degree heating weeks ({inputs['dhw']:.1f}) above NOAA bleaching threshold")
        elif inputs.get("dhw", 0) >= 4:
            drivers.append(f"degree heating weeks ({inputs['dhw']:.1f}) approaching stress levels")
        if inputs.get("temperature", 0) >= 29:
            drivers.append(f"temperature ({inputs['temperature']:.1f}°C) above 28°C stress threshold")
        if inputs.get("ssta", 0) >= 1.0:
            drivers.append(f"sea surface anomaly ({inputs['ssta']:.2f}°C) significantly elevated")
        if inputs.get("turbidity", 0) >= 0.12:
            drivers.append(f"turbidity ({inputs['turbidity']:.3f}) above typical reef levels")

    drivers_text = "; ".join(drivers) if drivers else "none identified"

    # Signal agreement
    if xgb_prob is None:
        agreement = "image-only prediction, no environmental data"
    elif cnn_prob < 0.5 and xgb_prob < 0.5:
        agreement = "both image and environment agree coral is healthy"
    elif cnn_prob >= 0.5 and xgb_prob >= 0.5:
        agreement = "both image and environment indicate bleaching stress"
    elif cnn_prob < 0.5:
        agreement = f"image ({cnn_prob:.2f}) suggests healthy but environment ({xgb_prob:.2f}) shows stress — image given more weight"
    else:
        agreement = f"image ({cnn_prob:.2f}) shows bleaching but environment ({xgb_prob:.2f}) is moderate — image given more weight"

    env_score = f"{xgb_prob:.2f}" if xgb_prob is not None else "N/A"
    prompt = f"""You are a marine biologist assistant reporting coral bleaching assessment results.

Assessment data:
- Prediction: {prediction.upper()}
- Risk level: {risk}
- Image analysis: {img_label} (score: {cnn_prob:.2f})
- Environmental analysis: {env_label} (score: {env_score})
- Key stress drivers: {drivers_text}
- Signal agreement: {agreement}

Write a clear, factual explanation in 2-3 sentences.
Rules:
- Do not add any information not present above
- Do not use words like "significant", "crucial", "notably", "it is worth noting"
- Do not start with "Based on" or "The assessment"
- Be direct and plain — this is shown to a non-expert user
- End with one actionable recommendation if risk is Moderate or High
"""

    try:
        response = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=120,
        )
        return response.choices[0].message.content.strip()

    except Exception as e:
        # Rule-based fallback — never fails
        base = f"The coral is predicted as {prediction} with {risk.lower()} risk. "
        base += f"Image analysis shows {img_label}. "
        if xgb_prob is not None:
            base += f"Environmental conditions indicate {env_label}."
        if risk in ("Moderate", "High"):
            base += " We recommend continued monitoring and reporting to local reef authorities."
        return base


# ── API endpoints ─────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {
        "status": "running",
        "model":  "coral bleaching detector v2",
        "inputs": {
            "image":       "coral photo (jpg/png)",
            "temperature": "sea temperature in Celsius (e.g. 27.5)",
            "dhw":         "degree heating weeks from NOAA CoralWatch (e.g. 4.2)",
            "ssta":        "sea surface temp anomaly in Celsius, negative=cooler (e.g. 0.8)",
            "turbidity":   "water turbidity 0-1, typical reef: 0.02-0.10 (e.g. 0.05)",
            "sheltered":   "1 if sheltered site, 0 if exposed to open ocean",
            "windspeed":   "wind speed in knots or m/s (e.g. 6)",
        }
    }


@app.post("/predict")
async def predict(
    image:       UploadFile = File(...),
    temperature: float = Form(None),
    dhw:         float = Form(None),
    ssta:        float = Form(None),
    turbidity:   float = Form(None),
    sheltered:   int   = Form(None),
    windspeed:   float = Form(None),
):
    # Save uploaded image to temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        tmp.write(await image.read())
        tmp_path = tmp.name

    processed_path = tmp_path + "_clean.jpg"

    try:
        # DIP preprocessing
        dip_result = process_image(tmp_path)
        cv2.imwrite(processed_path, dip_result["clean_bgr"])

        # CNN inference
        cnn_prob = get_cnn_prob(processed_path)

        # XGB inference — only if all tabular inputs provided
        tabular_complete = all(
            v is not None
            for v in [temperature, dhw, ssta, turbidity, sheltered, windspeed]
        )

        if tabular_complete:
            xgb_prob   = get_xgb_prob(temperature, dhw, ssta, turbidity, sheltered, windspeed)
            final_prob = fuse(cnn_prob, xgb_prob)
            mode       = "multimodal"
            inputs     = dict(
                temperature=temperature, dhw=dhw, ssta=ssta,
                turbidity=turbidity, sheltered=sheltered, windspeed=windspeed,
            )
        else:
            xgb_prob   = None
            final_prob = cnn_prob
            mode       = "image_only"
            inputs     = None

        prediction  = "bleached" if final_prob > 0.5 else "healthy"
        risk        = get_risk(final_prob)
        explanation = generate_explanation(prediction, risk, cnn_prob, xgb_prob, inputs)

    finally:
        # Always clean up temp files
        for p in [tmp_path, processed_path]:
            try:
                os.remove(p)
            except Exception:
                pass

    return {
        "mode":        mode,
        "cnn_prob":    round(cnn_prob, 4),
        "xgb_prob":    round(xgb_prob, 4) if xgb_prob is not None else None,
        "final_prob":  round(final_prob, 4),
        "risk":        risk,
        "prediction":  prediction,
        "explanation": explanation,
    }