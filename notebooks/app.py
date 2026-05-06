import streamlit as st
import cv2, torch, numpy as np, pickle, requests, base64
import torchvision.models as models
import torchvision.transforms as T
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
GEMINI_API_KEY = "YOUR_GEMINI_API_KEY_HERE"

val_transform = T.Compose([
    T.ToPILImage(), T.Resize((224, 224)),
    T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

st.set_page_config(page_title="Coral Bleaching Detector", page_icon="🪸", layout="wide")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;600;800&display=swap');
    html, body, [class*="css"] { font-family: 'Syne', sans-serif; }
    .stApp { background: linear-gradient(135deg, #0a0f1e 0%, #0d1f2d 50%, #091a1a 100%); color: #e8f4f0; }
    .hero { text-align: center; padding: 2.5rem 1rem 1.5rem; }
    .hero h1 { font-family: 'Syne', sans-serif; font-size: 3rem; font-weight: 800; background: linear-gradient(90deg, #00e5cc, #00b4d8, #48cae4); -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin-bottom: 0.3rem; letter-spacing: -1px; }
    .hero p { color: #6bbfb5; font-size: 1.05rem; font-family: 'Space Mono', monospace; }
    .result-card { background: rgba(255,255,255,0.04); border: 1px solid rgba(0,229,204,0.2); border-radius: 16px; padding: 1.5rem; margin-bottom: 1rem; }
    .bleached-badge { background: linear-gradient(135deg, #ff4d4d, #c0392b); color: white; padding: 0.6rem 1.5rem; border-radius: 50px; font-size: 1.3rem; font-weight: 800; display: inline-block; letter-spacing: 1px; }
    .healthy-badge { background: linear-gradient(135deg, #00e5cc, #00b09b); color: #0a0f1e; padding: 0.6rem 1.5rem; border-radius: 50px; font-size: 1.3rem; font-weight: 800; display: inline-block; letter-spacing: 1px; }
    .metric-box { background: rgba(0,229,204,0.06); border: 1px solid rgba(0,229,204,0.15); border-radius: 12px; padding: 1rem; text-align: center; }
    .metric-label { font-family: 'Space Mono', monospace; font-size: 0.7rem; color: #6bbfb5; text-transform: uppercase; letter-spacing: 2px; margin-bottom: 0.3rem; }
    .metric-value { font-size: 1.6rem; font-weight: 800; color: #00e5cc; }
    .section-title { font-family: 'Space Mono', monospace; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 3px; color: #00e5cc; margin-bottom: 1rem; border-bottom: 1px solid rgba(0,229,204,0.2); padding-bottom: 0.5rem; }
    .info-box { background: rgba(0,229,204,0.05); border-left: 3px solid #00e5cc; border-radius: 0 8px 8px 0; padding: 1rem 1.2rem; margin-top: 1rem; font-family: 'Space Mono', monospace; font-size: 0.82rem; color: #9ecfca; line-height: 1.8; }
    .ai-box { background: rgba(0,229,204,0.04); border: 1px solid rgba(0,229,204,0.25); border-radius: 16px; padding: 1.5rem; margin-top: 1rem; font-size: 0.95rem; line-height: 1.8; color: #d0ece8; }
    .ai-title { font-family: 'Space Mono', monospace; font-size: 0.7rem; text-transform: uppercase; letter-spacing: 3px; color: #00e5cc; margin-bottom: 0.8rem; }
    div[data-testid="stNumberInput"] label { color: #9ecfca !important; font-family: 'Space Mono', monospace; font-size: 0.8rem; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_cnn():
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 1)
    model.load_state_dict(torch.load("best_model.pt", map_location=DEVICE))
    model.eval()
    return model.to(DEVICE)


@st.cache_resource
def load_xgb_env():
    try:
        with open("xgb_env_model.pkl", "rb") as f:
            return pickle.load(f)
    except:
        return None


def get_gradcam(model, img_tensor):
    model.eval()
    saved = {}
    h1 = model.layer4[-1].register_forward_hook(lambda m,i,o: saved.update({"acts": o.detach()}))
    h2 = model.layer4[-1].register_full_backward_hook(lambda m,gi,go: saved.update({"grads": go[0].detach()}))
    model.zero_grad()
    model(img_tensor.unsqueeze(0).to(DEVICE)).backward()
    h1.remove(); h2.remove()
    acts  = saved["acts"][0].cpu().numpy()
    grads = saved["grads"][0].cpu().numpy()
    cam   = (grads.mean(axis=(1,2))[:, None, None] * acts).sum(axis=0)
    cam   = np.maximum(cam, 0)
    cam   = (cam - cam.min()) / (cam.max() + 1e-8)
    return cv2.resize(cam, (224, 224))


def denormalize(tensor):
    img = tensor.permute(1,2,0).numpy().astype(np.float64)
    return np.clip(img * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406], 0, 1)


def gemini_summary(img_rgb, cnn_pred, cnn_conf, xgb_pred, fused_pred,
                   temp, turbidity, windspeed, tsi, ssta):
    label       = "Bleached" if cnn_pred == 1 else "Healthy"
    xgb_label   = "Bleached" if xgb_pred == 1 else "Healthy"
    fused_label = "Bleached" if fused_pred == 1 else "Healthy"

    _, buf  = cv2.imencode(".jpg", cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
    img_b64 = base64.b64encode(buf).decode("utf-8")

    prompt = f"""You are a marine biology AI assistant analyzing a coral reef image.

The AI system produced these results:
- CNN Prediction: {label} (confidence: {cnn_conf*100:.1f}%)
- XGBoost Prediction (environmental): {xgb_label}
- Fusion Prediction: {fused_label}

Environmental parameters entered by the user:
- Temperature: {temp}°C
- Turbidity: {turbidity}
- Windspeed: {windspeed} knots
- Thermal Stress Index: {tsi}
- SSTA: {ssta}°C

Write a concise 3-4 sentence scientific summary that:
1. States whether the coral appears healthy or bleached
2. References the environmental conditions and whether they indicate stress
3. Notes whether the image and environmental models agree
4. Gives a brief ecological observation

Write in plain paragraph form, no bullet points."""

    payload  = {"contents": [{"parts": [{"text": prompt}, {"inline_data": {"mime_type": "image/jpeg", "data": img_b64}}]}]}
    url      = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={GEMINI_API_KEY}"
    response = requests.post(url, json=payload, timeout=30)

    if response.status_code == 200:
        return response.json()["candidates"][0]["content"]["parts"][0]["text"]
    else:
        return f"Gemini API error {response.status_code}: {response.text}"

st.markdown("""
<div class="hero">
    <h1>🪸 DeepReef AI</h1>
    <p>Explainable coral bleaching detection using deep learning</p>
</div>
""", unsafe_allow_html=True)

uploaded = st.file_uploader("Upload a coral image (JPG or PNG)", type=["jpg", "jpeg", "png"])

st.markdown("<br>", unsafe_allow_html=True)
st.markdown('<div class="section-title">Environmental Parameters (Optional)</div>', unsafe_allow_html=True)
st.caption("Enter field measurements if available. Leave as default if unknown — XGBoost will use these values.")

e1, e2, e3, e4, e5 = st.columns(5)
with e1: temp     = st.number_input("Temperature (°C)",        value=27.0, step=0.1, format="%.1f")
with e2: turbidity= st.number_input("Turbidity (0–1)",          value=0.05, step=0.01, format="%.2f")
with e3: windspeed= st.number_input("Windspeed (knots)",        value=7.0,  step=0.5, format="%.1f")
with e4: tsi      = st.number_input("Thermal Stress Index",     value=0.0,  step=0.1, format="%.2f")
with e5: ssta     = st.number_input("SSTA (°C)",                value=0.0,  step=0.1, format="%.2f")

if uploaded is None:
    st.markdown("""
    <div class="info-box">
    → Upload a coral image above to run the analysis<br>
    → CNN classifies the image visually<br>
    → XGBoost uses the environmental values you entered above<br>
    → Gemini AI generates a plain-English summary combining both
    </div>
    """, unsafe_allow_html=True)

if uploaded:
    file_bytes = np.frombuffer(uploaded.read(), np.uint8)
    img_bgr    = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    img_rgb    = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_tensor = val_transform(img_rgb)

    try:
        cnn_model = load_cnn()
    except Exception as e:
        st.error(f"Could not load best_model.pt — make sure it's in the same folder. Error: {e}")
        st.stop()

    xgb_bundle = load_xgb_env()

    with torch.no_grad():
        cnn_prob = torch.sigmoid(cnn_model(img_tensor.unsqueeze(0).to(DEVICE))).item()

    cnn_pred = 1 if cnn_prob > 0.5 else 0
    cnn_conf = cnn_prob if cnn_pred == 1 else 1 - cnn_prob

    cam     = get_gradcam(cnn_model, img_tensor)
    orig    = denormalize(img_tensor)
    heat    = cv2.cvtColor(cv2.applyColorMap(np.uint8(255*cam), cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB) / 255.0
    overlay = np.clip(0.65*orig + 0.35*heat, 0, 1)

    xgb_pred  = 0
    xgb_prob  = 0.5
    fused_pred = cnn_pred
    fused_prob = cnn_prob

    if xgb_bundle:
        xgb_model  = xgb_bundle["model"]
        env_input  = np.array([[temp, turbidity, windspeed, tsi, ssta]])
        xgb_prob   = xgb_model.predict_proba(env_input)[0][1]
        xgb_pred   = 1 if xgb_prob > 0.5 else 0
        fused_prob  = 0.6 * cnn_prob + 0.4 * xgb_prob
        fused_pred  = 1 if fused_prob > 0.5 else 0

    col_img, col_results = st.columns([1, 1], gap="large")

    with col_img:
        st.markdown('<div class="section-title">Uploaded Image</div>', unsafe_allow_html=True)
        st.image(img_rgb, use_container_width=True)

    with col_results:
        st.markdown('<div class="section-title">Prediction</div>', unsafe_allow_html=True)
        badge = "bleached-badge" if cnn_pred == 1 else "healthy-badge"
        label = "🔴 BLEACHED" if cnn_pred == 1 else "🟢 HEALTHY"
        st.markdown(f'<div class="result-card" style="text-align:center"><div class="{badge}">{label}</div></div>', unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f'<div class="metric-box"><div class="metric-label">CNN Confidence</div><div class="metric-value">{cnn_conf*100:.1f}%</div></div>', unsafe_allow_html=True)
        with c2:
            st.markdown(f'<div class="metric-box"><div class="metric-label">Bleach Probability</div><div class="metric-value">{cnn_prob*100:.1f}%</div></div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.progress(float(cnn_conf))

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-title">All Models</div>', unsafe_allow_html=True)

        if xgb_bundle:
            m1, m2, m3 = st.columns(3)
            for col, name, pred, prob in zip(
                [m1, m2, m3],
                ["CNN", "XGBoost (env)", "Fusion"],
                [cnn_pred, xgb_pred, fused_pred],
                [cnn_prob, xgb_prob, fused_prob]
            ):
                color = "#ff4d4d" if pred == 1 else "#00e5cc"
                with col:
                    st.markdown(f"""
                    <div class="metric-box">
                        <div class="metric-label">{name}</div>
                        <div style="font-size:1rem;font-weight:700;color:{color};margin-top:0.3rem">{"Bleached" if pred==1 else "Healthy"}</div>
                        <div style="font-family:'Space Mono',monospace;font-size:0.72rem;color:#6bbfb5;margin-top:0.2rem">{prob*100:.1f}%</div>
                    </div>""", unsafe_allow_html=True)
        else:
            st.info("xgb_env_model.pkl not found — only CNN prediction shown.")

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-title">GradCAM — Where the model is looking</div>', unsafe_allow_html=True)
    g1, g2, g3 = st.columns(3)
    g1.image(orig,    caption="Original",        use_container_width=True, clamp=True)
    g2.image(cam,     caption="GradCAM Heatmap", use_container_width=True, clamp=True)
    g3.image(overlay, caption="Overlay",         use_container_width=True, clamp=True)
    st.markdown("""
    <div class="info-box">
    🔴 Red regions → where the model focuses most &nbsp;|&nbsp; 🔵 Blue regions → ignored areas<br>
    A good heatmap is centered on the coral body, not the water or background.
    </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-title">AI Summary — Gemini Analysis</div>', unsafe_allow_html=True)

    if GEMINI_API_KEY == "YOUR_GEMINI_API_KEY_HERE":
        st.warning("Add your Gemini API key to the top of app.py to enable AI-generated summaries.")
    else:
        with st.spinner("Generating summary..."):
            summary = gemini_summary(img_rgb, cnn_pred, cnn_conf, xgb_pred, fused_pred,
                                     temp, turbidity, windspeed, tsi, ssta)
        st.markdown(f'<div class="ai-box"><div class="ai-title">🤖 Gemini Summary</div>{summary}</div>', unsafe_allow_html=True)