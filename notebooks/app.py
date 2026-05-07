import streamlit as st
import cv2, torch, numpy as np, pickle, requests, base64
import matplotlib
matplotlib.use("Agg")
from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image

DEVICE           = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OPENROUTER_KEY   = "your openrouter key"
MODEL_NAME       = "./best_vit_model"

st.set_page_config(page_title="DeepReef AI", page_icon="🪸", layout="wide")
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;600;800&display=swap');
    html, body, [class*="css"] { font-family: 'Syne', sans-serif; }
    .stApp { background: linear-gradient(135deg, #0a0f1e 0%, #0d1f2d 50%, #091a1a 100%); color: #e8f4f0; }
    .hero { text-align: center; padding: 2.5rem 1rem 1.5rem; }
    .hero h1 { font-size: 3rem; font-weight: 800; background: linear-gradient(90deg, #00e5cc, #00b4d8, #48cae4); -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin-bottom: 0.3rem; }
    .hero p { color: #6bbfb5; font-size: 1.05rem; font-family: 'Space Mono', monospace; }
    .result-card { background: rgba(255,255,255,0.04); border: 1px solid rgba(0,229,204,0.2); border-radius: 16px; padding: 1.5rem; margin-bottom: 1rem; }
    .bleached-badge { background: linear-gradient(135deg, #ff4d4d, #c0392b); color: white; padding: 0.6rem 1.5rem; border-radius: 50px; font-size: 1.3rem; font-weight: 800; display: inline-block; }
    .healthy-badge { background: linear-gradient(135deg, #00e5cc, #00b09b); color: #0a0f1e; padding: 0.6rem 1.5rem; border-radius: 50px; font-size: 1.3rem; font-weight: 800; display: inline-block; }
    .metric-box { background: rgba(0,229,204,0.06); border: 1px solid rgba(0,229,204,0.15); border-radius: 12px; padding: 1rem; text-align: center; }
    .metric-label { font-family: 'Space Mono', monospace; font-size: 0.7rem; color: #6bbfb5; text-transform: uppercase; letter-spacing: 2px; margin-bottom: 0.3rem; }
    .metric-value { font-size: 1.6rem; font-weight: 800; color: #00e5cc; }
    .section-title { font-family: 'Space Mono', monospace; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 3px; color: #00e5cc; margin-bottom: 1rem; border-bottom: 1px solid rgba(0,229,204,0.2); padding-bottom: 0.5rem; }
    .info-box { background: rgba(0,229,204,0.05); border-left: 3px solid #00e5cc; border-radius: 0 8px 8px 0; padding: 1rem 1.2rem; margin-top: 1rem; font-family: 'Space Mono', monospace; font-size: 0.82rem; color: #9ecfca; line-height: 1.8; }
    .ai-box { background: rgba(0,229,204,0.04); border: 1px solid rgba(0,229,204,0.25); border-radius: 16px; padding: 1.5rem; margin-top: 1rem; font-size: 0.95rem; line-height: 1.8; color: #d0ece8; }
    .ai-title { font-family: 'Space Mono', monospace; font-size: 0.7rem; text-transform: uppercase; letter-spacing: 3px; color: #00e5cc; margin-bottom: 0.8rem; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
    model     = AutoModelForImageClassification.from_pretrained(MODEL_NAME).to(DEVICE)
    model.eval()
    return processor, model


@st.cache_resource
def load_xgb():
    try:
        with open("xgb_env_model.pkl", "rb") as f:
            return pickle.load(f)
    except:
        return None


def get_gradcam(model, pixel_values):
    model.eval()
    saved = {}

    def fwd(m, i, o):  saved["acts"]  = o.detach()
    def bwd(m, gi, go): saved["grads"] = go[0].detach()

    h1 = model.vit.encoder.layer[-2].register_forward_hook(fwd)
    h2 = model.vit.encoder.layer[-2].register_full_backward_hook(bwd)

    model.zero_grad()
    out       = model(pixel_values=pixel_values)
    class_idx = out.logits.argmax(dim=-1).item()
    out.logits[0, class_idx].backward()

    h1.remove(); h2.remove()

    acts    = saved["acts"][0, 1:].cpu()
    grads   = saved["grads"][0, 1:].cpu()
    weights = grads.mean(dim=0)
    cam     = (acts * weights).sum(dim=-1).numpy()
    cam     = np.maximum(cam, 0)
    cam     = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    size    = int(np.sqrt(len(cam)))
    return cv2.resize(cam.reshape(size, size), (224, 224))


def openrouter_summary(img_rgb, vit_pred, vit_conf, xgb_pred, fused_pred,
                       temp, turbidity, windspeed, tsi, ssta):
    _, buf  = cv2.imencode(".jpg", cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
    img_b64 = base64.b64encode(buf).decode("utf-8")

    prompt = f"""You are a marine biology AI assistant analyzing a coral reef image.

Results:
- ViT (NOAA pretrained): {"Bleached" if vit_pred==1 else "Healthy"} ({vit_conf*100:.1f}% confidence)
- XGBoost (environmental): {"Bleached" if xgb_pred==1 else "Healthy"}
- Fusion: {"Bleached" if fused_pred==1 else "Healthy"}

Environmental data: Temperature {temp}°C, Turbidity {turbidity}, Windspeed {windspeed} knots, TSI {tsi}, SSTA {ssta}°C

Write 3-4 sentences: state the diagnosis, reference environmental conditions, note if models agree, give an ecological observation. Plain paragraph, no bullet points."""

    response = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {OPENROUTER_KEY}",
            "Content-Type": "application/json"
        },
        json={
            "model": "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text",       "text": prompt},
                        {"type": "image_url",  "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}}
                    ]
                }
            ]
        },
        timeout=30
    )

    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"]
    return f"OpenRouter error {response.status_code}: {response.text}"


st.markdown("""
<div class="hero">
    <h1>🪸 DeepReef AI</h1>
    <p>NOAA ViT · XGBoost Environmental · NVIDIA AI Explainability</p>
</div>
""", unsafe_allow_html=True)

uploaded = st.file_uploader("Upload a coral image (JPG or PNG)", type=["jpg", "jpeg", "png"])

st.markdown("<br>", unsafe_allow_html=True)
st.markdown('<div class="section-title">Environmental Parameters (Optional)</div>', unsafe_allow_html=True)
st.caption("Enter field measurements if available — XGBoost uses these to assess environmental stress.")

e1, e2, e3, e4, e5 = st.columns(5)
with e1: temp      = st.number_input("Temperature (°C)",     value=27.0, step=0.1,  format="%.1f")
with e2: turbidity = st.number_input("Turbidity (0–1)",      value=0.05, step=0.01, format="%.2f")
with e3: windspeed = st.number_input("Windspeed (knots)",    value=7.0,  step=0.5,  format="%.1f")
with e4: tsi       = st.number_input("Thermal Stress Index", value=0.0,  step=0.1,  format="%.2f")
with e5: ssta      = st.number_input("SSTA (°C)",            value=0.0,  step=0.1,  format="%.2f")

if uploaded is None:
    st.markdown("""
    <div class="info-box">
    → Upload a coral image above to run the full analysis<br>
    → NOAA pretrained ViT classifies the image visually<br>
    → XGBoost uses the environmental values you entered<br>
    → Fusion combines both for the final prediction<br>
    → NVIDIA AI generates a plain-English scientific summary
    </div>
    """, unsafe_allow_html=True)

if uploaded:
    file_bytes = np.frombuffer(uploaded.read(), np.uint8)
    img_bgr    = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    img_rgb    = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_pil    = Image.fromarray(img_rgb)

    try:
        processor, vit_model = load_model()
    except Exception as e:
        st.error(f"Could not load model. Error: {e}")
        st.stop()

    xgb_bundle   = load_xgb()
    inputs       = processor(images=img_pil, return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(DEVICE)

    with torch.no_grad():
        out   = vit_model(pixel_values=pixel_values)
        probs = torch.softmax(out.logits, dim=-1)[0]

    label2id  = vit_model.config.label2id
    bleach_id = label2id.get("CORAL_BL", label2id.get("1", 1))

    vit_prob = probs[bleach_id].item()
    vit_pred = 1 if vit_prob > 0.5 else 0
    vit_conf = probs[vit_pred].item()

    cam     = get_gradcam(vit_model, pixel_values)
    orig    = np.array(img_pil.resize((224, 224))).astype(np.float32) / 255.0
    heat    = cv2.cvtColor(cv2.applyColorMap(np.uint8(255*cam), cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB) / 255.0
    overlay = np.clip(0.65*orig + 0.35*heat, 0, 1)

    xgb_pred = 0; xgb_prob = 0.5; fused_pred = vit_pred; fused_prob = vit_prob

    if xgb_bundle:
        xgb_prob   = xgb_bundle["model"].predict_proba([[temp, turbidity, windspeed, tsi, ssta]])[0][1]
        xgb_pred   = 1 if xgb_prob > 0.5 else 0
        fused_prob  = 0.6 * vit_prob + 0.4 * xgb_prob
        fused_pred  = 1 if fused_prob > 0.5 else 0

    col_img, col_results = st.columns([1, 1], gap="large")

    with col_img:
        st.markdown('<div class="section-title">Uploaded Image</div>', unsafe_allow_html=True)
        st.image(img_rgb, use_container_width=True)

    with col_results:
        st.markdown('<div class="section-title">Prediction</div>', unsafe_allow_html=True)
        badge = "bleached-badge" if vit_pred == 1 else "healthy-badge"
        st.markdown(f'<div class="result-card" style="text-align:center"><div class="{badge}">{"🔴 BLEACHED" if vit_pred==1 else "🟢 HEALTHY"}</div></div>', unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        with c1: st.markdown(f'<div class="metric-box"><div class="metric-label">ViT Confidence</div><div class="metric-value">{vit_conf*100:.1f}%</div></div>', unsafe_allow_html=True)
        with c2: st.markdown(f'<div class="metric-box"><div class="metric-label">Bleach Probability</div><div class="metric-value">{vit_prob*100:.1f}%</div></div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.progress(float(vit_conf))
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-title">All Models</div>', unsafe_allow_html=True)

        m1, m2, m3 = st.columns(3)
        for col, name, pred, prob in zip(
            [m1, m2, m3],
            ["ViT (NOAA)", "XGBoost (env)", "Fusion"],
            [vit_pred, xgb_pred, fused_pred],
            [vit_prob, xgb_prob, fused_prob]
        ):
            color = "#ff4d4d" if pred == 1 else "#00e5cc"
            with col:
                st.markdown(f"""<div class="metric-box">
                    <div class="metric-label">{name}</div>
                    <div style="font-size:1rem;font-weight:700;color:{color};margin-top:0.3rem">{"Bleached" if pred==1 else "Healthy"}</div>
                    <div style="font-family:'Space Mono',monospace;font-size:0.72rem;color:#6bbfb5;margin-top:0.2rem">{prob*100:.1f}%</div>
                </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-title">GradCAM — Where the ViT is looking</div>', unsafe_allow_html=True)
    g1, g2, g3 = st.columns(3)
    g1.image(orig, caption="Original", use_container_width=True, clamp=True)
    g2.image(cv2.cvtColor(cv2.applyColorMap(np.uint8(255*cam), cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB),
             caption="GradCAM Heatmap", use_container_width=True)
    g3.image(overlay, caption="Overlay", use_container_width=True, clamp=True)
    st.markdown("""<div class="info-box">
    🔴 Red regions → where the ViT focuses most &nbsp;|&nbsp; 🔵 Blue regions → ignored<br>
    ViT works on 16×16 patches — heatmap shows which patches drove the prediction.
    </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-title">AI Summary — NVIDIA Analysis</div>', unsafe_allow_html=True)

    if OPENROUTER_KEY == "YOUR_OPENROUTER_API_KEY_HERE":
        st.warning("Add your OpenRouter API key to the top of app.py to enable AI summaries.")
    else:
        with st.spinner("Generating summary..."):
            summary = openrouter_summary(img_rgb, vit_pred, vit_conf, xgb_pred, fused_pred,
                                         temp, turbidity, windspeed, tsi, ssta)
        st.markdown(f'<div class="ai-box"><div class="ai-title">🤖 NVIDIA AI Summary</div>{summary}</div>', unsafe_allow_html=True)