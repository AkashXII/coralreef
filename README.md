# DeepReef AI

DeepReef AI is a multimodal coral reef health analysis system that combines deep learning-based image classification, environmental stress modeling, and visual explainability to detect coral bleaching from underwater photographs.

---

## Overview

Coral bleaching is one of the most ecologically significant indicators of reef degradation, driven by prolonged thermal stress and environmental disturbance. DeepReef AI addresses the challenge of scalable reef monitoring by automating bleaching detection through two independent signal sources — visual analysis of coral imagery and evaluation of in-situ environmental parameters — fused into a single prediction with interpretable outputs.

---

## System Architecture

```
Coral Image
     │
     ▼
NOAA Vision Transformer (ViT)
     │
     ├── GradCAM Heatmap
     │
     ▼
Image-based Prediction (P1)

Environmental Parameters
(Temperature, Turbidity, Windspeed, TSI, SSTA)
     │
     ▼
XGBoost Environmental Model
     │
     ▼
Environment-based Prediction (P2)

     ▼
Fusion: 0.6 × P1 + 0.4 × P2
     ▼
Final Coral Health Prediction
     ▼
NVIDIA AI Ecological Summary
```

---

## Features

**Visual Classification**
A Vision Transformer pretrained by NOAA on 10,419 annotated coral reef images classifies uploaded photographs as healthy or bleached. The model operates on 16×16 image patches and achieves 90.3% test accuracy on the NOAA PIFSC ESD Coral Bleaching Dataset.

**Environmental Stress Analysis**
An XGBoost classifier independently evaluates user-supplied environmental measurements — sea surface temperature, turbidity, windspeed, thermal stress index, and SSTA — trained on the Global Coral Bleaching Database of 1,252 reef site observations.

**Multimodal Fusion**
Image and environmental predictions are combined through a weighted average, with the ViT model receiving 60% weight and XGBoost 40%, reflecting the relative predictive strength of each modality.

**GradCAM Explainability**
Gradient-weighted Class Activation Mapping is applied to the final transformer encoder layer, producing spatial attention heatmaps that highlight which regions of the image drove the classification decision.

**AI-Generated Summaries**
NVIDIA Llama 3.1 Nemotron (via OpenRouter) synthesises the classification results, environmental conditions, and model agreement into a concise ecological summary for each analysis.

---

## Project Structure

```
coralreef/
├── dataset/
│   ├── bleached/
│   ├── healthy/
│   └── coral_bleaching_cleaned.csv
├── notebooks/
│   ├── best_vit_model/
│   ├── noaa_coral_dataset/
│   ├── app.py
│   ├── train_vit.py
│   ├── xgb_env_model.pkl
│   └── requirements.txt
├── venv/
└── split_data.py
```

---

## Installation

Clone the repository and navigate to the notebooks directory:

```bash
git clone <your-repo-url>
cd coralreef/notebooks
```

Create and activate a virtual environment:

```bash
python -m venv venv
source ../venv/bin/activate        # Linux / Mac
..\venv\Scripts\activate           # Windows
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Running the Application

```bash
streamlit run app.py
```

The application launches at `http://localhost:8501`.

Add your OpenRouter API key to line 4 of `app.py` to enable AI-generated summaries:

```python
OPENROUTER_KEY = "your_key_here"
```

---

## Training

To retrain the ViT or rebuild the XGBoost model:

```bash
python train_vit.py
```

This downloads the NOAA dataset automatically, fine-tunes the ViT for 10 epochs, evaluates on the held-out test set, and saves `best_vit_model/` and `xgb_env_model.pkl` to the working directory.

---

## Model Performance

| Model | Accuracy | Notes |
|---|---|---|
| ViT (NOAA pretrained) | 90.3% | Test set, 1,274 images |
| XGBoost (environmental) | ~72% | 5 environmental features |
| Fusion | 84.8% | 0.6 ViT + 0.4 XGBoost |

---

## Environmental Parameter Reference

| Parameter | Unit | Safe Range | Bleaching Risk |
|---|---|---|---|
| Temperature | °C | 24 – 28 | > 28 |
| Turbidity | 0 – 1 scale | 0.02 – 0.10 | > 0.15 |
| Windspeed | knots | 4 – 10 | < 3 |
| Thermal Stress Index | — | 0 – 1 | > 3 |
| SSTA | °C | -0.5 to +0.5 | > +1.0 |

---

## Limitations

The system performs best on close-up coral imagery with visible surface texture under clear water conditions. Accuracy degrades on wide-angle reef scenes, heavily occluded subjects, and images captured in turbid or low-light environments. The XGBoost model was trained predominantly on Hawaiian Archipelago survey data and may not generalise to reefs in other geographic regions without retraining.

---

## Tech Stack

| Component | Technology |
|---|---|
| Image classification | NOAA ViT (HuggingFace Transformers) |
| Environmental model | XGBoost |
| Explainability | GradCAM |
| AI summaries | NVIDIA Llama 3.1 Nemotron via OpenRouter |
| Interface | Streamlit |
| Image processing | OpenCV, PIL |

---

## Dataset Sources

- **Image data**: NMFS-OSI/NOAA-PIFSC-ESD-CORAL-Bleaching-Dataset — 10,419 annotated 224×224 coral patches from NOAA Ecosystem Sciences Division surveys of the Hawaiian Archipelago (2014–2019)
- **Environmental data**: Global Coral Bleaching Database — 1,252 reef site observations with associated environmental measurements

---

## Research Motivation

Large-scale reef monitoring is constrained by the manual effort required to annotate survey imagery. DeepReef AI explores the use of multimodal AI — combining visual and environmental signals — to automate bleaching assessment while maintaining interpretability through gradient-based visualisation. The system is intended as a research and demonstration tool rather than a production monitoring system.

---

## License

This project is intended for educational and research purposes.
