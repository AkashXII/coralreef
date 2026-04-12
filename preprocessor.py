"""
preprocessing.py — DIP Preprocessing Module
Coral Bleaching Detection System

Stages:
    1. Blue-shift correction      (LAB colorspace white balance)
    2. Bilateral filtering        (noise removal, edge-preserving)
    3. CLAHE contrast enhancement (local contrast for underwater imagery)
    4. GLCM texture extraction    (statistical texture analysis)
    5. rCBI heatmap generation    (per-pixel index, visualization only)

Usage — process entire dataset:
    python preprocessing.py --input data/raw --output data/clean

Usage — single image (from notebook or another script):
    from preprocessing import process_image
    result = process_image("path/to/coral.jpg")
"""

import os
import argparse
import json
import numpy as np
import cv2
from skimage.feature import graycomatrix, graycoprops
from skimage.color import rgb2gray
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

#Blue-shift correction
def correct_blue_shift(img_bgr: np.ndarray) -> np.ndarray:
    """
    Underwater photos are blue-tinted because water absorbs red/green
    light faster than blue within the first few meters of depth.

    Fix: convert to LAB colorspace and re-center the A and B channels
    so their mean sits at 128 (neutral grey). This is the Grey World
    white balance assumption.

      L = lightness        (leave untouched)
      A = green-red axis   (re-center)
      B = blue-yellow axis (re-center — this is where the cast lives)

    Args:
        img_bgr: OpenCV BGR image, uint8
    Returns:
        Blue-shift corrected BGR image, uint8
    """
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    L, A, B = cv2.split(lab)

    A_corrected = np.clip(A - A.mean() + 128.0, 0, 255)
    B_corrected = np.clip(B - B.mean() + 128.0, 0, 255)

    lab_corrected = cv2.merge([L, A_corrected, B_corrected]).astype(np.uint8)
    return cv2.cvtColor(lab_corrected, cv2.COLOR_LAB2BGR)


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 2 — Bilateral filtering (noise removal)
# ─────────────────────────────────────────────────────────────────────────────

def apply_bilateral_filter(img_bgr: np.ndarray) -> np.ndarray:
    """
    Underwater images contain speckle noise from light scattering
    through water particles. A standard Gaussian blur removes noise
    but also blurs coral texture edges — which are exactly what the
    CNN needs to distinguish healthy vs bleached coral.

    Bilateral filtering smooths noise in flat uniform regions but
    preserves sharp edges by only averaging pixels that are both
    spatially close AND similar in intensity.

    Parameters:
        d=9           — diameter of pixel neighbourhood considered
        sigmaColor=75 — intensity similarity tolerance
                        (higher = smoother, less edge-preserving)
        sigmaSpace=75 — spatial distance tolerance
                        (higher = farther pixels can influence result)

    Args:
        img_bgr: Blue-shift corrected BGR image, uint8
    Returns:
        Noise-reduced BGR image, uint8
    """
    return cv2.bilateralFilter(img_bgr, d=5, sigmaColor=40, sigmaSpace=40)


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 3 — CLAHE contrast enhancement
# ─────────────────────────────────────────────────────────────────────────────

def apply_clahe(img_bgr: np.ndarray) -> np.ndarray:
    """
    Underwater images are naturally low contrast — water scatters and
    absorbs light, reducing the difference between bright and dark regions.
    Low contrast makes it harder for the CNN to see coral texture patterns.

    CLAHE = Contrast Limited Adaptive Histogram Equalization.

    Unlike regular histogram equalization (which works globally and
    often over-amplifies highlights), CLAHE divides the image into
    small tiles and equalizes each independently, then blends them.
    This enhances local contrast — subtle coral texture becomes more
    visible without blowing out bright regions.

    clipLimit=2.0 prevents over-amplifying noise in uniform regions.
    tileGridSize=(8,8) sets the tile size for local equalization.

    Applied to L channel in LAB space only — so colour (A, B channels)
    is preserved. Only lightness contrast is enhanced.

    Args:
        img_bgr: Bilateral-filtered BGR image, uint8
    Returns:
        Contrast-enhanced BGR image, uint8
    """
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)

    clahe     = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    L_enhanced = clahe.apply(L)

    lab_enhanced = cv2.merge([L_enhanced, A, B])
    return cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 4 — GLCM texture feature extraction
# ─────────────────────────────────────────────────────────────────────────────

def extract_glcm_features(img_bgr: np.ndarray) -> dict:
    """
    Extracts texture features from the fully preprocessed image
    using the Grey-Level Co-occurrence Matrix (GLCM).

    Why texture matters for bleaching:
      Healthy coral has regular repeating ridge/polyp patterns
      Bleached coral loses structural regularity and pigment

    Features (mean across 4 directions for rotation invariance):
      contrast    — local intensity variation  (high = rough texture)
      correlation — linear dependency          (high = regular pattern)
      energy      — sum of squared elements    (high = uniform texture)
      homogeneity — closeness to diagonal      (high = smooth)

    Args:
        img_bgr: Fully preprocessed BGR image (after stages 1-3)
    Returns:
        dict of 4 GLCM texture features
    """
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    grey    = (rgb2gray(img_rgb) * 255).astype(np.uint8)

    # Quantize to 64 grey levels — reduces matrix from 256x256 to 64x64
    grey_quantized = (grey // 4).astype(np.uint8)

    glcm = graycomatrix(
        grey_quantized,
        distances=[1],
        angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
        levels=64,
        symmetric=True,
        normed=True,
    )

    return {
        "glcm_contrast":    float(graycoprops(glcm, "contrast").mean()),
        "glcm_correlation": float(graycoprops(glcm, "correlation").mean()),
        "glcm_energy":      float(graycoprops(glcm, "energy").mean()),
        "glcm_homogeneity": float(graycoprops(glcm, "homogeneity").mean()),
    }


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 5 — rCBI heatmap (visualization only)
# ─────────────────────────────────────────────────────────────────────────────

def generate_rcbi_heatmap(img_bgr: np.ndarray) -> tuple:
    """
    Computes the relative Coral Bleaching Index per pixel and
    generates a false-color heatmap for web app visualization.

    rCBI = (R - B) / (R + B + epsilon)

    Note: after blue-shift correction, higher rCBI empirically
    corresponds to warmer pigmented regions, lower rCBI to cooler
    depigmented regions. Used for spatial visualization only —
    not as a classification feature.

    Colormap RdYlBu_r:
      Red  → high rCBI (warm/pigmented)
      Blue → low rCBI  (cool/depigmented)

    Args:
        img_bgr: Fully preprocessed BGR image
    Returns:
        rcbi_map: Per-pixel rCBI float32 array H×W, range [-1, 1]
        overlay:  Image with heatmap alpha-blended at 50%, uint8
    """
    img_rgb  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    R, B, eps = img_rgb[:,:,0], img_rgb[:,:,2], 1e-6

    rcbi_map  = (R - B) / (R + B + eps)
    rcbi_norm = (np.clip(rcbi_map, -0.5, 0.5) + 0.5)

    cmap        = plt.get_cmap("RdYlBu_r")
    colored_bgr = cv2.cvtColor(
        (cmap(rcbi_norm)[:, :, :3] * 255).astype(np.uint8),
        cv2.COLOR_RGB2BGR
    )

    overlay = cv2.addWeighted(img_bgr, 0.5, colored_bgr, 0.5, 0)
    return rcbi_map, overlay


# ─────────────────────────────────────────────────────────────────────────────
# Main processing function
# ─────────────────────────────────────────────────────────────────────────────

def process_image(image_path: str, save_heatmap_path: str = None) -> dict:
    """
    Runs all 5 DIP stages on a single image.

    Pipeline order:
        raw image
            → Stage 1: blue-shift correction  (correct_blue_shift)
            → Stage 2: bilateral filter        (apply_bilateral_filter)
            → Stage 3: CLAHE enhancement       (apply_clahe)
            → Stage 4: GLCM features           (extract_glcm_features)
            → Stage 5: rCBI heatmap            (generate_rcbi_heatmap)

    Args:
        image_path:        Path to raw coral image
        save_heatmap_path: If provided, saves heatmap PNG here

    Returns dict:
        clean_bgr     — fully preprocessed image (BGR uint8)
        glcm_features — dict of 4 texture features
        rcbi_map      — per-pixel rCBI float32 H×W
        heatmap_bgr   — rCBI overlay image (BGR uint8)
        rcbi_mean     — mean rCBI over full image
    """
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")

    # Stage 1
    corrected = correct_blue_shift(img_bgr)

    # Stage 2
    denoised  = apply_bilateral_filter(corrected)

    # Stage 3
    clean_bgr = apply_clahe(denoised)

    # Stage 4
    glcm_features = extract_glcm_features(clean_bgr)

    # Stage 5
    rcbi_map, heatmap_bgr = generate_rcbi_heatmap(clean_bgr)
    rcbi_mean = float(rcbi_map.mean())

    if save_heatmap_path:
        cv2.imwrite(save_heatmap_path, heatmap_bgr)

    return {
        "clean_bgr":     clean_bgr,
        "glcm_features": glcm_features,
        "rcbi_map":      rcbi_map,
        "heatmap_bgr":   heatmap_bgr,
        "rcbi_mean":     rcbi_mean,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Dataset batch processing
# ─────────────────────────────────────────────────────────────────────────────

def process_dataset(input_dir: str, output_dir: str):
    """
    Processes all images in input_dir maintaining folder structure.
    Expects:
        input_dir/train/healthy/
        input_dir/train/bleached/
        input_dir/val/healthy/
        input_dir/val/bleached/

    Saves:
        output_dir/train/healthy/       ← cleaned images
        output_dir/train_heatmaps/      ← rCBI overlays
        output_dir/dip_features.json    ← GLCM + rCBI per image
    """
    classes      = ["healthy", "bleached"]
    all_features = {}

    for split in ["train", "val"]:
        for cls in classes:
            in_path   = os.path.join(input_dir,  split, cls)
            out_path  = os.path.join(output_dir, split, cls)
            heat_path = os.path.join(output_dir, split + "_heatmaps", cls)

            os.makedirs(out_path,  exist_ok=True)
            os.makedirs(heat_path, exist_ok=True)

            image_files = [
                f for f in os.listdir(in_path)
                if f.lower().endswith((".jpg", ".jpeg", ".png"))
            ]

            print(f"\nProcessing {split}/{cls} — {len(image_files)} images")

            for fname in image_files:
                src  = os.path.join(in_path,   fname)
                dst  = os.path.join(out_path,  fname)
                hdst = os.path.join(heat_path,
                                    fname.replace(".", "_heatmap."))
                try:
                    result = process_image(src, save_heatmap_path=hdst)
                    cv2.imwrite(dst, result["clean_bgr"])

                    all_features[f"{split}/{cls}/{fname}"] = {
                        **result["glcm_features"],
                        "rcbi_mean": result["rcbi_mean"],
                        "label":     1 if cls == "bleached" else 0,
                    }

                except Exception as e:
                    print(f"  ERROR: {fname} — {e}")

    features_path = os.path.join(output_dir, "dip_features.json")
    with open(features_path, "w") as f:
        json.dump(all_features, f, indent=2)

    print(f"\nDone. Cleaned images → {output_dir}")
    print(f"DIP features       → {features_path}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="DIP preprocessing for coral bleaching dataset"
    )
    parser.add_argument("--input",  required=True,
                        help="Root dir with train/val/healthy/bleached")
    parser.add_argument("--output", required=True,
                        help="Output dir for cleaned images and heatmaps")
    args = parser.parse_args()
    process_dataset(args.input, args.output)