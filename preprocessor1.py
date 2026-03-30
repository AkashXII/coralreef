"""
preprocessing.py — DIP Preprocessing Module
Coral Bleaching Detection System

Stages:
    1. Blue-shift correction      (colorspace transform + histogram stretch)
    2. ROI masking                (HSV segmentation + morphological cleanup)
    3. GLCM texture extraction    (statistical texture analysis)
    4. rCBI heatmap generation    (per-pixel index + pseudocolor mapping)

Usage — process entire dataset:
    python preprocessing.py --input data/raw --output data/clean

Usage — process single image:
    from preprocessing import process_image
    result = process_image("path/to/coral.jpg")
"""

import os
import argparse
import numpy as np
import cv2
from skimage.feature import graycomatrix, graycoprops
from skimage.color import rgb2gray
import matplotlib
# headless — no display needed
import matplotlib.pyplot as plt
from PIL import Image


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 1 — Blue-shift correction
# ─────────────────────────────────────────────────────────────────────────────

def correct_blue_shift(img_bgr: np.ndarray) -> np.ndarray:
    """
    Underwater photos are blue-tinted because water absorbs red/green light.
    We fix this using the Grey World assumption in LAB colorspace:
      - L channel = lightness  (we leave this alone)
      - A channel = green-red axis
      - B channel = blue-yellow axis  (this is where the blue cast lives)

    We stretch A and B so their mean sits at 128 (neutral grey).
    This is called white balance correction.

    Args:
        img_bgr: OpenCV BGR image, uint8

    Returns:
        Corrected BGR image, uint8
    """
    # Convert BGR → LAB
    # LAB separates color from lightness — ideal for color correction
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)

    L, A, B = cv2.split(lab)

    # Stretch A and B channels so their mean = 128 (neutral)
    # Formula: channel = channel - mean(channel) + 128
    # This removes the color cast without changing lightness
    A_corrected = A - A.mean() + 128.0
    B_corrected = B - B.mean() + 128.0

    # Clip back to valid LAB range [0, 255]
    A_corrected = np.clip(A_corrected, 0, 255)
    B_corrected = np.clip(B_corrected, 0, 255)

    # Merge and convert back to BGR
    lab_corrected = cv2.merge([L, A_corrected, B_corrected]).astype(np.uint8)
    corrected_bgr = cv2.cvtColor(lab_corrected, cv2.COLOR_LAB2BGR)

    return corrected_bgr


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 2 — ROI masking
# ─────────────────────────────────────────────────────────────────────────────

def extract_coral_roi(img_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Isolates the coral region using HSV color segmentation.

    Why HSV? HSV separates hue (what color) from saturation (how vivid)
    and value (how bright). This makes it much easier to define color
    ranges than RGB where all three channels change together.

    Coral hues in HSV:
      - Healthy coral: warm browns/yellows/oranges → Hue 5-40°, high saturation
      - Bleached coral: white/grey → any hue, very LOW saturation, HIGH value
      - Water: blue → Hue 90-130°  (we exclude this)
      - Rocks: dark grey → low value (we exclude this)

    After masking we apply morphological opening (erode then dilate)
    to remove small noise blobs and smooth the mask edges.

    Args:
        img_bgr: Blue-shift corrected BGR image

    Returns:
        mask:        Binary mask (255 = coral, 0 = background), uint8
        masked_img:  Original image with background zeroed out, uint8
    """
    # Convert to HSV
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    # OpenCV HSV ranges: H [0-179], S [0-255], V [0-255]

    # Mask 1 — healthy/partially healthy coral (warm hues, vivid)
    lower_coral = np.array([5,  40,  40])
    upper_coral = np.array([40, 255, 255])
    mask_healthy = cv2.inRange(hsv, lower_coral, upper_coral)

    # Mask 2 — bleached coral (any hue, low saturation, high brightness)
    # Low saturation = white/grey = bleached
    lower_bleached = np.array([0,   0, 160])
    upper_bleached = np.array([179, 60, 255])
    mask_bleached = cv2.inRange(hsv, lower_bleached, upper_bleached)

    # Combine both masks — covers healthy AND bleached regions
    mask_combined = cv2.bitwise_or(mask_healthy, mask_bleached)

    # Morphological opening: erode then dilate
    # Removes small noise (specks of sand, highlights) while
    # keeping the large connected coral region intact
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask_opened = cv2.morphologyEx(mask_combined, cv2.MORPH_OPEN, kernel)

    # Morphological closing: dilate then erode
    # Fills small holes inside the coral region
    mask_clean = cv2.morphologyEx(mask_opened, cv2.MORPH_CLOSE, kernel)

    # Keep only the largest connected component
    # (removes small isolated blobs that aren't the main coral)
    mask_clean = _keep_largest_component(mask_clean)

    # Apply mask to original image
    masked_img = cv2.bitwise_and(img_bgr, img_bgr, mask=mask_clean)

    return mask_clean, masked_img


def _keep_largest_component(mask: np.ndarray) -> np.ndarray:
    """
    Finds all connected white blobs in the mask and keeps only the largest.
    Removes isolated noise pixels that passed the color threshold.
    """
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask, connectivity=8
    )

    if num_labels <= 1:
        # No components found — return original mask unchanged
        return mask

    # stats[:,4] = area of each component. Skip label 0 (background)
    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    clean_mask = np.zeros_like(mask)
    clean_mask[labels == largest_label] = 255

    return clean_mask


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 3 — GLCM texture feature extraction
# ─────────────────────────────────────────────────────────────────────────────

def extract_glcm_features(img_bgr: np.ndarray,
                           mask: np.ndarray) -> dict[str, float]:
    """
    Extracts texture features from the coral ROI using the
    Grey-Level Co-occurrence Matrix (GLCM).

    What is GLCM?
    For each pair of pixels (at a given distance and direction),
    GLCM counts how often value i appears next to value j.
    The resulting matrix encodes the texture pattern of the image.

    Why does this matter for bleaching?
    Healthy coral has regular, repeating ridge/polyp patterns →
      high energy, high correlation, low contrast
    Bleached coral loses pigment and structure →
      texture becomes flat, homogeneous, low energy

    Features extracted:
      contrast    — local intensity variation (high = rough texture)
      correlation — linear dependency between pixels (high = regular pattern)
      energy      — sum of squared elements (high = uniform/regular texture)
      homogeneity — closeness of distribution to diagonal (high = smooth)

    Args:
        img_bgr: Corrected BGR image
        mask:    Binary coral mask

    Returns:
        dict of 4 texture features (mean across 4 directions)
    """
    # Convert to greyscale — GLCM works on intensity values
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    grey = (rgb2gray(img_rgb) * 255).astype(np.uint8)

    # Apply mask — only analyse coral pixels
    grey_masked = grey.copy()
    grey_masked[mask == 0] = 0

    # Quantize to 64 grey levels for efficiency
    # (reduces GLCM from 256×256 to 64×64)
    grey_quantized = (grey_masked // 4).astype(np.uint8)

    # Compute GLCM at distance=1, 4 directions (0°, 45°, 90°, 135°)
    # Multiple directions makes features rotation-invariant
    distances  = [1]
    angles     = [0, np.pi/4, np.pi/2, 3*np.pi/4]

    glcm = graycomatrix(
        grey_quantized,
        distances=distances,
        angles=angles,
        levels=64,
        symmetric=True,
        normed=True,      # normalize so values sum to 1
    )

    # Extract properties — shape is (1, 4): 1 distance × 4 angles
    # We take the mean across all 4 angles for rotation invariance
    features = {
        "glcm_contrast":    float(graycoprops(glcm, "contrast").mean()),
        "glcm_correlation": float(graycoprops(glcm, "correlation").mean()),
        "glcm_energy":      float(graycoprops(glcm, "energy").mean()),
        "glcm_homogeneity": float(graycoprops(glcm, "homogeneity").mean()),
    }

    return features


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 4 — rCBI heatmap visualization
# ─────────────────────────────────────────────────────────────────────────────

def generate_rcbi_heatmap(img_bgr: np.ndarray,
                           mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Computes the relative Coral Bleaching Index per pixel and
    generates a false-color heatmap overlaid on the original image.

    rCBI = (R - B) / (R + B + ε)

    Why this works:
      Healthy coral contains zooxanthellae (symbiotic algae) that
      absorb red light. When they die (bleaching), red reflectance
      increases and blue stays the same → rCBI goes UP.
      So: high rCBI = likely bleached, low rCBI = likely healthy.

    Colormap: RdYlBu_r
      Blue  → low rCBI  → healthy
      Yellow → mid rCBI → partial bleaching
      Red   → high rCBI → bleached

    Args:
        img_bgr: Corrected BGR image
        mask:    Binary coral mask

    Returns:
        rcbi_map:   Raw rCBI values per pixel, float32, shape H×W
        overlay:    Original image with heatmap blended on top, uint8
    """
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

    R = img_rgb[:, :, 0]
    B = img_rgb[:, :, 2]
    eps = 1e-6

    # Compute rCBI — values in [-1, 1]
    rcbi_map = (R - B) / (R + B + eps)

    # Zero out non-coral pixels
    rcbi_masked = rcbi_map.copy()
    rcbi_masked[mask == 0] = np.nan   # NaN = transparent in colormap

    # Normalize rCBI to [0, 1] for colormap application
    # We clip to [-0.5, 0.5] to focus the color range on meaningful values
    rcbi_norm = np.clip(rcbi_masked, -0.5, 0.5)
    rcbi_norm = (rcbi_norm + 0.5)    # shift to [0, 1]

    # Apply RdYlBu_r colormap (reversed: blue=low, red=high)
    cmap = plt.get_cmap("RdYlBu_r")
    colored = cmap(rcbi_norm)         # H×W×4 RGBA, float [0,1]

    # Convert to uint8 BGR for OpenCV blending
    colored_bgr = cv2.cvtColor(
        (colored[:, :, :3] * 255).astype(np.uint8),
        cv2.COLOR_RGB2BGR
    )

    # Alpha blend: 50% original, 50% heatmap — only on coral region
    alpha = 0.5
    overlay = img_bgr.copy()
    coral_pixels = mask > 0
    overlay[coral_pixels] = cv2.addWeighted(
        img_bgr, 1 - alpha,
        colored_bgr, alpha,
        0
    )[coral_pixels]

    return rcbi_map, overlay


# ─────────────────────────────────────────────────────────────────────────────
# Main processing function
# ─────────────────────────────────────────────────────────────────────────────

def process_image(image_path: str,
                  save_heatmap_path: str = None) -> dict:
    """
    Runs all 4 DIP stages on a single image.

    Args:
        image_path:        Path to raw coral image
        save_heatmap_path: If given, saves heatmap PNG to this path

    Returns dict with:
        clean_bgr    — blue-shift corrected image (numpy BGR)
        mask         — coral ROI binary mask (numpy)
        glcm_features— dict of 4 texture features
        rcbi_map     — per-pixel rCBI values (numpy float32)
        heatmap_bgr  — rCBI overlay image (numpy BGR)
        rcbi_mean    — mean rCBI over coral region (useful feature for XGBoost)
        coral_coverage — fraction of image that is coral (quality check)
    """
    # Load image
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")

    # ── Stage 1: Blue-shift correction ──
    clean_bgr = correct_blue_shift(img_bgr)

    # ── Stage 2: ROI masking ──
    mask, masked_bgr = extract_coral_roi(clean_bgr)

    # Quality check — if coral coverage is very low, flag this image
    coral_coverage = (mask > 0).sum() / mask.size

    # ── Stage 3: GLCM texture features ──
    glcm_features = extract_glcm_features(masked_bgr, mask)

    # ── Stage 4: rCBI heatmap ──
    rcbi_map, heatmap_bgr = generate_rcbi_heatmap(clean_bgr, mask)

    # Mean rCBI over coral region only (good feature for XGBoost)
    coral_pixels = rcbi_map[mask > 0]
    rcbi_mean = float(coral_pixels.mean()) if len(coral_pixels) > 0 else 0.0

    # Save heatmap if path provided
    if save_heatmap_path:
        cv2.imwrite(save_heatmap_path, heatmap_bgr)

    return {
        "clean_bgr":       clean_bgr,
        "mask":            mask,
        "glcm_features":   glcm_features,
        "rcbi_map":        rcbi_map,
        "heatmap_bgr":     heatmap_bgr,
        "rcbi_mean":       rcbi_mean,
        "coral_coverage":  coral_coverage,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Dataset batch processing
# ─────────────────────────────────────────────────────────────────────────────

def process_dataset(input_dir: str, output_dir: str):
    """
    Processes all images in input_dir (expects healthy/ and bleached/ subfolders)
    Saves cleaned images and heatmaps to output_dir.
    Prints a warning for any image with coral_coverage < 10%.
    """
    import json

    classes = ["healthy", "bleached"]
    all_features = {}

    for split in ["train", "val"]:
        for cls in classes:
            in_path  = os.path.join(input_dir,  split, cls)
            out_path = os.path.join(output_dir, split, cls)
            heat_path = os.path.join(output_dir, split + "_heatmaps", cls)

            os.makedirs(out_path,  exist_ok=True)
            os.makedirs(heat_path, exist_ok=True)

            image_files = [
                f for f in os.listdir(in_path)
                if f.lower().endswith((".jpg", ".jpeg", ".png"))
            ]

            print(f"\nProcessing {split}/{cls} — {len(image_files)} images")

            for fname in image_files:
                src  = os.path.join(in_path, fname)
                dst  = os.path.join(out_path, fname)
                hdst = os.path.join(heat_path, fname.replace(".", "_heatmap."))

                try:
                    result = process_image(src, save_heatmap_path=hdst)

                    # Save cleaned image
                    cv2.imwrite(dst, result["clean_bgr"])

                    # Warn if barely any coral detected
                    if result["coral_coverage"] < 0.10:
                        print(f"  LOW COVERAGE WARNING: {fname} "
                              f"({result['coral_coverage']:.1%} coral pixels)")

                    # Store GLCM + rCBI features keyed by filename
                    all_features[f"{split}/{cls}/{fname}"] = {
                        **result["glcm_features"],
                        "rcbi_mean": result["rcbi_mean"],
                        "coral_coverage": result["coral_coverage"],
                        "label": 1 if cls == "bleached" else 0,
                    }

                except Exception as e:
                    print(f"  ERROR processing {fname}: {e}")

    # Save all extracted features to JSON
    # These can optionally be added to XGBoost tabular features
    features_path = os.path.join(output_dir, "dip_features.json")
    with open(features_path, "w") as f:
        json.dump(all_features, f, indent=2)

    print(f"\nDone. Cleaned images saved to: {output_dir}")
    print(f"DIP features saved to: {features_path}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="DIP preprocessing for coral bleaching dataset"
    )
    parser.add_argument("--input",  required=True,
                        help="Root data dir with train/val/healthy/bleached")
    parser.add_argument("--output", required=True,
                        help="Output dir for cleaned images + heatmaps")
    args = parser.parse_args()

    process_dataset(args.input, args.output)