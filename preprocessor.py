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
def correct_blue_shift(img_bgr: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    L, A, B = cv2.split(lab)

    A_corrected = np.clip(A - A.mean() + 128.0, 0, 255)
    B_corrected = np.clip(B - B.mean() + 128.0, 0, 255)

    lab_corrected = cv2.merge([L, A_corrected, B_corrected]).astype(np.uint8)
    return cv2.cvtColor(lab_corrected, cv2.COLOR_LAB2BGR)

def apply_bilateral_filter(img_bgr: np.ndarray) -> np.ndarray:
    return cv2.bilateralFilter(img_bgr, d=5, sigmaColor=40, sigmaSpace=40)


def apply_clahe(img_bgr: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)

    clahe     = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    L_enhanced = clahe.apply(L)

    lab_enhanced = cv2.merge([L_enhanced, A, B])
    return cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)


def extract_glcm_features(img_bgr: np.ndarray) -> dict:
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    grey    = (rgb2gray(img_rgb) * 255).astype(np.uint8)
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

def generate_rcbi_heatmap(img_bgr: np.ndarray) -> tuple:
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

def process_image(image_path: str, save_heatmap_path: str = None) -> dict:

    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")
    corrected = correct_blue_shift(img_bgr)
    denoised  = apply_bilateral_filter(corrected)
    clean_bgr = apply_clahe(denoised)
    glcm_features = extract_glcm_features(clean_bgr)
#not added to the dashboard, could add later on ig?
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

def process_dataset(input_dir: str, output_dir: str):
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