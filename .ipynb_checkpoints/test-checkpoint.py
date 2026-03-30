# ── Install dependencies if needed ──────────────────────────────────────────
# Run this cell once, then restart kernel
# !pip install opencv-python scikit-image matplotlib pillow numpy

import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys

# ── Point this to wherever your preprocessing.py lives ──────────────────────
sys.path.insert(0, "/path/to/your/preprocessing.py".rsplit("/", 1)[0])
from preprocessing import (
    correct_blue_shift,
    extract_coral_roi,
    extract_glcm_features,
    generate_rcbi_heatmap,
)

# ── CHANGE THIS to any image you want to test ────────────────────────────────
IMAGE_PATH = "/path/to/your/coral_image.jpg"


# ── Helper ───────────────────────────────────────────────────────────────────
def bgr_to_rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


# ── Run all stages ───────────────────────────────────────────────────────────
img_bgr    = cv2.imread(IMAGE_PATH)
assert img_bgr is not None, f"Could not load image at: {IMAGE_PATH}"

corrected        = correct_blue_shift(img_bgr)
mask, masked     = extract_coral_roi(corrected)
glcm             = extract_glcm_features(masked, mask)
rcbi_map, heatmap = generate_rcbi_heatmap(corrected, mask)

coral_pixels = rcbi_map[mask > 0]
rcbi_mean    = float(coral_pixels.mean()) if len(coral_pixels) > 0 else 0.0
coverage     = (mask > 0).sum() / mask.size


# ── Display ──────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle(f"DIP Preprocessing — {IMAGE_PATH.split('/')[-1]}", fontsize=13)

panels = [
    (bgr_to_rgb(img_bgr),    "Original (raw)"),
    (bgr_to_rgb(corrected),  "Stage 1: Blue-shift corrected"),
    (mask,                   "Stage 2: ROI mask (binary)"),
    (bgr_to_rgb(masked),     "Stage 2: Masked image"),
    (bgr_to_rgb(heatmap),    f"Stage 4: rCBI heatmap  (mean={rcbi_mean:.3f})"),
    (rcbi_map,               "Stage 4: Raw rCBI values"),
]

for ax, (img, title) in zip(axes.flat, panels):
    if img.ndim == 2:
        if img.dtype == np.float32 or img.max() <= 1.0:
            im = ax.imshow(img, cmap="RdYlBu_r", vmin=-0.5, vmax=0.5)
            plt.colorbar(im, ax=ax, fraction=0.046)
        else:
            ax.imshow(img, cmap="gray")
    else:
        ax.imshow(img)
    ax.set_title(title, fontsize=10)
    ax.axis("off")

plt.tight_layout()
plt.show()


# ── Print feature values ─────────────────────────────────────────────────────
print("\nGLCM Texture Features:")
for k, v in glcm.items():
    print(f"  {k}: {v:.4f}")
print(f"\nrCBI mean (coral region): {rcbi_mean:.4f}")
print(f"Coral coverage:           {coverage:.1%}")


# ── Optional: view with cv2.imshow ───────────────────────────────────────────
# Uncomment these lines if you want the OpenCV popup window
# Note: cv2.imshow may not work in all Jupyter environments (works in VS Code)

# cv2.imshow("Original",        img_bgr)
# cv2.imshow("Blue corrected",  corrected)
# cv2.imshow("ROI mask",        mask)
# cv2.imshow("Masked image",    masked)
# cv2.imshow("rCBI heatmap",    heatmap)
# cv2.waitKey(0)
# cv2.destroyAllWindows()