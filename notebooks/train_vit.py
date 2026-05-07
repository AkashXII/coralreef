import torch, numpy as np, matplotlib.pyplot as plt, pickle, os, cv2
import pandas as pd
from huggingface_hub import snapshot_download
from transformers import AutoImageProcessor, AutoModelForImageClassification
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from PIL import Image

DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_NAME = "NMFS-OSI/noaa-esd-coral-bleaching-vit-classifier-v1"
EPOCHS     = 10
BATCH_SIZE = 8
LR         = 2e-5
CSV_PATH   = "../dataset/coral_bleaching_cleaned.csv"
DATA_DIR   = "./noaa_coral_dataset"

print(f"Using: {DEVICE}")

if not os.path.exists(DATA_DIR):
    print("Downloading NOAA dataset...")
    snapshot_download(
        repo_id="NMFS-OSI/NOAA-PIFSC-ESD-CORAL-Bleaching-Dataset",
        repo_type="dataset",
        local_dir=DATA_DIR
    )

processor = AutoImageProcessor.from_pretrained(MODEL_NAME)


def find_images(data_dir):
    paths, labels = [], []
    label_map = {"CORAL": 0, "CORAL_BL": 1}
    for split in ["train", "validation", "test"]:
        split_dir = os.path.join(data_dir, split)
        if not os.path.exists(split_dir):
            split_dir = data_dir
        for folder, label in label_map.items():
            folder_path = os.path.join(split_dir, folder)
            if not os.path.exists(folder_path):
                continue
            for fname in os.listdir(folder_path):
                if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                    paths.append(os.path.join(folder_path, fname))
                    labels.append(label)
    return paths, labels


class CoralDataset(Dataset):
    def __init__(self, paths, labels):
        self.paths  = paths
        self.labels = labels

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img    = Image.open(self.paths[idx]).convert("RGB")
        inputs = processor(images=img, return_tensors="pt")
        pv     = inputs["pixel_values"].squeeze(0)
        return pv, torch.tensor(self.labels[idx], dtype=torch.long)


all_paths, all_labels = find_images(DATA_DIR)
print(f"Total images found: {len(all_paths)}")
print(f"Bleached: {sum(all_labels)}  Healthy: {len(all_labels)-sum(all_labels)}")

tr_p, tmp_p, tr_l, tmp_l = train_test_split(all_paths, all_labels, test_size=0.3, random_state=42, stratify=all_labels)
vl_p, te_p, vl_l, te_l   = train_test_split(tmp_p, tmp_l, test_size=0.5, random_state=42, stratify=tmp_l)

train_loader = DataLoader(CoralDataset(tr_p, tr_l), batch_size=BATCH_SIZE, shuffle=True,  num_workers=4, pin_memory=True)
val_loader   = DataLoader(CoralDataset(vl_p, vl_l), batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
test_loader  = DataLoader(CoralDataset(te_p, te_l), batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

model = AutoModelForImageClassification.from_pretrained(
    MODEL_NAME, num_labels=2,
    id2label={0: "CORAL", 1: "CORAL_BL"},
    label2id={"CORAL": 0, "CORAL_BL": 1},
    ignore_mismatched_sizes=True
).to(DEVICE)

model.vit.encoder.gradient_checkpointing = True
torch.backends.cuda.matmul.allow_tf32 = True

optimizer = AdamW(model.parameters(), lr=LR, weight_decay=0.01)
scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)


def run_eval(loader):
    model.eval()
    correct = total = 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for pv, labels in loader:
            pv, labels = pv.to(DEVICE), labels.to(DEVICE)
            preds = model(pixel_values=pv).logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total   += labels.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    return correct / total, np.array(all_preds), np.array(all_labels)


best_val_acc = 0.0

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for pv, labels in train_loader:
        pv, labels = pv.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad(set_to_none=True)
        loss = model(pixel_values=pv, labels=labels).loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
    torch.cuda.empty_cache()

    val_acc, _, _ = run_eval(val_loader)
    scheduler.step()
    print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {total_loss/len(train_loader):.4f} | Val Acc: {val_acc:.4f}")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        model.save_pretrained("best_vit_model")
        processor.save_pretrained("best_vit_model")

print(f"\nBest Val Acc: {best_val_acc:.4f}")

model = AutoModelForImageClassification.from_pretrained("best_vit_model").to(DEVICE)
test_acc, test_preds, test_labels = run_eval(test_loader)
print(f"Test Accuracy: {test_acc:.4f}")
print(classification_report(test_labels, test_preds, target_names=["Healthy", "Bleached"]))

disp = ConfusionMatrixDisplay(confusion_matrix(test_labels, test_preds), display_labels=["Healthy", "Bleached"])
disp.plot(cmap="Blues")
plt.title("Confusion Matrix — ViT (NOAA)")
plt.tight_layout()
plt.savefig("cm_vit.png", dpi=150)
plt.show()


def get_gradcam(model, pixel_values):
    model.eval()
    saved = {}

    def fwd_hook(module, inp, out):
        saved["acts"] = out.detach()

    def bwd_hook(module, grad_input, grad_output):
        saved["grads"] = grad_output[0].detach()

    h1 = model.vit.encoder.layer[-2].register_forward_hook(fwd)
    h2 = model.vit.encoder.layer[-2].register_full_backward_hook(bwd)

    model.zero_grad()
    out = model(pixel_values=pixel_values)

    class_idx = out.logits.argmax(dim=-1).item()
    score = out.logits[0, class_idx]
    score.backward()

    h1.remove()
    h2.remove()

    acts = saved["acts"][0, 1:].cpu()   # [n_patches, hidden]
    grads = saved["grads"][0, 1:].cpu() # [n_patches, hidden]

    weights = grads.mean(dim=0)         # [hidden]
    cam = (acts * weights).sum(dim=-1)   # [n_patches]

    cam = cam.numpy()
    cam = np.maximum(cam, 0)
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

    size = int(np.sqrt(len(cam)))
    return cv2.resize(cam.reshape(size, size), (224, 224))


def show_gradcam(model, img_path, label, idx):
    img_pil = Image.open(img_path).convert("RGB")
    inputs  = processor(images=img_pil, return_tensors="pt")
    pv      = inputs["pixel_values"].to(DEVICE)
    cam     = get_gradcam(model, pv)
    orig    = np.array(img_pil.resize((224, 224))).astype(np.float32) / 255.0
    heat    = cv2.cvtColor(cv2.applyColorMap(np.uint8(255*cam), cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB) / 255.0
    overlay = np.clip(0.65*orig + 0.35*heat, 0, 1)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, img, title in zip(axes, [orig, cam, overlay],
        [f"Original ({'Bleached' if label==1 else 'Healthy'})", "GradCAM", "Overlay"]):
        ax.imshow(img, cmap="jet" if title == "GradCAM" else None)
        ax.set_title(title)
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(f"gradcam_vit_{idx}.png", dpi=150)
    plt.show()


for i, idx in enumerate([0, 100, 500]):
    if idx < len(te_p):
        show_gradcam(model, te_p[idx], te_l[idx], i)


if os.path.exists(CSV_PATH):
    df    = pd.read_csv(CSV_PATH)
    feats = ["Temperature_Mean", "Turbidity", "Windspeed", "Thermal_Stress_Index", "SSTA"]
    X     = df[feats].values
    y     = (df["Percent_Bleaching"] > 20).astype(int).values
    Xt, Xv, yt, yv = train_test_split(X, y, test_size=0.2, random_state=42)
    xgb = XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.05, eval_metric="logloss")
    xgb.fit(Xt, yt, eval_set=[(Xv, yv)], verbose=False)
    print(f"XGBoost Accuracy: {(xgb.predict(Xv)==yv).mean():.4f}")
    with open("xgb_env_model.pkl", "wb") as f:
        pickle.dump({"model": xgb, "features": feats}, f)
    print("Saved: best_vit_model/  xgb_env_model.pkl")
else:
    print(f"CSV not found at {CSV_PATH} — skipping XGBoost")