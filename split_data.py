import os, shutil, random

random.seed(42)

for label in ['healthy', 'bleached']:
    images = os.listdir(f'dataset/{label}')
    random.shuffle(images)
    
    split = int(0.8 * len(images))
    train_imgs = images[:split]
    val_imgs = images[split:]
    
    os.makedirs(f'dataset/train/{label}', exist_ok=True)
    os.makedirs(f'dataset/val/{label}', exist_ok=True)
    
    for img in train_imgs:
        shutil.copy(f'dataset/{label}/{img}', f'dataset/train/{label}/{img}')
    for img in val_imgs:
        shutil.copy(f'dataset/{label}/{img}', f'dataset/val/{label}/{img}')

print("Done!")
print(f"Train healthy: {len(os.listdir('dataset/train/healthy'))}")
print(f"Train bleached: {len(os.listdir('dataset/train/bleached'))}")
print(f"Val healthy: {len(os.listdir('dataset/val/healthy'))}")
print(f"Val bleached: {len(os.listdir('dataset/val/bleached'))}")