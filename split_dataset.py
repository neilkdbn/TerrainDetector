import os
import shutil
import random

# Change this to your raw dataset path
SOURCE_DIR = "raw_dataset"

BASE_DIR = "dataset"

CLASSES = ["smooth", "gravel", "sand", "rock"]

SPLIT_RATIO = (0.7, 0.2, 0.1)

for cls in CLASSES:
    src = os.path.join(SOURCE_DIR, cls)
    images = os.listdir(src)
    random.shuffle(images)

    total = len(images)
    train_end = int(SPLIT_RATIO[0] * total)
    val_end = train_end + int(SPLIT_RATIO[1] * total)

    splits = {
        "train": images[:train_end],
        "val": images[train_end:val_end],
        "test": images[val_end:]
    }

    for split, imgs in splits.items():
        dest_dir = os.path.join(BASE_DIR, split, cls)
        os.makedirs(dest_dir, exist_ok=True)

        for img in imgs:
            shutil.copy(
                os.path.join(src, img),
                os.path.join(dest_dir, img)
            )

print("Dataset split complete!")