import os
import random
import shutil

shutil.rmtree("data/train_balanced", ignore_errors=True)
# Paths
source_dir = "data/raw_dataset"
target_dir = "data/train_balanced"

classes = ["rock", "sand", "gravel", "smooth"]

# Limits
limits = {
    "rock": 150,
    "sand": 150,
    "gravel": None,
    "smooth": None
}

for cls in classes:
    src_path = os.path.join(source_dir, cls)
    tgt_path = os.path.join(target_dir, cls)

    os.makedirs(tgt_path, exist_ok=True)

    images = os.listdir(src_path)

    # Shuffle for randomness
    random.shuffle(images)

    # Apply limit if needed
    if limits[cls]:
        images = images[:limits[cls]]

    # Copy selected images
    for img in images:
        src_file = os.path.join(src_path, img)
        tgt_file = os.path.join(tgt_path, img)
        shutil.copy(src_file, tgt_file)

print("Downscaling complete.")