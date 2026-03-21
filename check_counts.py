import os

for cls in ["gravel", "rock", "sand", "smooth"]:
    path = f"data/train_balanced/{cls}"
    print(cls, len(os.listdir(path)))