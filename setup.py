import os

folder_names =  [
    "src/data/Input", 
    "src/data/Preprocessed/", 
    "src/results/models/", 
    "src/results/preprocessing/"
    ]

for folder in folder_names:
    if not os.path.exists(folder):
        os.makedirs(folder)
        print(f"📁 {folder} créé")
    else:
        print(f"✅ {folder} déjà présent")