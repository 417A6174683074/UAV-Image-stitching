from datasets import load_dataset
from pathlib import Path
import requests
from tqdm import tqdm

# Charger le dataset
dataset = load_dataset("RussRobin/Aerial234", split="train")

# Créer un dossier local pour stocker les images
save_dir = Path("aerial234_images")
save_dir.mkdir(parents=True, exist_ok=True)

# Télécharger chaque image
for i, example in enumerate(tqdm(dataset)):
    img_url = example["image"]["url"]
    img_ext = img_url.split(".")[-1]
    img_path = save_dir / f"img_{i:05d}.{img_ext}"

    if not img_path.exists():
        r = requests.get(img_url)
        with open(img_path, "wb") as f:
            f.write(r.content)

print(f"Téléchargement terminé : {len(dataset)} images dans {save_dir}")
