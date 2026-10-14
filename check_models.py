
from pathlib import Path
import numpy as np
import tensorflow as tf
from PIL import Image

ROOT = Path(__file__).resolve().parent
IMAGE_PATH = ROOT / "data" / "raw" / "chest_xray" / "test" / "NORMAL" / "IM-0001-0001.jpeg"

MODEL_PATHS = [
    ROOT / "models" / "best_model.keras",
    ROOT / "models" / "medical_image_model.keras",
]

if not IMAGE_PATH.exists():
    raise FileNotFoundError(
        f"Image not found: {IMAGE_PATH}\n"
        "Update IMAGE_PATH to the exact path of your test image."
    )

image = Image.open(IMAGE_PATH).convert("RGB").resize((224, 224))
batch = np.expand_dims(np.asarray(image, dtype=np.float32), axis=0)

for path in MODEL_PATHS:
    print(f"\nModel: {path.name}")
    model = tf.keras.models.load_model(path)
    score = float(model.predict(batch, verbose=0)[0][0])
    print(f"Raw pneumonia score: {score:.6f}")
    print(f"Classification at threshold 0.50: "
          f"{'PNEUMONIA' if score >= 0.50 else 'NORMAL'}")

