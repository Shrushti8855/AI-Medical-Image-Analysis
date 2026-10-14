
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight

from src.data import load_datasets
from src.model import build_model, unfreeze_for_fine_tuning

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "reports"


def get_class_weights(train_ds):
    """Calculate class weights from training labels only."""
    labels = []

    for _, batch_labels in train_ds:
        labels.extend(
            batch_labels.numpy().reshape(-1).astype(int).tolist()
        )

    classes = np.array([0, 1])
    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=np.array(labels),
    )

    class_weights = {
        int(class_id): float(weight)
        for class_id, weight in zip(classes, weights)
    }

    print("\nClass weights:")
    print(f"NORMAL (0): {class_weights[0]:.4f}")
    print(f"PNEUMONIA (1): {class_weights[1]:.4f}")

    return class_weights


def train_model(initial_epochs=5, fine_tune_epochs=5):
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    train_ds, val_ds, _ = load_datasets()
    class_weights = get_class_weights(train_ds)

    model = build_model()

    best_model_path = MODEL_DIR / "best_model.keras"
    final_model_path = MODEL_DIR / "medical_image_model.keras"
    log_path = OUTPUT_DIR / "training_log.csv"

    # Use a fresh checkpoint for this training run.
    if best_model_path.exists():
        best_model_path.unlink()

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(best_model_path),
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=2,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.CSVLogger(
            str(log_path),
            append=False,
        ),
    ]

    print("\n--- Phase 1: Train classification head ---")

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=initial_epochs,
        class_weight=class_weights,
        callbacks=callbacks,
    )

    print("\n--- Phase 2: Fine-tune MobileNetV2 ---")

    model = unfreeze_for_fine_tuning(
        model,
        trainable_layers=30,
    )

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=fine_tune_epochs,
        class_weight=class_weights,
        callbacks=callbacks,
    )

    best_model = tf.keras.models.load_model(best_model_path)
    best_model.save(final_model_path)

    print("\nTraining finished.")
    print(f"Best checkpoint: {best_model_path}")
    print(f"Final model: {final_model_path}")
    print(f"Training log: {log_path}")

    return best_model


if __name__ == "__main__":
    train_model(initial_epochs=5, fine_tune_epochs=5)
