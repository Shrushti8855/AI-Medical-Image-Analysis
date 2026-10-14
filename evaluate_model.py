
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

from src.data import load_datasets

PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_ROOT / "models" / "best_model.keras"


def evaluate_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    _, _, test_ds = load_datasets()
    model = tf.keras.models.load_model(MODEL_PATH)

    true_labels = []
    probabilities = []

    print("\nEvaluating the complete test dataset...")

    for images, labels in test_ds:
        predictions = model.predict(images, verbose=0).ravel()
        true_labels.extend(labels.numpy().ravel().astype(int))
        probabilities.extend(predictions)

    y_true = np.array(true_labels)
    y_prob = np.array(probabilities)
    threshold = 0.50
    y_pred = (y_prob >= threshold).astype(int)

    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])

    print(f"\nTotal test images: {len(y_true)}")
    print(f"Threshold: {threshold:.2f}")
    print(f"Test accuracy: {accuracy_score(y_true, y_pred):.4f}")

    print("\nClassification report:")
    print(
        classification_report(
            y_true,
            y_pred,
            labels=[0, 1],
            target_names=["NORMAL", "PNEUMONIA"],
            zero_division=0,
        )
    )

    print("Confusion matrix (rows = actual, columns = predicted):")
    print(matrix)

    tn, fp, fn, tp = matrix.ravel()
    print("\nError analysis:")
    print(f"NORMAL correctly classified: {tn}")
    print(f"NORMAL incorrectly classified as PNEUMONIA: {fp}")
    print(f"PNEUMONIA incorrectly classified as NORMAL: {fn}")
    print(f"PNEUMONIA correctly classified: {tp}")


if __name__ == "__main__":
    evaluate_model()
