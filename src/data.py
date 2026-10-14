
from pathlib import Path

import tensorflow as tf
from sklearn.model_selection import train_test_split

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16
SEED = 42

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "raw" / "chest_xray"

CLASS_NAMES = ["NORMAL", "PNEUMONIA"]


def collect_image_paths(directory):
    """Collect image paths and labels in a consistent class order."""
    paths = []
    labels = []

    for label, class_name in enumerate(CLASS_NAMES):
        class_dir = directory / class_name

        if not class_dir.exists():
            raise FileNotFoundError(
                f"Required class directory not found: {class_dir}"
            )

        for path in sorted(class_dir.iterdir()):
            if path.is_file() and path.suffix.lower() in {
                ".jpg", ".jpeg", ".png"
            }:
                paths.append(str(path))
                labels.append(label)

    return paths, labels


def make_dataset(paths, labels, training=False):
    """Load and batch images from a list of paths and labels."""
    path_ds = tf.data.Dataset.from_tensor_slices(paths)
    label_ds = tf.data.Dataset.from_tensor_slices(
        tf.cast(labels, tf.float32)
    )

    dataset = tf.data.Dataset.zip((path_ds, label_ds))

    def load_image(path, label):
        image = tf.io.read_file(path)
        image = tf.io.decode_image(
            image, channels=3, expand_animations=False
        )
        image.set_shape([None, None, 3])
        image = tf.image.resize(image, IMAGE_SIZE)
        return image, tf.reshape(label, [1])

    dataset = dataset.map(
        load_image,
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    if training:
        dataset = dataset.shuffle(
            buffer_size=len(paths),
            seed=SEED,
            reshuffle_each_iteration=True,
        )

    return dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)


def load_datasets():
    """Create a stratified train/validation split and a separate test set."""
    train_dir = DATA_DIR / "train"
    test_dir = DATA_DIR / "test"

    for directory in [train_dir, test_dir]:
        if not directory.exists():
            raise FileNotFoundError(
                f"Required dataset directory not found: {directory}"
            )

    all_paths, all_labels = collect_image_paths(train_dir)

    train_paths, val_paths, train_labels, val_labels = (
        train_test_split(
            all_paths,
            all_labels,
            test_size=0.2,
            random_state=SEED,
            stratify=all_labels,
        )
    )

    test_paths, test_labels = collect_image_paths(test_dir)

    print("\nDataset distribution:")
    for name, labels in [
        ("Train", train_labels),
        ("Validation", val_labels),
        ("Test", test_labels),
    ]:
        normal_count = sum(label == 0 for label in labels)
        pneumonia_count = sum(label == 1 for label in labels)
        print(
            f"{name}: NORMAL={normal_count}, "
            f"PNEUMONIA={pneumonia_count}, "
            f"TOTAL={len(labels)}"
        )

    train_ds = make_dataset(
        train_paths, train_labels, training=True
    )
    val_ds = make_dataset(
        val_paths, val_labels
    )
    test_ds = make_dataset(
        test_paths, test_labels
    )

    return train_ds, val_ds, test_ds
