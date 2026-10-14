
import tensorflow as tf
from tensorflow.keras import layers, models

IMAGE_SIZE = (224, 224)


def build_model():
    """Build a MobileNetV2 binary chest X-ray classifier."""

    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(*IMAGE_SIZE, 3),
        include_top=False,
        weights="imagenet",
    )

    # Freeze the pretrained feature extractor initially.
    base_model.trainable = False

    inputs = layers.Input(shape=(*IMAGE_SIZE, 3))

    # Scale pixel values from [0, 255] to [-1, 1].
    x = layers.Rescaling(
        scale=1.0 / 127.5,
        offset=-1,
    )(inputs)

    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(
        1,
        activation="sigmoid",
        name="pneumonia_probability",
    )(x)

    model = models.Model(inputs, outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=0.001
        ),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ],
    )

    return model


def unfreeze_for_fine_tuning(model, trainable_layers=30):
    """Unfreeze selected MobileNetV2 layers for fine-tuning."""

    base_model = next(
        layer
        for layer in model.layers
        if isinstance(layer, tf.keras.Model)
    )

    base_model.trainable = True

    # Keep earlier feature layers frozen.
    for layer in base_model.layers[:-trainable_layers]:
        layer.trainable = False

    # Keep Batch Normalization layers frozen for stability.
    for layer in base_model.layers:
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=1e-5
        ),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ],
    )

    return model