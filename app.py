
from pathlib import Path

import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_ROOT / "models" / "best_model.keras"
IMAGE_SIZE = (224, 224)

st.set_page_config(
    page_title="AI Medical Image Analysis",
    page_icon="🩻",
    layout="centered",
)

st.title("🩻 AI Medical Image Analysis")
st.write(
    "Educational chest X-ray image classification using "
    "MobileNetV2 transfer learning."
)

st.warning(
    "Educational demonstration only. This model is not a "
    "medical diagnostic tool and must not be used for "
    "medical decisions."
)


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}"
        )
    return tf.keras.models.load_model(MODEL_PATH)


uploaded_file = st.file_uploader(
    "Upload a chest X-ray image",
    type=["jpg", "jpeg", "png"],
)

threshold = st.slider(
    "Experimental classification threshold",
    min_value=0.10,
    max_value=0.90,
    value=0.50,
    step=0.05,
    help="This setting is for demonstration only, not clinical use.",
)

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")

        st.subheader("Uploaded image")
        st.image(image, caption="Uploaded chest X-ray", width=400)

       
        image_array = np.asarray(image, dtype=np.float32)
        image_tensor = tf.convert_to_tensor(image_array)
        image_tensor = tf.image.resize(image_tensor, IMAGE_SIZE)
        image_batch = tf.expand_dims(image_tensor, axis=0).numpy()


        if st.button("Analyze image", type="primary"):
            with st.spinner("Analyzing image..."):
                model = load_model()
                score = float(
                    model.predict(image_batch, verbose=0)[0][0]
                )

                if score >= threshold:
                    prediction = "PNEUMONIA"
                else:
                    prediction = "NORMAL"

            st.subheader("Model output")

            if prediction == "PNEUMONIA":
                st.error(f"Model classification: {prediction}")
            else:
                st.success(f"Model classification: {prediction}")

            st.metric(
                "Pneumonia model score",
                f"{score * 100:.2f}%",
            )

            st.caption(
                f"Experimental threshold: {threshold:.2f}. "
                "The score is not a calibrated probability or "
                "a diagnosis. Image quality and dataset differences "
                "can affect predictions."
            )

    except Exception as error:
        st.error(f"Unable to process this image: {error}")

st.divider()
st.caption(
    "Project: AI Medical Image Analysis | "
    "Model: MobileNetV2 | Classes: NORMAL and PNEUMONIA"
)
