"""Streamlit interface for the trained CIFAR-10 MobileNetV2 model."""

import json
from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image


# Keep the two model artifacts beside this file for local and cloud deployment.
APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / "model.keras"
CLASS_NAMES_PATH = APP_DIR / "class_names.json"

st.set_page_config(page_title="Klasifikasi Gambar CNN", page_icon="🖼️", layout="centered")
st.title("Klasifikasi Gambar dengan CNN")
st.write(
    "Unggah gambar untuk melihat prediksi salah satu dari 10 kelas CIFAR-10. "
    "Model dilatih pada gambar kecil 32×32; foto dengan objek utama yang jelas "
    "biasanya lebih mudah dikenali."
)


@st.cache_resource(show_spinner="Memuat model...")
def load_model_and_labels():
    """Load artifacts once per Streamlit process; fail with a useful message."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"File model tidak ditemukan: {MODEL_PATH.name}")
    if not CLASS_NAMES_PATH.exists():
        raise FileNotFoundError(f"File label tidak ditemukan: {CLASS_NAMES_PATH.name}")
    with CLASS_NAMES_PATH.open("r", encoding="utf-8") as file:
        names = json.load(file)
    model = tf.keras.models.load_model(MODEL_PATH)
    return model, names


try:
    model, class_names = load_model_and_labels()
except (FileNotFoundError, OSError, ValueError) as error:
    st.error(f"Aplikasi belum siap: {error}")
    st.info(
        "Jalankan seluruh notebook training_cnn.ipynb, lalu letakkan model.keras "
        "dan class_names.json di folder yang sama dengan app.py."
    )
    st.stop()


# Accept common RGB image types; conversion to RGB aligns the model's 3-channel input.
uploaded_file = st.file_uploader(
    "Pilih gambar",
    type=["jpg", "jpeg", "png", "webp"],
    help="Model menerima RGB. Gambar grayscale dan transparan otomatis dikonversi.",
)

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(
            image,
            caption="Gambar yang diunggah",
            width="stretch",
            alt="Gambar RGB yang akan diprediksi",
        )

        # Pass original RGB pixels in 0–255. The saved model performs the same
        # resize-to-96 and MobileNetV2 normalization used during training.
        image_array = np.asarray(image, dtype=np.float32)
        probabilities = model.predict(image_array[None, ...], verbose=0)[0]
        predicted_index = int(np.argmax(probabilities))
        predicted_label = class_names[str(predicted_index)]
        confidence = float(probabilities[predicted_index])

        st.subheader("Hasil prediksi")
        st.metric("Kelas", predicted_label, f"Confidence {confidence:.1%}")
        st.caption(
            "Confidence berasal dari skor softmax dan tidak menjamin bahwa "
            "probabilitas model sudah terkalibrasi."
        )

        # Show three strongest classes so the user can inspect close alternatives.
        top_indices = np.argsort(probabilities)[::-1][:3]
        top_results = {
            class_names[str(int(index))]: float(probabilities[index])
            for index in top_indices
        }
        st.write("Tiga kelas teratas")
        st.bar_chart(
            top_results,
            horizontal=True,
            x_label="Confidence",
            alt="Confidence untuk tiga kelas prediksi teratas",
        )
    except (OSError, ValueError, tf.errors.InvalidArgumentError) as error:
        st.error(f"Gambar tidak dapat diproses: {error}")

st.divider()
st.caption("Proyek pembelajaran CNN Week 8 • Dataset CIFAR-10")
