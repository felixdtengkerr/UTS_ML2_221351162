import streamlit as st
import tensorflow as tf
import numpy as np
import joblib

# Load scaler dan label encoder
scaler = joblib.load("scaler.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# Load model TFLite
interpreter = tf.lite.Interpreter(model_path="anemia-model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Judul Aplikasi
st.title("Prediksi Anemia dari Citra Mata")
st.write("Masukkan parameter hasil ekstraksi citra untuk memprediksi kondisi anemia.")

# Input fitur sesuai dengan dataset
sex = st.selectbox("Jenis Kelamin", ['Laki-laki', 'Perempuan'])  # 0 atau 1
hb = st.number_input("Kadar Hemoglobin (Hb)", min_value=0.0, max_value=20.0, value=13.0)
red_pct = st.number_input("Persentase Piksel Merah (%)", min_value=0.0, max_value=100.0, value=30.0)
green_pct = st.number_input("Persentase Piksel Hijau (%)", min_value=0.0, max_value=100.0, value=30.0)
blue_pct = st.number_input("Persentase Piksel Biru (%)", min_value=0.0, max_value=100.0, value=30.0)

# Encode gender: misal Laki-laki = 1, Perempuan = 0
sex_encoded = 1 if sex == 'Laki-laki' else 0

# Gabungkan semua input menjadi array
input_data = np.array([[sex_encoded, hb, red_pct, green_pct, blue_pct]])

# Prediksi ketika tombol ditekan
if st.button("Prediksi Anemia"):
    try:
        input_scaled = scaler.transform(input_data).astype(np.float32)
        interpreter.set_tensor(input_details[0]['index'], input_scaled)
        interpreter.invoke()
        prediction = interpreter.get_tensor(output_details[0]['index'])

        predicted_label = np.argmax(prediction)
        anemia_status = label_encoder.inverse_transform([predicted_label])[0]

        st.success(f"Hasil Prediksi: **{anemia_status.upper()}**")
    except ValueError as e:
        st.error(f"Terjadi kesalahan saat preprocessing: {e}")
