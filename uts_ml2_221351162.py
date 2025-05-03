import streamlit as st
import tensorflow as tf
import numpy as np
import joblib

# Load scaler dan label encoder
scaler = joblib.load('scaler.pkl')
label_encoder = joblib.load('label_encoder.pkl')

# Load model TFLite
interpreter = tf.lite.Interpreter(model_path="india-rental-house-price (1).tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

st.title("Klasifikasi Kota Berdasarkan Data Properti")
st.write("Masukkan parameter properti untuk memprediksi kota (Mumbai, Delhi, atau Pune).")

# Input numerik (disesuaikan dengan sample_input di notebook)
area = st.number_input("Area (sqft)", min_value=10.0, max_value=10000.0, value=78.0)
bedrooms = st.number_input("Jumlah Kamar Tidur", min_value=0, max_value=20, value=2)
bathrooms = st.number_input("Jumlah Kamar Mandi", min_value=0, max_value=20, value=2)
price = st.number_input("Harga (Lakh)", min_value=1.0, max_value=10000.0, value=20.13)
location_score = st.number_input("Skor Lokasi", min_value=0.0, max_value=100.0, value=81.60)
distance_to_city_center = st.number_input("Jarak ke Pusat Kota (km)", min_value=0.0, max_value=100.0, value=7.62)
public_transport_score = st.number_input("Akses Transportasi Umum", min_value=0.0, max_value=500.0, value=262.71)

# Dummy values untuk kolom lain (yang di-label encode)
# Jika Anda tahu kolom kategorikal lainnya, bisa tambahkan selectbox yang sesuai
encoded_categoricals = [0] * 8  # placeholder untuk fitur kategorikal

# Gabungkan semua input
input_data = np.array([[area, bedrooms, bathrooms, price, location_score,
                        distance_to_city_center, public_transport_score] + encoded_categoricals])

if st.button("Prediksi Kota"):
    try:
        input_scaled = scaler.transform(input_data).astype(np.float32)
        interpreter.set_tensor(input_details[0]['index'], input_scaled)
        interpreter.invoke()
        prediction = interpreter.get_tensor(output_details[0]['index'])

        predicted_label = np.argmax(prediction)
        city = label_encoder.inverse_transform([predicted_label])[0]

        st.success(f"Kota diprediksi: **{city}**")
    except ValueError as e:
        st.error(f"Terjadi kesalahan saat transformasi atau prediksi: {e}")
