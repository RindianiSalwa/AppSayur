import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import streamlit as st
import requests
from PIL import Image
import numpy as np
from io import BytesIO

# Load model
model = load_model('models/final_model.h5')

# Dictionary kandungan gizi
kandungan = {
    "Brokoli": "Vitamin C, Vitamin K, Serat, Folat, Antioksidan",
    "Capsicum": "Vitamin A, Vitamin C, Vitamin B6, Folat, Antioksidan",
    "Kacang_Polong": "Protein, Serat, Vitamin K, Vitamin B1, Fosfor",
    "Kembang_Kol": "Vitamin C, Vitamin K, Folat, Serat, Antioksidan",
    "Kentang": "Karbohidrat, Vitamin C, Vitamin B6, Kalium, Serat",
    "Kubis": "Vitamin C, Vitamin K, Folat, Mangan, Serat",
    "Labu_Botol": "Vitamin C, Vitamin B, Magnesium, Kalsium, Serat",
    "Labu_Kabocha_Hijau": "Vitamin A, Vitamin C, Beta-Karoten, Serat",
    "Labu_Pahit": "Vitamin C, Vitamin A, Zat Besi, Kalium, Antioksidan",
    "Lobak": "Vitamin C, Folat, Kalium, Serat",
    "Pepaya": "Vitamin C, Vitamin A, Folat, Serat, Enzim Papain",
    "Terong_Hijau": "Serat, Vitamin B1, Vitamin B6, Folat, Mangan",
    "Timun": "Air, Vitamin K, Vitamin C, Magnesium, Kalium",
    "Tomat": "Likopen, Vitamin C, Vitamin K, Folat, Kalium",
    "Wortel": "Vitamin A, Beta-Karoten, Serat, Kalium",
}

# Fungsi prediksi
def predict_species(img):
    img = img.resize((224, 224))  
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0

    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions)

    labels = {
        0: 'Brokoli', 1: 'Capsicum', 2: 'Kacang_Polong', 3: 'Kembang_Kol',
        4: 'Kentang', 5: 'Kubis', 6: 'Labu_Botol', 7: 'Labu_Kabocha_Hijau',
        8: 'Labu_Pahit', 9: 'Lobak', 10: 'Pepaya', 11: 'Terong_Hijau',
        12: 'Timun', 13: 'Tomat', 14: 'Wortel',
    }

    predicted_species = labels.get(predicted_class, 'Tidak Diketahui')
    predicted_probability = predictions[0][predicted_class] * 100  

    if predicted_probability <= 90:
        return "⚠️ Gambar ini tidak termasuk jenis sayuran yang telah di dukung."
    else:
        kand = kandungan.get(predicted_species, "Informasi kandungan belum tersedia.")
        return f"""
        ✅ Termasuk Jenis Sayuran **{predicted_species}**  
        🔢 Akurasi: {predicted_probability:.2f}%  
        🥗 Kandungan: {kand}
        """

# ========================== STREAMLIT UI ==========================

# Judul
st.title("Vegetable Classification App 🥦🍅")
st.subheader("Klasifikasi Sayuran Dengan Menggunakan MobileNet")
st.caption("Kelompok 2 Pagi A Khasanah-Rindiani-Salsabilla")

st.markdown("---")

# Bagian gambar contoh
st.info("Di bawah ini adalah contoh gambar sayuran yang berhasil diprediksi dengan benar oleh sistem:")
image_path = "assets/output.png"
st.image(image_path, caption="Contoh Sayuran", use_container_width=True)

st.markdown("---")

# Bagian input gambar
st.header("📂 Form Input Data Gambar")
input_options = st.selectbox("Pilih Metode Input:", ["Upload Gambar", "URL Gambar"])

if input_options == "Upload Gambar":
    uploaded_file = st.file_uploader("Upload gambar sayuran (.jpg/.jpeg/.png)", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        st.image(uploaded_file, caption="✅ Gambar berhasil diupload", use_container_width=True)
        if st.button("🔍 Prediksi dari Upload"):
            img = Image.open(uploaded_file)
            result = predict_species(img)
            st.success(result)

elif input_options == "URL Gambar":
    url = st.text_input("Masukkan URL gambar sayuran:")
    if url:
        try:
            response = requests.get(url, stream=True)
            img = Image.open(BytesIO(response.content))
            st.image(img, caption="✅ Gambar berhasil diambil", use_container_width=True)
            if st.button("🔍 Prediksi dari URL"):
                result = predict_species(img)
                st.success(result)
        except:
            st.error("❌ URL tidak valid atau tidak bisa diakses.")

st.markdown("---")

# Bagian tentang
st.header("ℹ️ Tentang Aplikasi")
st.write("""
Aplikasi ini dibuat untuk melakukan **klasifikasi jenis sayuran** 
menggunakan metode **MobileNet**.

**Fitur utama**:
- Upload gambar sayuran dari device
- Input URL gambar
- Menampilkan hasil prediksi jenis sayuran dan kandungan gizinya

**Dikembangkan oleh**: Kelompok 2 Pagi A Khasanah-Rindiani-Salsabilla
""")
