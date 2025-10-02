import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import streamlit as st
import requests
from PIL import Image
import numpy as np
from io import BytesIO

# Load the pre-trained model
model = load_model('models/final_model.h5')

# Fungsi untuk melakukan prediksi
def predict_species(img):
    # Praproses gambar
    img = img.resize((224, 224))  
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0

    # Melakukan prediksi
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions)

    # Get the class label
    labels = {
        0: 'Brokoli',
        1: 'Capsicum',
        2: 'Kacang_Polong',
        3: 'Kembang_Kol',
        4: 'Kentang',
        5: 'Kubis',
        6: 'Labu_Botol',
        7: 'Labu_Kabocha_Hijau',
        8: 'Labu_Pahit',
        9: 'Lobak',
        10: 'Pepaya',
        11: 'Terong_Hijau',
        12: 'Timun',
        13: 'Tomat',
        14: 'Wortel',
    }

    predicted_species = labels.get(predicted_class, 'Tidak Diketahui')

    # Mendapatkan probabilitas kelas yang diprediksi
    predicted_probability = predictions[0][predicted_class] * 100  # Konversi ke persentase

    if predicted_probability <= 90:
        return "Gambar ini Tidak termasuk jenis sayuran yang telah di dukung."
    else:
        return f" Termasuk Jenis Sayuran {predicted_species}, dengan nilai akurasi sebesar {predicted_probability:.2f}%."


# Streamlit UI

st.markdown('''
    <div style="text-align: center;">
        <h2>
            <i>Computer Vision</i> <br> 
            Klasifikasi Sayuran Dengan Menggunakan Metode MobileNet
        </h2>
        <span>Kelompok 4 Pagi A</span>
    </div>
    <hr style="height:5px;border-width:0;color:gray;background-color:gray">''', 
    unsafe_allow_html=True)

# Direktori gambar Anda
image_path = "assets/output.png"

st.write("Di Bawah ini adalah beberapa gambar jenis sayuran yang hampir di prediksi dengan BENAR oleh sistem : ")

# Menampilkan gambar dengan Streamlit
st.image(image_path, caption='Gambar Sayuran Brokoli, Capsicum, Kacang Polong, Kembang Kol, Kentang, Kubis, Labu Botol, Labu Kabocha Hijau, Labu Pahit, Lobak, Pepaya, Terong Hijau, Timun, Tomat, dan Wortel.', use_column_width=True)

st.markdown('''
        <hr style="height:5px;border-width:0;color:gray;background-color:gray">
        ''', 
    unsafe_allow_html=True)

st.title(":pencil: Form Input Data Gambar")

input_options = ['Pilih Salah Satu', 'Upload Gambar', 'URL Gambar']
selected_input_option = st.selectbox("Pilih Salah Satu Jenis Inputan Data Gambar :", input_options)

if selected_input_option == 'Upload Gambar':
    st.subheader("Form Upload Data Gambar")
    st.markdown('''
    :red[*Note :] Format Gambar Yang Di Dukung adalah .jpg dan .jpeg''', unsafe_allow_html=True)
    # Option to upload image file
    uploaded_file = st.file_uploader("Input Data Gambar Sayuran:", type=["jpg", "jpeg"])
    if uploaded_file is not None:
        # Display the uploaded image
        st.image(uploaded_file, caption="Gambar Telah Sukses Di Upload!", use_column_width=True)

        # Make predictions when the user clicks the button
        if st.button("Predict"):
            # Convert the uploaded file to Pillow Image
            img = Image.open(uploaded_file)
            predicted_species = predict_species(img)
            st.subheader("Didapat Hasil :")
            st.success(f"{predicted_species}")
elif selected_input_option == 'URL Gambar':
    st.subheader("Form URL Data Gambar")
    st.markdown('''
    :red[*Note :] Setelah URL Data Gambar Telah Di Inputkan Klik Tombol Enter Pada Keyboard!''', unsafe_allow_html=True)
    # Option to upload image through URL
    image_url = st.text_input("Input URL Data Gambar Sayuran:")
    if image_url:
        try:
            # Fetch the image from the URL
            response = requests.get(image_url, stream=True)
            response.raise_for_status()

            # Open and display the image
            img = Image.open(BytesIO(response.content))
            st.image(img, caption="Gambar Telah Sukses Di Upload!", use_column_width=True)

            # Make predictions when the user clicks the button
            if st.button("Predict"):
                predicted_species = predict_species(img)
                st.subheader("Didapat Hasil :")
                st.success(f"{predicted_species}")
        except Exception as e:
            st.error(f"Error: {e}")
else:
    st.warning('Silakan Pilih Salah Satu Jenis Inputan Gambar.')


