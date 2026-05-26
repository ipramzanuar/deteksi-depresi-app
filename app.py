from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg19 import preprocess_input
import numpy as np
import os
import sqlite3
from datetime import datetime

app = Flask(__name__)

# Load model
model = load_model('Model/model_deteksi_depresi_1_vgg19.keras')

# Inisialisasi database SQLite
def init_db():
    conn = sqlite3.connect('hasil_prediksi.db')
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS hasil_prediksi (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            nama_file TEXT,
            hasil TEXT,
            akurasi REAL,
            waktu TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()

# Fungsi untuk memprediksi gambar
def predict_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    preds = model.predict(img_array)
    prob = preds[0][0]
    label = "Depresi" if prob > 0.5 else "Tidak Depresi"
    accuracy = round(prob * 100, 2)
    return label, accuracy

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detect', methods=['GET', 'POST'])
def detect():
    prediction = None
    accuracy = None
    image_path = None

    if request.method == 'POST':
        file = request.files['image']
        if file:
            img_path = os.path.join('static', file.filename)
            file.save(img_path)
            prediction, accuracy = predict_image(img_path)
            image_path = file.filename

            # Simpan ke database SQLite
            conn = sqlite3.connect('hasil_prediksi.db')
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO hasil_prediksi (nama_file, hasil, akurasi)
                VALUES (?, ?, ?)
            """, (file.filename, prediction, accuracy))
            conn.commit()
            conn.close()

    return render_template('deteksi.html', prediction=prediction, accuracy=accuracy, image_path=image_path)

# Jalankan aplikasi
if __name__ == '__main__':
    init_db()
    app.run(debug=True)
