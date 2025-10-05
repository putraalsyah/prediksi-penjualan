import pandas as pd
import os
import joblib
from flask import Flask, request, jsonify
from flask_cors import CORS
from sklearn.linear_model import LinearRegression

# Inisialisasi aplikasi Flask
app = Flask(__name__)

# Konfigurasi CORS untuk mengizinkan permintaan dari GitHub Pages Anda
CORS(app, resources={r"/predict": {"origins": "https://putraalsyah.github.io"}})

# --- FUNGSI UNTUK MELATIH DAN MENYIMPAN MODEL ---
def train_and_save_model(data_path='advertising.csv', model_path='advertising_sales_model.pkl'):
    """
    Fungsi ini membaca dataset, melatih model regresi linear sederhana,
    dan menyimpannya ke dalam sebuah file.
    """
    print(f"Membaca dataset dari {data_path}...")
    try:
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        print(f"Error: File '{data_path}' tidak ditemukan. Pastikan file tersebut ada di direktori yang sama.")
        return None

    print("Melatih model regresi linear...")
    # Memisahkan fitur (X) dan target (y)
    # Kita hanya menggunakan 'TV' sebagai fitur sesuai dengan notebook
    X = df[['TV']]
    y = df['Sales']

    # Membuat dan melatih model
    model = LinearRegression()
    model.fit(X, y)
    print("Model berhasil dilatih.")

    # Menyimpan model yang sudah dilatih
    joblib.dump(model, model_path)
    print(f"Model berhasil disimpan di '{model_path}'.")
    return model

# --- MEMUAT MODEL SAAT APLIKASI DIMULAI ---
MODEL_FILENAME = 'advertising_sales_model.pkl'
DATA_FILENAME = 'advertising.csv'
model = None

# Cek apakah file model sudah ada
if os.path.exists(MODEL_FILENAME):
    print(f"Memuat model yang sudah ada dari '{MODEL_FILENAME}'...")
    model = joblib.load(MODEL_FILENAME)
    print("Model berhasil dimuat.")
else:
    print(f"File model '{MODEL_FILENAME}' tidak ditemukan.")
    # Cek apakah dataset ada untuk memulai pelatihan
    if os.path.exists(DATA_FILENAME):
        print("Memulai proses pelatihan model baru...")
        model = train_and_save_model(DATA_FILENAME, MODEL_FILENAME)
    else:
        print(f"Error: Dataset '{DATA_FILENAME}' tidak ditemukan. Tidak dapat melatih model baru.")

# --- API ENDPOINT UNTUK PREDIKSI ---
@app.route('/predict', methods=['POST'])
def predict():
    """
    Endpoint API untuk menerima budget iklan TV dan mengembalikan prediksi penjualan.
    """
    if model is None:
        return jsonify({'error': 'Model tidak tersedia. Periksa log server.'}), 500

    try:
        # Mendapatkan data JSON dari request
        data = request.get_json()
        
        # Mengambil nilai budget TV dan mengubahnya menjadi float
        tv_budget = float(data['tv_budget'])
        
        # Membuat array 2D untuk prediksi, karena model scikit-learn mengharapkannya
        prediction_input = [[tv_budget]]
        
        # Melakukan prediksi menggunakan model
        predicted_sales = model.predict(prediction_input)
        
        # Mengembalikan hasil prediksi dalam format JSON
        return jsonify({'predicted_sales': round(predicted_sales[0], 2)})

    except Exception as e:
        # Menangani error jika terjadi
        return jsonify({'error': str(e)}), 400

# --- MENJALANKAN SERVER FLASK ---
if __name__ == '__main__':
    # Menjalankan aplikasi di localhost pada port 5000
    # Opsi threaded=False penting agar model bisa dimuat sebelum request pertama
    print("Menjalankan server Flask di http://127.0.0.1:5000")
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=False)

