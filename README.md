**1. Dataset**

Data masih dummy menggunakan dari kaggle dataset
https://www.kaggle.com/datasets/aprabowo/indonesia-tourism-destination/code

Perlu dibuat dataset yang asli (tidak dummy) dari scraping data di website. Bisa lewat platform tempat wisata tripadvisor, google maps dengan google maps reviews scraper atau scrape it cloud.
Dengan intinya konsep kolom-kolom dalam datasetnya seperti yang dari kaggle, ada tempat wisata sebagai item dan pengguna sebagai user.
Rencananya fokus 3 daerah spesifik di tempat wisata (JAKARTA, BANDUNG, SURABAYA)

**Notes : Dibawah Ini hanya saran kalau bisa lebih bagus diubah saja agar hasil training modelnya bagus dan logis, tidak overfitting**

**2. Struktur Dataset:**

User ID: Identitas unik pengguna.
Place ID: Identitas unik tempat wisata.
Nama Tempat Wisata: Nama tempat (contoh: Monas, Tangkuban Perahu).
Deskripsi : Penjelasan singkat tempatnya.
Kategori: Jenis wisata (sejarah, alam, modern).
Lokasi: Kota tempat wisata.
Rating: Nilai ulasan pengguna (1-5).
Ulasan: Komentar dari pengguna.
Harga Tiket: Kisaran harga masuk (jika ada).

Inputnya:
User : Pengguna 
Item : Destinasi wisata

**3. Sumber Data:**

Google Maps: Mengambil data ulasan, rating, lokasi.
TripAdvisor: Untuk kategori, harga tiket, fasilitas.
(Bisa variasi lain yang penting tidak manipulasi)


**4. Model Hybrid Content Based Filtering & Deep Matrix Factorization**

Nantinya data tersebut digunakan untuk sistem rekomendasi dengan hybrid 2 model, content based filtering dan deep matrix factorization. karena menggunakan deep learning agar tidak overfitting perlu pake data yang berariatif dan jumlah besar
(Contoh bisa diliat di folder model)

- Hybrid Model
Penggabungan:
Gabungkan skor prediksi dari CBF dan DeepMF menggunakan metode ensemble berbasis prediksi rating

**5. Tools**
- Pandas & NumPy untuk preprocessing.
- TensorFlow  untuk membangun model.
- Scikit-Learn untuk evaluasi metrik.

Utamanya hasil modelnya bagus, bisa belajar dengan baik (tanpa manipulasi) dan datasetnya juga harus bagus & banyak biar memperbaikin performa modelnya
