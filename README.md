# Eye Disease Classification: Automatic Detection Using CNN and InceptionV3

## Deskripsi Proyek

Proyek ini bertujuan untuk mengembangkan sistem klasifikasi citra yang dapat mengenali dan membedakan gambar fundus mata. Sistem ini dirancang untuk mendeteksi jenis penyakit mata tertentu, seperti **Normal**, **Diabetic Retinopathy**, **Glaucoma**, dan **Cataract**, sebagai bagian dari upaya mendukung teknologi di bidang kesehatan. Proyek ini hanya ditujukan untuk tujuan edukasi dan tidak dapat menggantikan diagnosis medis profesional.

---

## Deskripsi Dataset

Dataset yang digunakan dalam proyek ini berasal dari [Kaggle - Eye Disease Classification Dataset](https://www.kaggle.com/datasets/gunavenkatdoddi/eye-diseases-classification/data). Dataset ini terdiri dari 4.217 citra fundus mata yang terbagi dalam empat kategori: **Normal**, **Diabetic Retinopathy**, **Glaucoma**, dan **Cataract**. Untuk memastikan kualitas model, citra telah melalui tahap preprocessing seperti resizing, normalisasi, dan augmentasi.

---

## Deskripsi Model

Proyek ini menggunakan dua model utama untuk tugas klasifikasi.

### Convolutional Neural Network (CNN)
CNN adalah model custom yang dirancang untuk mengekstraksi fitur penting dari gambar fundus mata. Dengan arsitektur sederhana, CNN memberikan hasil yang layak pada tugas klasifikasi, meskipun memiliki keterbatasan dalam hal performa. Model ini kurang mampu menangani distribusi data yang kompleks, terutama pada kelas tertentu seperti **Normal**.

### InceptionV3
InceptionV3 adalah model pre-trained dengan arsitektur yang lebih kompleks dan efisien. Model ini dirancang untuk menangani berbagai skala fitur melalui kombinasi filter konvolusi yang berbeda. Dibandingkan dengan CNN, InceptionV3 menunjukkan performa yang jauh lebih baik dengan distribusi metrik evaluasi yang lebih seimbang dan akurasi lebih tinggi. Namun, model ini membutuhkan daya komputasi yang lebih besar.

---

## Hasil dan Analisis

### Perbandingan Performa Model

| Model           | Akurasi | Kelebihan                          | Kekurangan                             |
| --------------- | ------- | ---------------------------------- | -------------------------------------- |
| **CNN**         | 69%     | Arsitektur sederhana               | Recall rendah pada kelas Normal        |
| **InceptionV3** | 82%     | Akurasi tinggi dan metrik seimbang | Membutuhkan daya komputasi lebih besar |

### Hasil Evaluasi Model

#### CNN

| Metrik    | Cataract | Diabetic Retinopathy | Glaucoma | Normal |
| --------- | -------- | -------------------- | -------- | ------ |
| Precision | 0.63     | 0.96                 | 0.60     | 0.72   |
| Recall    | 0.92     | 0.65                 | 0.82     | 0.38   |
| F1-Score  | 0.74     | 0.78                 | 0.69     | 0.50   |
| Accuracy  | 0.69     |                      |          |        |

#### InceptionV3

| Metrik    | Cataract | Diabetic Retinopathy | Glaucoma | Normal |
| --------- | -------- | -------------------- | -------- | ------ |
| Precision | 0.95     | 0.93                 | 0.92     | 0.63   |
| Recall    | 0.90     | 0.79                 | 0.63     | 0.93   |
| F1-Score  | 0.92     | 0.86                 | 0.75     | 0.75   |
| Accuracy  | 0.82     |                      |          |        |

### Perbandingan performa model
- Inceptionv3 memberikan performa yang lebih baik secara keseluruhan dengan akurasi 82% dibandingkan 69% pada CNN.
- Inceptionv3 unggul dalam precision pada hampir semua kelas yang menunjukkan bahwa model ini lebih efektif dalam mengurangi false positives.
- Untuk kelas Normal, Inceptionv3 jauh lebih baik dalam recall, tetapi precision CNN lebih tinggi.
- Secara keseluruhan, Inceptionv3 lebih baik dibandingkan CNN, terutama dalam menangani distribusi metrik yang lebih seimbang

---

## Langkah Instalasi

Ikuti langkah-langkah berikut untuk menjalankan proyek ini:

### Persyaratan Sistem

- **Python**: Versi 3.7 atau yang lebih baru

### Langkah-langkah Instalasi

1. Clone repository:

   ```bash
   git clone https://github.com/Nadirafrqn/Eye-diseases-classification.git
   cd Eye-diseases-classification
   ```

2. (Opsional) Buat dan aktifkan virtual environment:

   ```bash
   python -m venv venv
   source venv/bin/activate  # Untuk Windows: venv\Scripts\activate
   ```

3. Instal dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Jalankan aplikasi:

   ```bash
   streamlit run app.py
   ```

