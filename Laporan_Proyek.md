# Laporan Proyek Prediksi Diabetes

**Nama Proyek:** Sistem Prediksi Diabetes Menggunakan Algoritma Machine Learning

**Nama Anda:** [Isi dengan Nama Anda]

**Tanggal:** 26 Mei 2025


## 1. Domain Proyek

### Latar Belakang
Diabetes mellitus adalah penyakit kronis yang ditandai dengan kadar gula darah tinggi dan telah menjadi masalah kesehatan global dengan prevalensi yang terus meningkat. Menurut International Diabetes Federation (IDF) dan World Health Organization (WHO), jutaan orang di seluruh dunia hidup dengan diabetes, dan banyak di antaranya tidak menyadari kondisi mereka. Diabetes yang tidak terkontrol dapat menyebabkan komplikasi serius seperti penyakit jantung, stroke, gagal ginjal, kerusakan saraf, dan kebutaan. Deteksi dini dan intervensi yang tepat waktu sangat krusial untuk mencegah atau menunda komplikasi ini serta meningkatkan kualitas hidup pasien.

### Mengapa dan Bagaimana Masalah Harus Diselesaikan
Masalah tingginya prevalensi diabetes dan risiko komplikasinya memerlukan solusi proaktif untuk identifikasi dini individu yang berisiko. Diagnosis konvensional seringkali baru dilakukan setelah gejala muncul atau melalui pemeriksaan rutin yang mungkin tidak diakses oleh semua orang secara berkala. Dengan memanfaatkan kemajuan teknologi, khususnya *machine learning*, kita dapat mengembangkan sistem prediksi yang mampu menganalisis berbagai faktor risiko dan data biomedis pasien untuk mengidentifikasi kemungkinan status diabetes (non-diabetik, prediabetik, atau diabetik) secara lebih efisien dan potensial lebih dini.

Penyelesaian masalah ini melalui pendekatan machine learning dapat memberikan beberapa manfaat:
1.  **Deteksi Dini:** Membantu tenaga medis dan individu untuk mengidentifikasi risiko diabetes lebih awal.
2.  **Intervensi Tepat Waktu:** Memungkinkan penerapan strategi pencegahan atau manajemen diabetes lebih cepat.
3.  **Pengurangan Biaya Kesehatan:** Mencegah komplikasi jangka panjang dapat mengurangi beban biaya perawatan.
4.  **Alat Bantu Keputusan:** Menyediakan alat bantu bagi tenaga medis dalam proses skrining dan diagnosis.

### Hasil Riset Terkait atau Referensi
Saya mendapatkan referensi ini dari mahasiswa UGM pendeteksian diabetes menggunakan Algoritma Logistic Regression [Jurnal](https://journal.ugm.ac.id/v3/JNTETI/article/download/3586/1646).

* [Referensi 1](https://www.voaindonesia.com/a/jumlah-penderita-diabetes-di-indonesia-terus-meningkat/7870777.html)
* [Referensi 2](https://www.tempo.co/gaya-hidup/gaya-hidup-tak-sehat-picu-naiknya-kasus-diabetes-di-usia-muda-1192505)

---

## 2. Business Understanding

### Proses Klarifikasi Masalah
Proses klarifikasi masalah dimulai dengan memahami tantangan utama dalam penanganan diabetes, yaitu keterlambatan diagnosis dan kebutuhan akan metode skrining yang efektif. Diperlukan sebuah sistem yang dapat memprediksi status diabetes seseorang berdasarkan data klinis dan demografis.

### Problem Statements
1.  Bagaimana cara mengembangkan model prediktif yang akurat untuk mengklasifikasikan status diabetes individu (Non-diabetic, Prediabetic, Diabetic)?
2.  Algoritma machine learning manakah yang memberikan performa terbaik untuk tugas klasifikasi ini pada dataset yang digunakan?
3.  Faktor atau fitur apa saja yang paling berpengaruh dalam memprediksi status diabetes?

### Goals
1.  Membangun dan mengevaluasi beberapa model machine learning (Decision Tree, Random Forest, Logistic Regression) untuk memprediksi status diabetes.
2.  Membandingkan kinerja model-model tersebut menggunakan metrik evaluasi yang relevan untuk menentukan model terbaik.
3.  Memberikan insight mengenai faktor-faktor yang signifikan dalam prediksi diabetes.

### Solution Statement
Untuk mencapai tujuan di atas, diajukan solusi sebagai berikut:
1.  **Pengembangan Model Klasifikasi Multikelas:** Mengimplementasikan setidaknya tiga algoritma klasifikasi (Decision Tree, Random Forest, Logistic Regression) untuk memprediksi status diabetes.
2.  **Analisis Komparatif dan Pemilihan Model Terbaik:** Melakukan analisis komparatif terhadap hasil evaluasi ketiga model. Model dengan performa terbaik akan dipilih sebagai solusi akhir.
3.  **Pengukuran Kinerja:** Kinerja solusi akan diukur menggunakan akurasi, presisi, recall, F1-score, dan confusion matrix.

---

## 3. Data Understanding

### Sumber Data
Dataset ini diperoleh dari Kaggle:
[Diabetes Prediction Dataset](https://www.kaggle.com/datasets/marshalpatel3558/diabetes-prediction-dataset-legit-dataset/data)

### Kondisi Data
1. Jumlah Baris dan kolom:
   Dataset memiliki 1000 baris dan 14 kolom.
2. Nilai yang hilang (Missing Value):
   Dari hasil eksplorasi data, tidak ada (0) missing value pada data.
3. Data Duplikat:
   Dari hasil ekplorasi data, tidak ada (0) data yang terduplikat.
4. Unique Class:
   Pada kolom Gender dan CLASS terdapat bebrapa kategori yang terulang.

### Informasi Data
Dataset yang digunakan berisi 1000 baris dan 14 kolom. 

### Uraian Seluruh Variabel/Fitur
1.  `ID`: Pengidentifikasi unik untuk setiap rekaman dalam kumpulan data.
2.  `No_Pation`: Pengidentifikasi lain untuk pasien. Bisa berupa nomor pasien atau ID rekaman.
3.  `Gender`: Jenis kelamin pasien (kemudian di-encode).
4.  `AGE`: Usia pasien dalam tahun (Numerik, int64).
5.  `Urea`: Kadar urea dalam darah (Numerik, float64).
6.  `Cr` (Creatinine): Kadar kreatinin dalam darah (Numerik, int64).
7.  `HbA1c`: Rata-rata kadar gula darah 2-3 bulan terakhir (Numerik, float64).
8.  `Chol` (Cholesterol): Kadar kolesterol total (Numerik, float64).
9.  `TG` (Triglycerides): Kadar trigliserida (Numerik, float64).
10.  `HDL`: Kadar kolesterol HDL (Numerik, float64).
11.  `LDL`: Kadar kolesterol LDL (Numerik, float64).
12. `VLDL`: Kadar kolesterol VLDL (Numerik, float64).
13. `BMI`: Indeks Massa Tubuh (Numerik, float64).
14. `CLASS`: Variabel target status diabetes (Kategorikal, kemudian di-encode menjadi 0, 1, 2).


## 4. Data Preparation

1.  **Penghapusan Kolom Identifier:**
    * **Teknik:** Kolom `ID` dan `No_Pation` dihapus.
    * **Alasan:** Tidak memiliki nilai prediktif.
2.  **Encoding Variabel Target (`CLASS`):**
    * **Teknik:** `LabelEncoder` digunakan untuk mengubah `CLASS` (misal, 'N', 'P', 'Y') menjadi numerik (0, 1, 2).
    * **Alasan:** Algoritma machine learning memerlukan input numerik.
3.  **Encoding Fitur Kategorikal (`Gender`):**
    * **Teknik:** `LabelEncoder` digunakan untuk mengubah `Gender` (misal, 'Male', 'Female') menjadi numerik (0, 1).
    * **Alasan:** Fitur input juga harus numerik.
    ```python
    # label_encoder = LabelEncoder()
    # df['Gender'] = label_encoder.fit_transform(df['Gender'])
    # df['CLASS'] = label_encoder.fit_transform(df['CLASS'])
    ```
4.  **Pemisahan Fitur (X) dan Target (y):**
    * **Teknik:** Dataset dipisahkan menjadi matriks fitur `X` dan vektor target `y`.
    * **Alasan:** Standar untuk membedakan variabel independen dan dependen.
5.  **Pembagian Data menjadi Set Latih dan Uji:**
    * **Teknik:** Dataset dibagi menjadi 70% data latih dan 30% data uji (`test_size=0.3`, `random_state=42`).
    * **Alasan:** Untuk melatih model dan mengevaluasinya pada data yang belum pernah dilihat.
    ```python
    # X = df.drop('CLASS', axis=1)
    # y = df['CLASS']
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    ```
---

## 5. Modeling

Tiga algoritma klasifikasi diimplementasikan: Decision Tree, Random Forest, dan Logistic Regression.

1.  **Decision Tree Classifier:**
    * **Parameter:** `random_state=42`, `max_depth=5`.
    * **Kelebihan:** Mudah diinterpretasi, cepat.
    * **Kekurangan:** Cenderung overfitting jika tidak dibatasi.
    ```python
    # dt_classifier = DecisionTreeClassifier(random_state=42, max_depth=5)
    # dt_classifier.fit(X_train, y_train)
    ```
2.  **Random Forest Classifier:**
    * **Parameter:** `n_estimators=100`, `random_state=42`.
    * **Kelebihan:** Lebih robust terhadap overfitting, akurasi tinggi.
    * **Kekurangan:** Kurang interpretatif, lebih banyak sumber daya.
    ```python
    # rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    # rf_model.fit(X_train, y_train)
    ```
3.  **Logistic Regression:**
    * **Parameter:** `random_state=42`, `max_iter=1000`.
    * **Kelebihan:** Sederhana, cepat, output probabilitas.
    * **Kekurangan:** Asumsi linearitas, kurang baik untuk pola non-linear.
    ```python
    # # logreg_model = LogisticRegression(random_state=42, max_iter=1000)
    # # logreg_model.fit(X_train, y_train)
    ```

### Pemilihan Model Terbaik
Berdasarkan evaluasi, **Decision Tree (`max_depth=5`)** dan **Random Forest** menunjukkan performa terbaik dan hampir setara (~98.67% akurasi). Decision Tree unggul dalam recall untuk kelas Diabetic, sementara Random Forest unggul dalam presisi untuk kelas yang sama. Keduanya jauh lebih baik daripada Logistic Regression untuk dataset ini.

---

## 6. Evaluation

### Metrik Evaluasi yang Digunakan
1.  **Akurasi:** Proporsi prediksi benar secara keseluruhan.
2.  **Presisi:** TP / (TP + FP). Akurasi prediksi positif.
3.  **Recall (Sensitivity):** TP / (TP + FN). Kemampuan menemukan semua kasus positif.
4.  **F1-Score:** 2 * (Presisi * Recall) / (Presisi + Recall). Keseimbangan presisi dan recall.
5.  **Confusion Matrix:** Tabel visualisasi prediksi benar dan salah per kelas.

### Hasil Proyek Berdasarkan Metrik Evaluasi

| Metrik                 | Decision Tree (`max_depth=5`) | Random Forest (`n_estimators=100`) | Logistic Regression |
| ---------------------- | ----------------------------- | ------------------------------------ | ------------------- |
| **Akurasi Keseluruhan**| **~98.67%** | **~98.67%** | ~94.33%             |
| **Kelas 0 (Non-diabetic):** |                               |                                      |                     |
| Presisi                | 0.99                          | 0.99                                 | 0.97                |
| Recall                 | 0.99                          | 1.00                                 | 0.98                |
| F1-score               | 0.99                          | 0.99                                 | 0.97                |
| **Kelas 1 (Prediabetic):**|                               |                                      |                     |
| Presisi                | 0.97                          | 0.97                                 | 0.84                |
| Recall                 | 0.94                          | 0.94                                 | 0.89                |
| F1-score               | 0.96                          | 0.96                                 | 0.86                |
| **Kelas 2 (Diabetic):** |                               |                                      |                     |
| Presisi                | 0.91                          | **1.00** | 0.50                |
| Recall                 | **1.00** | 0.90                                 | 0.30                |
| F1-score               | 0.95                          | 0.95                                 | 0.38                |

**Analisis Hasil:**
* Decision Tree dan Random Forest sangat unggul.
* Untuk Kelas 2 (Diabetic):
    * Decision Tree: Recall 100% (menemukan semua kasus Diabetic).
    * Random Forest: Presisi 100% (semua yang diprediksi Diabetic benar-benar Diabetic).
* Logistic Regression kurang perform, terutama untuk Kelas 2.

**Confusion Matrix Decision Tree:**

[[252   1   1]
[  2  34   0]
[  0   0  10]]


**Confusion Matrix Random Forest:**

[[253   1   0]
[  2  34   0]
[  1   0   9]]

---
## 9. Prediksi Diabetes 
Untuk menguji kinerja model terhadap data baru, dilakukan prediksi menggunakan data pasien acak yang dibuat secara sintetik. Data ini mencakup berbagai fitur medis seperti usia, kadar gula darah (HbA1c), kolesterol, dan lainnya.

1. Membuat Data Pasien Secara Acak
```
random_sample = pd.DataFrame({
    'Gender': [random.choice(['M', 'F'])], 
    'AGE': [np.random.randint(20, 70)],
    'Urea': [np.random.uniform(2.0, 8.0)],
    'Cr': [np.random.randint(30, 100)],
    'HbA1c': [np.random.uniform(3.5, 10.0)],
    'Chol': [np.random.uniform(3.0, 7.0)],
    'TG': [np.random.uniform(0.5, 5.0)],
    'HDL': [np.random.uniform(0.5, 2.0)],
    'LDL': [np.random.uniform(1.0, 5.0)],
    'VLDL': [np.random.uniform(0.1, 1.5)],
    'BMI': [np.random.uniform(18, 35)]
})
```
Kode ini membuat 1 baris data acak untuk seorang pasien dengan berbagai nilai medis. Misalnya:

* Umur (AGE): antara 20 hingga 70 tahun
* HbA1c: antara 3.5 hingga 10.0 (indikator penting untuk diabetes)
* BMI: Body Mass Index

2. Menampilkan Data
```
print("Sampel Data Acak (Sebelum Encoding Gender):")
print(random_sample)
```
Ini hanya untuk menampilkan data acak yang dihasilkan sebelum diolah.

3. Encoding Gender
```
random_sample['Gender'] = random_sample['Gender'].map({'M': 1, 'F': 0})
```
Model ML hanya bisa membaca angka. Maka 'M' diubah menjadi 1, dan 'F' jadi 0.

4. Prediksi dengan Model
a. Decision Tree
```
dt_pred = dt_classifier.predict(random_sample)
dt_proba = dt_classifier.predict_proba(random_sample)
```
b. Random Forest
```
rf_pred = rf_model.predict(random_sample)
rf_proba = rf_model.predict_proba(random_sample)
```
c. Logistic Regression
```
logreg_pred = logreg_model.predict(random_sample)
logreg_proba = logreg_model.predict_proba(random_sample)
```

5. Mapping Angka ke Label Kelas
```
class_mapping = {0: 'No Diabetes', 1: 'Prediabetes', 2: 'Diabetes'}
```
Untuk menampilkan prediksi dalam bentuk yang bisa dipahami manusia.


6. Menampilkan Hasil
```
print("===== PREDIKSI DENGAN BERBAGAI MODEL =====")
...
print(f"Decision Tree Prediction: {class_mapping[dt_pred[0]]}")
print(f"Decision Tree Probabilities: {dict(zip(class_mapping.values(), dt_proba[0]))}\n")
```

7. Hasil
```
Sampel Data Acak (Sebelum Encoding Gender):
  Gender  AGE      Urea  Cr     HbA1c      Chol        TG       HDL       LDL  \
0      F   33  2.100224  51  3.562133  6.184824  0.545463  0.554187  3.365697   

      VLDL        BMI  
0  0.85876  28.830565  
===== PREDIKSI DENGAN BERBAGAI MODEL =====
Decision Tree Prediction: No Diabetes
Decision Tree Probabilities: {'No Diabetes': np.float64(1.0), 'Prediabetes': np.float64(0.0), 'Diabetes': np.float64(0.0)}

Random Forest Prediction: No Diabetes
Random Forest Probabilities: {'No Diabetes': np.float64(0.58), 'Prediabetes': np.float64(0.16), 'Diabetes': np.float64(0.26)}

Logistic Regression Prediction: No Diabetes
Logistic Regression Probabilities: {'No Diabetes': np.float64(0.436707456370631), 'Prediabetes': np.float64(0.24169677739617182), 'Diabetes': np.float64(0.321595766233197)}
```

---

## 8. Kesimpulan
Proyek ini berhasil membangun model machine learning yang efektif untuk prediksi diabetes berdasarkan dataset yang digunakan. Model Decision Tree dan Random Forest menunjukkan performa yang sangat tinggi dengan tingkat akurasi sekitar 98.67%, mengungguli model Logistic Regression secara signifikan.
* Decision Tree memiliki keunggulan pada recall untuk kelas Diabetic, menjadikannya sangat andal dalam mendeteksi kasus positif.
* Random Forest menunjukkan performa presisi yang lebih tinggi, mengurangi kemungkinan prediksi positif palsu (false positive).
  
Kedua model ini membuktikan bahwa pendekatan berbasis pohon sangat cocok untuk permasalahan klasifikasi diabetes. Hasil ini menunjukkan potensi besar model ini sebagai alat bantu deteksi dini diabetes yang cepat dan akurat, yang dapat mendukung pengambilan keputusan di bidang kesehatan.

---



