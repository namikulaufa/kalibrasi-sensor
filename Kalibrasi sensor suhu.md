# Kalibrasi sensor suhu

## PROMPT 1

sensors-23-07347.pdfPDFBayangkan anda mahasiswa elektronika dan Instrumentasi universitas Gadjah Mada semester 5, hari ini anda akan berkuliah mata kuliah Instrumentasi dan Kendali Cerdas. Hari ini dosen memberikan tugas untuk mengerjakan suatu tugas. Tugas ini dikerjakan secara individu. Instruksi tugas yaitu sebagai berikut

Anda diberikan tugas untuk mendemokan apa yang terdapat pada paper yang telah dilampirkan. Jadi istilahnya anda akan menjalankan project berdasarkan referensi paper yang telah dilampirkan. Tunjukkan kepada saya bagaimana prosesnya mulai dari persiapan bahan alat, beserta semua yang perlu dipersiapkan termasuk kode program apabila perlu. Langkah langkah pengerjaan juga tolong ditunjukkan untuk pengerjaan tugas ini. Dan pada akhir berikan goals dengan bahasa yang mudah dipahami apa guna melakukan project berdasarkan paper ini.

Telah dilampirkan materi berupa paper referensi pada file sensors-23-07347.pdf). Gunakan file ini sebagai referensi pengerjaan tugas.

Lakukan pengerjaan tugas tersebut, pastikan tidak ada tulisan yang terlewat. Tidak perlu menggunakan kata kata kiasan dengan bahasa yang indah. Hindari struktur kalimat antitesis seperti 'ini bukan hanya X tapi Y'. Hindari juga penulisan awal kalimat 'ini adalah'.

## RESPONSE 1

Tugas mata kuliahmu intinya: **mendemokan kalibrasi sensor suhu berbasis machine learning** seperti pada paper  
“Machine‑Learning‑Based Calibration of Temperature Sensors” (Sensors 2023, 23, 7347).

sensors-23-07347

Aku susun langkahnya seperti kamu beneran mau ngerjain di lab: mulai dari persiapan alat, pengambilan data, pemodelan (linear, polinomial, ANN), sampai siap demo di kelas.

___________

1\. Inti isi paper yang harus kamu tiru
---------------------------------------

Dari paper:

*   Sensor yang dikalibrasi: **tiga buah Pt100 tiga‑kawat** (sensor A, B, C).
    
*   Lingkungan uji: **air check bin / environmental chamber** (gambar 1 di halaman 2) dengan kontrol suhu dari **−50 sampai 150 °C**.
    
*   Sifat Pt100: resistansi mengikuti
    
    Rt\=R0(1+αt)R\_t = R\_0 (1 + \\alpha t)Rt​\=R0​(1+αt)
    
    dengan α ≈ 0,00392 °C⁻¹.
    
*   Data diambil tiap **0,2 °C** di rentang −50 sampai 140 °C.
    
*   Galat didefinisikan
    
    E\=Emeasure−EactualE = E\_{measure} - E\_{actual}E\=Emeasure​−Eactual​
*   Dipakai **tiga metode kalibrasi**:
    
    1.  **Linear regression**
        
        Y\=wX+bY = wX + bY\=wX+b
    2.  **Polynomial regression** (misal orde 2)
        
        Y\=β0+β1X+β2X2Y = \\beta\_0 + \\beta\_1 X + \\beta\_2 X^2Y\=β0​+β1​X+β2​X2
    3.  **Neural network BP (ANN)**
        
        *   Arsitektur: 1 input (suhu terukur), 1 hidden layer (16 neuron, aktivasi ReLU), 1 output (suhu terkoreksi).
            
        *   Training dengan backpropagation untuk meminimalkan error (MSE).
            
*   Hasil utama:
    
    *   Sebelum kalibrasi, **galat bisa sampai ±2,74 °C**, 3σ ≈ 0,61 °C di rentang −40–140 °C.
        
    *   Setelah kalibrasi ANN, 3σ turun ke **0,167 °C**, dan di rentang 25–75 °C galat maksimum **< 0,05 °C**.
        
    *   Dibanding linear dan polinomial, **ANN punya MAE, MSE lebih kecil dan R² paling tinggi**.
        
    *   Data per sensor: 829, 891, dan 924 sampel di rentang −50–150 °C.
        

Tugasmu: membuat **versi mini** dari eksperimen ini (konsep sama, skala boleh disesuaikan lab kampus) lalu mendemokan proses + hasil.

___________

2\. Tujuan teknis tugas (supaya jelas dari awal)
------------------------------------------------

Tujuan yang akan kamu capai:

1.  Mengukur keluaran beberapa sensor Pt100 tiga‑kawat di rentang suhu tertentu.
    
2.  Membandingkan pembacaan sensor dengan suhu referensi.
    
3.  Menghitung galat tiap sensor.
    
4.  Melatih tiga model kalibrasi:
    
    *   Linear regression
        
    *   Polynomial regression
        
    *   Artificial Neural Network (ANN, backprop 3‑layer)
        
5.  Mengaplikasikan model terlatih untuk:
    
    *   Mengoreksi pembacaan sensor secara offline (di Python)
        
    *   Opsional: mengimplementasikan koefisien/weight ke mikrokontroler.
        
6.  Menunjukkan bahwa **ANN memberi kalibrasi paling akurat** untuk sensor suhu murah.
    

___________

3\. Persiapan alat dan bahan
----------------------------

### 3.1. Hardware

Usahakan sedekat mungkin dengan setup di paper; kalau ada keterbatasan di lab, kamu bisa catat sebagai “modifikasi”.

**Sensor dan lingkungan uji**

*   3× **Pt100 tiga‑kawat** (sensor A, B, C)
    
*   1× **ruang uji suhu**:
    
    *   Ideal: environmental chamber / air check bin −40 sampai 120 °C
        
    *   Alternatif (kalau chamber tidak ada): kombinasi
        
        *   Water bath + es untuk suhu rendah (0–40 °C)
            
        *   Oven laboratorik untuk suhu tinggi (40–120 °C)
            
*   **Sensor referensi**:
    
    *   Bisa pakai:
        
        *   Pt100 kelas lebih tinggi yang sudah dikalibrasi
            
        *   Termometer digital dengan spesifikasi akurasi bagus
            
    *   Pada paper, suhu referensi diambil dari set‑point dan pembacaan chamber.
        

**Rangkaian pengkondisi sinyal**

Untuk tiap Pt100:

*   Sumber arus konstan (sekitar 1 mA).
    
*   Rangkaian pengukuran 3‑kawat Pt100:
    
    *   Bisa gunakan modul siap pakai, contoh: MAX31865 (RTD‑to‑digital).
        
    *   Atau rangkaian sendiri:
        
        *   Sumber arus konstan
            
        *   Penguat instrumen / differential amplifier
            
        *   Low‑pass filter
            
*   Mikrokontroler / DAQ:
    
    *   Arduino / STM32 / ESP32, atau NI‑DAQ yang terhubung ke PC.
        
    *   Minimal 3 kanal input untuk 3 sensor (atau 1 modul per sensor via SPI/I²C).
        

**Lain‑lain**

*   Kabel, breadboard / PCB, konektor.
    
*   PC / laptop untuk logging data.
    
*   Alat keselamatan: sarung tangan, penjepit, dsb (kalau main suhu tinggi / rendah).
    

### 3.2. Software

Di laptop/PC:

*   Python 3.x (Anaconda lebih praktis)
    
*   Library:
    
    ```bash
    pip install numpy pandas matplotlib scikit-learn tensorflow
    ```
    
*   Editor/IDE: VS Code / Jupyter Notebook.
    
*   Arduino IDE / PlatformIO (kalau pakai Arduino/STM32).
    

___________

4\. Teori singkat yang kamu butuhkan
------------------------------------

### 4.1. Pt100 dan rangkaian 3‑kawat

*   Pt100: resistor platina 100 Ω pada 0 °C.
    
*   Hubungan resistansi–suhu (aproksimasi linear):
    
    Rt\=R0(1+αt)R\_t = R\_0 (1 + \\alpha t)Rt​\=R0​(1+αt)
    
    dengan:
    
    *   RtR\_tRt​: resistansi pada suhu t
        
    *   R0R\_0R0​: resistansi di 0 °C (100 Ω)
        
    *   α\\alphaα: koefisien suhu (sekitar 0,00392 °C⁻¹).
        
*   Cara ukur:
    
    *   Alirkan arus konstan I lewat Pt100
        
    *   Ukur tegangan V
        
    *   Hitung resistansi:
        
        Rt\=VIR\_t = \\frac{V}{I}Rt​\=IV​
    *   Konversi ke suhu:
        
        t≈Rt−R0αR0t \\approx \\frac{R\_t - R\_0}{\\alpha R\_0}t≈αR0​Rt​−R0​​
*   3‑kawat dipakai agar efek resistansi kabel bisa dikompensasi secara listrik. Untuk kalibrasi ini, kamu menganggap pembacaan “suhu sensor” sudah keluar dari rangkaian.
    

### 4.2. Definisi galat

Galat untuk tiap titik data:

E\=Emeasure−EactualE = E\_{measure} - E\_{actual}E\=Emeasure​−Eactual​

*   EmeasureE\_{measure}Emeasure​: suhu yang dibaca sensor Pt100.
    
*   EactualE\_{actual}Eactual​: suhu referensi (dari chamber atau termometer referensi).
    

Galat dipakai sebagai dasar evaluasi model (sebelum & sesudah kalibrasi).

### 4.3. Model kalibrasi

#### 4.3.1. Linear regression

*   Mengasumsikan hubungan **linier** antara suhu sensor dan suhu aktual:
    
    Y\=wX+bY = wX + bY\=wX+b
    *   X: suhu terukur sensor
        
    *   Y: suhu terkoreksi (mendekati suhu aktual)
        
    *   w, b: parameter yang dicari dari data.
        

Paper memberi contoh hasil fitting untuk salah satu sensor:

Y\=0,9887X−0,577Y = 0{,}9887 X - 0{,}577Y\=0,9887X−0,577

#### 4.3.2. Polynomial regression

*   Masih model linear dalam parameter, tetapi dengan fitur **X, X², X³, …**
    
*   Contoh orde 2:
    
    Y\=β0+β1X+β2X2Y = \\beta\_0 + \\beta\_1 X + \\beta\_2 X^2Y\=β0​+β1​X+β2​X2
*   Orde dipilih dengan cross‑validation. Orde terlalu rendah → tidak menangkap non‑linearitas; terlalu tinggi → overfitting.
    

Contoh rumus di paper:

Y\=0,9903X2−0,00001807X−0,5636Y = 0{,}9903 X^2 - 0{,}00001807 X - 0{,}5636Y\=0,9903X2−0,00001807X−0,5636

#### 4.3.3. Neural Network (BP ANN)

Gambar 2 pada halaman 4 menunjukkan struktur ANN: input layer → hidden layer → output layer.

Persamaan neuron hidden ke‑h:

Sh\=f(∑i\=1qWihXi−θh)S\_h = f\\left(\\sum\_{i=1}^{q} W\_{ih} X\_i - \\theta\_h\\right)Sh​\=f(i\=1∑q​Wih​Xi​−θh​)

*   Xᵢ: input
    
*   Wᵢₕ: bobot input‑hidden
    
*   θₕ: bias hidden
    
*   f: aktivasi (ReLU).
    

Output neuron j:

Yj\=f(∑h\=1dWhjSh−θj)Y\_j = f\\left(\\sum\_{h=1}^{d} W\_{hj} S\_h - \\theta\_j\\right)Yj​\=f(h\=1∑d​Whj​Sh​−θj​)

Error total:

E\=12∑j\=1m(tj−yj)2E = \\frac{1}{2} \\sum\_{j=1}^{m} (t\_j - y\_j)^2E\=21​j\=1∑m​(tj​−yj​)2

Dalam tugas ini:

*   Input: satu nilai (suhu terukur).
    
*   Hidden layer: 16 neuron, aktivasi ReLU.
    
*   Output: satu nilai (suhu terkoreksi), tanpa aktivasi (linear).
    
*   Loss: MSE (mean squared error).
    
*   Optimizer: misalnya Adam.
    

### 4.4. Indeks kinerja (MAE, MSE, R²)

Untuk menilai model, paper memakai:

*   **MAE**:
    
    MAE\=1n∑i\=1n∣yactual,i−ypredicted,i∣\\text{MAE} = \\frac{1}{n} \\sum\_{i=1}^n \\left|y\_{actual,i} - y\_{predicted,i}\\right|MAE\=n1​i\=1∑n​∣yactual,i​−ypredicted,i​∣
*   **MSE**:
    
    MSE\=1n∑i\=1n(yactual,i−ypredicted,i)2\\text{MSE} = \\frac{1}{n} \\sum\_{i=1}^n \\left(y\_{actual,i} - y\_{predicted,i}\\right)^2MSE\=n1​i\=1∑n​(yactual,i​−ypredicted,i​)2
*   **R²**:
    
    R2\=1−∑i\=1n(yactual,i−ypredicted,i)2∑i\=1n(yactual,i−yˉactual)2R^2 = 1 - \\frac{\\sum\_{i=1}^n (y\_{actual,i} - y\_{predicted,i})^2}{\\sum\_{i=1}^n (y\_{actual,i} - \\bar{y}\_{actual})^2}R2\=1−∑i\=1n​(yactual,i​−yˉ​actual​)2∑i\=1n​(yactual,i​−ypredicted,i​)2​

Nilai bagus kalau **MAE dan MSE kecil, R² mendekati 1**.

___________

5\. Langkah eksperimen: dari setup sampai dataset
-------------------------------------------------

### 5.1. Merakit sensor dan sistem akuisisi

1.  Pasang 3 sensor Pt100 tiga‑kawat (A, B, C) pada holder agar ujungnya berada di area udara yang sama dalam chamber.
    
2.  Sambungkan tiap sensor ke modul pengkondisi (misal MAX31865 atau rangkaian 3‑kawat buatanmu).
    
3.  Hubungkan modul ke mikrokontroler / DAQ:
    
    *   Kalau MAX31865 → SPI/I²C.
        
    *   Kalau penguat analog → ke ADC.
        
4.  Pastikan mikrokontroler terhubung ke PC via USB dan bisa mengirim data serial.
    
5.  Lakukan uji awal di suhu ruang:
    
    *   Catat T\_sensor\_A, T\_sensor\_B, T\_sensor\_C, dan T\_ref kamar.
        
    *   Pastikan pembacaan konsisten dan beda antar sensor tidak aneh sekali.
        

### 5.2. Menyiapkan format data

Gunakan file CSV dengan kolom seperti:

```text
T_actual,T_A,T_B,T_C
25.0,24.3,24.7,25.2
30.0,29.5,29.8,30.4
...
```

*   `T_actual`: suhu referensi (dari chamber atau termometer referensi).
    
*   `T_A/B/C`: suhu dari masing‑masing sensor.
    

### 5.3. Prosedur pengambilan data

Usaha untuk meniru pola di paper (rentang luas + interval cukup rapat).

1.  Tentukan **rentang dan step suhu** yang realistis di lab:
    
    *   Contoh: −10 sampai 100 °C dengan step 2 °C.
        
    *   Kalau chamber kampus bisa lebih lebar, bisa mendekati −40 sampai 120 °C.
        
2.  Untuk tiap set‑point suhu:
    
    1.  Atur chamber ke suhu target (misal 30 °C).
        
    2.  Tunggu sampai suhu stabil (monitor reading referensi ±0,1 °C).
        
    3.  Ambil N kali pembacaan (misal 50 sampel) dari ketiga sensor dan referensi.
        
    4.  Hitung rata‑rata masing‑masing, tulis ke file CSV sebagai satu baris data.
        
3.  Ulangi sampai seluruh rentang tercakup.
    
4.  Targetkan **ratusan titik data** per sensor (paper punya 800+ sampel per sensor).
    

### 5.4. Menghitung galat awal (sebelum kalibrasi)

Gunakan Python untuk:

```python
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("data_pt100.csv")

for sensor in ["T_A", "T_B", "T_C"]:
    df[f"err_{sensor}"] = df[sensor] - df["T_actual"]

# Plot error vs T_actual seperti Gambar 3 di paper
plt.figure()
for sensor in ["T_A", "T_B", "T_C"]:
    plt.plot(df["T_actual"], df[f"err_{sensor}"], label=sensor)
plt.xlabel("Temperature (°C)")
plt.ylabel("Error (°C)")
plt.legend()
plt.title("Measurement error of uncalibrated sensors")
plt.show()

# Hitung mean dan 3σ
for sensor in ["T_A", "T_B", "T_C"]:
    e = df[f"err_{sensor}"]
    mean = e.mean()
    sigma = e.std()
    print(sensor, "mean =", mean, "std =", sigma, "3σ =", 3*sigma)
```

Output ini mirip dengan Gambar 3 dan tabel awal di paper:

*   Di paper, 3σ sekitar 0,61 °C untuk uncalibrated di rentang −40–140 °C.
    

___________

6\. Pembuatan model kalibrasi di Python
---------------------------------------

Untuk kesederhanaan, kerjakan **per sensor**. Contoh di bawah untuk sensor A (`T_A`). Nanti ulangi untuk B dan C.

### 6.1. Menyiapkan dataset

```python
from sklearn.model_selection import train_test_split

# X: suhu terukur sensor A
X = df[["T_A"]].values  # bentuk (n_samples, 1)
y = df["T_actual"].values  # suhu referensi

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```

### 6.2. Model 1 – Linear Regression

```python
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)

y_pred_train = lin_reg.predict(X_train)
y_pred_test = lin_reg.predict(X_test)

def eval_model(name, y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print(f"{name}: MAE={mae:.4f}, MSE={mse:.4f}, R2={r2:.7f}")

print("Koefisien linear:")
print("w =", lin_reg.coef_[0])
print("b =", lin_reg.intercept_)

eval_model("Linear train", y_train, y_pred_train)
eval_model("Linear test", y_test, y_pred_test)
```

Bandingkan w dan b yang kamu dapat dengan contoh paper (salah satu sensor: w ≈ 0,9887, b ≈ −0,577).

Setelah itu, hitung galat baru setelah kalibrasi:

```python
df["T_A_lin"] = lin_reg.predict(df[["T_A"]])
df["err_A_lin"] = df["T_A_lin"] - df["T_actual"]
```

Plot error linear seperti Gambar 6–7:

```python
plt.figure()
plt.plot(df["T_actual"], df["err_T_A"], label="original")
plt.plot(df["T_actual"], df["err_A_lin"], label="linear")
plt.xlabel("Temperature (°C)")
plt.ylabel("Error (°C)")
plt.legend()
plt.title("Sensor A: error before/after linear regression")
plt.show()
```

### 6.3. Model 2 – Polynomial Regression

Gunakan orde 2 dulu (seperti contoh di paper).

```python
from sklearn.preprocessing import PolynomialFeatures

poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly_train = poly.fit_transform(X_train)  # fitur: [X, X^2]
X_poly_test = poly.transform(X_test)

poly_reg = LinearRegression()
poly_reg.fit(X_poly_train, y_train)

y_pred_train_poly = poly_reg.predict(X_poly_train)
y_pred_test_poly = poly_reg.predict(X_poly_test)

print("Koefisien polynomial (orde 2):")
print("beta0 (intercept) =", poly_reg.intercept_)
print("beta =", poly_reg.coef_)  # [beta1, beta2]

eval_model("Poly train", y_train, y_pred_train_poly)
eval_model("Poly test", y_test, y_pred_test_poly)
```

Terapkan ke seluruh data:

```python
df["T_A_poly"] = poly_reg.predict(poly.transform(df[["T_A"]]))
df["err_A_poly"] = df["T_A_poly"] - df["T_actual"]
```

Plot seperti Gambar 8–9:

```python
plt.figure()
plt.plot(df["T_actual"], df["err_T_A"], label="original")
plt.plot(df["T_actual"], df["err_A_poly"], label="polynomial")
plt.xlabel("Temperature (°C)")
plt.ylabel("Error (°C)")
plt.legend()
plt.title("Sensor A: error before/after polynomial regression")
plt.show()
```

### 6.4. Model 3 – ANN (BP Neural Network)

Gunakan TensorFlow/Keras.

```python
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping

# Opsional: normalisasi input biar training lebih stabil
X_mean = X_train.mean()
X_std = X_train.std()
X_train_n = (X_train - X_mean) / X_std
X_test_n = (X_test - X_mean) / X_std

model = Sequential([
    Dense(16, activation="relu", input_shape=(1,)),  # hidden layer 16 neuron
    Dense(1)  # output linear
])

model.compile(optimizer="adam", loss="mse", metrics=["mae"])

es = EarlyStopping(monitor="val_loss", patience=20, restore_best_weights=True)

history = model.fit(
    X_train_n, y_train,
    validation_data=(X_test_n, y_test),
    epochs=500,
    batch_size=32,
    callbacks=[es],
    verbose=0
)

# Evaluasi
y_pred_train_ann = model.predict(X_train_n).flatten()
y_pred_test_ann = model.predict(X_test_n).flatten()

eval_model("ANN train", y_train, y_pred_train_ann)
eval_model("ANN test", y_test, y_pred_test_ann)
```

Terapkan ke seluruh data:

```python
X_all_n = (df[["T_A"]].values - X_mean) / X_std
df["T_A_ann"] = model.predict(X_all_n).flatten()
df["err_A_ann"] = df["T_A_ann"] - df["T_actual"]
```

Plot error seperti Gambar 4–5:

```python
plt.figure()
plt.plot(df["T_actual"], df["err_T_A"], label="original")
plt.plot(df["T_actual"], df["err_A_ann"], label="ANN")
plt.xlabel("Temperature (°C)")
plt.ylabel("Error (°C)")
plt.legend()
plt.title("Sensor A: error before/after ANN")
plt.show()
```

### 6.5. Ringkasan performa ketiga model

Buat tabel seperti pada Tabel 2 di paper (MSE, MAE, R²). Contoh kode:

```python
import pandas as pd

rows = []

def collect_row(method, y_true_train, y_pred_train, y_true_test, y_pred_test):
    rows.append({
        "Method": method + " (train)",
        "MAE": mean_absolute_error(y_true_train, y_pred_train),
        "MSE": mean_squared_error(y_true_train, y_pred_train),
        "R2": r2_score(y_true_train, y_pred_train)
    })
    rows.append({
        "Method": method + " (test)",
        "MAE": mean_absolute_error(y_true_test, y_pred_test),
        "MSE": mean_squared_error(y_true_test, y_pred_test),
        "R2": r2_score(y_true_test, y_pred_test)
    })

collect_row("Linear", y_train, y_pred_train, y_test, y_pred_test)
collect_row("Polynomial", y_train, y_pred_train_poly, y_test, y_pred_test_poly)
collect_row("ANN", y_train, y_pred_train_ann, y_test, y_pred_test_ann)

summary = pd.DataFrame(rows)
print(summary)
```

Bandingkan pola angkanya dengan Tabel 2 di paper:

*   **ANN** harusnya memberi MAE dan MSE paling kecil, R² paling mendekati 1.
    
*   Linear paling sederhana dan biasanya performanya paling jelek untuk data non‑linier penuh.
    

Lakukan hal yang sama untuk sensor B (`T_B`) dan C (`T_C`) sehingga kamu punya tabel besar mirip tabel 2.

___________

7\. Implementasi kalibrasi di sistem nyata
------------------------------------------

Setelah model terbentuk, kalibrasi bisa dipakai:

### 7.1. Linear dan polynomial di mikrokontroler

Cukup kirim koefisien w, b (atau β₀, β₁, β₂) dan pakai rumus di kode C++ Arduino.

Contoh untuk linear:

```cpp
float calibrate_linear(float T_meas) {
    const float w = 0.9887;   // ganti dengan hasil training
    const float b = -0.577;   // ganti juga
    return w * T_meas + b;
}
```

Contoh untuk polynomial orde 2:

```cpp
float calibrate_poly(float T_meas) {
    const float beta0 = -0.5636;
    const float beta1 = -0.00001807;
    const float beta2 = 0.9903;
    return beta0 + beta1 * T_meas + beta2 * T_meas * T_meas;
}
```

### 7.2. ANN di mikrokontroler (versi sederhana)

Setelah training, ambil bobot dan bias:

```python
weights = model.get_weights()
W1, b1, W2, b2 = weights  # karena 2 layer: input->hidden, hidden->output
print(W1.shape)  # (1, 16)
print(b1.shape)  # (16,)
print(W2.shape)  # (16, 1)
print(b2.shape)  # (1,)
```

Salin ke kode C++:

```cpp
// Contoh, isi angka diganti hasil training (disingkat di sini)
const float W1[16] = { /* ... */ };
const float b1[16] = { /* ... */ };
const float W2[16] = { /* ... */ };
const float b2 = /* ... */;
const float X_mean = /* dari Python */;
const float X_std  = /* dari Python */;

float relu(float x) {
    return x > 0 ? x : 0;
}

float calibrate_ann(float T_meas) {
    // normalisasi input
    float x = (T_meas - X_mean) / X_std;

    float hidden[16];
    for (int h = 0; h < 16; ++h) {
        float z = W1[h] * x + b1[h];
        hidden[h] = relu(z);
    }

    float y = b2;
    for (int h = 0; h < 16; ++h) {
        y += W2[h] * hidden[h];
    }
    return y; // suhu terkoreksi
}
```

Dengan ini, saat demo kamu bisa:

*   Tampilkan pembacaan raw sensor vs pembacaan yang sudah dikalibrasi ANN di serial monitor / LCD.
    

___________

8\. Skenario demo di kelas
--------------------------

Supaya tugasmu enak dinilai, susun sesi demo seperti ini:

1.  **Slide 1–2: Latar belakang**
    
    *   Banyak sensor suhu murah akurasinya jelek.
        
    *   Paper ini mengusulkan **kalibrasi berbasis ANN** untuk memperbaiki akurasi Pt100 tiga‑kawat.
        
2.  **Slide 3: Setup eksperimen**
    
    *   Tampilkan foto/gambar: Pt100, chamber, rangkaian (mirip Gambar 1 di paper).
        
    *   Jelaskan rentang suhu dan cara pengambilan data (step, jumlah sampel).
        
3.  **Slide 4: Galat sebelum kalibrasi**
    
    *   Plot error vs T (seperti Gambar 3).
        
    *   Tunjukkan 3σ dan nilai maksimum error.
        
4.  **Slide 5–7: Tiga model kalibrasi**
    
    *   Linear regression: tunjukkan rumus w dan b + plot perbandingan.
        
    *   Polynomial regression: tunjukkan koefisien orde 2 + plot.
        
    *   ANN: tunjukkan arsitektur 1‑16‑1 (gambar mirip Gambar 2) + plot error setelah ANN (Gambar 4/5 style).
        
5.  **Slide 8: Tabel kinerja**
    
    *   Tabel MAE, MSE, R² untuk A, B, C dan 3 metode (meniru Tabel 2).
        
    *   Sorot baris ANN yang paling baik.
        
6.  **Live demo singkat**
    
    *   Masukkan sensor ke chamber / media bersuhu stabil (misal 40 °C).
        
    *   Tampilkan:
        
        *   `T_A` (raw)
            
        *   `T_A_lin`
            
        *   `T_A_poly`
            
        *   `T_A_ann`
            
    *   Bandingkan dengan `T_actual`.
        
7.  **Kesimpulan lisan**
    
    *   Jelaskan bahwa ANN mengurangi galat jadi jauh lebih kecil, terutama di rentang suhu kerja utama.
        

___________

9\. Goal project dengan bahasa sederhana
----------------------------------------

Supaya mudah diucapkan ke dosen, kamu bisa rangkum tujuan project seperti ini:

1.  **Membuat sensor suhu murah menjadi jauh lebih akurat**  
    Pt100 tiga‑kawat yang murah biasanya punya error sampai beberapa derajat, apalagi di suhu ekstrem. Dengan kalibrasi berbasis data, pembacaannya bisa didekatkan ke suhu sebenarnya.
    
2.  **Membandingkan tiga pendekatan kalibrasi**  
    Kamu membandingkan:
    
    *   Model linier sederhana
        
    *   Model polinomial
        
    *   Model neural network  
        sehingga terlihat saat apa model sederhana cukup, dan kapan perlu ANN.
        
3.  **Menerapkan konsep machine learning di dunia instrumentasi**  
    Project ini menghubungkan teori ML (regresi, ANN, error metrics) dengan praktik pengukuran fisik: desain rangkaian, pengambilan data, dan kalibrasi.
    
4.  **Menurunkan galat dan meningkatkan kepercayaan pada data sensor**  
    Setelah kalibrasi, error rata‑rata turun drastis (di paper: MAE turun ke sekitar 0,1 °C dan 3σ ke 0,167 °C). Itu berarti sistem monitoring/ kontrol yang memakai sensor ini bisa mengambil keputusan yang lebih tepat.
    
5.  **Memberi dasar untuk sistem kendali cerdas**  
    Pada mata kuliah Instrumentasi dan Kendali Cerdas, sensor yang akurat sangat penting. Kalau input sensornya akurat, algoritma kendali cerdas di atasnya (misal fuzzy, PID adaptif, RL) bisa bekerja lebih baik.
    

Kalimat penutup yang bisa kamu pakai saat presentasi:

> “Project ini menunjukkan bahwa dengan memanfaatkan model machine learning, terutama neural network, pembacaan sensor suhu murah bisa dikoreksi sehingga mendekati suhu sebenarnya. Pendekatan seperti ini bisa diterapkan ke banyak jenis sensor lain supaya sistem instrumentasi dan kendali di industri lebih andal dan hemat biaya.”

## PROMPT 2

Karena meterbatasan alat di labo. Untuk mikrokontrollernya alhamdulillah terdapat Arduino. Sehingga dari sini apakah project ini tetap dapat berjalan, dan apabila memang dapat berjalan proses apa saja yang perlu diganti karena hanya memiliki sensor dht11 ini

## RESPONSE 2

Project‑nya masih sangat bisa jalan dengan Arduino + DHT11 🙂  
Kuncinya: tujuan tugas digeser sedikit dari **“mengulang eksperimen Pt100 persis”** menjadi **“mendemokan konsep kalibrasi sensor suhu berbasis machine learning seperti di paper”**, tapi dengan sensor yang tersedia.

Paper tetap jadi referensi teori, rumus, dan alur kalibrasinya.

sensors-23-07347

___________

1\. Pemetaan: paper vs kondisi lab kamu
---------------------------------------

Supaya jelas dulu apa yang berubah:

| Bagian | Di paper | Di lab kamu |
| --- | --- | --- |
| Sensor | 3× Pt100 3‑kawat (analog, resistansi) | 1× DHT11 (digital, temp + RH) |
| Rentang suhu | −50 s.d. 150 °C, step 0,2 °C | Kira‑kira 0 s.d. 50 °C (range kerja DHT11) |
| Lingkungan uji | Air check bin / chamber terkontrol | Kombinasi: suhu kamar, dekat es, dekat air hangat/panas < 50 °C, ruangan AC, dll |
| Akuisisi data | Rangkaian arus konstan + DAQ | Arduino membaca DHT11 via library |
| Jumlah sensor | 3 sensor (A,B,C) | Minimal 1 sensor DHT11 (kalau bisa pinjam 2–3 modul DHT11 malah bagus) |
| Algoritma | Linear, polynomial, ANN 3‑layer | Sama persis, dijalankan di Python |

Jadi yang berubah hanya **fisik sensornya dan cara buat variasi suhu**.  
Bagian **algoritma kalibrasi (rumus linear, polynomial, ANN, MAE/MSE/R²)** tetap mengacu ke paper.

___________

2\. Tujuan baru yang realistis
------------------------------

Tujuan yang kamu bilang ke dosen kurang lebih:

> Mengkalibrasi sensor suhu DHT11 murah dengan tiga metode (linear regression, polynomial regression, dan ANN) berdasarkan alur pada paper “Machine‑Learning‑Based Calibration of Temperature Sensors”, lalu membandingkan performanya.

___________

3\. Perubahan utama dibanding paper
-----------------------------------

### 3.1. Sensor dan teori dasar

Di laporan:

*   Bagian “Pt100 temperature sensor” dari paper tetap bisa dipakai sebagai **latar belakang umum** tentang pentingnya kalibrasi suhu.
    
    sensors-23-07347
    
*   Bagian “Materials and Methods – Sensors for calibration” kamu ganti menjadi penjelasan DHT11:
    
    Poin minimal yang ditulis:
    
    *   DHT11 mengukur suhu dan kelembapan dengan output digital.
        
    *   Rentang suhu kerja sekitar 0–50 °C, akurasi tipikal sekitar ±2 °C.
        
    *   Output data dikirim ke mikrokontroler dengan protokol 1‑wire.
        
*   Tidak perlu bahas arus konstan, perhitungan resistansi seperti Pt100, karena DHT11 langsung memberi angka suhu.
    

### 3.2. Rentang dan cara variasi suhu

*   Paper: −50 s.d. 140 °C dengan chamber, step 0,2 °C.
    
    sensors-23-07347
    
*   Kamu: rentang realistis DHT11, misalnya 15–45 °C, dengan beberapa titik:
    
    Contoh titik suhu:
    
    *   ~18–22 °C: suhu kamar biasa.
        
    *   ~15–18 °C: ruangan AC dingin / dekat jendela malam hari.
        
    *   ~25–30 °C: ruangan agak panas / dekat lampu / dekat adaptor laptop.
        
    *   ~35–40 °C: box tertutup dengan gelas air hangat di dalamnya.
        
    *   ~45–50 °C: box tertutup dengan gelas air lebih panas (jangan sampai sensor langsung kena uap panas ekstrim).
        

Sensor DHT11 jangan kena air langsung, taruh di udara di atas permukaan air hangat/air dingin di wadah tertutup.

### 3.3. Sensor referensi

Paper pakai suhu actual dari chamber sebagai referensi.

Di lab kamu:

*   Pakai **termometer referensi** apa pun yang tersedia:
    
    *   Termometer digital klinis (yang biasa untuk badan).
        
    *   Termometer ruangan.
        
    *   Modul sensor lain yang katanya lebih akurat (misal DS18B20 kalau ada).
        
*   Di data, pakai kolom:
    
    *   `T_ref` = suhu dari termometer referensi.
        
    *   `T_dht` = suhu dari DHT11.
        

___________

4\. Setup hardware baru
-----------------------

### 4.1. Alat yang dibutuhkan

*   1× Arduino Uno/Nano/yang sejenis.
    
*   1× DHT11 (lebih enak versi modul yang sudah ada resistor pull‑up).
    
*   1× termometer referensi.
    
*   Kabel jumper dan breadboard.
    
*   Beberapa wadah/wadah plastik bertutup untuk eksperimen suhu berbeda.
    
*   Laptop dengan Arduino IDE + Python.
    

### 4.2. Wiring DHT11 ke Arduino

Contoh (untuk modul 3 pin):

*   VCC → 5V
    
*   GND → GND
    
*   DATA → pin digital 2 (bebas, asal konsisten dengan kode)
    

___________

5\. Kode Arduino untuk logging data
-----------------------------------

Gunakan library DHT dari Adafruit (bisa pasang via Library Manager).

```cpp
#include "DHT.h"

#define DHTPIN 2      // pin data DHT11
#define DHTTYPE DHT11

DHT dht(DHTPIN, DHTTYPE);

void setup() {
  Serial.begin(9600);
  dht.begin();
  // header data, supaya gampang jadi CSV
  Serial.println("T_ref,T_dht");
}

void loop() {
  float t = dht.readTemperature(); // Celsius
  if (isnan(t)) {
    // gagal baca, skip
    return;
  }

  // Untuk setiap kondisi suhu:
  // - kamu baca T_dht dari sini
  // - kamu baca T_ref dari termometer manual
  // - kamu ketik T_ref di Serial Monitor dan Arduino cetak baris lengkap
  //   supaya tidak perlu edit banyak di Excel

  // Sementara, tampilkan cuma T_dht dulu
  Serial.print("?,"); // placeholder T_ref, nanti diganti manual di Excel/Notepad
  Serial.println(t);

  delay(2000); // baca tiap 2 detik
}
```

Cara praktis:

1.  Buka Serial Monitor.
    
2.  Biarkan Arduino mencetak `?,T_dht`.
    
3.  Copy log ke file `.txt`, simpan.
    
4.  Di Excel/LibreOffice:
    
    *   Pisahkan kolom dengan delimiter koma.
        
    *   Ganti `?` per baris dengan nilai `T_ref` yang kamu ukur manual.
        

Atau kalau mau rapi, kamu bisa tuliskan `T_ref` lewat **input serial** ke Arduino, tapi untuk tugas ini cara manual sudah cukup.

Format akhir file CSV:

```text
T_ref,T_dht
24.5,23.7
24.5,23.8
30.2,28.9
...
```

___________

6\. Prosedur pengambilan data (versi DHT11)
-------------------------------------------

Satu siklus data:

1.  Pilih kondisi suhu (misal box dengan air hangat).
    
2.  Letakkan DHT11 dan termometer referensi dalam box yang sama (di udara, bukan terendam).
    
3.  Tutup box, tunggu sampai pembacaan **stabil** (misal perubahan <0,2 °C selama 1–2 menit).
    
4.  Catat:
    
    *   `T_ref` dari termometer.
        
    *   `T_dht` yang terlihat di Serial Monitor (boleh ambil beberapa sampel, lalu dirata‑ratakan).
        
5.  Ulangi untuk beberapa kondisi suhu berbeda sampai punya, misal:
    
    *   8–10 titik suhu × 10 sampel per titik = 80–100 pasang data.
        

Kalau sempat, bagus kalau:

*   Ada data di ujung bawah (~15–18 °C) dan ujung atas (~45–50 °C), supaya model bisa melihat nonlinieritas di tepi rentang.
    

___________

7\. Pengolahan data & kalibrasi di Python
-----------------------------------------

### 7.1. Buka data dan cek error awal

```python
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("dht11_data.csv")  # kolom: T_ref, T_dht
df["err_raw"] = df["T_dht"] - df["T_ref"]

print(df.describe())

plt.scatter(df["T_ref"], df["err_raw"])
plt.xlabel("T_ref (°C)")
plt.ylabel("Error DHT11 (°C)")
plt.title("Error DHT11 sebelum kalibrasi")
plt.show()
```

Bagian ini setara dengan **Figure 3** di paper yang menunjukkan error sensor sebelum kalibrasi.

sensors-23-07347

### 7.2. Split train/test

```python
from sklearn.model_selection import train_test_split

X = df[["T_dht"]].values   # input: suhu dari DHT11
y = df["T_ref"].values     # target: suhu referensi

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```

### 7.3. Model 1 – Linear Regression (sesuai persamaan (2) di paper)

Rumus di paper:

Y\=wX+bY = w X + bY\=wX+b

```python
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

lin = LinearRegression()
lin.fit(X_train, y_train)

y_pred_train_lin = lin.predict(X_train)
y_pred_test_lin  = lin.predict(X_test)

def eval_model(name, y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    r2  = r2_score(y_true, y_pred)
    print(f"{name}: MAE={mae:.4f}, MSE={mse:.4f}, R2={r2:.6f}")

print("Koef linear: w =", lin.coef_[0], "b =", lin.intercept_)
eval_model("Linear train", y_train, y_pred_train_lin)
eval_model("Linear test",  y_test,  y_pred_test_lin)
```

### 7.4. Model 2 – Polynomial Regression (persamaan (3) di paper)

```python
from sklearn.preprocessing import PolynomialFeatures

poly = PolynomialFeatures(degree=2, include_bias=False)
X_train_poly = poly.fit_transform(X_train)
X_test_poly  = poly.transform(X_test)

poly_reg = LinearRegression()
poly_reg.fit(X_train_poly, y_train)

y_pred_train_poly = poly_reg.predict(X_train_poly)
y_pred_test_poly  = poly_reg.predict(X_test_poly)

print("Koef poly:", "intercept =", poly_reg.intercept_, "coef =", poly_reg.coef_)
eval_model("Poly train", y_train, y_pred_train_poly)
eval_model("Poly test",  y_test,  y_pred_test_poly)
```

### 7.5. Model 3 – ANN (persamaan (4)–(6) di paper)

Kamu mengikuti setting paper: 1 input → 1 hidden layer (16 neuron, ReLU) → 1 output.

sensors-23-07347

```python
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping

# Normalisasi input
X_mean = X_train.mean()
X_std  = X_train.std()
X_train_n = (X_train - X_mean) / X_std
X_test_n  = (X_test  - X_mean) / X_std

model = Sequential([
    Dense(16, activation="relu", input_shape=(1,)),
    Dense(1)  # output linear
])

model.compile(optimizer="adam", loss="mse", metrics=["mae"])

es = EarlyStopping(monitor="val_loss", patience=20, restore_best_weights=True)

history = model.fit(
    X_train_n, y_train,
    validation_data=(X_test_n, y_test),
    epochs=500,
    batch_size=16,
    callbacks=[es],
    verbose=0
)

y_pred_train_ann = model.predict(X_train_n).flatten()
y_pred_test_ann  = model.predict(X_test_n).flatten()

eval_model("ANN train", y_train, y_pred_train_ann)
eval_model("ANN test",  y_test,  y_pred_test_ann)
```

### 7.6. Bandingkan error setelah kalibrasi (mirip Figure 4–9 di paper)

```python
import matplotlib.pyplot as plt

# Terapkan model ke semua data
df["T_lin"]  = lin.predict(df[["T_dht"]])
df["T_poly"] = poly_reg.predict(poly.transform(df[["T_dht"]]))

X_all_n = (df[["T_dht"]].values - X_mean) / X_std
df["T_ann"]  = model.predict(X_all_n).flatten()

df["err_lin"]  = df["T_lin"]  - df["T_ref"]
df["err_poly"] = df["T_poly"] - df["T_ref"]
df["err_ann"]  = df["T_ann"]  - df["T_ref"]

plt.figure()
plt.scatter(df["T_ref"], df["err_raw"],  label="Original", s=10)
plt.scatter(df["T_ref"], df["err_lin"],  label="Linear",   s=10)
plt.scatter(df["T_ref"], df["err_poly"], label="Poly",     s=10)
plt.scatter(df["T_ref"], df["err_ann"],  label="ANN",      s=10)
plt.axhline(0, color="black", linewidth=0.5)
plt.xlabel("T_ref (°C)")
plt.ylabel("Error (°C)")
plt.legend()
plt.title("Perbandingan error DHT11 sebelum/sesudah kalibrasi")
plt.show()
```

Kalau model tersusun dengan benar, pola hasilnya akan mirip tabel 2 di paper: semua metode menurunkan MAE dan MSE, ANN biasanya paling kecil.

sensors-23-07347

___________

8\. Penyesuaian laporan / presentasi ke dosen
---------------------------------------------

Hal yang sebaiknya kamu tulis dan ucapkan jelas:

1.  Jelaskan singkat isi paper:
    
    *   Tiga metode kalibrasi: linear, polynomial, ANN.
        
    *   Mereka menggunakan Pt100 3‑kawat dengan chamber suhu lebar.
        
2.  Jelaskan keterbatasan alat di lab:
    
    *   Hanya tersedia Arduino dan sensor DHT11.
        
3.  Jelaskan langkah adaptasi:
    
    *   Sensor diganti ke DHT11, rentang suhu disesuaikan 0–50 °C.
        
    *   Lingkungan suhu dibentuk dengan box + air hangat/dingin, AC, dll.
        
    *   Algoritma kalibrasi dan metrik (MAE, MSE, R²) tetap sama seperti di paper.
        
4.  Tampilkan grafik:
    
    *   Error sebelum kalibrasi.
        
    *   Error setelah linear, polynomial, ANN.
        
    *   Tabel kecil MAE–MSE–R² untuk ketiga metode.
        

___________

9\. Jawaban langsung atas pertanyaanmu
--------------------------------------

*   **Project tetap bisa berjalan** dan masih sangat relevan dengan isi paper.
    
*   **Bagian yang diganti / disederhanakan:**
    
    1.  Pt100 + rangkaian arus konstan → DHT11 langsung ke Arduino.
        
    2.  Air check bin suhu −50 s.d. 150 °C → beberapa kondisi suhu sederhana 15–45 °C.
        
    3.  Tiga sensor A,B,C → minimal satu sensor DHT11 (kalau bisa pinjam beberapa, kamu bisa menyebutnya sensor A,B,C juga).
        
    4.  Sumber data referensi dari set‑point chamber → termometer referensi manual.
        
*   **Bagian yang tetap sama dengan paper:**
    
    1.  Konsep kalibrasi berbasis ML.
        
    2.  Penggunaan model linear regression, polynomial regression, dan ANN tiga lapis dengan 16 neuron hidden.
        
    3.  Evaluasi pakai MAE, MSE, dan R².
        

Projectmu jadi demonstrasi: **sejauh mana machine learning bisa meningkatkan akurasi sensor DHT11 murah**, dengan alur yang mengikuti paper Pt100 tadi.

