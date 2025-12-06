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

