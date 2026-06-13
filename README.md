# Simulasi Kuantum 1D (Quantum Tunneling & Scattering)

Repositori ini berisi simulasi numerik persamaan Schrödinger bergantung waktu (TDSE) dalam 1 dimensi untuk memodelkan perambatan paket gelombang kuantum (Gaussian wave packet) yang menabrak penghalang potensial (Gaussian potential barrier).

Simulasi ini diselesaikan menggunakan metode **Split-Step Fourier (SSFM)** yang sangat stabil dan akurat.

## Fitur Utama
- **Animasi Live (GUI Mode)**: Jika dijalankan di komputer lokal (Mac/PC), simulasi akan menampilkan visualisasi pergerakan gelombang secara real-time.
- **Headless Mode**: Jika dijalankan di server atau latar belakang, script otomatis mendeteksi ketiadaan GUI dan langsung menyimpan grafik hasil simulasi ke bentuk gambar.
- **Visualisasi Ilmiah**: Menghasilkan 2 jenis plot grafik presisi tinggi.

---

## Hasil Grafik Simulasi

### 1. Peta Evolusi Ruang-Waktu (Space-Time Evolution)
Menampilkan seluruh proses perambatan dan hamburan gelombang kuantum secara kontinu seiring waktu:
![Peta Evolusi Ruang-Waktu](quantum_space_time.png)

### 2. Keadaan Akhir (Final State)
Menampilkan bentuk paket gelombang (keadaan probabilitas $|\Psi(x)|^2$ dan bagian riil $\text{Re}(\Psi)$) setelah menabrak penghalang potensial:
![Keadaan Akhir](quantum_simulation.png)

---

## Cara Menjalankan

### 1. Install Dependensi
Pastikan Anda memiliki pustaka Python yang dibutuhkan:
```bash
pip install numpy matplotlib scipy
```

### 2. Jalankan Script
Jalankan perintah berikut di terminal Anda untuk melihat animasi live:
```bash
python3 quantum_simulation.py
```
*(Tutup jendela grafis untuk menyelesaikan simulasi dan menyimpan gambar).*
