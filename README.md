# Lab Pantulan Bola (RK4)

[![CI](https://github.com/zamzamzamakhsyari01-prog/lab-pantulan-bola/actions/workflows/ci.yml/badge.svg)](https://github.com/zamzamzamakhsyari01-prog/lab-pantulan-bola/actions)
![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

Aplikasi desktop (Tkinter) untuk praktikum **pantulan bola**: simulasi numerik dengan **Runge-Kutta orde 4**, pelacakan bola dari video/kamera dengan OpenCV, lalu estimasi **koefisien restitusi (e)** dan validasi model terhadap data eksperimen.

## Fitur

- **Simulasi RK4** bola jatuh + pantul, hambatan udara linier/kuadratik, tumbukan tepat di lantai.
- **Akuisisi data**: kamera langsung, berkas video, urutan gambar, CSV, serial Arduino/ESP32.
- **Pelacakan**: otomatis (warna/terang/gelap/gerak), semi-otomatis (CSRT/KCF/MIL), dan manual per frame.
- **Kalibrasi** penggaris + sumbu x–y (x = lantai) yang bisa diseret langsung pada pratinjau.
- **Analisis**: fit parabola per penerbangan; *e* dari tinggi, kecepatan, dan waktu terbang; estimasi g.
- **Validasi**: RMSE, MAE, R², residual, optimasi parameter, uji konvergensi RK4.
- **Ekspor**: grafik PNG, Excel, foto dokumentasi video, laporan TXT.
- **Pelepas bola otomatis** lewat gearbox ESP32 (`arduino/gearbox_esp32`).

## Versi web (tanpa instalasi)

Buka **https://zamzamzamakhsyari01-prog.github.io/lab-pantulan-bola/** — simulasi RK4 dan analisis koefisien restitusi dari CSV berjalan langsung di browser (Pyodide). Data tidak dikirim ke server; riwayat tersimpan di perangkat Anda. Pelacakan video, kamera, dan gearbox ESP32 hanya ada di aplikasi desktop.

Kode situs ada di `docs/`. Setelah mengubah `core_fisika.py`, jalankan `python scripts/sync_web.py`.

## Instalasi

```bash
git clone https://github.com/zamzamzamakhsyari01-prog/lab-pantulan-bola.git
cd lab-pantulan-bola
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[all]"
```

Tanpa Excel/serial: `pip install -e .`. Tkinter biasanya sudah ikut Python (Linux: `sudo apt install python3-tk`).

## Menjalankan

```bash
lab-pantulan          # atau: python -m lab_pantulan
```

## Memakai sebagai pustaka

```python
from lab_pantulan import core_fisika as cf

model = cf.buat_model("pantulan", dict(m=0.06, e=0.8, c=0.0, x0=1.0, v0=0.0))
T, Y = cf.simulasi(model, 4.0, 0.001)
print(model.info())
```

## Tes

```bash
pip install -e ".[dev]"
pytest
```

## Struktur

```
src/lab_pantulan/
  core_fisika.py   # RK4, model pantulan, analisis & validasi
  akuisisi.py      # kalibrasi, pelacak, sumber video/serial, CSV
  app.py           # GUI Tkinter
docs/                    # situs GitHub Pages (Pyodide + Plotly)
arduino/gearbox_esp32/   # sketch pelepas bola
tests/
```

## Catatan video

Rekaman HP berkodek H.265/HEVC sering gagal dibuka OpenCV. Konversi dulu:

```bash
ffmpeg -i masukan.mp4 -c:v libx264 -pix_fmt yuv420p keluaran.mp4
```

## Lisensi

MIT — lihat [LICENSE](LICENSE).
