## module akuisisi
"""
akuisisi.py
===========
Sumber data eksperimen:
  * Kamera langsung (webcam / kamera HP via DroidCam-IP)
  * Berkas video (mp4, avi, mov, ...)
  * Impor gambar dari perangkat (satu atau banyak berkas / urutan frame)
  * Berkas CSV hasil rekaman sebelumnya
  * Serial Arduino (sensor ultrasonik HC-SR04, potensiometer, dsb.)

Pelacakan objek memakai OpenCV:
  - mode 'warna'  : segmentasi HSV (pilih warna dengan klik pada frame)
  - mode 'terang' : objek paling terang
  - mode 'gelap'  : objek paling gelap
  - mode 'gerak'  : pengurangan latar (background subtraction)
"""

import os
import csv
import glob
import numpy as np

try:
    import cv2
    CV2_ADA = True
except Exception:                                  # pragma: no cover
    CV2_ADA = False

try:
    import serial                                  # pyserial (opsional)
    SERIAL_ADA = True
except Exception:
    SERIAL_ADA = False


EKSTENSI_GAMBAR = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.webp')
EKSTENSI_VIDEO = ('.mp4', '.avi', '.mov', '.mkv', '.wmv', '.m4v', '.mpg', '.mpeg')


# =====================================================================
# KALIBRASI PIKSEL -> METER
# =====================================================================
class Kalibrasi:
    """Kalibrasi piksel -> meter dari DUA alat yang diseret langsung pada pratinjau.

    1. PENGGARIS (skala)  : garis dari titik 0 ke titik ujung. Panjang nyatanya (cm)
                            diisi pengguna dan bebas divariasikan (10, 20, 30, 50 cm, ...).
                            Hasil: px_per_m.
    2. SUMBU x-y          : titik asal (origin) + sudut sumbu x terhadap tepi gambar.
                            SUMBU x ADALAH LANTAI (y = 0); sumbu y tegak lurus x, positif
                            ke atas. Asal dan sudut dapat digeser/diputar seperti sumbu
                            koordinat pada Tracker, sehingga kamera yang agak miring
                            (roll) tidak membuat gerak vertikal bola terbaca sebagai gerak
                            horizontal.

    r_bola : jari-jari bola (m). y yang dilaporkan = tinggi pusat bola di atas lantai
             dikurangi r_bola, jadi y = 0 tepat saat bola menyentuh lantai (sesuai model:
             pantulan terjadi di y = 0).
    """

    def __init__(self, px_per_m=1000.0, origin=(0.0, 0.0), r_bola=0.0):
        self.px_per_m = float(px_per_m)
        self._origin = (float(origin[0]), float(origin[1]))
        self.lantai_diset = (self._origin != (0.0, 0.0))   # sumbu x (lantai) ditentukan pengguna
        self._origin_sementara = False                    # titik nol darurat (posisi bola pertama)
        self.skala_diset = False                          # skala berasal dari kalibrasi
        self.panjang_ref = 0.0                            # panjang nyata penggaris (m)
        self.r_bola = float(r_bola)
        self._ux, self._uy = 0.0, 1.0                     # arah "atas" (kanan, atas)
        self.p0 = None                                    # titik 0 penggaris (piksel gambar)
        self.p1 = None                                    # titik ujung penggaris (piksel gambar)

    # ---- titik nol (asal sumbu) ----
    @property
    def origin(self):
        return self._origin

    @origin.setter
    def origin(self, nilai):
        """Set asal sumbu (titik di lantai) secara eksplisit oleh pengguna."""
        self._origin = (float(nilai[0]), float(nilai[1]))
        self.lantai_diset = True
        self._origin_sementara = False

    def origin_sementara(self, cx, cy):
        """Titik nol DARURAT = posisi bola pertama (hanya agar angka tidak besar).
        Tidak dianggap kalibrasi lantai; dipakai bila pengguna belum mengatur sumbu."""
        if not self.lantai_diset and not self._origin_sementara:
            self._origin = (float(cx), float(cy))
            self._origin_sementara = True

    def reset_sementara(self):
        if self._origin_sementara and not self.lantai_diset:
            self._origin = (0.0, 0.0)
            self._origin_sementara = False

    # ---- orientasi sumbu ----
    @property
    def sudut_x_deg(self):
        """Sudut sumbu x (= lantai) terhadap garis mendatar gambar, derajat.
        0 = lantai datar; positif = ujung kanan sumbu x naik di layar."""
        return float(np.degrees(np.arctan2(-self._ux, self._uy))) + 0.0     # + 0.0: hindari "-0.0"

    @sudut_x_deg.setter
    def sudut_x_deg(self, derajat):
        th = np.radians(float(derajat))
        self._ux, self._uy = -float(np.sin(th)), float(np.cos(th))

    def atur_sumbu(self, asal, sudut_x_deg):
        """Tetapkan sumbu x-y sekaligus: asal (piksel) dan sudut sumbu x (derajat)."""
        self.origin = asal
        self.sudut_x_deg = sudut_x_deg

    def arah_atas_px(self):
        """Vektor satuan sumbu y ('atas') dalam koordinat piksel gambar (x kanan, y bawah)."""
        return (self._ux, -self._uy)

    def arah_samping_px(self):
        """Vektor satuan sumbu x (searah lantai) dalam koordinat piksel gambar."""
        return (self._uy, self._ux)

    # ---- penggaris (skala) ----
    def panjang_px(self):
        if self.p0 is None or self.p1 is None:
            return 0.0
        return float(np.hypot(self.p1[0] - self.p0[0], self.p1[1] - self.p0[1]))

    def hitung_skala(self):
        """Skala (px/m) dari penggaris yang tersimpan dan panjang nyatanya (panjang_ref)."""
        if self.p0 is None or self.p1 is None:
            raise ValueError('Seret penggaris pada pratinjau (dari titik 0 ke ujung) dahulu.')
        if self.panjang_ref <= 0:
            raise ValueError('Isi "Panjang acuan" (> 0) sesuai jarak titik 0 ke ujung penggaris.')
        d = self.panjang_px()
        if d < 5:
            raise ValueError('Penggaris terlalu pendek (< 5 piksel).')
        self.px_per_m = d / self.panjang_ref
        self.skala_diset = True
        return self.px_per_m

    def atur_penggaris(self, p0, p1, panjang_m):
        """Simpan penggaris (titik 0 -> ujung) dan hitung skalanya."""
        self.p0 = (float(p0[0]), float(p0[1]))
        self.p1 = (float(p1[0]), float(p1[1]))
        self.panjang_ref = float(panjang_m)
        return self.hitung_skala()

    def tukar_penggaris(self):
        """Tukar titik 0 dengan titik ujung."""
        self.p0, self.p1 = self.p1, self.p0

    def hapus_penggaris(self):
        self.p0 = self.p1 = None
        self.skala_diset = False
        self.panjang_ref = 0.0

    def sumbu_dari_penggaris(self):
        """Pasang sumbu x-y mengikuti penggaris yang sudah digambar (jalan pintas):
        asal = titik 0 penggaris.
          * penggaris lebih mendatar : tergeletak di lantai -> sumbu x searah penggaris
          * penggaris lebih tegak    : berdiri di lantai    -> sumbu y searah penggaris
        Mengembalikan sudut sumbu x (derajat)."""
        if self.p0 is None or self.p1 is None:
            raise ValueError('Gambar penggaris dahulu.')
        vx = float(self.p1[0] - self.p0[0])          # kanan
        vy = float(self.p0[1] - self.p1[1])          # atas
        if abs(vx) < 1e-9 and abs(vy) < 1e-9:
            raise ValueError('Penggaris terlalu pendek.')
        phi = float(np.degrees(np.arctan2(vy, vx)))
        if abs(vx) >= abs(vy):                       # mendatar
            th = phi if vx >= 0 else (phi - 180.0 if phi > 0 else phi + 180.0)
        else:                                        # tegak
            th = phi - 90.0 if vy > 0 else phi + 90.0
        self.atur_sumbu(self.p0, th)
        return th

    # ---- kompatibilitas versi lama ----
    def dari_dua_titik(self, p1, p2, panjang_m):
        """Hanya skala (kompatibel dengan versi lama)."""
        d = float(np.hypot(p2[0] - p1[0], p2[1] - p1[1]))
        if panjang_m > 0 and d > 0:
            self.px_per_m = d / panjang_m
            self.skala_diset = True
        return self.px_per_m

    # ---- konversi ----
    def ke_meter(self, cx, cy):
        """Piksel -> (x_m, y_m). x searah sumbu x (lantai), y ke ATAS positif tegak lurus
        lantai, y = 0 saat bola menyentuh lantai. Boleh skalar maupun array numpy."""
        ox, oy = self._origin
        dx = np.asarray(cx, dtype=float) - ox
        dy = oy - np.asarray(cy, dtype=float)            # ke atas positif
        y_px = dx * self._ux + dy * self._uy              # searah sumbu y
        x_px = dx * self._uy - dy * self._ux              # searah sumbu x (lantai)
        x_m = x_px / self.px_per_m
        y_m = y_px / self.px_per_m - self.r_bola
        if np.ndim(x_m) == 0:
            return float(x_m), float(y_m)
        return x_m, y_m

    def ke_piksel(self, x_m, y_m):
        """Kebalikan ke_meter (untuk menggambar lapisan kalibrasi)."""
        ox, oy = self._origin
        x_px = x_m * self.px_per_m
        y_px = (y_m + self.r_bola) * self.px_per_m
        dx = x_px * self._uy + y_px * self._ux
        dy = -x_px * self._ux + y_px * self._uy
        return ox + dx, oy - dy


# =====================================================================
# PELACAK OBJEK
# =====================================================================
class PelacakObjek:
    def __init__(self, mode='warna', toleransi=(12, 80, 80), luas_min=30):
        self.mode = mode
        self.toleransi = toleransi
        self.luas_min = int(luas_min)
        self.hsv_bawah = np.array([0, 120, 70])
        self.hsv_atas = np.array([12, 255, 255])
        self.roi = None                   # (x, y, w, h) pembatas pencarian
        self._latar = None
        self.terakhir = None

    # ---- pengaturan warna dari klik pengguna ----
    def ambil_warna(self, frame, x, y, r=5):
        if not CV2_ADA:
            return
        h, w = frame.shape[:2]
        x0, x1 = max(0, x - r), min(w, x + r + 1)
        y0, y1 = max(0, y - r), min(h, y + r + 1)
        if x1 <= x0 or y1 <= y0:
            return
        petak = cv2.cvtColor(frame[y0:y1, x0:x1], cv2.COLOR_BGR2HSV).reshape(-1, 3).astype(float)
        # hue bersifat melingkar (merah ada di 0 dan 179) -> pakai rata-rata sudut
        sudut = petak[:, 0] * 2.0 * np.pi / 180.0
        hue = (np.degrees(np.arctan2(np.mean(np.sin(sudut)), np.mean(np.cos(sudut)))) / 2.0) % 180.0
        sat = float(np.median(petak[:, 1]))
        val = float(np.median(petak[:, 2]))
        dh, ds, dv = self.toleransi
        self.hue_tengah = float(hue)
        self.hsv_bawah = np.array([max(hue - dh, 0), max(sat - ds, 40), max(val - dv, 40)])
        self.hsv_atas = np.array([min(hue + dh, 179), 255, 255])
        self.mode = 'warna'
        return np.array([hue, sat, val])

    def reset(self):
        self._latar = None
        self.terakhir = None

    # ---- pelacakan satu frame ----
    def proses(self, frame):
        """Kembalikan (cx, cy, mask, frame_anotasi). (None, None, ...) bila gagal."""
        if not CV2_ADA:
            return None, None, None, frame
        kerja = frame
        ox = oy = 0
        if self.roi:
            x, y, w, h = [int(v) for v in self.roi]
            kerja = frame[y:y + h, x:x + w]
            ox, oy = x, y

        if self.mode == 'warna':
            hsv = cv2.cvtColor(kerja, cv2.COLOR_BGR2HSV)
            lo, hi = self.hsv_bawah.astype(float), self.hsv_atas.astype(float)
            s_min, v_min = lo[1], lo[2]
            pusat = getattr(self, 'hue_tengah', (lo[0] + hi[0]) / 2.0)
            dh = max((hi[0] - lo[0]) / 2.0, self.toleransi[0])
            h1, h2 = pusat - dh, pusat + dh
            mask = cv2.inRange(hsv,
                               np.array([max(h1, 0), s_min, v_min], np.uint8),
                               np.array([min(h2, 179), 255, 255], np.uint8))
            if h1 < 0:                      # hue membungkus lewat 0 (merah)
                mask |= cv2.inRange(hsv, np.array([180 + h1, s_min, v_min], np.uint8),
                                    np.array([179, 255, 255], np.uint8))
            if h2 > 179:                    # membungkus lewat 179
                mask |= cv2.inRange(hsv, np.array([0, s_min, v_min], np.uint8),
                                    np.array([h2 - 180, 255, 255], np.uint8))
        elif self.mode in ('terang', 'gelap'):
            abu = cv2.cvtColor(kerja, cv2.COLOR_BGR2GRAY)
            abu = cv2.GaussianBlur(abu, (7, 7), 0)
            if self.mode == 'terang':
                _, mask = cv2.threshold(abu, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            else:
                _, mask = cv2.threshold(abu, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        else:                                    # 'gerak'
            abu = cv2.cvtColor(kerja, cv2.COLOR_BGR2GRAY)
            abu = cv2.GaussianBlur(abu, (7, 7), 0).astype(np.float32)
            if self._latar is None:
                self._latar = abu.copy()
                return None, None, None, frame
            beda = cv2.absdiff(abu, self._latar).astype(np.uint8)
            cv2.accumulateWeighted(abu, self._latar, 0.05)
            _, mask = cv2.threshold(beda, 18, 255, cv2.THRESH_BINARY)

        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))

        kontur, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        kontur = [c for c in kontur if cv2.contourArea(c) >= self.luas_min]
        anotasi = frame.copy()
        if not kontur:
            return None, None, mask, anotasi

        c = max(kontur, key=cv2.contourArea)
        M = cv2.moments(c)
        if M['m00'] == 0:
            return None, None, mask, anotasi
        cx = M['m10'] / M['m00'] + ox
        cy = M['m01'] / M['m00'] + oy
        self.terakhir = (cx, cy)

        (xr, yr), rad = cv2.minEnclosingCircle(c)
        cv2.circle(anotasi, (int(xr + ox), int(yr + oy)), int(rad), (0, 255, 255), 2)
        cv2.circle(anotasi, (int(cx), int(cy)), 4, (0, 0, 255), -1)
        cv2.putText(anotasi, f'({cx:.0f},{cy:.0f})', (int(cx) + 8, int(cy) - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        if self.roi:
            x, y, w, h = [int(v) for v in self.roi]
            cv2.rectangle(anotasi, (x, y), (x + w, y + h), (255, 0, 0), 1)
        return cx, cy, mask, anotasi


class PelacakCSRT:
    """
    Pelacakan SEMI-OTOMATIS: pengguna menandai objek satu kali (ROI),
    selanjutnya OpenCV mengikuti objek itu sendiri antar frame.
    Antarmukanya sama dengan PelacakObjek sehingga bisa dipakai lacak_sumber().
    """

    URUTAN = ('TrackerCSRT_create', 'TrackerKCF_create', 'TrackerMIL_create')

    def __init__(self, algoritma=None):
        self.algoritma = algoritma
        self.tracker = None
        self.roi = None
        self.terakhir = None
        self.nama = '-'

    def _buat(self):
        if not CV2_ADA:
            raise RuntimeError('OpenCV belum terpasang.')
        kandidat = [self.algoritma] if self.algoritma else list(self.URUTAN)
        for nama in kandidat:
            for induk in (cv2, getattr(cv2, 'legacy', None)):
                if induk is not None and hasattr(induk, nama):
                    self.nama = nama.replace('Tracker', '').replace('_create', '')
                    return getattr(induk, nama)()
        raise RuntimeError('Tidak ada algoritma pelacak (CSRT/KCF/MIL) pada OpenCV ini.')

    def mulai(self, frame, roi):
        """roi = (x, y, lebar, tinggi) hasil seretan pengguna."""
        x, y, w, h = [int(v) for v in roi]
        if w < 4 or h < 4:
            raise ValueError('Kotak ROI terlalu kecil.')
        self.tracker = self._buat()
        self.tracker.init(frame, (x, y, w, h))
        self.roi = (x, y, w, h)
        self.terakhir = (x + w / 2.0, y + h / 2.0)
        return self.nama

    def reset(self):
        self.tracker = None
        self.terakhir = None

    def proses(self, frame):
        anotasi = frame.copy()
        if self.tracker is None:
            cv2.putText(anotasi, 'Pilih ROI objek lalu tekan Kunci Objek',
                        (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
            return None, None, None, anotasi
        ok, kotak = self.tracker.update(frame)
        if not ok:
            cv2.putText(anotasi, 'Objek hilang — kunci ulang', (10, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            return None, None, None, anotasi
        x, y, w, h = [float(v) for v in kotak]
        cx, cy = x + w / 2.0, y + h / 2.0
        self.terakhir = (cx, cy)
        cv2.rectangle(anotasi, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 255), 2)
        cv2.circle(anotasi, (int(cx), int(cy)), 4, (0, 0, 255), -1)
        cv2.putText(anotasi, f'{self.nama} ({cx:.0f},{cy:.0f})', (int(x), max(int(y) - 8, 14)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        return cx, cy, None, anotasi


def gambar_penanda(frame, titik, aktif=None, riwayat=(), kursor=None, zoom=4):
    """Gambar jejak titik manual, garis bidik, dan kaca pembesar pada frame."""
    if not CV2_ADA or frame is None:
        return frame
    keluar = frame.copy()
    H, W = keluar.shape[:2]
    for (px, py) in riwayat:
        cv2.circle(keluar, (int(px), int(py)), 2, (0, 180, 255), -1)
    if titik is not None:
        cv2.drawMarker(keluar, (int(titik[0]), int(titik[1])), (0, 0, 255),
                       cv2.MARKER_CROSS, 22, 2)
        cv2.circle(keluar, (int(titik[0]), int(titik[1])), 10, (0, 255, 0), 1)
    if kursor is not None:
        kx, ky = int(kursor[0]), int(kursor[1])
        if 0 <= kx < W and 0 <= ky < H:
            cv2.line(keluar, (kx, 0), (kx, H), (0, 255, 0), 1)
            cv2.line(keluar, (0, ky), (W, ky), (0, 255, 0), 1)
            # kaca pembesar di pojok kanan atas
            r = max(int(min(W, H) * 0.06), 12)
            x0, x1 = max(kx - r, 0), min(kx + r, W)
            y0, y1 = max(ky - r, 0), min(ky + r, H)
            petak = frame[y0:y1, x0:x1]
            if petak.size:
                sisi = min(int(2 * r * zoom), min(W, H) // 2)
                besar = cv2.resize(petak, (sisi, sisi), interpolation=cv2.INTER_NEAREST)
                cv2.line(besar, (sisi // 2, 0), (sisi // 2, sisi), (0, 255, 0), 1)
                cv2.line(besar, (0, sisi // 2), (sisi, sisi // 2), (0, 255, 0), 1)
                keluar[4:4 + sisi, W - sisi - 4:W - 4] = besar
                cv2.rectangle(keluar, (W - sisi - 4, 4), (W - 4, 4 + sisi), (0, 255, 0), 1)
    if aktif:
        cv2.putText(keluar, aktif, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    return keluar


def gambar_kalibrasi(frame, kal):
    """Lapisan kalibrasi di atas frame: SUMBU x = LANTAI (y=0), sumbu y, dan penggaris.
    Mengembalikan salinan; frame asli tidak diubah. (Pratinjau interaktif di aplikasi
    menggambar lapisan yang sama pada kanvas Tk supaya bisa diseret.)"""
    if not CV2_ADA or frame is None:
        return frame
    ada_penggaris = kal.p0 is not None and kal.p1 is not None
    if not (kal.lantai_diset or ada_penggaris):
        return frame
    keluar = frame.copy()
    H, W = keluar.shape[:2]
    if kal.lantai_diset:
        ox, oy = kal.origin
        ax_, ay_ = kal.arah_atas_px()
        sx, sy = kal.arah_samping_px()
        L = 2.0 * float(np.hypot(W, H))

        def garis(tinggi_px, warna, tebal):
            bx, by = ox + tinggi_px * ax_, oy + tinggi_px * ay_
            cv2.line(keluar, (int(bx - L * sx), int(by - L * sy)),
                     (int(bx + L * sx), int(by + L * sy)), warna, tebal)

        garis(0.0, (0, 200, 0), 2)                                  # sumbu x = lantai (y=0)
        if kal.r_bola > 0:                                          # level pusat saat kontak
            garis(kal.r_bola * kal.px_per_m, (255, 140, 0), 1)
        pan = 110.0
        cv2.arrowedLine(keluar, (int(ox), int(oy)), (int(ox + pan * ax_), int(oy + pan * ay_)),
                        (0, 160, 255), 2, tipLength=0.15)           # sumbu y
        cv2.putText(keluar, 'x = lantai (y=0)',
                    (int(min(max(ox + 130 * sx, 5), W - 130)), int(min(max(oy + 130 * sy + 16, 14), H - 4))),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 0), 1)
        cv2.putText(keluar, 'y', (int(ox + (pan + 12) * ax_), int(oy + (pan + 12) * ay_)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 160, 255), 1)
        cv2.circle(keluar, (int(ox), int(oy)), 5, (60, 60, 255), -1)
    if ada_penggaris:
        a = (int(kal.p0[0]), int(kal.p0[1]))
        b = (int(kal.p1[0]), int(kal.p1[1]))
        cv2.line(keluar, a, b, (255, 229, 0), 2)
        cv2.circle(keluar, a, 5, (255, 229, 0), -1)
        cv2.circle(keluar, b, 5, (255, 229, 0), -1)
        cv2.putText(keluar, '0', (a[0] + 8, a[1] + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 229, 0), 1)
        if kal.panjang_ref > 0:
            cv2.putText(keluar, f'{kal.panjang_ref * 100:g} cm', (b[0] + 8, b[1] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 229, 0), 1)
    cv2.putText(keluar, f'Kal: {kal.px_per_m:.0f} px/m | sumbu x {kal.sudut_x_deg:+.1f} deg',
                (10, H - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
    return keluar


# =====================================================================
# SUMBER DATA
# =====================================================================
class SumberVideo:
    """Pembungkus seragam untuk kamera, berkas video, dan urutan gambar."""

    def __init__(self, tipe, sumber, fps=30.0):
        self.tipe = tipe                 # 'kamera' | 'video' | 'gambar'
        self.sumber = sumber
        self.fps = float(fps) if fps else 30.0
        self.cap = None
        self.berkas = []
        self.i = 0
        self.t0 = 0.0
        self.backend = '-'
        self.rotasi = 0                  # metadata putaran (video dari HP)
        self.ukuran = (0, 0)
        self.pesan = ''
        self._buka()

    # ---- pembantu ----
    @staticmethod
    def _daftar_backend(tipe):
        nama = (['CAP_FFMPEG', 'CAP_ANY', 'CAP_MSMF', 'CAP_GSTREAMER', 'CAP_IMAGES']
                if tipe == 'video' else
                ['CAP_ANY', 'CAP_DSHOW', 'CAP_MSMF', 'CAP_V4L2', 'CAP_AVFOUNDATION'])
        keluar = []
        for n in nama:
            if hasattr(cv2, n):
                keluar.append((n, getattr(cv2, n)))
        return keluar or [('CAP_ANY', 0)]

    def _putar(self, frame):
        if frame is None or not self.rotasi:
            return frame
        if self.rotasi == 90:
            return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        if self.rotasi == 180:
            return cv2.rotate(frame, cv2.ROTATE_180)
        if self.rotasi == 270:
            return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        return frame

    def diagnosa(self):
        """Ringkasan kondisi sumber — dipakai untuk pesan galat yang jelas."""
        return {'tipe': self.tipe, 'backend': self.backend, 'fps': round(self.fps, 3),
                'jumlah_frame': self.jumlah_frame, 'ukuran': self.ukuran,
                'rotasi': self.rotasi, 'pesan': self.pesan}

    def _buka(self):
        if not CV2_ADA:
            raise RuntimeError('OpenCV (cv2) belum terpasang. Jalankan: pip install opencv-python')
        if self.tipe in ('kamera', 'video'):
            if self.tipe == 'video' and not os.path.isfile(str(self.sumber)):
                raise RuntimeError(f'Berkas tidak ditemukan: {self.sumber}')
            target = (int(self.sumber) if (self.tipe == 'kamera' and str(self.sumber).isdigit())
                      else self.sumber)
            galat = []
            for nama, kode in self._daftar_backend(self.tipe):
                try:
                    cap = cv2.VideoCapture(target, kode) if kode else cv2.VideoCapture(target)
                except Exception as exc:
                    galat.append(f'{nama}: {exc}'); continue
                if not cap.isOpened():
                    cap.release(); galat.append(f'{nama}: tidak terbuka'); continue
                ok, frame = cap.read()          # WAJIB bisa membaca 1 frame nyata
                if not ok or frame is None or frame.size == 0:
                    cap.release(); galat.append(f'{nama}: terbuka tetapi frame kosong'); continue
                self.cap = cap
                self.backend = nama
                self.ukuran = (int(frame.shape[1]), int(frame.shape[0]))
                if float(frame.mean()) < 1.0:
                    self.pesan = ('Frame pertama seluruhnya hitam — periksa video atau '
                                  'gunakan codec lain.')
                break
            if self.cap is None:
                raise RuntimeError(
                    ('Video tidak dapat didekode oleh OpenCV: ' if self.tipe == 'video'
                     else 'Kamera tidak dapat dibuka: ') + str(self.sumber) +
                    '\n\nPercobaan: ' + '; '.join(galat) +
                    ('\n\nSaran: banyak rekaman HP memakai codec H.265/HEVC yang tidak '
                     'didukung OpenCV. Konversikan dahulu ke H.264/MP4, misalnya:\n'
                     '  ffmpeg -i masukan.mp4 -c:v libx264 -pix_fmt yuv420p keluaran.mp4\n'
                     'Alternatif lain: ekspor video menjadi urutan gambar lalu pakai '
                     '"Impor gambar (banyak)".' if self.tipe == 'video' else ''))
            f = self.cap.get(cv2.CAP_PROP_FPS)
            if f and 1 < f < 1000:
                self.fps = float(f)
            try:
                if hasattr(cv2, 'CAP_PROP_ORIENTATION_META'):
                    self.rotasi = int(self.cap.get(cv2.CAP_PROP_ORIENTATION_META) or 0) % 360
            except Exception:
                self.rotasi = 0
            if self.tipe == 'video':
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        else:                                       # urutan gambar
            if isinstance(self.sumber, (list, tuple)):
                self.berkas = list(self.sumber)
            elif os.path.isdir(self.sumber):
                self.berkas = sorted(
                    [p for p in glob.glob(os.path.join(self.sumber, '*'))
                     if p.lower().endswith(EKSTENSI_GAMBAR)])
            else:
                self.berkas = [self.sumber]
            if not self.berkas:
                raise RuntimeError('Tidak ada berkas gambar yang ditemukan.')

    @property
    def jumlah_frame(self):
        if self.tipe == 'gambar':
            return len(self.berkas)
        if self.cap:
            n = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            return n if n > 0 else -1
        return -1

    def baca(self):
        """Kembalikan (ok, frame, t) dengan t dalam detik."""
        if self.tipe == 'gambar':
            if self.i >= len(self.berkas):
                return False, None, 0.0
            frame = cv2.imread(self.berkas[self.i])
            t = self.i / self.fps
            self.i += 1
            return frame is not None, frame, t
        gagal = 0
        ok, frame = self.cap.read()
        while (not ok or frame is None) and gagal < 5:     # lewati frame rusak
            gagal += 1
            self.i += 1
            ok, frame = self.cap.read()
        if not ok or frame is None:
            return False, None, 0.0
        frame = self._putar(frame)
        if self.tipe == 'video':
            ms = self.cap.get(cv2.CAP_PROP_POS_MSEC)
            t = ms / 1000.0 if ms and ms > 0 else self.i / self.fps
        else:
            t = self.i / self.fps
        self.i += 1
        return True, frame, t

    def bisa_lompat(self):
        """True bila sumber mendukung akses frame acak (video / urutan gambar)."""
        return self.tipe in ('video', 'gambar')

    def ambil_frame(self, i):
        """Ambil frame ke-i tanpa mengubah alur baca berurutan. -> (ok, frame, t)."""
        i = int(max(i, 0))
        if self.tipe == 'gambar':
            if i >= len(self.berkas):
                return False, None, 0.0
            frame = cv2.imread(self.berkas[i])
            return frame is not None, frame, i / self.fps
        if self.tipe == 'video':
            n = self.jumlah_frame
            if n > 0 and i >= n:
                return False, None, 0.0
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ok, frame = self.cap.read()
            if not ok or frame is None:                    # sebagian codec gagal seek
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                for _ in range(i + 1):
                    ok, frame = self.cap.read()
                    if not ok:
                        break
            if not ok or frame is None:
                return False, None, 0.0
            frame = self._putar(frame)
            ms = self.cap.get(cv2.CAP_PROP_POS_MSEC)
            t = ms / 1000.0 if ms and ms > 0 else i / self.fps
            return True, frame, t
        # kamera: tidak bisa melompat, ambil frame terkini
        ok, frame, _ = self.baca()
        return ok, frame, i / self.fps

    def lepas(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None


class SumberSerial:
    """
    Pembaca Arduino. Format baris yang diterima:
        "1.234,0.0215"  -> waktu(s), nilai
        "0.0215"        -> nilai saja (waktu dihitung dari pencacah)
    """

    def __init__(self, port, baud=9600, fps=50.0, skala=1.0):
        if not SERIAL_ADA:
            raise RuntimeError('Modul pyserial belum terpasang. Jalankan: pip install pyserial')
        self.ser = serial.Serial(port, int(baud), timeout=0.05)
        self.fps = float(fps)
        self.skala = float(skala)
        self.i = 0

    @staticmethod
    def daftar_port():
        if not SERIAL_ADA:
            return []
        try:
            from serial.tools import list_ports
            return [p.device for p in list_ports.comports()]
        except Exception:
            return []

    def baca(self):
        """Kembalikan daftar (t, nilai) yang tersedia di buffer."""
        keluar = []
        try:
            while self.ser.in_waiting:
                baris = self.ser.readline().decode('utf-8', 'ignore').strip()
                if not baris:
                    continue
                bagian = [b for b in baris.replace(';', ',').replace('\t', ',').split(',') if b]
                try:
                    if len(bagian) >= 2:
                        t, v = float(bagian[0]), float(bagian[1])
                        if t > 1000:               # kemungkinan millis()
                            t /= 1000.0
                    else:
                        v = float(bagian[0])
                        t = self.i / self.fps
                except ValueError:
                    continue
                self.i += 1
                keluar.append((t, v * self.skala))
        except Exception:
            pass
        return keluar

    def lepas(self):
        try:
            self.ser.close()
        except Exception:
            pass


# =====================================================================
# PELACAKAN OTOMATIS SELURUH BERKAS (batch)
# =====================================================================
def lacak_sumber(sumber, pelacak, kalibrasi, maks_frame=100000, callback=None):
    """
    Lacak seluruh frame dari SumberVideo (video / urutan gambar).
    Kembalikan dict berisi t, x_m, y_m, cx_px, cy_px, dan jumlah frame gagal.
    """
    t_l, x_l, y_l, cx_l, cy_l, i_l = [], [], [], [], [], []
    gagal = 0
    n = 0
    while n < maks_frame:
        ok, frame, t = sumber.baca()
        if not ok:
            break
        cx, cy, _, anotasi = pelacak.proses(frame)
        n += 1
        if cx is None:
            gagal += 1
        else:
            kalibrasi.origin_sementara(cx, cy)      # hanya bila lantai belum dikalibrasi
            xm, ym = kalibrasi.ke_meter(cx, cy)
            t_l.append(t); x_l.append(xm); y_l.append(ym)
            cx_l.append(cx); cy_l.append(cy)
            i_l.append(max(sumber.i - 1, 0))           # nomor frame (untuk foto dokumentasi)
        if callback is not None and n % 5 == 0:
            callback(n, anotasi)
    return {'t': np.array(t_l), 'x': np.array(x_l), 'y': np.array(y_l),
            'cx': np.array(cx_l), 'cy': np.array(cy_l),
            'i': np.array(i_l, dtype=int), 'n_frame': n, 'n_gagal': gagal}


# =====================================================================
# BACA / TULIS CSV
# =====================================================================
def simpan_csv(nama, t, x, y=None, header_tambahan=None):
    with open(nama, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        if header_tambahan:
            for baris in header_tambahan:
                w.writerow(['#', baris])
        if y is None:
            w.writerow(['t_s', 'sinyal'])
            for a, b in zip(t, x):
                w.writerow([f'{a:.6f}', f'{b:.6f}'])
        else:
            w.writerow(['t_s', 'x_m', 'y_m'])
            for a, b, c in zip(t, x, y):
                w.writerow([f'{a:.6f}', f'{b:.6f}', f'{c:.6f}'])
    return nama


def muat_csv(nama):
    """Baca CSV fleksibel: kolom pertama waktu, kolom berikutnya data."""
    t, kolom = [], []
    with open(nama, 'r', encoding='utf-8', errors='ignore') as f:
        for baris in f:
            baris = baris.strip()
            if not baris or baris.startswith('#'):
                continue
            bagian = [b for b in baris.replace(';', ',').replace('\t', ',').split(',') if b != '']
            try:
                angka = [float(b) for b in bagian]
            except ValueError:
                continue                     # baris header
            if len(angka) < 2:
                continue
            t.append(angka[0])
            kolom.append(angka[1:])
    if not t:
        raise RuntimeError('CSV tidak berisi data numerik yang valid.')
    t = np.array(t)
    A = np.array(kolom)
    hasil = {'t': t, 'x': A[:, 0]}
    hasil['y'] = A[:, 1] if A.shape[1] > 1 else None
    return hasil
