"""
LAB GETARAN & GERAK — PANTULAN BOLA — Analisis Numerik (metode RK4)
=========================================================================================
Versi ini HANYA berisi sistem Pantulan (Bola Jatuh Bebas). Sistem Pegas dan
Osilasi/Bandul telah dihapus agar aplikasi lebih ringkas dan fokus. Semua
parameter fisik (massa, koefisien restitusi, redaman, gravitasi, kondisi
awal, jenis hambatan udara) tetap bisa diatur bebas sesuai kebutuhan.

Satu jendela, lima tab:
  1. Parameter & Prediksi Ideal
  2. Eksperimen (Kamera / Impor Gambar / Video / Arduino / CSV)
  3. Analisis Data (analisis puncak & koefisien restitusi e)
  4. Validasi Model RK4 (model terkoreksi + validasi, grafik, residual, data keseluruhan)
  5. Bantuan

Jalankan:  python app_lab_getaran.py
Kebutuhan: numpy, scipy, matplotlib, opencv-python, pillow
           (pyserial opsional — dibutuhkan untuk Arduino & kendali gearbox)
           (openpyxl opsional — dibutuhkan untuk "Simpan Semua (Grafik + Excel)")
"""

import os
import re
import csv
import time
import datetime
import traceback
import tempfile
import shutil

import numpy as np

import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import matplotlib
matplotlib.use('TkAgg')
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

from . import core_fisika as cf
from . import akuisisi as ak

try:
    from PIL import Image, ImageTk
    PIL_ADA = True
except Exception:
    PIL_ADA = False


JUDUL = 'Lab Getaran & Gerak — Pantulan Bola (Metode RK4)'
KUNCI_SISTEM = 'pantulan'
NAMA_SISTEM = 'Pantulan (Bola Jatuh Bebas)'
LABEL_SINYAL = 'Ketinggian y (m)'


def _aman_excel(nilai):
    """Ubah nilai NaN/Inf menjadi sel kosong (None) sebelum ditulis ke Excel.
    openpyxl MENOLAK menyimpan workbook yang berisi float non-terhingga
    (NaN/±Inf) — inilah penyebab paling umum "Simpan Semua (Grafik + Excel)"
    gagal total tanpa pesan yang jelas, terutama saat data eksperimen
    bermasalah (mis. korelasi/­metrik jadi NaN karena sinyal pantulan yang
    kacau). Fungsi ini dipakai di setiap baris yang ditulis ke workbook."""
    if isinstance(nilai, (float, np.floating)):
        try:
            if not np.isfinite(nilai):
                return None
        except Exception:
            return None
    return nilai


def _baris_aman(daftar):
    """Bungkus satu baris nilai (list) dengan _aman_excel untuk setiap sel."""
    return [_aman_excel(v) for v in daftar]


def _jarak_titik_segmen(p, a, b):
    """Jarak titik p ke ruas garis a-b (semuanya dalam koordinat yang sama)."""
    px, py = p
    ax, ay = a
    bx, by = b
    dx, dy = bx - ax, by - ay
    d2 = dx * dx + dy * dy
    if d2 < 1e-12:
        return float(np.hypot(px - ax, py - ay))
    u = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / d2))
    return float(np.hypot(px - (ax + u * dx), py - (ay + u * dy)))


def _sudut_x_dari_arah(dx, dy):
    """Sudut sumbu x (derajat) dari vektor layar (dx ke kanan, dy ke BAWAH).
    Sumbu x selalu diarahkan ke kanan (-90..90 derajat) agar sumbu y tetap menunjuk ke atas."""
    th = float(np.degrees(np.arctan2(-dy, dx)))
    return ((th + 90.0) % 180.0) - 90.0


# =====================================================================
# -------------------------------------------------------------------------
# Alat bantu PUNCAK & LEMBAH manual (Tab 3) dan peta waktu RK4 (Tab 4)
# -------------------------------------------------------------------------
def _snap_ekstrem(t, s, x, jenis, hw, lo_min=0):
    """Klik di waktu x -> indeks puncak ('P') atau lembah ('V') lokal terdekat.
    Dicari di jendela +-hw sampel (tidak boleh lebih kiri dari lo_min), lalu 'mendaki'
    sampai benar-benar ekstrem lokal. Mengembalikan None bila klik di kiri lo_min."""
    t = np.asarray(t, float); s = np.asarray(s, float)
    n = len(t)
    if n == 0:
        return None
    i0 = int(np.argmin(np.abs(t - x)))
    if i0 < lo_min:
        return None
    lo, hi = max(i0 - hw, lo_min), min(i0 + hw + 1, n)
    seg = s[lo:hi]
    j = lo + int(np.argmax(seg) if jenis == 'P' else np.argmin(seg))
    for _ in range(n):                                   # mendaki ke ekstrem lokal
        cand = j
        for k in (j - 1, j + 1):
            if lo_min <= k < n:
                if (s[k] > s[cand]) if jenis == 'P' else (s[k] < s[cand]):
                    cand = k
        if cand == j:
            break
        j = cand
    return j


def _nomori_pl(entri, s):
    """entri = [(jenis, idx, t)] hasil klik berurutan -> [(jenis, no, t, idx)].
    Penomoran sama dengan Tab 1/4: P0 = titik lepas, V1 = tumbukan pertama, P1 = puncak
    pantulan ke-1, V2, P2, ...  Klik pertama berupa puncak TERTINGGI di awal data dianggap
    titik lepas (P0)."""
    out = []
    if not entri:
        return out
    s = np.asarray(s, float)
    j0, i0, _ = entri[0]
    if j0 == 'P':
        smax, smin = float(np.max(s)), float(np.min(s))
        lepas = (i0 <= 0.25 * len(s)) and (s[i0] >= smax - 0.03 * max(smax - smin, 1e-12))
        pn, vn = (0, 1) if lepas else (1, 2)
    else:
        pn, vn = 1, 1
    for jenis, idx, tt in entri:
        if jenis == 'P':
            out.append(('P', pn, float(tt), int(idx))); pn += 1
        else:
            out.append(('V', vn, float(tt), int(idx))); vn += 1
    return out


def _tonggak_model(T, Y, pk):
    """Waktu puncak & lembah (kontak lantai) model RK4 -> {('P', n): t, ('V', n): t}.
    pk = cf.puncak_simulasi(T, Y) = [(t, y, jenis)], jenis 'lepas' = P0.
    Lembah ke-n = posisi terendah di antara puncak ke-(n-1) dan ke-n."""
    T = np.asarray(T, float); Y = np.asarray(Y, float)
    nomor, c = [], 0
    for tt, yy, jn in pk:
        if jn == 'lepas':
            nomor.append(0)
        else:
            c += 1; nomor.append(c)
    out = {}
    for i, (tt, yy, jn) in enumerate(pk):
        if nomor[i] >= 1:
            out[('P', nomor[i])] = float(tt)
        if i >= 1 and nomor[i] == nomor[i - 1] + 1:
            m = (T >= pk[i - 1][0]) & (T <= tt)
            if int(m.sum()) >= 3:
                Ts, Ys = T[m], Y[m, 0]
                out[('V', nomor[i])] = float(Ts[int(np.argmin(Ys))])
    return out


def _peta_linear(a, b):
    """Peta waktu linear  t_data = a*T + b  -> (fwd, inv, dfwd)."""
    return (lambda x: a * np.asarray(x, float) + b,
            lambda y: (np.asarray(y, float) - b) / a,
            lambda x: np.full(np.shape(x), float(a)))


def _peta_tonggak(pas, lereng=(0.5, 2.0)):
    """pas = [(jenis, no, t_data, T_model)]. Peta waktu sepotong-sepotong yang melewati
    SEMUA pasangan puncak/lembah. Di luar rentang tonggak peta mengikuti kemiringan global
    (kuadrat terkecil) -> 'sesudah itu model mengikuti saja'. Tonggak sesudah potongan yang
    kemiringannya tidak wajar dibuang. Mengembalikan dict atau (None, pesan)."""
    pas = sorted(pas, key=lambda q: q[2])
    keep = []
    for q in pas:
        if not keep or (q[2] > keep[-1][2] + 1e-6 and q[3] > keep[-1][3] + 1e-6):
            keep.append(q)
    n_buang = 0
    for i in range(1, len(keep)):
        sl = (keep[i][2] - keep[i - 1][2]) / (keep[i][3] - keep[i - 1][3])
        if not (lereng[0] <= sl <= lereng[1]):
            n_buang = len(keep) - i
            keep = keep[:i]
            break
    if len(keep) < 2:
        return None, 'kurang dari 2 pasangan puncak/lembah yang cocok'
    Td = np.array([q[2] for q in keep]); Tm = np.array([q[3] for q in keep])
    ag = float(np.polyfit(Tm, Td, 1)[0])
    if not (np.isfinite(ag) and ag > 0):
        return None, 'kemiringan peta waktu tidak valid'
    sl = np.diff(Td) / np.diff(Tm)

    def fwd(x):
        x = np.asarray(x, float)
        y = np.interp(x, Tm, Td)
        y = np.where(x < Tm[0], Td[0] + ag * (x - Tm[0]), y)
        return np.where(x > Tm[-1], Td[-1] + ag * (x - Tm[-1]), y)

    def inv(y):
        y = np.asarray(y, float)
        x = np.interp(y, Td, Tm)
        x = np.where(y < Td[0], Tm[0] + (y - Td[0]) / ag, x)
        return np.where(y > Td[-1], Tm[-1] + (y - Td[-1]) / ag, x)

    def dfwd(x):
        x = np.asarray(x, float)
        i = np.clip(np.searchsorted(Tm, x, side='right') - 1, 0, len(sl) - 1)
        d = sl[i]
        d = np.where((x < Tm[0]) | (x > Tm[-1]), ag, d)
        return d

    return {'fwd': fwd, 'inv': inv, 'dfwd': dfwd, 'a': ag, 'n': len(keep),
            'nP': sum(1 for q in keep if q[0] == 'P'),
            'nV': sum(1 for q in keep if q[0] == 'V'),
            'n_buang': n_buang, 't_ref': float(Td[0]), 'pasangan': keep}, ''


class Aplikasi(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title(JUDUL)
        self.geometry('1330x850')
        self.minsize(1100, 700)

        # ---------- penyimpanan bersama ----------
        self.data = {'t': None, 's': None, 'x': None, 'y': None, 'sumber': '-'}
        self.hasil = {}          # hasil analisis
        self.prediksi = None     # (T, Y, model) prediksi ideal
        self.validasi = None     # hasil validasi terakhir

        # ---------- objek akuisisi ----------
        self.sumber = None
        self.serial = None
        self.servo_ser = None            # koneksi serial khusus gearbox (bila terpisah dari Arduino sensor)
        self._servo_berbagi = False      # True bila gearbox memakai koneksi self.serial yang sama
        self._vid_writer = None          # cv2.VideoWriter aktif saat merekam video dari kamera
        self._rekam_video_aktif = False
        self._rekam_video_path = None
        self._rekam_video_n = 0
        self.pel_auto = ak.PelacakObjek()
        self.pel_csrt = ak.PelacakCSRT()
        self.pelacak = self.pel_auto
        self.cara = 'otomatis'          # 'otomatis' | 'semi' | 'manual'
        self.titik_manual = {}          # {indeks_frame: (t, cx_px, cy_px)}
        self.kal_frame = {}             # {indeks_frame: snapshot penggaris + sumbu x-y} (mode manual)
        self.manual_i = 0
        self._dok_sumber = None          # (tipe, argumen) sumber video/gambar untuk foto dokumentasi
        self.t_frame = 0.0
        self._geser_slider = False
        self._kursor = None
        self._geser_gambar = (0.0, 0.0)
        self.kal = ak.Kalibrasi(px_per_m=1000.0)
        self.frame_akhir = None
        self.merekam = False
        self.pratinjau_aktif = False
        self._buf = self._buf_kosong()
        self._pangkas_ref = None         # titik lepas (waktu mutlak) agar grafik langsung tidak bergeser
        self.puncak_manual = None        # None = puncak otomatis; daftar waktu = pilihan pengguna (Tab 3)
        self._t_idx3 = None              # waktu sampel puncak yang sedang dipakai analisis
        self._r3 = None                  # hasil analisis_pantulan terakhir (dipakai Tab 3 & 4)
        self.puncak_tambah = []          # waktu puncak TAMBAHAN hasil klik (Tab 3/4); puncak lain tidak berubah
        self.pl_manual = []              # klik berurutan puncak/lembah (Tab 3): [(jenis 'P'|'V', indeks, waktu)]
        self._pl_n = 0                   # jumlah sampel data saat klik (untuk membuang pilihan basi)
        self._id_validasi = None         # jadwal render ulang Tab 4 saat slider digeser
        self._k_validasi = None          # faktor renggang-rapat yang terakhir dipakai Tab 4
        self.ax4_utama = None            # sumbu grafik atas Tab 4 (untuk klik puncak)
        self._sig_analisis = None        # tanda data pada analisis terakhir (untuk analisis otomatis)
        self._skala_pratinjau = 1.0
        self._roi_awal = None
        self._seret = None               # seretan penggaris / sumbu yang sedang berlangsung
        self._t_hitung = 0.0             # pembatas laju hitung-ulang saat sumbu diseret (manual)
        self._foto = None
        self._pesan_sinyal = ('', '#0a5')

        self.var = {}            # StringVar parameter
        self._bangun()
        self.protocol('WM_DELETE_WINDOW', self._tutup)

    # =================================================================
    # KERANGKA
    # =================================================================
    def _bangun(self):
        atas = ttk.Frame(self, padding=(8, 6))
        atas.pack(fill='x')
        ttk.Label(atas, text=f'SISTEM: {NAMA_SISTEM}', font=('Segoe UI', 10, 'bold')).pack(side='left')
        self.lbl_status = ttk.Label(atas, text='Siap. Belum ada data eksperimen.',
                                    foreground='#0a5')
        self.lbl_status.pack(side='left', padx=20)
        ttk.Button(atas, text='Reset Semua', command=self._reset).pack(side='right')
        ttk.Button(atas, text='💾 Simpan Semua (Grafik + Excel)',
                   command=self._simpan_semua).pack(side='right', padx=6)

        self.nb = ttk.Notebook(self)
        self.nb.pack(fill='both', expand=True, padx=6, pady=(0, 6))
        self.tab1 = ttk.Frame(self.nb); self.nb.add(self.tab1, text=' 1. Parameter & Prediksi Ideal ')
        self.tab2 = ttk.Frame(self.nb); self.nb.add(self.tab2, text=' 2. Eksperimen / Akuisisi ')
        self.tab3 = ttk.Frame(self.nb); self.nb.add(self.tab3, text=' 3. Analisis Data ')
        self.tab4 = ttk.Frame(self.nb); self.nb.add(self.tab4, text=' 4. Validasi Model RK4 ')
        self.tab5 = ttk.Frame(self.nb); self.nb.add(self.tab5, text=' Bantuan ')

        self._tab_parameter()
        self._tab_eksperimen()
        self._tab_analisis()
        self._tab_validasi()
        self._tab_bantuan()
        self._ringkas_koreksi()
        self.nb.bind('<<NotebookTabChanged>>', self._tab_berganti)

    # ---------- utilitas ----------
    def _status(self, teks, warna='#0a5'):
        self.lbl_status.config(text=teks, foreground=warna)
        self.update_idletasks()

    def _f(self, kunci, bawaan=0.0):
        try:
            return float(str(self.var[kunci].get()).replace(',', '.'))
        except Exception:
            return float(bawaan)

    def _baris(self, induk, baris, teks, kunci, bawaan, satuan=''):
        lbl = ttk.Label(induk, text=teks)
        lbl.grid(row=baris, column=0, sticky='w', pady=2)
        self.var[kunci] = tk.StringVar(value=str(bawaan))
        ent = ttk.Entry(induk, textvariable=self.var[kunci], width=12)
        ent.grid(row=baris, column=1, sticky='w', padx=4)
        lbl2 = ttk.Label(induk, text=satuan, foreground='#666')
        lbl2.grid(row=baris, column=2, sticky='w')
        return (lbl, ent, lbl2)

    def _daftar_com(self):
        """Kembalikan daftar port COM/serial yang terdeteksi di sistem saat ini
        (butuh pyserial). Kembalikan list kosong bila pyserial tak tersedia atau
        tak ada port terdeteksi."""
        try:
            from serial.tools import list_ports
            return [p.device for p in list_ports.comports()]
        except Exception:
            return []

    def _kotak_port(self, induk, baris, teks, kunci, bawaan='', satuan=''):
        """Baris label + Combobox pilihan port COM (klik ▾ untuk memilih dari port
        yang terdeteksi) beserta tombol 🔄 untuk memindai ulang. Tetap bisa diketik
        manual bila port yang diinginkan belum tersambung/terdeteksi."""
        ttk.Label(induk, text=teks).grid(row=baris, column=0, sticky='w', pady=2)
        self.var[kunci] = tk.StringVar(value=bawaan)
        cmb = ttk.Combobox(induk, textvariable=self.var[kunci], width=10,
                           values=self._daftar_com())
        cmb.grid(row=baris, column=1, sticky='w', padx=4)
        cmb.bind('<Button-1>', lambda e, c=cmb: c.config(values=self._daftar_com()))
        f = ttk.Frame(induk); f.grid(row=baris, column=2, sticky='w')
        ttk.Button(f, text='🔄', width=3,
                  command=lambda c=cmb: c.config(values=self._daftar_com())).pack(side='left')
        if satuan:
            ttk.Label(f, text=satuan, foreground='#666').pack(side='left', padx=2)
        return cmb

    def _kanvas(self, induk, ukuran=(7.6, 6.2)):
        fig = Figure(figsize=ukuran, dpi=96)
        kanvas = FigureCanvasTkAgg(fig, master=induk)
        kanvas.get_tk_widget().pack(fill='both', expand=True)
        bar = NavigationToolbar2Tk(kanvas, induk)
        bar.update()
        return fig, kanvas

    def _panel_gulir(self, induk, lebar=340):
        """
        Panel sisi kiri yang bisa DIGULIR VERTIKAL — dipakai untuk semua tab
        agar tombol/kontrol di bagian bawah tetap terjangkau walau jendela
        diperkecil. Kembalikan frame isi tempat widget-widget diletakkan
        seperti biasa (padding sudah disertakan).
        """
        wadah = ttk.Frame(induk)
        wadah.pack(side='left', fill='y')
        kanvas = tk.Canvas(wadah, highlightthickness=0, bd=0, width=lebar)
        sb = ttk.Scrollbar(wadah, orient='vertical', command=kanvas.yview)
        kanvas.configure(yscrollcommand=sb.set)
        kanvas.pack(side='left', fill='y')
        sb.pack(side='left', fill='y')
        isi = ttk.Frame(kanvas, padding=8)
        id_jendela = kanvas.create_window((0, 0), window=isi, anchor='nw')

        def _atur_scrollregion(_=None):
            kanvas.configure(scrollregion=kanvas.bbox('all'))
        isi.bind('<Configure>', _atur_scrollregion)

        def _atur_lebar(ev):
            kanvas.itemconfig(id_jendela, width=max(ev.width, lebar))
        kanvas.bind('<Configure>', _atur_lebar)

        def _roda(ev):
            if not kanvas.bbox('all') or kanvas.bbox('all')[3] <= kanvas.winfo_height():
                return                       # konten muat semua, tak perlu digulir
            langkah = -1 * (ev.delta // 120) if getattr(ev, 'delta', 0) else \
                      (1 if getattr(ev, 'num', 0) == 5 else -1)
            kanvas.yview_scroll(int(langkah), 'units')

        def _pasang_roda(_):
            kanvas.bind_all('<MouseWheel>', _roda)      # Windows / macOS
            kanvas.bind_all('<Button-4>', _roda)         # Linux — gulir atas
            kanvas.bind_all('<Button-5>', _roda)         # Linux — gulir bawah

        def _lepas_roda(_):
            kanvas.unbind_all('<MouseWheel>')
            kanvas.unbind_all('<Button-4>')
            kanvas.unbind_all('<Button-5>')

        kanvas.bind('<Enter>', _pasang_roda)
        kanvas.bind('<Leave>', _lepas_roda)
        return isi

    # =================================================================
    # TAB 1 — PARAMETER & PREDIKSI IDEAL
    # =================================================================
    def _tab_parameter(self):
        kiri = self._panel_gulir(self.tab1, lebar=330)
        kanan = ttk.Frame(self.tab1)
        kanan.pack(side='left', fill='both', expand=True)

        b = ttk.LabelFrame(kiri, text='Parameter Sistem (Pantulan Bola)', padding=8)
        b.pack(fill='x')
        self._baris(b, 0, 'Massa bola m', 'm', 0.060, 'kg')
        self._baris(b, 1, 'Koef. restitusi e', 'e', 0.80, '(0-1)')
        self._baris(b, 2, 'Ketinggian awal y0', 'x0', 1.000, 'm')
        self._baris(b, 3, 'Kecepatan awal v0', 'v0', 0.0, 'm/s')
        self._baris(b, 4, 'Konstanta redaman udara c', 'c', 0.0, 'N·s/m (kg/s)')
        self._baris(b, 5, 'Gravitasi g', 'g', 9.81, 'm/s²')
        ttk.Label(b, text='Hambatan udara').grid(row=6, column=0, sticky='w', pady=2)
        self.var['hambatan'] = tk.StringVar(value='linier')
        ttk.Combobox(b, textvariable=self.var['hambatan'], state='readonly', width=12,
                     values=['linier', 'kuadratik']).grid(row=6, column=1, columnspan=2,
                                                          sticky='w', padx=4)

        c = ttk.LabelFrame(kiri, text='Pengaturan Numerik', padding=8)
        c.pack(fill='x', pady=6)
        self._baris(c, 0, 'Waktu akhir', 't_akhir', 6.0, 's')
        self._baris(c, 1, 'Langkah h', 'h', 0.005, 's')
        ttk.Label(c, text='Metode: RK4 (Runge-Kutta orde 4)', foreground='#1a4f8a'
                  ).grid(row=2, column=0, columnspan=3, sticky='w', pady=2)

        tombol = ttk.Frame(kiri); tombol.pack(fill='x', pady=6)
        ttk.Button(tombol, text='▶  Hitung & Plot Prediksi Ideal',
                   command=self._jalankan_prediksi).pack(fill='x')
        ttk.Button(tombol, text='Simpan prediksi ke CSV',
                   command=self._simpan_prediksi).pack(fill='x', pady=3)

        self.lbl_info1 = ttk.Label(kanan, text='', foreground='#1a4f8a',
                                   font=('Consolas', 9), padding=4)
        self.lbl_info1.pack(fill='x')
        self.fig1, self.kanvas1 = self._kanvas(kanan)

    def _parameter(self, pakai_koreksi=False):
        """Kumpulkan parameter model. pakai_koreksi=True -> ambil hasil analisis bila ada."""
        p = {'m': self._f('m', 0.06), 'e': self._f('e', 0.8),
             'x0': self._f('x0', 1.0), 'v0': self._f('v0', 0.0),
             'c': self._f('c', 0.0), 'g': self._f('g', 9.81),
             'hambatan': self.var['hambatan'].get()}
        if pakai_koreksi:
            h = self.hasil
            # y0 (ketinggian awal) SELALU dari Tab 1 dan berlaku di semua tab; hasil
            # analisis/optimasi tidak boleh menimpanya. Hanya e, c, v0 yang dikoreksi.
            for kunci_h, kunci_p in (('e_terkoreksi', 'e'), ('c_terkoreksi', 'c'),
                                     ('v0_data', 'v0')):
                if kunci_h in h and np.isfinite(h[kunci_h]):
                    p[kunci_p] = float(h[kunci_h])
            # opsi Tab 4: y0 model = tinggi awal data (Tab 2), supaya model dan data mulai
            # dari ketinggian yang sama (selisih y0 membuat seluruh pantulan bergeser fase)
            y0d = self._y0_data()
            if y0d is not None and self._samakan_y0():
                p['x0'] = y0d
        return p

    def _y0_data(self):
        """Tinggi awal data eksperimen (median 3 sampel pertama) = angka 'tinggi awal' di Tab 2."""
        s = self.data.get('s')
        if s is None or len(s) < 3:
            return None
        return float(np.median(np.asarray(s, dtype=float)[:3]))

    def _samakan_y0(self):
        v = self.var.get('samakan_y0')
        return bool(v.get()) if v is not None else False

    def _jalankan_prediksi(self):
        """Prediksi ideal MURNI dengan RK4 (parameter Tab 1) — TIDAK dipengaruhi data
        eksperimen, jadi isian "Waktu akhir" selalu dipakai apa adanya. Grafik posisi dan
        kecepatan otomatis memuat titik PUNCAK (v = 0) dan titik TUMBUKAN dengan lantai.
        Perbandingan dengan data eksperimen ada di Tab 4."""
        try:
            p = self._parameter(pakai_koreksi=False)
            t_akhir = max(self._f('t_akhir', 6.0), 1e-3)
            model = cf.buat_model(KUNCI_SISTEM, p)
            h = max(self._f('h', 0.005), 1e-6)
            T, Y = cf.simulasi(model, t_akhir, h)
            self.prediksi = (T, Y, model)
            puncak = cf.puncak_simulasi(T, Y)
            tumb = cf.tumbukan_simulasi(T, Y, model.e)

            # label puncak: 'L' = titik lepas, P1, P2, ... = puncak sesudah pantulan ke-n
            label, n_p = [], 0
            for _, _, jenis in puncak:
                if jenis == 'lepas':
                    label.append('L')
                else:
                    n_p += 1
                    label.append(f'P{n_p}')

            self.fig1.clear()
            ax1 = self.fig1.add_subplot(2, 1, 1)
            ax2 = self.fig1.add_subplot(2, 1, 2, sharex=ax1)

            ax1.plot(T, Y[:, 0], '-', lw=1.4, color='#1f77b4', label='posisi (RK4)')
            ana = model.solusi_analitik(T)
            if ana is not None:
                ax1.plot(T, ana, '--', lw=1.0, color='crimson', label='analitik (ideal)')
            ax1.axhline(0, color='k', lw=0.7, ls='--', alpha=0.4, label='lantai y=0')
            tp = np.array([q[0] for q in puncak])
            yp = np.array([q[1] for q in puncak])
            if len(puncak):
                ax1.plot(tp, yp, 'rv', ms=7, label=f'puncak ({len(puncak)})')
                for k, (tt, yy, lb) in enumerate(zip(tp, yp, label)):
                    if k >= 12:
                        break
                    teks = f'{lb}\n{yy:.3f} m' if k < 6 else lb
                    ax1.annotate(teks, (tt, yy), textcoords='offset points', xytext=(0, 7),
                                 ha='center', fontsize=7, color='#8b0000')
                for tt in tp[:30]:                       # garis bantu: puncak sama di kedua grafik
                    ax1.axvline(tt, color='r', ls=':', lw=0.6, alpha=0.45)
                    ax2.axvline(tt, color='r', ls=':', lw=0.6, alpha=0.45)
            ax1.set_title('POSISI TERHADAP WAKTU — ▼ = puncak tiap pantulan')
            ax1.set_ylabel(LABEL_SINYAL); ax1.grid(True); ax1.legend(fontsize=8)

            v = Y[:, 1]
            ax2.axhline(0, color='k', lw=0.8)
            ax2.fill_between(T, 0, v, where=v > 0, color='#2e8b57', alpha=0.18,
                             interpolate=True, label='bergerak naik (v > 0)')
            ax2.fill_between(T, 0, v, where=v < 0, color='#d2691e', alpha=0.18,
                             interpolate=True, label='bergerak turun (v < 0)')
            ax2.plot(T, v, '-', color='#d2691e', lw=1.2)
            if len(puncak):
                ax2.plot(tp, np.zeros_like(tp), 'rv', ms=7, label='puncak (v = 0)')
            if tumb:
                tt_ = np.array([q[0] for q in tumb])
                vb_ = np.array([q[1] for q in tumb])
                va_ = np.array([q[2] for q in tumb])
                ax2.vlines(tt_, vb_, va_, color='gray', ls=':', lw=0.8)
                ax2.plot(tt_, vb_, 'bv', ms=5, label='tepat sebelum tumbukan')
                ax2.plot(tt_, va_, '^', color='green', ms=5, label='tepat sesudah tumbukan')
                for k in range(min(len(tumb), 5)):
                    ax2.annotate(f'{vb_[k]:.2f}', (tt_[k], vb_[k]), textcoords='offset points',
                                 xytext=(-14, -3), ha='right', fontsize=7, color='blue')
                    ax2.annotate(f'{va_[k]:.2f}', (tt_[k], va_[k]), textcoords='offset points',
                                 xytext=(14, 2), ha='left', fontsize=7, color='green')
            ax2.set_title('KECEPATAN TERHADAP WAKTU — v = 0 di setiap puncak')
            ax2.set_ylim(top=max(float(np.max(v)), 0.1) * 1.5)
            ax2.set_xlabel('Waktu (s)'); ax2.set_ylabel('Kecepatan (m/s)\n(+ naik, − turun)')
            ax2.grid(True); ax2.legend(fontsize=7, ncol=3, loc='upper right')

            self.fig1.tight_layout()
            self.kanvas1.draw()

            teks = model.info()
            if ana is not None:
                mk_a = cf.metrik_validasi(ana, Y[:, 0])
                teks += f"\nGalat numerik vs analitik: RMSE={mk_a['RMSE']:.3e}  maks={mk_a['maks']:.3e}"
            if puncak:
                ring = ', '.join(f'{lb}: {yy:.3f} m @ {tt:.2f} s'
                                 for lb, yy, tt in list(zip(label, yp, tp))[:5])
                teks += f"\nPuncak ({len(puncak)}): {ring}" + (' …' if len(puncak) > 5 else '')
            ada_eks = self.data.get('t') is not None and len(self.data['t']) >= 5
            if ada_eks:
                teks += '\n\nData eksperimen tersedia — bandingkan dengan model RK4 di Tab 4.'
            self.lbl_info1.config(text=teks)
            self._status(f'Prediksi RK4 selesai — {len(T)} titik, {len(puncak)} puncak, '
                         f'waktu akhir = {t_akhir:g} s.')
        except Exception as exc:
            self._galat(exc)

    def _simpan_prediksi(self):
        if not self.prediksi:
            messagebox.showwarning('Belum ada', 'Hitung prediksi ideal terlebih dahulu.')
            return
        nama = filedialog.asksaveasfilename(defaultextension='.csv',
                                            filetypes=[('CSV', '*.csv')])
        if not nama:
            return
        T, Y, _ = self.prediksi
        with open(nama, 'w', newline='', encoding='utf-8') as f:
            w = csv.writer(f)
            w.writerow(['t_s', 'posisi', 'kecepatan'])
            for a, b, c in zip(T, Y[:, 0], Y[:, 1]):
                w.writerow([f'{a:.6f}', f'{b:.6f}', f'{c:.6f}'])
        self._status(f'Prediksi disimpan: {os.path.basename(nama)}')

    # =================================================================
    # TAB 2 — EKSPERIMEN / AKUISISI
    # =================================================================
    def _tab_eksperimen(self):
        kiri = self._panel_gulir(self.tab2, lebar=340)
        kanan = ttk.Frame(self.tab2); kanan.pack(side='left', fill='both', expand=True)

        s = ttk.LabelFrame(kiri, text='Sumber Data', padding=8); s.pack(fill='x')
        self.var['sumber'] = tk.StringVar(value='Kamera')
        ttk.Combobox(s, textvariable=self.var['sumber'], state='readonly', width=24,
                     values=['Kamera', 'Berkas video', 'Impor gambar (banyak)',
                             'Berkas CSV', 'Arduino (serial)']
                     ).grid(row=0, column=0, columnspan=3, sticky='w')
        self._baris(s, 1, 'Indeks kamera', 'kam_idx', 0, '(0,1,2…)')
        self._baris(s, 2, 'FPS / laju cuplik', 'fps', 60.0, 'Hz')
        self.var['berkas'] = tk.StringVar(value='')
        ttk.Button(s, text='Pilih berkas…', command=self._pilih_berkas
                   ).grid(row=3, column=0, sticky='w', pady=3)
        ttk.Label(s, textvariable=self.var['berkas'], width=26, foreground='#555'
                  ).grid(row=3, column=1, columnspan=2, sticky='w')
        self._kotak_port(s, 4, 'Port serial', 'port', 'COM3', '')
        self._baris(s, 5, 'Baud rate', 'baud', 9600, '')

        d = ttk.LabelFrame(kiri, text='Pelacakan Objek', padding=8); d.pack(fill='x', pady=6)
        ttk.Label(d, text='Cara pelacakan', font=('Segoe UI', 9, 'bold')
                  ).grid(row=0, column=0, columnspan=3, sticky='w')
        self.var['cara'] = tk.StringVar(value='Otomatis (deteksi objek)')
        cmb_cara = ttk.Combobox(d, textvariable=self.var['cara'], state='readonly', width=28,
                                values=['Otomatis (deteksi objek)',
                                        'Semi-otomatis (kunci objek/ROI)',
                                        'Manual (klik tiap frame)'])
        cmb_cara.grid(row=1, column=0, columnspan=3, sticky='w', pady=(0, 6))
        cmb_cara.bind('<<ComboboxSelected>>', lambda e: self._ganti_cara())

        self.var['mode_lacak'] = tk.StringVar(value='warna')
        ttk.Label(d, text='Mode').grid(row=2, column=0, sticky='w')
        self.cmb_mode = ttk.Combobox(d, textvariable=self.var['mode_lacak'], state='readonly',
                                     width=14, values=['warna', 'terang', 'gelap', 'gerak'])
        self.cmb_mode.grid(row=2, column=1, sticky='w', padx=4)
        self.cmb_mode.bind('<<ComboboxSelected>>',
                           lambda e: setattr(self.pel_auto, 'mode', self.var['mode_lacak'].get()))
        self._baris(d, 3, 'Toleransi warna (H)', 'tol_h', 12, '')
        self._baris(d, 4, 'Luas minimum', 'luas_min', 30, 'px²')

        ttk.Label(d, text='Klik pada pratinjau untuk:').grid(row=5, column=0, columnspan=3,
                                                             sticky='w', pady=(6, 0))
        self.var['mode_klik'] = tk.StringVar(value='warna')
        for i, (nilai, teks) in enumerate([('tandai', 'Tandai posisi objek (manual)'),
                                           ('warna', 'Ambil warna objek'),
                                           ('roi', 'Pilih ROI / kotak objek (seret)'),
                                           ('penggaris', 'Penggaris: SERET dari titik 0 ke ujung'),
                                           ('sumbu', 'Sumbu x–y (x = lantai): seret asal / ujung x')]):
            ttk.Radiobutton(d, text=teks, value=nilai, variable=self.var['mode_klik'],
                            command=self._mode_klik_berubah
                            ).grid(row=6 + i, column=0, columnspan=3, sticky='w')

        kal = ttk.LabelFrame(kiri, text='Kalibrasi Penggaris, Sumbu x–y & Sinyal', padding=8)
        kal.pack(fill='x')
        ttk.Label(kal, text='1) Pilih klik "Penggaris", lalu SERET pada pratinjau dari titik 0 ke ujung '
                            'penggaris dan isi panjang nyatanya.\n'
                            '2) Pilih klik "Sumbu x–y": seret ⊕ (asal) untuk menggeser, seret ● (ujung x) '
                            'untuk memutar. Sumbu x = LANTAI (y = 0), sumbu y tegak lurus ke atas.\n'
                            'Otomatis / semi: atur SEBELUM merekam. Manual: boleh SESUDAH menandai titik.',
                  foreground='#555', wraplength=290, justify='left'
                  ).grid(row=0, column=0, columnspan=3, sticky='w', pady=(0, 4))
        ttk.Label(kal, text='Panjang acuan').grid(row=1, column=0, sticky='w', pady=2)
        self.var['panjang_acuan'] = tk.StringVar(value='30')
        cmb_p = ttk.Combobox(kal, textvariable=self.var['panjang_acuan'], width=10,
                             values=['5', '10', '20', '30', '50', '100'])
        cmb_p.grid(row=1, column=1, sticky='w', padx=4)
        ttk.Label(kal, text='cm (titik 0 → ujung)', foreground='#666').grid(row=1, column=2, sticky='w')
        for peristiwa in ('<<ComboboxSelected>>', '<Return>', '<FocusOut>', '<KeyRelease>'):
            cmb_p.bind(peristiwa, self._ubah_panjang_acuan)
        self._baris(kal, 2, 'Skala', 'px_per_m', 1000.0, 'px/m')
        self._baris(kal, 3, 'Jari-jari bola', 'r_bola_cm', 0.0, 'cm')
        _, ent_sudut, _ = self._baris(kal, 4, 'Sudut sumbu x', 'sudut_x', 0.0, '° (0 = lantai datar)')
        ent_sudut.bind('<Return>', lambda e: self._terapkan_kalibrasi_ke_data(True))
        ent_sudut.bind('<FocusOut>', lambda e: self._terapkan_kalibrasi_ke_data(True))
        for baris, (teks, perintah) in enumerate([
                ('⇄ Tukar titik 0 ↔ ujung penggaris', self._tukar_penggaris),
                ('⌖ Pasang sumbu mengikuti penggaris', self._sumbu_dari_penggaris),
                ('↔ Datarkan sumbu x (0°)', self._datarkan_sumbu),
                ('🗑 Hapus penggaris & sumbu', self._hapus_kalibrasi)], start=5):
            ttk.Button(kal, text=teks, command=perintah
                       ).grid(row=baris, column=0, columnspan=3, sticky='we', pady=1)
        ttk.Button(kal, text='↻ Terapkan skala / sudut / jari-jari ke data',
                   command=self._terapkan_kalibrasi_ke_data
                   ).grid(row=9, column=0, columnspan=3, sticky='we', pady=3)
        self.lbl_kal = ttk.Label(kal, text='', wraplength=290, justify='left')
        self.lbl_kal.grid(row=10, column=0, columnspan=3, sticky='w', pady=(0, 4))
        ttk.Label(kal, text='Sumbu sinyal').grid(row=11, column=0, sticky='w')
        self.var['sumbu'] = tk.StringVar(value='Y (vertikal)')
        ttk.Combobox(kal, textvariable=self.var['sumbu'], state='readonly', width=16,
                     values=['Y (vertikal)', 'X (horizontal)', 'Otomatis (varians terbesar)']
                     ).grid(row=11, column=1, columnspan=2, sticky='w', padx=4)
        ttk.Label(kal, text='Referensi nol').grid(row=12, column=0, sticky='w')
        self.var['referensi'] = tk.StringVar(value='kalibrasi (sumbu x)')
        ttk.Combobox(kal, textvariable=self.var['referensi'], state='readonly', width=20,
                     values=['kalibrasi (sumbu x)', 'mentah', 'rata-rata', 'minimum (lantai)']
                     ).grid(row=12, column=1, columnspan=2, sticky='w', padx=4)
        self.var['haluskan'] = tk.StringVar(value='0')
        ttk.Label(kal, text='Perataan (titik)').grid(row=13, column=0, sticky='w')
        ttk.Entry(kal, textvariable=self.var['haluskan'], width=8).grid(row=13, column=1, sticky='w', padx=4)
        self.var['pangkas_lepas'] = tk.BooleanVar(value=True)
        ttk.Checkbutton(kal, text='Pangkas data sebelum bola dilepas (t = 0 saat lepas)',
                        variable=self.var['pangkas_lepas']).grid(row=14, column=0, columnspan=3, sticky='w')
        self.var['buang_lonjakan'] = tk.BooleanVar(value=True)
        ttk.Checkbutton(kal, text='Buang lonjakan satu-frame (salah deteksi)',
                        variable=self.var['buang_lonjakan']).grid(row=15, column=0, columnspan=3, sticky='w')

        # ---------- kendali gearbox (motor DC + driver L298N lewat Arduino) ----------
        sv = ttk.LabelFrame(kiri, text='Kendali Gearbox (Pemicu Rekam)', padding=8)
        sv.pack(fill='x', pady=6)
        self.var['servo_aktif'] = tk.BooleanVar(value=True)
        ttk.Checkbutton(sv, text='Putar gearbox (buka) otomatis saat "Mulai Rekam"',
                        variable=self.var['servo_aktif']).grid(row=0, column=0, columnspan=3,
                                                                sticky='w')
        self._kotak_port(sv, 1, 'Port gearbox', 'servo_port', '', '')
        ttk.Label(sv, text='(kosongkan = pakai port Arduino)', foreground='#666'
                  ).grid(row=2, column=0, columnspan=3, sticky='w')
        self._baris(sv, 3, 'Baud gearbox', 'servo_baud', 9600, '')
        self._baris(sv, 4, 'Perintah BUKA (90°)', 'servo_buka', '1', '')
        self._baris(sv, 5, 'Perintah TUTUP (kembali)', 'servo_tutup', '0', '')
        bsv = ttk.Frame(sv); bsv.grid(row=6, column=0, columnspan=3, sticky='w', pady=(4, 0))
        ttk.Button(bsv, text='🔌 Sambung', width=11, command=self._sambung_servo).pack(side='left')
        ttk.Button(bsv, text='🔓 Uji Buka', width=11, command=self._servo_buka).pack(side='left', padx=2)
        ttk.Button(bsv, text='🔒 Uji Tutup', width=11, command=self._servo_tutup).pack(side='left')
        self.lbl_servo = ttk.Label(sv, text='Gearbox: belum tersambung.', foreground='#666',
                                   wraplength=290, justify='left')
        self.lbl_servo.grid(row=7, column=0, columnspan=3, sticky='w', pady=(4, 0))

        t = ttk.Frame(kiri); t.pack(fill='x', pady=8)
        ttk.Button(t, text='📷  Buka & Pratinjau', command=self._buka_sumber).pack(fill='x')
        ttk.Button(t, text='⏺  Mulai Rekam', command=self._mulai_rekam).pack(fill='x', pady=2)
        ttk.Button(t, text='⏹  Berhenti', command=self._berhenti).pack(fill='x')
        ttk.Separator(t).pack(fill='x', pady=4)
        ttk.Label(t, text='Kamera → rekam video dulu, lalu track manual per frame:',
                  foreground='#555', wraplength=280, justify='left').pack(fill='x')
        ttk.Button(t, text='🎥  Rekam Video dari Kamera',
                   command=self._mulai_rekam_video).pack(fill='x', pady=2)
        ttk.Button(t, text='⏹🎥  Selesai Rekam & Buka untuk Track',
                   command=self._berhenti_rekam_video).pack(fill='x')
        ttk.Separator(t).pack(fill='x', pady=4)
        ttk.Button(t, text='⚙  Proses Seluruh Berkas (otomatis)',
                   command=self._proses_batch).pack(fill='x', pady=2)
        ttk.Separator(t).pack(fill='x', pady=4)
        ttk.Button(t, text='💾  Simpan data ke CSV', command=self._simpan_data).pack(fill='x')
        ttk.Button(t, text='📂  Muat data dari CSV', command=self._muat_data).pack(fill='x', pady=2)
        ttk.Button(t, text='🧪  Buat data sintetis (uji coba)',
                   command=self._data_sintetis).pack(fill='x')
        ttk.Button(t, text='🗑  Hapus data', command=self._hapus_data).pack(fill='x', pady=2)

        atas = ttk.Frame(kanan); atas.pack(fill='both', expand=True)
        kiri_prev = ttk.LabelFrame(atas, text='Pratinjau Kamera / Gambar')
        kiri_prev.pack(side='left', fill='both', expand=True, padx=4, pady=4)
        self.kanvas_prev = tk.Canvas(kiri_prev, bg='#222', width=520, height=390,
                                     highlightthickness=0)
        self.kanvas_prev.pack(fill='both', expand=True)
        self.kanvas_prev.bind('<Button-1>', self._klik_prev)
        self.kanvas_prev.bind('<B1-Motion>', self._seret_prev)
        self.kanvas_prev.bind('<ButtonRelease-1>', self._lepas_prev)
        self.kanvas_prev.bind('<Motion>', self._gerak_kursor)
        self.kanvas_prev.bind('<Leave>', lambda e: setattr(self, '_kursor', None))
        self.bar_manual = ttk.LabelFrame(kiri_prev, text='Pelacakan Manual / Semi-otomatis',
                                         padding=4)
        b1 = ttk.Frame(self.bar_manual); b1.pack(fill='x')
        for teks, langkah in (('⏮', -1000000), ('◀◀ −10', -10), ('◀ −1', -1),
                              ('+1 ▶', 1), ('+10 ▶▶', 10)):
            ttk.Button(b1, text=teks, width=7,
                       command=lambda n=langkah: self._geser_frame(n)).pack(side='left', padx=1)
        ttk.Label(b1, text='Lompat tiap').pack(side='left', padx=(10, 2))
        self.var['lompat'] = tk.StringVar(value='1')
        ttk.Entry(b1, textvariable=self.var['lompat'], width=4).pack(side='left')
        ttk.Label(b1, text='frame').pack(side='left', padx=2)

        self.var_slider = tk.DoubleVar(value=0)
        self.slider = ttk.Scale(self.bar_manual, from_=0, to=1, variable=self.var_slider,
                                command=self._slider_frame)
        self.slider.pack(fill='x', pady=3)

        b2 = ttk.Frame(self.bar_manual); b2.pack(fill='x')
        self.btn_putar = ttk.Button(b2, text='▶ Putar', width=9, command=self._putar_jeda)
        self.btn_putar.pack(side='left', padx=2)
        ttk.Button(b2, text='🔒 Kunci Objek (semi-otomatis)',
                   command=self._kunci_objek).pack(side='left', padx=2)
        ttk.Button(b2, text='↶ Hapus titik terakhir',
                   command=self._hapus_titik).pack(side='left', padx=2)
        ttk.Button(b2, text='Kosongkan tanda',
                   command=self._kosongkan_tanda).pack(side='left', padx=2)
        ttk.Button(b2, text='✔ Selesai & Perbarui Data',
                   command=lambda: self._perbarui_sinyal()).pack(side='left', padx=2)

        b3 = ttk.Frame(self.bar_manual); b3.pack(fill='x', pady=(3, 0))
        ttk.Button(b3, text='↺ Kalibrasi frame ini ikut frame sebelumnya',
                   command=self._ikuti_sebelumnya).pack(side='left', padx=2)
        self.var['alur_frame'] = tk.BooleanVar(value=True)
        self.var['ikut_penggaris'] = tk.BooleanVar(value=True)
        ttk.Button(b3, text='Lanjut ke frame berikutnya ▶',
                   command=self._maju_alur).pack(side='left', padx=2)
        ttk.Checkbutton(b3, text='Alur: titik → sumbu',
                        variable=self.var['alur_frame']).pack(side='left', padx=6)
        ttk.Checkbutton(b3, text='Penggaris ikut sumbu',
                        variable=self.var['ikut_penggaris']).pack(side='left')

        self.lbl_prev = ttk.Label(kiri_prev, text='Belum ada sumber terbuka.', foreground='#666')
        self.lbl_prev.pack(fill='x')
        self.bind('<Left>', lambda e: self._geser_frame(-1))
        self.bind('<Right>', lambda e: self._geser_frame(1))
        self.bind('<Control-z>', lambda e: self._hapus_titik())

        kanan_plot = ttk.LabelFrame(atas, text='Data Terekam')
        kanan_plot.pack(side='left', fill='both', expand=True, padx=4, pady=4)
        self.fig2, self.kanvas2 = self._kanvas(kanan_plot, ukuran=(5.2, 4.2))
        self._ringkas_kalibrasi()

    # ---------- cara pelacakan ----------
    def _ganti_cara(self):
        pilih = self.var['cara'].get()
        self.cara = ('manual' if pilih.startswith('Manual')
                     else 'semi' if pilih.startswith('Semi') else 'otomatis')
        self.pelacak = self.pel_csrt if self.cara == 'semi' else self.pel_auto
        self.cmb_mode.config(state='readonly' if self.cara == 'otomatis' else 'disabled')
        if self.cara == 'manual':
            self.var['mode_klik'].set('tandai')
            self._tampil_bar(True)
            self.lbl_prev.config(
                text='Manual: klik posisi objek pada pratinjau → frame maju otomatis '
                     '(panah ←/→ untuk berpindah, Ctrl+Z untuk membatalkan). Penggaris & sumbu x–y '
                     'boleh diatur/diubah SESUDAH menandai (klik "Sumbu x–y"). Alur tiap frame: klik titik → atur '
                     'sumbu x–y (penggaris ikut) → tombol Lanjut ke frame berikutnya.')
        elif self.cara == 'semi':
            self.var['mode_klik'].set('roi')
            self._tampil_bar(True)
            self.lbl_prev.config(text='Semi-otomatis: atur penggaris & sumbu x–y DULU, lalu seret kotak '
                                      'pada objek, tekan "Kunci Objek", lalu Mulai Rekam / Proses Seluruh Berkas.')
        else:
            self.var['mode_klik'].set('warna')
            self._tampil_bar(self.sumber is not None and self.sumber.bisa_lompat())
            self.lbl_prev.config(text='Otomatis: atur penggaris & sumbu x–y (x = lantai) pada pratinjau '
                                      'SEBELUM merekam, lalu klik objek untuk mengambil warnanya.')
        self._atur_kursor_klik()
        self._label_kalibrasi()
        self._status(f'Cara pelacakan: {pilih}')

    def _tampil_bar(self, tampil):
        if tampil:
            if not self.bar_manual.winfo_manager():
                self.bar_manual.pack(fill='x', before=self.lbl_prev)
        else:
            self.bar_manual.pack_forget()

    def _n_frame(self):
        if self.sumber is None:
            return 0
        n = self.sumber.jumlah_frame
        return n if n and n > 0 else 0

    def _tampil_manual(self, gambar_saja=False):
        """Tampilkan satu frame (telusur) beserta tanda yang sudah ada."""
        if self.sumber is None:
            return
        if gambar_saja and self.frame_akhir is not None:
            frame, t, ok = self.frame_akhir, self.t_frame, True
        else:
            ok, frame, t = self.sumber.ambil_frame(self.manual_i)
        if not ok or frame is None:
            self._pesan_kanvas(f'Frame {self.manual_i} tidak dapat dibaca.')
            self._status('Frame tidak tersedia — coba konversi video ke H.264/MP4.', '#a60')
            return
        self.frame_akhir = frame
        self.t_frame = t
        self._muat_kal_frame(self.manual_i)       # penggaris & sumbu x-y milik frame ini
        titik = self.titik_manual.get(self.manual_i)
        riwayat = [(cx, cy) for (_, cx, cy) in self.titik_manual.values()]
        n = self._n_frame()
        an = ak.gambar_penanda(frame, (titik[1], titik[2]) if titik else None,
                               f'Frame {self.manual_i}' + (f'/{n-1}' if n else '') +
                               f'  t={t:.3f}s', riwayat,
                               kursor=getattr(self, '_kursor', None))
        self._tampil(an)
        self.lbl_prev.config(text=f'Frame {self.manual_i}' + (f' dari {n}' if n else '') +
                                  f' | t = {t:.3f} s | titik manual: {len(self.titik_manual)}'
                                  + ('  [sudah ditandai]' if titik else '')
                                  + self._info_kal_frame())
        if n:
            self._geser_slider = True
            self.slider.config(to=max(n - 1, 1))
            self.var_slider.set(self.manual_i)
            self._geser_slider = False

    def _putar_jeda(self):
        """Putar pratinjau video/gambar atau jeda kembali ke telusur frame."""
        if self.sumber is None:
            messagebox.showinfo('Pratinjau', 'Buka sumber terlebih dahulu.'); return
        if self.pratinjau_aktif:
            self.pratinjau_aktif = False
            self.btn_putar.config(text='▶ Putar')
            return
        self.pratinjau_aktif = True
        self.btn_putar.config(text='⏸ Jeda')
        if self.sumber.bisa_lompat():
            self._loop_putar()
        else:
            self._loop_frame()

    def _loop_putar(self):
        """Pemutaran video/urutan gambar; ikut melacak & merekam bila bukan mode manual."""
        if not self.pratinjau_aktif or self.sumber is None:
            return
        n = self._n_frame()
        if n and self.manual_i >= n:
            self.pratinjau_aktif = False
            self.btn_putar.config(text='▶ Putar')
            self.manual_i = max(n - 1, 0)
            if self.merekam:
                self.merekam = False
                self._perbarui_sinyal()
            return
        if self.cara == 'manual':
            self._tampil_manual()
        else:
            ok, frame, t = self.sumber.ambil_frame(self.manual_i)
            if not ok or frame is None:
                self.pratinjau_aktif = False
                self.btn_putar.config(text='▶ Putar')
                self._pesan_kanvas(f'Frame {self.manual_i} tidak dapat dibaca.')
                return
            self.frame_akhir, self.t_frame = frame, t
            cx, cy, _, anotasi = self.pelacak.proses(frame)
            if cx is not None:
                self.kal.origin_sementara(cx, cy)
                if self.merekam:
                    self._buf_tambah(t, cx, cy, idx=self.manual_i)
            self._tampil(anotasi)
            self.lbl_prev.config(
                text=f'Frame {self.manual_i}' + (f'/{n-1}' if n else '') + f' | t={t:.3f}s | ' +
                     ('objek tidak terdeteksi' if cx is None else f'({cx:.0f},{cy:.0f})') +
                     (f" | terekam {len(self._buf['t'])}" if self.merekam else ''))
            if self.merekam and len(self._buf['t']) % 20 == 0 and len(self._buf['t']) > 2:
                self._perbarui_sinyal(sementara=True)
            if not self._geser_slider and n:
                self._geser_slider = True
                self.slider.config(to=max(n - 1, 1)); self.var_slider.set(self.manual_i)
                self._geser_slider = False
        self.manual_i += 1
        self.after(max(int(1000 / max(self.sumber.fps, 1)), 12), self._loop_putar)

    def _geser_frame(self, langkah):
        if self.sumber is None:
            return
        if self.pratinjau_aktif:                 # jeda dulu bila sedang diputar
            self.pratinjau_aktif = False
            self.btn_putar.config(text='▶ Putar')
        if not self.sumber.bisa_lompat():
            self._status('Kamera langsung tidak bisa melompat frame — pakai video/gambar.', '#a60')
            return
        self._mode_tandai_bila_alur()
        n = self._n_frame()
        if langkah <= -1000000:
            self.manual_i = 0
        else:
            self.manual_i = max(0, self.manual_i + int(langkah))
        if n:
            self.manual_i = min(self.manual_i, n - 1)
        self._tampil_manual()

    def _slider_frame(self, _=None):
        if self._geser_slider or self.sumber is None or not self.sumber.bisa_lompat():
            return
        if self.pratinjau_aktif:
            self.pratinjau_aktif = False
            self.btn_putar.config(text='▶ Putar')
        i = int(float(self.var_slider.get()))
        if i != self.manual_i:
            self._mode_tandai_bila_alur()
            self.manual_i = i
            self._tampil_manual()

    def _pindah_ke_manual(self):
        """Klik 'Tandai posisi objek' = tracking manual. Pastikan 'Cara pelacakan' ikut
        berpindah ke Manual supaya penggaris & sumbu x–y TIDAK terkunci (kunci hanya
        berlaku untuk otomatis / semi-otomatis)."""
        if self.cara == 'manual':
            return
        self.var['cara'].set('Manual (klik tiap frame)')
        self._ganti_cara()

    def _tandai_manual(self, x, y):
        """Rekam posisi objek hasil klik pengguna pada frame yang sedang tampil."""
        if self.frame_akhir is None:
            return
        self._pindah_ke_manual()
        self.titik_manual[self.manual_i] = (self.t_frame, float(x), float(y))
        self.kal.origin_sementara(x, y)
        self._manual_ke_buf()
        if self.cara == 'manual' and self.var['alur_frame'].get():
            self.var['mode_klik'].set('sumbu'); self._atur_kursor_klik()
            self._tampil_manual()
            self._status(f'Frame {self.manual_i}: titik ditandai. Sekarang atur SUMBU x–y frame ini '
                         '(klik/seret asal, putar ujung x); penggaris ikut. Lalu tekan tombol Lanjut ke frame berikutnya.',
                         '#1a4f8a')
            return
        lompat = max(int(self._f('lompat', 1) or 1), 1)
        n = self._n_frame()
        if self.sumber is not None and self.sumber.bisa_lompat() and (not n or self.manual_i + lompat < n):
            self.manual_i += lompat
        self._tampil_manual()

    # ---------- kalibrasi PER FRAME (mode manual) ----------
    def _snap_kal(self):
        k = self.kal
        return {'origin': tuple(k._origin), 'ux': k._ux, 'uy': k._uy, 'px': k.px_per_m,
                'lantai': k.lantai_diset, 'sem': k._origin_sementara, 'skala': k.skala_diset,
                'p0': k.p0, 'p1': k.p1, 'panjang': k.panjang_ref}

    def _terap_snap(self, k, s):
        k._origin = tuple(s['origin']); k._ux, k._uy = s['ux'], s['uy']
        k.px_per_m = s['px']; k.lantai_diset = s['lantai']; k._origin_sementara = s['sem']
        k.skala_diset = s['skala']; k.p0, k.p1 = s['p0'], s['p1']; k.panjang_ref = s['panjang']

    def _kal_dari_snap(self, s):
        k = ak.Kalibrasi(px_per_m=s['px'])
        self._terap_snap(k, s)
        k.r_bola = self.kal.r_bola                  # jari-jari bola berlaku untuk semua frame
        return k

    def _snap_untuk(self, i):
        """(frame_pemilik, snapshot) untuk frame i: milik sendiri, atau ikut frame sebelumnya
        yang pernah diatur (sebelum frame pertama yang diatur, ikut yang pertama)."""
        if not self.kal_frame:
            return None
        kunci = sorted(self.kal_frame)
        sebelum = [j for j in kunci if j <= i]
        j = sebelum[-1] if sebelum else kunci[0]
        return j, self.kal_frame[j]

    def _muat_kal_frame(self, i):
        """Pasang penggaris + sumbu x-y milik frame i ke self.kal (hanya mode manual)."""
        if self.cara != 'manual' or self._seret is not None:
            return
        s = self._snap_untuk(i)
        if s is None or s[1] == self._snap_kal():
            return
        self._terap_snap(self.kal, s[1])
        self._ringkas_kalibrasi()

    def _pra_ubah_kal(self):
        """Dipanggil SEBELUM kalibrasi diubah: kalibrasi lama dijadikan dasar frame 0 supaya
        frame-frame awal tidak ikut berubah."""
        k = self.kal
        if self.cara == 'manual' and not self.kal_frame and (
                k.lantai_diset or k.skala_diset or k.p0 is not None):
            self.kal_frame[0] = self._snap_kal()

    def _simpan_kal_frame(self):
        """Simpan penggaris + sumbu x-y saat ini sebagai milik frame yang sedang tampil."""
        if self.cara == 'manual':
            self.kal_frame[self.manual_i] = self._snap_kal()

    def _info_kal_frame(self):
        s = self._snap_untuk(self.manual_i)
        if s is None:
            return ''
        return (' | kalibrasi: frame ini' if s[0] == self.manual_i
                else f' | kalibrasi: ikut frame {s[0]}')

    def _ikuti_sebelumnya(self):
        """Buang kalibrasi khusus frame ini sehingga mengikuti frame sebelumnya."""
        if self.manual_i not in self.kal_frame:
            self._status('Frame ini sudah mengikuti kalibrasi frame sebelumnya.')
            return
        if len(self.kal_frame) == 1:
            self._status('Ini satu-satunya kalibrasi (dasar untuk semua frame), tidak dapat dibuang.')
            return
        del self.kal_frame[self.manual_i]
        self._muat_kal_frame(self.manual_i)
        self._hitung_ulang_data()
        self._tampil_manual()
        self._status(f'Frame {self.manual_i} kini mengikuti kalibrasi frame sebelumnya.')

    def _manual_ke_buf(self):
        self._buf = self._buf_kosong()
        cache = {}
        for i in sorted(self.titik_manual):
            t, cx, cy = self.titik_manual[i]
            s = self._snap_untuk(i)                 # kalibrasi milik frame i (bila ada)
            kal = None
            if s is not None:
                if s[0] not in cache:
                    cache[s[0]] = self._kal_dari_snap(s[1])
                kal = cache[s[0]]
            self._buf_tambah(t, cx, cy, kal, idx=i)
        if len(self._buf['t']) >= 2:
            self._perbarui_sinyal(sementara=True)

    def _mode_tandai_bila_alur(self):
        if self.cara == 'manual' and self.var['alur_frame'].get() \
                and self.var['mode_klik'].get() != 'tandai':
            self.var['mode_klik'].set('tandai'); self._atur_kursor_klik()

    def _maju_alur(self):
        """Sesudah sumbu frame ini diatur: pindah ke frame berikutnya, siap menandai titik."""
        lompat = max(int(self._f('lompat', 1) or 1), 1)
        n = self._n_frame()
        self.var['mode_klik'].set('tandai'); self._atur_kursor_klik()
        if self.sumber is not None and self.sumber.bisa_lompat() and (not n or self.manual_i + lompat < n):
            self.manual_i += lompat
            self._tampil_manual()
            self._status(f'Frame {self.manual_i}: klik posisi objek.', '#1a4f8a')
        else:
            self._tampil_manual()
            self._status('Frame terakhir selesai. Tekan "Selesai & Perbarui Data".', '#0a5')

    def _ikut_penggaris(self, lama):
        """Penggaris ikut bergeser/berputar bersama sumbu x-y (mode manual)."""
        k = self.kal
        if self.cara != 'manual' or not self.var['ikut_penggaris'].get():
            return
        if k.p0 is None or k.p1 is None or not lama[2]:
            return
        (ox0, oy0), a0, _ = lama
        ox, oy = k.origin
        d = ((k.sudut_x_deg - a0 + 180.0) % 360.0) - 180.0
        c, sn = float(np.cos(np.radians(d))), float(np.sin(np.radians(d)))
        def pindah(p):
            dx, dy = p[0] - ox0, p[1] - oy0
            return (ox + dx * c + dy * sn, oy - dx * sn + dy * c)
        k.p0, k.p1 = pindah(k.p0), pindah(k.p1)

    def _hapus_titik(self):
        if not self.titik_manual:
            return
        i = self.manual_i if self.manual_i in self.titik_manual else max(self.titik_manual)
        self.titik_manual.pop(i, None)
        self.manual_i = i
        self._manual_ke_buf()
        self._tampil_manual()
        self._status(f'Titik pada frame {i} dihapus ({len(self.titik_manual)} tersisa).')

    def _kosongkan_tanda(self):
        if self.titik_manual and messagebox.askyesno('Kosongkan', 'Hapus semua tanda manual?'):
            self.titik_manual = {}
            self._buf = self._buf_kosong()
            self._pangkas_ref = None
            self.kal.reset_sementara()
            self._tampil_manual()
            self._status('Semua tanda manual dihapus.')

    def _kunci_objek(self):
        """Inisialisasi pelacak semi-otomatis dari kotak ROI yang diseret pengguna."""
        if self.frame_akhir is None:
            messagebox.showinfo('Kunci objek', 'Buka sumber dan tampilkan satu frame dahulu.')
            return
        roi = self.pel_csrt.roi or self.pel_auto.roi
        if not roi:
            messagebox.showinfo('Kunci objek',
                                'Pilih dahulu kotak objek: set klik ke "Pilih ROI / kotak objek" '
                                'lalu seret kotak pada pratinjau.')
            return
        try:
            nama = self.pel_csrt.mulai(self.frame_akhir, roi)
            self.pelacak = self.pel_csrt
            self.var['cara'].set('Semi-otomatis (kunci objek/ROI)')
            self.cara = 'semi'
            self._status(f'Objek dikunci dengan algoritma {nama}. Siap merekam.')
        except Exception as exc:
            self._galat(exc)

    # ---------- sumber ----------
    def _pilih_berkas(self):
        mode = self.var['sumber'].get()
        if mode == 'Impor gambar (banyak)':
            berkas = filedialog.askopenfilenames(
                title='Pilih gambar (urutan frame)',
                filetypes=[('Gambar', '*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp'),
                           ('Semua', '*.*')])
            if berkas:
                self._berkas = sorted(berkas)
                self.var['berkas'].set(f'{len(self._berkas)} gambar dipilih')
        elif mode == 'Berkas CSV':
            self._muat_data()
        else:
            berkas = filedialog.askopenfilename(
                title='Pilih video',
                filetypes=[('Video', '*.mp4 *.avi *.mov *.mkv *.m4v'), ('Semua', '*.*')])
            if berkas:
                self._berkas = berkas
                self.var['berkas'].set(os.path.basename(berkas))

    def _siapkan_pelacak(self):
        self.pel_auto.mode = self.var['mode_lacak'].get()
        self.pel_auto.luas_min = int(self._f('luas_min', 30))
        th = int(self._f('tol_h', 12))
        self.pel_auto.toleransi = (th, 80, 80)
        self.pelacak = self.pel_csrt if self.cara == 'semi' else self.pel_auto
        self._sinkron_kal_dari_form()

    def _buka_sumber(self):
        mode = self.var['sumber'].get()
        try:
            self._berhenti()
            self._siapkan_pelacak()
            if mode == 'Arduino (serial)':
                self.serial = ak.SumberSerial(self.var['port'].get(), int(self._f('baud', 9600)),
                                              fps=self._f('fps', 60.0))
                self.lbl_prev.config(text=f"Serial {self.var['port'].get()} terbuka. Tekan Mulai Rekam.")
                self._status('Port serial terbuka.')
                return
            if mode == 'Berkas CSV':
                self._muat_data(); return
            if mode == 'Kamera':
                self.sumber = ak.SumberVideo('kamera', int(self._f('kam_idx', 0)), self._f('fps', 60.0))
                self._dok_sumber = None
            elif mode == 'Berkas video':
                if not getattr(self, '_berkas', None):
                    messagebox.showwarning('Berkas', 'Pilih berkas video terlebih dahulu.'); return
                self.sumber = ak.SumberVideo('video', self._berkas, self._f('fps', 60.0))
                self._dok_sumber = ('video', self._berkas)
            else:
                if not getattr(self, '_berkas', None):
                    messagebox.showwarning('Berkas', 'Pilih gambar terlebih dahulu.'); return
                self.sumber = ak.SumberVideo('gambar', list(self._berkas), self._f('fps', 60.0))
                self._dok_sumber = ('gambar', list(self._berkas))
            self.pel_auto.reset()
            self.titik_manual = {}
            self.kal_frame = {}
            self.manual_i = 0
            self._status(f'Sumber "{mode}" terbuka — {self.sumber.jumlah_frame} frame.')
            if self.sumber.bisa_lompat():
                # video / urutan gambar: selalu mulai dari telusur frame demi frame
                self.pratinjau_aktif = False
                self.btn_putar.config(text='▶ Putar')
                self._tampil_bar(True)
                self._tampil_manual()
                d = self.sumber.diagnosa()
                self.var['fps'].set(f"{d['fps']:g}")
                info = (f"{d['backend']} | {d['ukuran'][0]}×{d['ukuran'][1]} px | "
                        f"{d['fps']:g} FPS | {d['jumlah_frame']} frame"
                        + (f" | putaran {d['rotasi']}°" if d['rotasi'] else ''))
                self._status(info)
                if d['pesan']:
                    messagebox.showwarning('Video', d['pesan'])
            else:
                self._tampil_bar(False)
                self.pratinjau_aktif = True
                self.btn_putar.config(text='⏸ Jeda')
                self._loop_frame()
        except Exception as exc:
            self._galat(exc)

    def _loop_frame(self):
        if not self.pratinjau_aktif or self.sumber is None:
            return
        ok, frame, t = self.sumber.baca()
        if not ok:
            if self.sumber.tipe in ('video', 'gambar'):
                self.pratinjau_aktif = False
                self.merekam = False
                self.lbl_prev.config(text='Berkas selesai diproses.')
                self._perbarui_sinyal()
                return
            self.after(30, self._loop_frame)
            return
        if self._rekam_video_aktif and self._vid_writer is not None:
            try:
                self._vid_writer.write(frame)
                self._rekam_video_n += 1
                if self._rekam_video_n % 10 == 0:
                    durasi = self._rekam_video_n / max(self._f('fps', 60.0), 1.0)
                    self._status(f'🎥 Merekam video kamera… {self._rekam_video_n} frame '
                                 f'(~{durasi:.1f} s)')
            except Exception:
                pass
        cx, cy, _, anotasi = self.pelacak.proses(frame)
        self.frame_akhir = frame
        if cx is not None:
            self.kal.origin_sementara(cx, cy)
            xm, ym = self.kal.ke_meter(cx, cy)
            if self.merekam:
                self._buf_tambah(t, cx, cy)
            self.lbl_prev.config(text=f'Frame {self.sumber.i} | piksel ({cx:.0f},{cy:.0f}) | '
                                      f'x={xm*100:.2f} cm  y={ym*100:.2f} cm | '
                                      f'terekam: {len(self._buf["t"])}')
        else:
            self.lbl_prev.config(text=f'Frame {self.sumber.i} | objek tidak terdeteksi — '
                                      f'atur mode/warna/ROI.')
        self._tampil(anotasi)
        if self.merekam and len(self._buf['t']) % 15 == 0 and len(self._buf['t']) > 2:
            self._perbarui_sinyal(sementara=True)
        jeda = 15 if self.sumber.tipe != 'kamera' else max(int(1000 / max(self.sumber.fps, 1)), 10)
        self.after(jeda, self._loop_frame)

    def _loop_serial(self):
        if not self.merekam or self.serial is None:
            return
        for t, v in self.serial.baca():
            self._buf['t'].append(t); self._buf['x'].append(0.0); self._buf['y'].append(v)
        self.lbl_prev.config(text=f'Serial — {len(self._buf["t"])} titik terekam.')
        if len(self._buf['t']) % 20 == 0 and len(self._buf['t']) > 2:
            self._perbarui_sinyal(sementara=True)
        self.after(30, self._loop_serial)

    # ---------- kendali gearbox (motor DC + driver L298N lewat Arduino) ----------
    def _sambung_servo(self):
        """Sambungkan koneksi serial khusus gearbox. Bila kolom 'Port gearbox'
        dikosongkan, gearbox akan mencoba memakai koneksi Arduino (self.serial) yang
        sudah dibuka lewat 'Buka & Pratinjau' — cocok bila sensor & driver motor
        berada pada papan Arduino yang sama.

        Catatan: membuka port serial membuat Arduino reset, sehingga sketch menganggap
        posisi = tertutup (posisiBuka = false). Pastikan gearbox memang berada di posisi
        awal (tertutup) sebelum menekan 'Sambung'."""
        port = self.var['servo_port'].get().strip()
        if not port:
            if self.serial is not None:
                self.servo_ser = None
                self._servo_berbagi = True
                self.lbl_servo.config(text='Gearbox: memakai koneksi Arduino yang sudah terbuka.',
                                      foreground='#0a5')
                self._status('Gearbox memakai koneksi serial Arduino.')
            else:
                self.lbl_servo.config(
                    text='Gearbox: isi Port gearbox, atau buka koneksi Arduino (serial) dahulu.',
                    foreground='#a60')
            return
        try:
            import serial
        except Exception:
            messagebox.showerror('Gearbox', 'Modul pyserial belum terpasang.\n\npip install pyserial')
            return
        try:
            if self.servo_ser is not None:
                try:
                    self.servo_ser.close()
                except Exception:
                    pass
            baud = int(self._f('servo_baud', 9600))
            self.servo_ser = serial.Serial(port, baud, timeout=1)
            time.sleep(2.0)   # beri waktu papan Arduino selesai reset setelah port dibuka
            self._servo_berbagi = False
            sapaan = self._baca_balasan_servo(tunggu=0.3)    # teks pembuka dari setup() sketch
            teks = f'Gearbox: tersambung di {port} ({baud} baud).'
            if sapaan:
                teks += f' Arduino: "{sapaan.splitlines()[0]}…"'
            self.lbl_servo.config(text=teks, foreground='#0a5')
            self._status(f'Gearbox tersambung di {port}.')
        except Exception as exc:
            self.servo_ser = None
            self._galat(exc)

    def _baca_balasan_servo(self, tunggu=1.2, selesai_bila_ada=False):
        """Baca balasan teks dari Arduino (mis. 'Motor berhenti pada posisi buka').
        Putaran gearbox memakai delay (waktu90 + waktuRem ≈ 350 ms) sebelum sketch
        membalas, jadi waktu tunggu dibuat cukup panjang. selesai_bila_ada=True:
        berhenti begitu satu baris lengkap diterima (tidak menunggu sampai batas waktu)."""
        ser = self.servo_ser
        if ser is None:
            return ''
        try:
            akhir = time.time() + tunggu
            buf = b''
            while time.time() < akhir:
                n = getattr(ser, 'in_waiting', 0)
                if n:
                    buf += ser.read(n)
                    if selesai_bila_ada and b'\n' in buf:
                        break
                else:
                    time.sleep(0.02)
            return buf.decode('utf-8', errors='ignore').strip()
        except Exception:
            return ''

    def _kirim_ke_arduino(self, perintah):
        """Coba kirim teks ke koneksi Arduino (akuisisi.SumberSerial) yang sudah
        terbuka, dengan mencoba beberapa nama atribut/metode umum agar tetap
        berfungsi walau detail internal akuisisi.py berbeda."""
        if self.serial is None:
            return False
        data = perintah.encode('utf-8')
        for atr in ('ser', 'koneksi', 'port', 'conn', '_ser', 'serial'):
            obj = getattr(self.serial, atr, None)
            if obj is not None and obj is not self.serial and hasattr(obj, 'write'):
                try:
                    obj.write(data)
                    return True
                except Exception:
                    pass
        for metode in ('tulis', 'kirim', 'write', 'send'):
            fn = getattr(self.serial, metode, None)
            if callable(fn):
                for arg in (perintah, data):
                    try:
                        fn(arg)
                        return True
                    except Exception:
                        continue
        return False

    def _kirim_servo(self, perintah):
        """Kirim perintah ke gearbox ('1' = buka 90°, '0' = kembali). Sketch Arduino
        menyimpan state posisiBuka, sehingga perintah yang sama dua kali berturut-turut
        diabaikan tanpa balasan."""
        if not perintah:
            return False
        if self.servo_ser is not None:
            try:
                self.servo_ser.reset_input_buffer()
            except Exception:
                pass
            try:
                self.servo_ser.write(perintah.encode('utf-8'))
                balasan = self._baca_balasan_servo(selesai_bila_ada=True)
                if balasan:
                    self.lbl_servo.config(
                        text=f'Gearbox: "{perintah}" terkirim → Arduino: "{balasan}"',
                        foreground='#0a5')
                else:
                    self.lbl_servo.config(
                        text=f'Gearbox: "{perintah}" terkirim, tanpa balasan. Mungkin gearbox SUDAH '
                             'di posisi itu (sketch mengabaikan perintah yang sama dua kali), '
                             'atau cek port/baud/kabel/catu daya motor.',
                        foreground='#a60')
                return True
            except Exception as exc:
                self._galat(exc)
                return False
        if self._servo_berbagi or self.serial is not None:
            ok = self._kirim_ke_arduino(perintah)
            if not ok:
                self.lbl_servo.config(
                    text='Gearbox: gagal mengirim lewat koneksi Arduino — isi Port gearbox terpisah.',
                    foreground='#c00')
            return ok
        self.lbl_servo.config(text='Gearbox: belum tersambung. Pilih Port gearbox lalu tekan "Sambung".',
                              foreground='#c00')
        return False

    def _servo_buka(self):
        return self._kirim_servo(self.var['servo_buka'].get() or '1')

    def _servo_tutup(self):
        return self._kirim_servo(self.var['servo_tutup'].get() or '0')

    # ---------- rekam video dari kamera (untuk track manual per frame) ----------
    def _mulai_rekam_video(self):
        """Rekam frame mentah dari KAMERA langsung ke berkas video (.mp4). Setelah
        'Selesai Rekam & Buka untuk Track' ditekan, video ini otomatis dibuka
        sebagai berkas video biasa (frame-seekable) sehingga bisa ditelusur dan
        dilacak MANUAL / semi-otomatis / otomatis per frame — bukan hanya video
        yang sudah ada sebelumnya."""
        if self.sumber is None or self.sumber.tipe != 'kamera':
            messagebox.showinfo('Rekam Video dari Kamera',
                                'Pilih sumber "Kamera", tekan "📷 Buka & Pratinjau" dahulu, '
                                'baru tekan tombol ini.')
            return
        if self._vid_writer is not None:
            messagebox.showinfo('Rekam Video dari Kamera', 'Sudah sedang merekam video.')
            return
        if self.frame_akhir is None:
            messagebox.showwarning('Rekam Video dari Kamera',
                                   'Belum ada gambar dari kamera — tunggu pratinjau muncul.')
            return
        try:
            import cv2
        except Exception as exc:
            self._galat(exc); return
        try:
            h, w = self.frame_akhir.shape[:2]
            fps = max(self._f('fps', 60.0), 1.0)
            folder = os.path.join(os.path.expanduser('~'), 'LabGetaran_Rekaman')
            os.makedirs(folder, exist_ok=True)
            nama = os.path.join(folder, f"rekam_{datetime.datetime.now():%Y%m%d_%H%M%S}.mp4")
            self._vid_writer = cv2.VideoWriter(nama, cv2.VideoWriter_fourcc(*'mp4v'),
                                               fps, (w, h))
            if not self._vid_writer.isOpened():
                self._vid_writer = None
                messagebox.showerror('Rekam Video dari Kamera',
                                     'Gagal membuat berkas video (codec mp4v tak tersedia di '
                                     'OpenCV Anda). Coba: pip install opencv-python terbaru.')
                return
            self._rekam_video_path = nama
            self._rekam_video_n = 0
            self._rekam_video_aktif = True
            if self.var['servo_aktif'].get():
                self._servo_buka()
            self._status(f'🎥 Mulai merekam video kamera → {os.path.basename(nama)}')
        except Exception as exc:
            self._vid_writer = None
            self._galat(exc)

    def _berhenti_rekam_video(self):
        if self._vid_writer is None:
            messagebox.showinfo('Rekam Video dari Kamera', 'Tidak sedang merekam video.')
            return
        self._rekam_video_aktif = False
        try:
            self._vid_writer.release()
        except Exception:
            pass
        self._vid_writer = None
        path, n = self._rekam_video_path, self._rekam_video_n
        self._status(f'🎥 Rekaman video selesai: {os.path.basename(path)} ({n} frame).')
        if n < 2:
            messagebox.showwarning('Rekam Video dari Kamera',
                                   'Rekaman terlalu pendek — tidak ada frame tersimpan.')
            return
        if not messagebox.askyesno(
                'Rekam Video dari Kamera',
                f'Rekaman selesai: {os.path.basename(path)} ({n} frame)\n\n'
                'Buka video ini sekarang untuk ditelusur & dilacak per frame '
                '(Manual / Semi-otomatis / Otomatis)?'):
            return
        self.var['sumber'].set('Berkas video')
        self._berkas = path
        self.var['berkas'].set(os.path.basename(path))
        self._buka_sumber()

    def _pesan_kanvas(self, teks):
        self.kanvas_prev.delete('all')
        self.kanvas_prev.create_text(max(self.kanvas_prev.winfo_width(), 300) // 2,
                                     max(self.kanvas_prev.winfo_height(), 200) // 2,
                                     text=teks, fill='#ddd', width=440, justify='center')

    def _tampil(self, frame_bgr):
        if frame_bgr is None or getattr(frame_bgr, 'size', 0) == 0:
            self._pesan_kanvas('Frame kosong — tidak ada gambar untuk ditampilkan.')
            return
        self.kanvas_prev.update_idletasks()
        lebar, tinggi = self.kanvas_prev.winfo_width(), self.kanvas_prev.winfo_height()
        if lebar < 50 or tinggi < 50:                 # widget belum sempat diukur
            lebar, tinggi = 520, 390
        h, w = frame_bgr.shape[:2]
        skala = min(lebar / w, tinggi / h)
        self._skala_pratinjau = skala
        self._geser_gambar = ((lebar - w * skala) / 2.0, (tinggi - h * skala) / 2.0)
        if not PIL_ADA:
            self._pesan_kanvas('Pillow belum terpasang sehingga pratinjau tidak bisa '
                               'digambar di jendela ini.\n\npip install pillow')
            try:
                import cv2
                cv2.imshow('Pratinjau (tekan q untuk tutup)', frame_bgr)
                cv2.waitKey(1)
            except Exception:
                pass
            return
        import cv2
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb).resize((max(int(w * skala), 1), max(int(h * skala), 1)))
        self._foto = ImageTk.PhotoImage(img)
        self.kanvas_prev.delete('all')
        self.kanvas_prev.create_image(self._geser_gambar[0], self._geser_gambar[1],
                                      anchor='nw', image=self._foto)
        self._gambar_overlay()                         # penggaris + sumbu x–y (x = lantai)

    # ---------- interaksi pratinjau ----------
    def _px_asli(self, ev):
        s = self._skala_pratinjau if self._skala_pratinjau > 0 else 1.0
        dx, dy = getattr(self, '_geser_gambar', (0.0, 0.0))
        return int((ev.x - dx) / s), int((ev.y - dy) / s)

    def _gerak_kursor(self, ev):
        """Garis bidik + kaca pembesar saat menandai manual; kursor berubah di atas handle
        penggaris / sumbu saat mode klik 'Penggaris' atau 'Sumbu x–y'."""
        mode = self.var['mode_klik'].get()
        if mode in ('penggaris', 'sumbu'):
            if self._seret is None:
                self._hover_kal(ev, mode)
            return
        if self.frame_akhir is None or mode != 'tandai':
            return
        if self.pratinjau_aktif:
            return
        self._kursor = self._px_asli(ev)
        self._tampil_manual(gambar_saja=True)

    def _klik_prev(self, ev):
        if self.frame_akhir is None:
            return
        mode = self.var['mode_klik'].get()
        if mode == 'penggaris':
            self._mulai_seret_penggaris(ev)
            return
        if mode == 'sumbu':
            self._mulai_seret_sumbu(ev)
            return
        x, y = self._px_asli(ev)
        if mode == 'tandai':
            self._tandai_manual(x, y)
        elif mode == 'warna':
            if not hasattr(self.pelacak, 'ambil_warna'):
                self.pelacak = self.pel_auto
            self.pelacak.ambil_warna(self.frame_akhir, x, y)
            self.var['mode_lacak'].set('warna')
            self._status(f'Warna objek diambil pada piksel ({x},{y}).')
        elif mode == 'roi':
            self._roi_awal = (x, y)

    def _seret_prev(self, ev):
        mode = self.var['mode_klik'].get()
        if mode in ('penggaris', 'sumbu') and self._seret:
            (self._seret_penggaris if mode == 'penggaris' else self._seret_sumbu)(ev)
            return
        if mode == 'roi' and self._roi_awal:
            s = self._skala_pratinjau
            self.kanvas_prev.delete('roi')
            self.kanvas_prev.create_rectangle(self._roi_awal[0] * s, self._roi_awal[1] * s,
                                              ev.x, ev.y, outline='#0f0', tags='roi')

    def _lepas_prev(self, ev):
        mode = self.var['mode_klik'].get()
        if mode in ('penggaris', 'sumbu') and self._seret:
            (self._lepas_penggaris if mode == 'penggaris' else self._lepas_sumbu)(ev)
            return
        if mode == 'roi' and self._roi_awal:
            x, y = self._px_asli(ev)
            x0, y0 = self._roi_awal
            roi = (min(x0, x), min(y0, y), abs(x - x0), abs(y - y0))
            if roi[2] < 8 or roi[3] < 8:
                self.pel_auto.roi = self.pel_csrt.roi = None
                self._status('ROI dibatalkan (terlalu kecil).')
            else:
                self.pel_auto.roi = self.pel_csrt.roi = roi
                self._status(f'ROI diset: {roi}' +
                             ('  — tekan "Kunci Objek"' if self.cara == 'semi' else ''))
            self._roi_awal = None

    # ---------- rekam ----------
    def _mulai_rekam(self):
        if self.cara == 'manual':
            messagebox.showinfo('Manual', 'Pada mode manual, data direkam setiap kali Anda '
                                          'mengklik posisi objek di pratinjau. Gunakan tombol '
                                          'panah/slider untuk berpindah frame.')
            return
        if self.cara == 'semi' and self.pel_csrt.tracker is None:
            messagebox.showinfo('Semi-otomatis', 'Seret kotak pada objek lalu tekan "Kunci Objek".')
            return
        if self.var['sumber'].get() != 'Arduino (serial)':
            if self.sumber is None:
                messagebox.showwarning('Sumber', 'Buka kamera/berkas terlebih dahulu.')
                return
            if not self._kal_siap():              # otomatis / semi: kalibrasi SEBELUM data diambil
                self._minta_kalibrasi()
                return
        self._buf = self._buf_kosong()
        self._pangkas_ref = None
        self.kal.reset_sementara()
        self.merekam = True
        self._label_kalibrasi()
        if self.var['servo_aktif'].get():
            self._servo_buka()
        if self.var['sumber'].get() == 'Arduino (serial)':
            if self.serial is None:
                messagebox.showwarning('Serial', 'Buka port serial terlebih dahulu.')
                self.merekam = False; return
            self._status('Merekam dari Arduino…')
            self._loop_serial()
        else:
            if self.sumber is None:
                messagebox.showwarning('Sumber', 'Buka kamera/berkas terlebih dahulu.')
                self.merekam = False; return
            if self.sumber.bisa_lompat():
                self.manual_i = 0
                if not self.pratinjau_aktif:
                    self.pratinjau_aktif = True
                    self.btn_putar.config(text='⏸ Jeda')
                    self._loop_putar()
            self._status('Merekam…')

    def _berhenti(self):
        self.merekam = False
        self.pratinjau_aktif = False
        pesan_video = None
        if self._vid_writer is not None:
            self._rekam_video_aktif = False
            try:
                self._vid_writer.release()
            except Exception:
                pass
            pesan_video = (f'Rekaman video kamera dihentikan paksa: '
                          f'{os.path.basename(self._rekam_video_path)} '
                          f'({self._rekam_video_n} frame). Lain kali gunakan tombol '
                          '"⏹🎥 Selesai Rekam & Buka untuk Track" agar otomatis dibuka.')
            self._vid_writer = None
        if self.sumber is not None:
            self.sumber.lepas(); self.sumber = None
        if self._buf['t']:
            self._perbarui_sinyal()
        self._status(pesan_video or 'Perekaman dihentikan.')

    def _proses_batch(self):
        """Lacak seluruh frame video/gambar tanpa pratinjau (lebih cepat)."""
        mode = self.var['sumber'].get()
        if mode not in ('Berkas video', 'Impor gambar (banyak)'):
            messagebox.showinfo('Batch', 'Mode ini untuk berkas video atau impor gambar.')
            return
        if self.cara == 'manual':
            messagebox.showinfo('Manual', 'Mode manual ditandai sendiri frame demi frame. '
                                          'Pilih "Otomatis" atau "Semi-otomatis" untuk batch.')
            return
        if not getattr(self, '_berkas', None):
            messagebox.showwarning('Berkas', 'Pilih berkas terlebih dahulu.'); return
        if not self._kal_siap():                  # otomatis / semi: kalibrasi SEBELUM data diambil
            self._minta_kalibrasi()
            return
        try:
            self._berhenti()
            self._siapkan_pelacak()
            self.kal.reset_sementara()
            tipe = 'video' if mode == 'Berkas video' else 'gambar'
            arg = self._berkas if tipe == 'video' else list(self._berkas)
            self._dok_sumber = (tipe, arg)
            sumber = ak.SumberVideo(tipe, arg, self._f('fps', 60.0))
            if self.cara == 'semi':
                roi = self.pel_csrt.roi
                if not roi:
                    messagebox.showinfo('Semi-otomatis', 'Seret dahulu kotak objek pada pratinjau.')
                    sumber.lepas(); return
                ok, frame0, _ = sumber.ambil_frame(0)
                if ok:
                    self.pel_csrt.mulai(frame0, roi)
                sumber.lepas()
                sumber = ak.SumberVideo(tipe, arg, self._f('fps', 60.0))
            self._status('Memproses seluruh frame…', '#a60')
            hasil = ak.lacak_sumber(sumber, self.pel_csrt if self.cara == 'semi' else self.pel_auto,
                                    self.kal,
                                    callback=lambda n, fr: (self._tampil(fr),
                                                            self.lbl_prev.config(text=f'Frame {n}…'),
                                                            self.update_idletasks()))
            sumber.lepas()
            self._buf = {'t': list(hasil['t']), 'x': list(hasil['x']), 'y': list(hasil['y']),
                         'cx': list(hasil['cx']), 'cy': list(hasil['cy']),
                         'i': [int(v) for v in hasil.get('i', [])]}
            self._perbarui_sinyal()
            self._status(f"Selesai: {hasil['n_frame']} frame, {hasil['n_gagal']} gagal deteksi.")
        except Exception as exc:
            self._galat(exc)

    # =================================================================
    # KALIBRASI PENGGARIS / KOORDINAT
    # =================================================================
    def _buf_kosong(self):
        return {'t': [], 'x': [], 'y': [], 'cx': [], 'cy': [], 'i': []}

    def _buf_tambah(self, t, cx, cy, kal=None, idx=-1):
        """Simpan satu sampel. PIKSEL MENTAH ikut disimpan sehingga data dapat dihitung
        ulang kapan saja bila kalibrasi (penggaris / lantai / jari-jari bola) berubah.
        idx = nomor frame video (-1 bila tidak diketahui), dipakai untuk foto dokumentasi."""
        b = self._buf
        xm, ym = (kal or self.kal).ke_meter(cx, cy)
        b['t'].append(t); b['x'].append(xm); b['y'].append(ym)
        b.setdefault('cx', []).append(float(cx))
        b.setdefault('cy', []).append(float(cy))
        b.setdefault('i', []).append(int(idx))

    def _buf_dari_piksel(self):
        b = self._buf
        n = len(b['t'])
        return n > 0 and len(b.get('cx', [])) == n and len(b.get('cy', [])) == n

    def _kal_berlaku(self):
        """True bila lantai sudah dikalibrasi DAN data berasal dari piksel hasil tracking."""
        return bool(self.kal.lantai_diset and self._buf_dari_piksel())

    def _kal_siap(self):
        """Penggaris (skala) DAN sumbu x–y (lantai) sudah diatur."""
        return bool(self.kal.skala_diset and self.kal.lantai_diset)

    def _minta_kalibrasi(self):
        messagebox.showwarning(
            'Atur penggaris & sumbu dahulu',
            'Pada mode OTOMATIS / SEMI-OTOMATIS, penggaris dan sumbu x–y harus diatur '
            'SEBELUM merekam / memproses berkas:\n\n'
            '1. Tekan "Buka & Pratinjau" sampai gambar tampil.\n'
            '2. Pilih klik "Penggaris", SERET dari titik 0 ke ujung penggaris, isi "Panjang acuan" (cm).\n'
            '3. Pilih klik "Sumbu x–y", letakkan asal di lantai dan putar sumbu x '
            'sejajar lantai (sumbu x = lantai, y = 0).\n\n'
            'Mode manual tidak wajib: kalibrasi boleh diatur sesudah titik ditandai.')

    def _sinkron_kal_dari_form(self):
        """Ambil nilai yang DIKETIK pengguna (skala, sudut sumbu x, jari-jari bola).
        Bila kalibrasi terkunci (otomatis/semi sesudah data diambil), skala & sudut yang
        diketik diabaikan. Mengembalikan (berubah, ditolak)."""
        k = self.kal
        berubah = ditolak = False
        terkunci = self._kal_terkunci()
        px = max(self._f('px_per_m', k.px_per_m), 1e-6)
        if abs(px - k.px_per_m) > 6e-4:                      # toleransi pembulatan tampilan (.3f)
            if terkunci:
                ditolak = True
            else:
                k.px_per_m = px
                k.skala_diset = True
                berubah = True
                if k.p0 is not None and k.p1 is not None:    # panjang nyata penggaris mengikuti skala
                    k.panjang_ref = k.panjang_px() / px
        ang = self._f('sudut_x', k.sudut_x_deg)
        if abs(ang - k.sudut_x_deg) > 6e-3:
            if terkunci:
                ditolak = True
            else:
                k.sudut_x_deg = ang
                berubah = True
        r = max(self._f('r_bola_cm', 0.0), 0.0) / 100.0
        if abs(r - k.r_bola) > 1e-9:
            k.r_bola = r
            berubah = True
        return berubah, ditolak

    def _label_kalibrasi(self):
        k = self.kal
        if not k.skala_diset and not k.lantai_diset:
            self.lbl_kal.config(text='Kalibrasi: BELUM. 1) seret penggaris (titik 0 → ujung), '
                                     '2) atur sumbu x–y (x = lantai).', foreground='#a60')
            return
        lengkap = k.skala_diset and k.lantai_diset
        asal = 'penggaris' if (k.p0 is not None and k.p1 is not None) else 'diketik'
        teks = (f'Skala {k.px_per_m:.1f} px/m ({asal}) | sumbu x = lantai '
                f'{"diset" if k.lantai_diset else "BELUM diset"} (sudut {k.sudut_x_deg:+.2f}°) | '
                f'r bola {k.r_bola * 100:.2f} cm')
        if self._kal_terkunci():
            teks += ' | 🔒 terkunci (otomatis): "Hapus data" untuk mengatur ulang'
        self.lbl_kal.config(text=teks, foreground='#0a5' if lengkap else '#a60')

    def _ringkas_kalibrasi(self, dorong_panjang=True):
        """`self.kal` adalah sumber kebenaran: dorong nilainya ke isian formulir + label."""
        k = self.kal
        self.var['px_per_m'].set(f'{k.px_per_m:.3f}')
        self.var['sudut_x'].set(f'{round(k.sudut_x_deg, 2) + 0.0:.2f}')
        if dorong_panjang and k.panjang_ref > 0:
            self.var['panjang_acuan'].set(f'{k.panjang_ref * 100:g}')
        self._label_kalibrasi()

    def _segarkan_pratinjau(self):
        if self.frame_akhir is None or self.pratinjau_aktif:
            return
        if self.sumber is not None and self.sumber.bisa_lompat():
            self._tampil_manual(gambar_saja=True)
        else:
            self._tampil(self.frame_akhir)

    def _hitung_ulang_data(self):
        """Hitung ulang x,y (meter) dari piksel mentah dengan kalibrasi terbaru."""
        if self.titik_manual:
            self._manual_ke_buf()
        elif self._buf_dari_piksel() and len(self._buf['t']) >= 2:
            b = self._buf
            xm, ym = self.kal.ke_meter(np.array(b['cx'], float), np.array(b['cy'], float))
            b['x'], b['y'] = list(xm), list(ym)
            self._perbarui_sinyal(sementara=True)

    def _hitung_ulang_langsung(self):
        """Saat sumbu/penggaris diseret pada mode MANUAL, grafik ikut berubah (seperti
        Tracker); dibatasi ~5 kali per detik supaya tetap ringan."""
        if self.cara != 'manual' or not (self.titik_manual or self._buf_dari_piksel()):
            return
        t = time.time()
        if t - self._t_hitung < 0.2:
            return
        self._t_hitung = t
        self._hitung_ulang_data()

    def _selesai_ubah_kal(self, dorong_panjang=True):
        """Dipanggil setelah penggaris / sumbu berubah: sinkronkan formulir, hitung ulang
        data dari piksel mentah, dan gambar ulang pratinjau."""
        self._simpan_kal_frame()
        self._ringkas_kalibrasi(dorong_panjang)
        self._hitung_ulang_data()
        self._segarkan_pratinjau()
        self._gambar_overlay()

    def _terapkan_kalibrasi_ke_data(self, hanya_bila_berubah=False):
        """Pakai nilai yang DIKETIK (skala, sudut sumbu x, jari-jari bola) lalu hitung ulang data."""
        self._pra_ubah_kal()
        berubah, ditolak = self._sinkron_kal_dari_form()
        if ditolak:
            self._ringkas_kalibrasi()
            self._status('Mode otomatis: skala & sudut sumbu terkunci sesudah data diambil. '
                         'Tekan "Hapus data" untuk mengatur ulang.', '#a60')
        if hanya_bila_berubah and not berubah:
            return
        self._selesai_ubah_kal()

    # ---------- penggaris & sumbu x–y yang diseret pada pratinjau ----------
    R_HANDLE = 12            # jari-jari tangkap handle (piksel kanvas)
    PANJANG_SUMBU = 110      # panjang panah sumbu x / y pada kanvas (piksel kanvas)

    def _kal_terkunci(self):
        """Otomatis / semi: penggaris & sumbu hanya boleh diatur SEBELUM data diambil
        (terkunci selama dan sesudah perekaman). Manual: boleh kapan saja, termasuk
        SESUDAH titik ditandai; data dihitung ulang dari piksel mentah."""
        if self.cara == 'manual' or self.titik_manual:
            return False                  # tracking manual (klik titik) -> tidak pernah dikunci
        return bool(self.merekam or self._buf_dari_piksel())

    def _kal_bisa_diubah(self):
        if not self._kal_terkunci():
            self._pra_ubah_kal()
            return True
        self._status('Mode otomatis: penggaris & sumbu x–y terkunci selama/sesudah data diambil. '
                     'Tekan "Hapus data", lalu atur ulang sebelum merekam.', '#a60')
        return False

    def _ke_kanvas(self, px, py):
        s = self._skala_pratinjau if self._skala_pratinjau > 0 else 1.0
        gx, gy = getattr(self, '_geser_gambar', (0.0, 0.0))
        return px * s + gx, py * s + gy

    def _px_asli_f(self, ev):
        """Koordinat piksel gambar (float), dijepit ke dalam gambar."""
        s = self._skala_pratinjau if self._skala_pratinjau > 0 else 1.0
        dx, dy = getattr(self, '_geser_gambar', (0.0, 0.0))
        return self._jepit((ev.x - dx) / s, (ev.y - dy) / s)

    def _jepit(self, fx, fy):
        if self.frame_akhir is not None:
            h, w = self.frame_akhir.shape[:2]
            fx = min(max(fx, 0.0), w - 1.0)
            fy = min(max(fy, 0.0), h - 1.0)
        return fx, fy

    def _handle_x_kanvas(self):
        ox, oy = self._ke_kanvas(*self.kal.origin)
        sx, sy = self.kal.arah_samping_px()
        return ox + self.PANJANG_SUMBU * sx, oy + self.PANJANG_SUMBU * sy

    def _panjang_acuan_m(self):
        return max(self._f('panjang_acuan', 0.0), 0.0) / 100.0

    def _atur_kursor_klik(self):
        m = self.var['mode_klik'].get()
        self.kanvas_prev.config(cursor='crosshair' if m in ('penggaris', 'sumbu') else '')

    def _mode_klik_berubah(self):
        self._atur_kursor_klik()
        if self.var['mode_klik'].get() == 'tandai':
            self._pindah_ke_manual()
        petunjuk = {
            'penggaris': 'Penggaris: tekan di titik 0, SERET ke ujung penggaris, lepas. Ujung & badan garis '
                         'bisa diseret lagi; panjang nyata diisi pada "Panjang acuan" (cm).',
            'sumbu': 'Sumbu x–y: seret ⊕ (asal) untuk menggeser, seret ● (ujung panah x) untuk memutar; '
                     'atau tekan-seret sepanjang lantai untuk menggambar sumbu baru. Sumbu x = lantai (y = 0).'}
        m = self.var['mode_klik'].get()
        if m in petunjuk:
            self._status(petunjuk[m], '#1a4f8a')

    def _hover_kal(self, ev, mode):
        kursor = 'crosshair'
        m, r, k = (ev.x, ev.y), self.R_HANDLE, self.kal
        if mode == 'penggaris' and k.p0 is not None and k.p1 is not None:
            a, b = self._ke_kanvas(*k.p0), self._ke_kanvas(*k.p1)
            if np.hypot(m[0] - a[0], m[1] - a[1]) <= r or np.hypot(m[0] - b[0], m[1] - b[1]) <= r:
                kursor = 'hand2'
            elif _jarak_titik_segmen(m, a, b) <= 8:
                kursor = 'fleur'
        elif mode == 'sumbu' and k.lantai_diset:
            o, h = self._ke_kanvas(*k.origin), self._handle_x_kanvas()
            if np.hypot(m[0] - h[0], m[1] - h[1]) <= r:
                kursor = 'hand2'
            elif np.hypot(m[0] - o[0], m[1] - o[1]) <= r:
                kursor = 'fleur'
        self.kanvas_prev.config(cursor=kursor)

    # ----- penggaris -----
    def _mulai_seret_penggaris(self, ev):
        if not self._kal_bisa_diubah():
            return
        k = self.kal
        m = (ev.x, ev.y)
        fx, fy = self._px_asli_f(ev)
        self._seret = None
        if k.p0 is not None and k.p1 is not None:
            a, b = self._ke_kanvas(*k.p0), self._ke_kanvas(*k.p1)
            if np.hypot(m[0] - a[0], m[1] - a[1]) <= self.R_HANDLE:
                self._seret = {'jenis': 'p0'}
            elif np.hypot(m[0] - b[0], m[1] - b[1]) <= self.R_HANDLE:
                self._seret = {'jenis': 'p1'}
            elif _jarak_titik_segmen(m, a, b) <= 8:
                self._seret = {'jenis': 'tubuh', 'awal': (fx, fy), 'p0': k.p0, 'p1': k.p1}
        if self._seret is None:                    # penggaris BARU: tekan = titik 0, seret = ujung
            k.p0 = (fx, fy)
            k.p1 = (fx, fy)
            self._seret = {'jenis': 'baru'}
        self._gambar_overlay()

    def _seret_penggaris(self, ev):
        s, k = self._seret, self.kal
        fx, fy = self._px_asli_f(ev)
        j = s['jenis']
        if j == 'p0':
            k.p0 = (fx, fy)
        elif j in ('p1', 'baru'):
            k.p1 = (fx, fy)
        else:                                      # geser seluruh penggaris
            h, w = self.frame_akhir.shape[:2]
            xs, ys = (s['p0'][0], s['p1'][0]), (s['p0'][1], s['p1'][1])
            dx = min(max(fx - s['awal'][0], -min(xs)), (w - 1.0) - max(xs))
            dy = min(max(fy - s['awal'][1], -min(ys)), (h - 1.0) - max(ys))
            k.p0 = (s['p0'][0] + dx, s['p0'][1] + dy)
            k.p1 = (s['p1'][0] + dx, s['p1'][1] + dy)
        L = self._panjang_acuan_m()
        if L > 0:                                  # skala ikut berubah saat diseret
            k.panjang_ref = L
            try:
                k.hitung_skala()
            except ValueError:
                pass
        self._simpan_kal_frame()
        self._ringkas_kalibrasi(dorong_panjang=False)
        self._hitung_ulang_langsung()
        self._gambar_overlay()

    def _lepas_penggaris(self, ev):
        s, self._seret = self._seret, None
        k = self.kal
        k.panjang_ref = self._panjang_acuan_m()
        try:
            k.hitung_skala()
        except ValueError as exc:
            if s['jenis'] == 'baru' and k.panjang_px() < 5:      # hanya klik, bukan seretan
                k.p0 = k.p1 = None
                pesan = 'Penggaris dibatalkan — SERET dari titik 0 ke ujung penggaris.'
            else:
                pesan = str(exc)
            self._selesai_ubah_kal()
            self._status(pesan, '#a60')
            return
        self._selesai_ubah_kal()
        lanjut = ('' if k.lantai_diset else
                  ' Berikutnya: pilih klik "Sumbu x–y" untuk menetapkan lantai (sumbu x).')
        self._status(f'Penggaris: {k.panjang_px():.1f} px = {k.panjang_ref * 100:g} cm → '
                     f'{k.px_per_m:.1f} px/m.{lanjut}')

    def _ubah_panjang_acuan(self, _=None):
        """Panjang acuan diubah/diketik: skala dihitung ulang dari penggaris yang sudah ada."""
        k = self.kal
        L = self._panjang_acuan_m()
        if L <= 0 or abs(L - k.panjang_ref) < 1e-9 or k.p0 is None or k.p1 is None:
            return
        if not self._kal_bisa_diubah():
            self.var['panjang_acuan'].set(f'{k.panjang_ref * 100:g}')
            return
        k.panjang_ref = L
        try:
            k.hitung_skala()
        except ValueError:
            return
        self._selesai_ubah_kal(dorong_panjang=False)

    def _tukar_penggaris(self):
        k = self.kal
        if k.p0 is None or k.p1 is None:
            messagebox.showinfo('Penggaris', 'Gambar penggaris dahulu: pilih klik "Penggaris" lalu seret.')
            return
        if not self._kal_bisa_diubah():
            return
        k.tukar_penggaris()
        self._selesai_ubah_kal()
        self._status('Titik 0 dan ujung penggaris ditukar.')

    # ----- sumbu x–y (x = lantai) -----
    def _mulai_seret_sumbu(self, ev):
        if not self._kal_bisa_diubah():
            return
        k = self.kal
        m = (ev.x, ev.y)
        fx, fy = self._px_asli_f(ev)
        self._seret = None
        if k.lantai_diset:
            hx, hy = self._handle_x_kanvas()
            ox, oy = self._ke_kanvas(*k.origin)
            if np.hypot(m[0] - hx, m[1] - hy) <= self.R_HANDLE:
                self._seret = {'jenis': 'sudut'}
            elif np.hypot(m[0] - ox, m[1] - oy) <= self.R_HANDLE:
                self._seret = {'jenis': 'asal', 'off': (k.origin[0] - fx, k.origin[1] - fy)}
        if self._seret is None:                    # sumbu BARU: tekan = asal, seret = arah lantai
            lama = (k.origin, k.sudut_x_deg, k.lantai_diset)
            k.origin = (fx, fy)
            self._ikut_penggaris(lama)
            self._seret = {'jenis': 'baru'}
        self._gambar_overlay()

    def _seret_sumbu(self, ev):
        s, k = self._seret, self.kal
        lama = (k.origin, k.sudut_x_deg, k.lantai_diset)
        if s['jenis'] == 'asal':
            fx, fy = self._px_asli_f(ev)
            k.origin = self._jepit(fx + s['off'][0], fy + s['off'][1])
        else:                                      # 'sudut' / 'baru': putar sumbu x mengikuti kursor
            ox, oy = self._ke_kanvas(*k.origin)
            if np.hypot(ev.x - ox, ev.y - oy) >= (6 if s['jenis'] == 'baru' else 3):
                k.sudut_x_deg = _sudut_x_dari_arah(ev.x - ox, ev.y - oy)
        self._ikut_penggaris(lama)
        self._simpan_kal_frame()
        self._ringkas_kalibrasi(dorong_panjang=False)
        self._hitung_ulang_langsung()
        self._gambar_overlay()

    def _lepas_sumbu(self, ev):
        self._seret = None
        k = self.kal
        self._selesai_ubah_kal()
        lanjut = ('' if k.skala_diset else
                  ' Berikutnya: pilih klik "Penggaris" lalu seret untuk menetapkan skala.')
        self._status(f'Sumbu x = lantai: asal ({k.origin[0]:.0f}, {k.origin[1]:.0f}) px | '
                     f'sudut {k.sudut_x_deg:+.2f}°.{lanjut}')
        if self.cara == 'manual' and self.var['alur_frame'].get() \
                and self.manual_i in self.titik_manual:
            self._status(f'Frame {self.manual_i}: sumbu & penggaris tersimpan. '
                         'Tekan tombol "Lanjut ke frame berikutnya ▶" bila sudah selesai.', '#0a5')

    def _sumbu_dari_penggaris(self):
        if not self._kal_bisa_diubah():
            return
        try:
            th = self.kal.sumbu_dari_penggaris()
        except ValueError as exc:
            messagebox.showinfo('Sumbu x–y', str(exc))
            return
        self._selesai_ubah_kal()
        self._status(f'Sumbu x–y dipasang mengikuti penggaris: asal = titik 0, sudut sumbu x {th:+.2f}°. '
                     'Seret ⊕ / ● bila perlu menyetel.')

    def _datarkan_sumbu(self):
        if not self._kal_bisa_diubah():
            return
        self.kal.sudut_x_deg = 0.0
        self._selesai_ubah_kal()
        self._status('Sumbu x (lantai) didatarkan: sudut 0°.')

    def _hapus_kalibrasi(self):
        if not self._kal_bisa_diubah():
            return
        k = self.kal
        if not (k.skala_diset or k.lantai_diset or k.p0 is not None):
            return
        if not messagebox.askyesno('Hapus kalibrasi', 'Hapus penggaris dan sumbu x–y?'):
            return
        self.kal = ak.Kalibrasi(px_per_m=1000.0)
        self.kal_frame = {}
        self.var['px_per_m'].set('1000.0')
        self.var['sudut_x'].set('0.0')
        self._sinkron_kal_dari_form()              # jari-jari bola tetap dari formulir
        if self.titik_manual:                      # titik manual: lantai darurat = titik pertama
            _, cx, cy = self.titik_manual[min(self.titik_manual)]
            self.kal.origin_sementara(cx, cy)
        self._selesai_ubah_kal()
        self._status('Penggaris & sumbu x–y dihapus.')

    # ----- lapisan gambar pada kanvas -----
    def _teks_ov(self, x, y, teks, warna, anchor='w'):
        c = self.kanvas_prev
        huruf = ('Segoe UI', 9, 'bold')
        c.create_text(x + 1, y + 1, text=teks, fill='#000000', anchor=anchor, font=huruf, tags='overlay')
        c.create_text(x, y, text=teks, fill=warna, anchor=anchor, font=huruf, tags='overlay')

    def _gambar_overlay(self):
        """Gambar penggaris dan sumbu x–y di atas pratinjau. SUMBU x = LANTAI (y = 0)."""
        c = self.kanvas_prev
        c.delete('overlay')
        if self.frame_akhir is None:
            return
        k = self.kal
        W, H = max(c.winfo_width(), 50), max(c.winfo_height(), 50)
        S = float(self.PANJANG_SUMBU)
        if k.lantai_diset:
            ox, oy = self._ke_kanvas(*k.origin)
            sx, sy = k.arah_samping_px()                 # arah sumbu x (lantai)
            ax, ay = k.arah_atas_px()                    # arah sumbu y (atas)
            L = 4.0 * (W + H)
            c.create_line(ox - L * sx, oy - L * sy, ox + L * sx, oy + L * sy,
                          fill='#22dd22', width=2, tags='overlay')           # lantai = sumbu x
            if k.r_bola > 0 and k.skala_diset:                               # level pusat saat kontak
                hh = k.r_bola * k.px_per_m * (self._skala_pratinjau or 1.0)
                bx, by = ox + hh * ax, oy + hh * ay
                c.create_line(bx - L * sx, by - L * sy, bx + L * sx, by + L * sy,
                              fill='#ffa733', dash=(4, 4), tags='overlay')
            c.create_line(ox, oy, ox + S * ax, oy + S * ay, fill='#ff9f1a', width=3,
                          arrow='last', tags='overlay')                       # sumbu y
            self._teks_ov(ox + (S + 14) * ax, oy + (S + 14) * ay, 'y', '#ff9f1a', 'center')
            hx, hy = ox + S * sx, oy + S * sy
            c.create_line(ox, oy, hx, hy, fill='#22dd22', width=3, arrow='last', tags='overlay')
            c.create_oval(hx - 6, hy - 6, hx + 6, hy + 6, fill='#22dd22', outline='white',
                          width=2, tags='overlay')                            # handle putar sumbu x
            self._teks_ov(hx + 16 * sx, hy + 16 * sy, 'x', '#22dd22', 'center')
            lx = min(max(ox + (S + 40) * sx - 14 * ax, 4), W - 160)
            ly = min(max(oy + (S + 40) * sy - 14 * ay, 10), H - 10)
            self._teks_ov(lx, ly, 'sumbu x = lantai (y = 0)', '#22dd22')
            c.create_oval(ox - 6, oy - 6, ox + 6, oy + 6, fill='#ff4d4d', outline='white',
                          width=2, tags='overlay')                            # handle geser asal
        if k.p0 is not None and k.p1 is not None:
            a, b = self._ke_kanvas(*k.p0), self._ke_kanvas(*k.p1)
            c.create_line(a[0], a[1], b[0], b[1], fill='#00e5ff', width=3, tags='overlay')
            d = float(np.hypot(b[0] - a[0], b[1] - a[1]))
            if d > 1:
                nx, ny = -(b[1] - a[1]) / d, (b[0] - a[0]) / d
                for p in (a, b):
                    c.create_line(p[0] - 9 * nx, p[1] - 9 * ny, p[0] + 9 * nx, p[1] + 9 * ny,
                                  fill='#00e5ff', width=3, tags='overlay')
            for p in (a, b):
                c.create_oval(p[0] - 6, p[1] - 6, p[0] + 6, p[1] + 6, fill='#00e5ff',
                              outline='white', width=2, tags='overlay')
            self._teks_ov(a[0] + 10, a[1] + 14, '0', '#00e5ff')
            self._teks_ov(b[0] + 10, b[1] - 12,
                          f'{k.panjang_ref * 100:g} cm' if k.panjang_ref > 0 else '? cm', '#00e5ff')
        if self._kal_terkunci() and (k.lantai_diset or k.p0 is not None):
            self._teks_ov(8, H - 12, '🔒 kalibrasi terkunci (otomatis) — "Hapus data" untuk mengubah',
                          '#dddddd')

    # ---------- olah sinyal ----------
    def _perkirakan_lantai(self, s, t=None):
        """Perkiraan level lantai (referensi nol) bila lantai BELUM dikalibrasi dengan
        penggaris/sumbu.

        Memakai NILAI TERENDAH data, bukan persentil. Persentil-2 dari data yang terus
        bertambah (saat perekaman) berubah setiap pembaruan: titik lembah pantulan hanya
        beberapa sampel (< 2 % data), sehingga nol grafik bergeser naik dan lembah yang
        berada di bawah persentil terpotong menjadi 0 (lembah \"hilang\"). Nilai terendah
        hanya bisa turun, jadi begitu bola menyentuh lantai nol tidak lagi bergeser naik.
        Satu titik salah-deteksi di bawah lantai (lonjakan tracker) dikecualikan lewat
        cf.buang_lonjakan sehingga tidak ikut menentukan referensi nol."""
        s = np.asarray(s, dtype=float)
        if len(s) < 5:
            return float(np.min(s))
        terendah = float(np.min(s))
        if t is not None and len(t) == len(s) and len(s) >= 7:
            try:
                ok = cf.buang_lonjakan(np.asarray(t, dtype=float), s - terendah)
                if np.any(ok):
                    return float(np.min(s[ok]))
            except Exception:
                pass
        return terendah

    def _sinyal_dari_xy(self, t, x, y):
        pilih = self.var['sumbu'].get()
        if pilih.startswith('X'):
            s = np.asarray(x)
        elif pilih.startswith('Y'):
            s = np.asarray(y)
        else:
            s = np.asarray(x) if np.var(x) > np.var(y) else np.asarray(y)
        ref = self.var['referensi'].get()
        kal_ok = self._kal_berlaku()
        pakai_persentil = ref.startswith('minimum') or (ref.startswith('kalibrasi') and not kal_ok)
        if ref == 'rata-rata':
            s = s - float(np.mean(s))
        elif pakai_persentil:
            s = np.clip(s - self._perkirakan_lantai(s, t), 0.0, None)
        # ref 'kalibrasi' dengan lantai terkalibrasi: y sudah relatif terhadap lantai -> TIDAK digeser
        jd = int(self._f('haluskan', 0) or 0)
        if jd > 1:
            s = cf.haluskan(s, jd)
            if pakai_persentil:
                s = np.clip(s, 0.0, None)
        return s

    def _perbarui_sinyal(self, sementara=False):
        if len(self._buf['t']) < 2:
            return
        t = np.array(self._buf['t'], dtype=float)
        x = np.array(self._buf['x'], dtype=float)
        y = np.array(self._buf['y'], dtype=float)
        kal_ok = self._kal_berlaku()
        # urutkan waktu & buang stempel waktu kembar (frame terduplikasi)
        urut = np.argsort(t, kind='stable')
        t, x, y = t[urut], x[urut], y[urut]
        t, iu = np.unique(t, return_index=True)
        x, y = x[iu], y[iu]
        if len(t) < 2:
            return
        s = self._sinyal_dari_xy(t, x, y)
        catatan = []
        if not kal_ok and self.var['referensi'].get().startswith('kalibrasi'):
            catatan.append('sumbu x (lantai) BELUM diatur (lantai diperkirakan dari titik terendah data)')
        # Tanpa kalibrasi: pastikan lembah tidak negatif. DENGAN kalibrasi lantai tidak boleh
        # digeser — kalau digeser, satu titik noise di bawah lantai merusak seluruh kalibrasi.
        if not kal_ok:
            lantai = self._perkirakan_lantai(s, t)
            if lantai < 0 or (np.min(s) < -1e-6):
                s = s - lantai
        # 1) buang lonjakan satu-frame
        n_buang = 0
        if self.var['buang_lonjakan'].get() and len(s) >= 8:
            ok = cf.buang_lonjakan(t, s)
            n_buang = int((~ok).sum())
            if n_buang:
                t, s, x, y = t[ok], s[ok], x[ok], y[ok]
        n_bawah = int(np.sum(s < -0.005))
        s_min = float(np.min(s))
        s = np.clip(s, 0.0, None)
        # 2) pangkas sebelum bola dilepas, lalu t = 0 saat lepas
        n_pangkas = 0
        if self.var['pangkas_lepas'].get() and len(s) >= 8:
            # Titik lepas dihitung dari seluruh data, sehingga berubah-ubah selama data masih
            # bertambah (grafik 'bergeser sendiri' ±1 frame, atau melompat beberapa detik saat
            # bola mulai jatuh). Maka selama pembaruan langsung (sementara) titik lepas yang
            # sudah ditetapkan dipakai ulang, dan dihitung ulang hanya pada pembaruan akhir.
            ref = self._pangkas_ref
            if sementara:
                if ref and abs(ref['t_awal'] - float(t[0])) < 1e-9 and ref['t_lepas'] <= float(t[-1]):
                    i0 = int(np.searchsorted(t, ref['t_lepas'] - 1e-9))
                else:
                    i0 = 0
            else:
                i0 = cf.indeks_lepas(s)
                self._pangkas_ref = ({'t_awal': float(t[0]), 't_lepas': float(t[i0])}
                                     if i0 > 0 else None)
            if i0 > 0 and len(s) - i0 >= 8:
                t, s, x, y = t[i0:], s[i0:], x[i0:], y[i0:]
                n_pangkas = i0
        t = t - t[0]
        y0 = float(np.median(s[:3]))
        if kal_ok:
            if n_bawah:
                catatan.append(f'{n_bawah} titik di bawah lantai (terendah {s_min * 100:.1f} cm): '
                               'periksa letak sumbu x (lantai) / jari-jari bola')
            elif s_min > 0.02:
                catatan.append(f'titik terendah masih {s_min * 100:.1f} cm di atas lantai: isi '
                               '"Jari-jari bola" atau periksa letak sumbu x (lantai)')
        self.data = {'t': t, 's': s, 'x': x, 'y': y,
                     'sumber': self.var['sumber'].get(), 'lantai_kal': kal_ok}
        if self.hasil:                       # e/c/v0 hasil analisis lama tidak berlaku untuk data baru
            self.hasil = {}
            self._ringkas_koreksi()
        self.fig2.clear()
        ax = self.fig2.add_subplot(2, 1, 1)
        ax.plot(t, s, '.-', ms=2, lw=0.9, color='#1f77b4')
        ax.axhline(0, color='k', lw=0.8, ls='--', alpha=0.5, label='lantai y=0')
        ax.legend(fontsize=8)
        ax.set_title(f'Sinyal Eksperimen — tinggi awal ≈ {y0 * 100:.1f} cm', fontsize=10)
        ax.set_ylabel(LABEL_SINYAL); ax.grid(True)
        ax.set_ylim(bottom=-0.02 * (np.max(s) - np.min(s) + 1e-9))  # sedikit ruang di bawah 0
        ax2 = self.fig2.add_subplot(2, 1, 2)
        ax2.plot(t, x * 100, '-', lw=0.8, label='x (cm)')
        ax2.plot(t, y * 100, '-', lw=0.8, label='y (cm)')
        ax2.set_xlabel('Waktu (s)'); ax2.legend(fontsize=8); ax2.grid(True)
        self.fig2.tight_layout(); self.kanvas2.draw()
        pesan = (f'Data eksperimen siap: {len(t)} titik, durasi {t[-1]:.2f} s, '
                 f'tinggi awal ≈ {y0 * 100:.1f} cm.')
        if n_pangkas:
            pesan += f' {n_pangkas} titik sebelum lepas dipangkas.'
        if n_buang:
            pesan += f' {n_buang} lonjakan dibuang.'
        if catatan:
            pesan += ' ⚠ ' + '; '.join(catatan)
        self._pesan_sinyal = (pesan, '#a60' if catatan else '#0a5')
        if not sementara:
            self._label_kalibrasi()
            self._gambar_overlay()
            self._status(*self._pesan_sinyal)

    def _data_sintetis(self):
        """Data uji coba bila kamera/berkas belum tersedia."""
        p = self._parameter()
        model = cf.buat_model(KUNCI_SISTEM, p)
        t_akhir = self._f('t_akhir', 6.0)
        T, Y = cf.simulasi(model, t_akhir, min(self._f('h', 0.005), 0.002))
        fps = max(self._f('fps', 60.0), 1.0)
        t = np.arange(0, T[-1], 1.0 / fps)
        s = np.interp(t, T, Y[:, 0])
        rng = np.random.default_rng(7)
        derau = 0.01 * (np.max(s) - np.min(s) + 1e-9)
        s = s + rng.normal(0, derau, len(s))
        self._buf = {'t': list(t), 'x': list(np.zeros_like(t)), 'y': list(s)}
        simpan = self.var['sumbu'].get(), self.var['referensi'].get()
        self.var['sumbu'].set('Y (vertikal)')
        self.var['referensi'].set('mentah')
        self._perbarui_sinyal()
        self.var['sumbu'].set(simpan[0]); self.var['referensi'].set(simpan[1])
        self.data['sumber'] = 'sintetis (uji coba)'
        self._status(f'Data sintetis dibuat ({len(t)} titik, derau 1%).')

    def _simpan_data(self):
        if self.data['t'] is None:
            messagebox.showwarning('Data', 'Belum ada data.'); return
        nama = filedialog.asksaveasfilename(defaultextension='.csv', filetypes=[('CSV', '*.csv')])
        if not nama:
            return
        ak.simpan_csv(nama, self.data['t'], self.data['x'], self.data['y'],
                      header_tambahan=[f"sistem={NAMA_SISTEM}",
                                       f"px_per_m={self.kal.px_per_m}",
                                       f"sudut_sumbu_x_deg={self.kal.sudut_x_deg:.3f}",
                                       f"jari2_bola_m={self.kal.r_bola}",
                                       f"lantai_terkalibrasi={bool(self.data.get('lantai_kal'))}",
                                       f"sumber={self.data['sumber']}"])
        self._status(f'Data disimpan: {os.path.basename(nama)}')

    def _muat_data(self):
        nama = filedialog.askopenfilename(filetypes=[('CSV', '*.csv *.txt'), ('Semua', '*.*')])
        if not nama:
            return
        try:
            d = ak.muat_csv(nama)
            y = d['y'] if d['y'] is not None else d['x']
            x = d['x'] if d['y'] is not None else np.zeros_like(d['t'])
            self._buf = {'t': list(d['t']), 'x': list(x), 'y': list(y)}
            self._perbarui_sinyal()
            self.data['sumber'] = os.path.basename(nama)
            self._status(f"CSV dimuat: {os.path.basename(nama)} ({len(d['t'])} titik).")
        except Exception as exc:
            self._galat(exc)

    def _hapus_data(self):
        self._buf = self._buf_kosong()
        self._pangkas_ref = None
        self.puncak_manual = None
        self.puncak_tambah = []
        self.pl_manual = []
        self._sig_analisis = None
        self._r3 = None
        for kunci in ('edit_puncak', 'edit_puncak4', 'edit_pl'):
            if kunci in self.var:
                self.var[kunci].set(False)
        self._label_pl()
        self.titik_manual = {}
        self.kal_frame = {}
        self.kal.reset_sementara()
        self.data = {'t': None, 's': None, 'x': None, 'y': None, 'sumber': '-'}
        self.hasil = {}
        self.fig2.clear(); self.kanvas2.draw()
        self._label_kalibrasi()
        self._gambar_overlay()
        self._status('Data dihapus.')

    # =================================================================
    # TAB 3 — ANALISIS DATA
    # =================================================================
    def _tab_analisis(self):
        kiri = self._panel_gulir(self.tab3, lebar=370)
        kanan = ttk.Frame(self.tab3); kanan.pack(side='left', fill='both', expand=True)

        # ---------------- panel kiri: pengaturan, tombol, ringkasan ----------------
        pen = ttk.LabelFrame(kiri, text='Pengaturan analisis', padding=6)
        pen.pack(fill='x')
        ttk.Label(pen, text='Lantai (y)').grid(row=0, column=0, sticky='w')
        self.var['mode_lantai3'] = tk.StringVar(value='y = 0 (sesuai Tab 2)')
        ttk.Combobox(pen, textvariable=self.var['mode_lantai3'], state='readonly', width=22,
                     values=['y = 0 (sesuai Tab 2)', 'otomatis dari data']
                     ).grid(row=0, column=1, columnspan=2, sticky='w', padx=4, pady=2)
        self._baris(pen, 1, 'y lantai manual (opsional)', 'lantai3', '', 'm')
        self._baris(pen, 2, 'Prominence puncak', 'prom3', 0.02, '(0–1)')
        ttk.Label(pen, text='Sumber: data eksperimen / sintetis aktif (Tab 2)',
                  foreground='#666').grid(row=3, column=0, columnspan=3, sticky='w', pady=(4, 0))

        ttk.Button(kiri, text='▶ Analisis Pantulan (puncak, tumbukan & e)', width=46,
                   command=self._analisis_pantulan).pack(pady=(8, 3))
        pl = ttk.LabelFrame(kiri, text='Puncak & lembah MANUAL (klik berurutan)', padding=6)
        pl.pack(fill='x', pady=(6, 2))
        self.var['edit_pl'] = tk.BooleanVar(value=False)
        ttk.Checkbutton(pl, text='✎ Klik: puncak → lembah → puncak → lembah …',
                        variable=self.var['edit_pl'],
                        command=self._ganti_mode_pl).pack(anchor='w')
        bar_m = ttk.Frame(pl); bar_m.pack(anchor='w', pady=(3, 0))
        ttk.Label(bar_m, text='Klik pertama =').pack(side='left')
        self.var['pl_mulai'] = tk.StringVar(value='puncak')
        ttk.Combobox(bar_m, textvariable=self.var['pl_mulai'], state='readonly', width=8,
                     values=['puncak', 'lembah']).pack(side='left', padx=4)
        ttk.Label(pl, wraplength=330, foreground='#666', justify='left',
                  text='Klik kiri di dekat puncak / lembah — jenisnya BERGANTIAN otomatis dan titik '
                       'menempel ke puncak/lembah data terdekat. Klik pertama di titik lepas = P0. '
                       'Klik kanan = urungkan titik terakhir. Setelah ≥ 2 puncak pantulan, hanya puncak '
                       'yang Anda klik yang dihitung (puncak otomatis diabaikan). Lembah dipakai Tab 4 '
                       'untuk mengunci waktu model.').pack(anchor='w', pady=(2, 2))
        self.lbl_pl = ttk.Label(pl, text='', foreground='#1a4f8a', wraplength=330, justify='left')
        self.lbl_pl.pack(anchor='w')
        bar_pl = ttk.Frame(pl); bar_pl.pack(anchor='w', pady=(3, 0))
        ttk.Button(bar_pl, text='↶ Urungkan titik terakhir',
                   command=self._urungkan_pl).pack(side='left')
        ttk.Button(bar_pl, text='🗑 Hapus semua klik',
                   command=self._hapus_pl).pack(side='left', padx=4)
        self._label_pl()

        ed = ttk.LabelFrame(kiri, text='Puncak', padding=6)
        ed.pack(fill='x', pady=(6, 2))
        self.var['edit_puncak'] = tk.BooleanVar(value=False)
        ttk.Checkbutton(ed, text='✎ Tambah puncak manual (klik pada grafik atas)',
                        variable=self.var['edit_puncak'],
                        command=self._ganti_mode_edit).pack(anchor='w')
        ttk.Label(ed, wraplength=330, foreground='#666', justify='left',
                  text='Klik kiri pada grafik = titik itu langsung dihitung sebagai puncak '
                       '(menempel ke titik tertinggi di sekitarnya). Puncak yang lain TIDAK '
                       'berubah atau hilang. Klik kanan dekat lingkaran merah muda = hapus '
                       'puncak tambahan itu.'
                  ).pack(anchor='w', pady=(2, 4))
        bar_p = ttk.Frame(ed); bar_p.pack(anchor='w')
        ttk.Button(bar_p, text='↶ Urungkan puncak terakhir',
                   command=self._urungkan_puncak).pack(side='left')
        ttk.Button(bar_p, text='↺ Kembali ke puncak otomatis',
                   command=self._puncak_otomatis).pack(side='left', padx=4)
        ttk.Button(kiri, text='➡ Terapkan e ke Validasi Model RK4 (Tab 4)', width=46,
                   command=self._terapkan_koreksi).pack(pady=3)
        ttk.Button(kiri, text='Salin ringkasan ke clipboard', width=46,
                   command=self._salin_hasil).pack(pady=3)
        ttk.Separator(kiri).pack(fill='x', pady=6)
        ttk.Label(kiri, text='Ringkasan hasil', font=('Segoe UI', 9, 'bold')).pack(anchor='w')
        self.txt3 = tk.Text(kiri, width=48, height=26, font=('Consolas', 9), wrap='word')
        self.txt3.pack(fill='both', expand=True, pady=4)

        # ---------------- panel kanan: sub-tab Grafik / Tabel / Perhitungan ----------------
        self.lbl3 = ttk.Label(kanan, text='Belum ada analisis. Tekan "▶ Analisis Pantulan".',
                              foreground='#1a4f8a', font=('Segoe UI', 10, 'bold'),
                              padding=(8, 4), justify='left')
        self.lbl3.pack(fill='x')
        self.lbl3.bind('<Configure>',
                       lambda ev: self.lbl3.config(wraplength=max(ev.width - 16, 200)))

        self.nb3 = ttk.Notebook(kanan)
        self.nb3.pack(fill='both', expand=True, padx=4, pady=(0, 4))
        t_graf = ttk.Frame(self.nb3); self.nb3.add(t_graf, text=' Grafik ')
        t_tab = ttk.Frame(self.nb3);  self.nb3.add(t_tab, text=' Tabel Puncak & Tumbukan ')
        t_rum = ttk.Frame(self.nb3);  self.nb3.add(t_rum, text=' Perhitungan & Rumus ')

        # -- grafik
        self.fig3, self.kanvas3 = self._kanvas(t_graf)
        self.ax3_utama = None
        self.kanvas3.mpl_connect('button_press_event', self._klik_grafik3)

        # -- tabel
        ttk.Label(t_tab, text='Puncak (hasil fit parabola)',
                  font=('Segoe UI', 9, 'bold')).pack(anchor='w', padx=4, pady=(6, 0))
        self.tv_puncak = self._tabel3(t_tab, (
            ('no', 'No', 45), ('jenis', 'Jenis', 85), ('t', 't puncak (s)', 95),
            ('h', 'h dari fit (m)', 105), ('h_sampel', 'h sampel mentah (m)', 130),
            ('n_fit', 'titik fit', 65), ('g', 'g fit (m/s²)', 90)))
        ttk.Label(t_tab, text='Tumbukan dengan lantai (koefisien restitusi e dengan tiga cara)',
                  font=('Segoe UI', 9, 'bold')).pack(anchor='w', padx=4, pady=(8, 0))
        self.tv_tumb = self._tabel3(t_tab, (
            ('no', 'No', 40), ('t', 't kontak (s)', 85), ('h1', 'h sebelum (m)', 90),
            ('h2', 'h sesudah (m)', 90), ('et', 'e (tinggi)', 75), ('v1', 'v sebelum', 75),
            ('v2', 'v sesudah', 75), ('ek', 'e (kecepatan)', 90), ('ew', 'e (waktu)', 75)))

        # -- perhitungan & rumus
        bar = ttk.Frame(t_rum); bar.pack(fill='x', padx=4, pady=4)
        ttk.Button(bar, text='Salin ke clipboard', command=self._salin_rincian).pack(side='left')
        ttk.Button(bar, text='Simpan TXT…', command=self._simpan_rincian).pack(side='left', padx=6)
        isi = ttk.Frame(t_rum); isi.pack(fill='both', expand=True)
        self.txt_rincian = tk.Text(isi, wrap='none', font=('Consolas', 10), padx=8, pady=6)
        sy = ttk.Scrollbar(isi, orient='vertical', command=self.txt_rincian.yview)
        sx = ttk.Scrollbar(isi, orient='horizontal', command=self.txt_rincian.xview)
        self.txt_rincian.configure(yscroll=sy.set, xscroll=sx.set)
        sy.pack(side='right', fill='y'); sx.pack(side='bottom', fill='x')
        self.txt_rincian.pack(fill='both', expand=True)

    def _tabel3(self, induk, spesifikasi):
        """Treeview dengan scrollbar; kolom menyesuaikan lebar panel."""
        wadah = ttk.Frame(induk); wadah.pack(fill='both', expand=True, padx=4, pady=2)
        tv = ttk.Treeview(wadah, columns=[k for k, _, _ in spesifikasi],
                          show='headings', height=8)
        for kod, teks, lebar in spesifikasi:
            tv.heading(kod, text=teks)
            tv.column(kod, width=lebar, minwidth=40, anchor='center')
        sb = ttk.Scrollbar(wadah, orient='vertical', command=tv.yview)
        tv.configure(yscroll=sb.set)
        tv.pack(side='left', fill='both', expand=True); sb.pack(side='left', fill='y')
        return tv

    def _tulis3(self, teks, bersih=False):
        if bersih:
            self.txt3.delete('1.0', 'end')
        self.txt3.insert('end', teks + '\n')
        self.txt3.see('end')

    def _cek_data(self):
        if self.data['t'] is None or len(self.data['t']) < 10:
            messagebox.showwarning('Data', 'Belum ada data eksperimen. '
                                            'Rekam di Tab 2 atau gunakan data sintetis.')
            return False
        return True

    def _tanda_data(self):
        d = self.data
        if d.get('t') is None or len(d['t']) < 10:
            return None
        t, y = np.asarray(d['t'], float), np.asarray(d['s'], float)
        return (len(t), round(float(t[-1]), 6), round(float(np.sum(y)), 6))

    def _tab_berganti(self, _ev=None):
        """Buka Tab 3 -> analisis langsung berjalan bila ada data baru (tanpa menekan tombol)."""
        try:
            if self.nb.select() != str(self.tab3):
                return
        except tk.TclError:
            return
        tanda = self._tanda_data()
        if tanda is not None and tanda != self._sig_analisis:
            self._analisis_pantulan(otomatis=True)

    def _atur_kursor(self):
        """Kursor silang pada grafik Tab 3 / Tab 4 selama salah satu mode klik aktif."""
        aktif3 = bool(self.var['edit_puncak'].get()) or bool(self.var['edit_pl'].get())
        aktif4 = bool(self.var['edit_puncak4'].get()) if 'edit_puncak4' in self.var else False
        for kanvas, aktif in ((self.kanvas3, aktif3), (getattr(self, 'kanvas4', None), aktif4)):
            if kanvas is None:
                continue
            try:
                kanvas.get_tk_widget().config(cursor='crosshair' if aktif else '')
            except tk.TclError:
                pass

    def _matikan_mode_klik(self, kecuali=None):
        for kunci in ('edit_puncak', 'edit_puncak4', 'edit_pl'):
            if kunci != kecuali and kunci in self.var:
                self.var[kunci].set(False)

    def _siapkan_puncak_manual(self):
        """Pastikan puncak otomatis sudah ada; puncak klik DITAMBAHKAN di atasnya."""
        if self._t_idx3 is None and self._tanda_data() is not None:
            self._analisis_pantulan(otomatis=True)

    def _ganti_mode_edit(self):
        aktif = self.var['edit_puncak'].get()
        if aktif:
            self._matikan_mode_klik(kecuali='edit_puncak')
            self._siapkan_puncak_manual()
        self._atur_kursor()
        if aktif:
            self.nb3.select(0)
            self._status('Mode tambah puncak: klik kiri pada grafik atas = tambah puncak (puncak lain tetap); '
                         'klik kanan dekat lingkaran merah muda = hapus puncak tambahan.',
                         '#1a4f8a')
        if self.hasil_pantulan_ada():
            self._analisis_pantulan(otomatis=True)

    def _ganti_mode_edit4(self):
        """Pilih puncak eksperimen secara manual langsung di grafik Tab 4."""
        aktif = self.var['edit_puncak4'].get()
        if aktif:
            if not self._cek_data():
                self.var['edit_puncak4'].set(False)
                return
            self._matikan_mode_klik(kecuali='edit_puncak4')
            self._siapkan_puncak_manual()
        self._atur_kursor()
        if aktif:
            self._status('Klik kiri pada grafik atas = tambah puncak (puncak lain tetap); '
                         'klik kanan dekat lingkaran merah muda = hapus puncak tambahan.', '#1a4f8a')
            self._analisis_pantulan(otomatis=True)
            self._validasi()

    def hasil_pantulan_ada(self):
        return self._sig_analisis is not None

    def _puncak_otomatis(self, dari_tab4=False):
        self.puncak_manual = None
        self.puncak_tambah = []
        self.pl_manual = []
        self.var['edit_puncak'].set(False)
        self.var['edit_pl'].set(False)
        self._label_pl()
        if 'edit_puncak4' in self.var:
            self.var['edit_puncak4'].set(False)
        self._atur_kursor()
        self._analisis_pantulan(otomatis=True)
        if dari_tab4 and self.data.get('t') is not None:
            self._validasi()

    def _zoom_aktif(self, kanvas):
        tb = getattr(kanvas, 'toolbar', None)                # abaikan klik saat zoom/geser aktif
        mode = getattr(getattr(tb, 'mode', ''), 'value', getattr(tb, 'mode', ''))
        return bool(mode)

    def _tol_puncak(self, t):
        return 2.0 * float(np.median(np.diff(t))) if len(t) > 1 else 0.0

    def _ubah_puncak_dari_klik(self, x, hapus=False):
        """Klik KIRI di waktu x (detik data): titik itu dihitung sebagai puncak TAMBAHAN —
        puncak lain tidak diubah / dihapus. Klik KANAN: hapus puncak tambahan terdekat
        (puncak otomatis tidak ikut terhapus)."""
        if self.data.get('t') is None:
            return False
        t = np.asarray(self.data['t'], float)
        dt = self._tol_puncak(t) / 2.0
        if hapus:
            if not self.puncak_tambah:
                self._status('Tidak ada puncak tambahan untuk dihapus.', '#a60')
                return False
            tol = max(0.03 * float(t[-1] - t[0]), 2.0 * dt)
            k = int(np.argmin([abs(v - x) for v in self.puncak_tambah]))
            if abs(self.puncak_tambah[k] - x) > tol:
                self._status('Klik kanan tepat di dekat lingkaran merah muda (puncak tambahan).', '#a60')
                return False
            self.puncak_tambah.pop(k)
            return True
        if self._t_idx3 is not None and len(self._t_idx3):
            if float(np.min(np.abs(np.asarray(self._t_idx3, float) - x))) <= 2.0 * dt:
                self._status('Titik itu sudah menjadi puncak.', '#a60')
                return False
        self.puncak_tambah.append(float(x))
        return True

    # ---------- puncak & lembah MANUAL: klik berurutan P - V - P - V ----------
    def _pl_valid(self):
        """Buang pilihan klik bila jumlah sampel data berubah (data baru)."""
        if self.pl_manual and (self.data.get('t') is None or len(self.data['t']) != self._pl_n):
            self.pl_manual = []
        return bool(self.pl_manual)

    def _pl_bernomor(self):
        if not self._pl_valid():
            return []
        return _nomori_pl(self.pl_manual, np.asarray(self.data['s'], float))

    def _pl_berikut(self):
        if self.pl_manual:
            return 'V' if self.pl_manual[-1][0] == 'P' else 'P'
        return 'V' if str(self.var['pl_mulai'].get()).startswith('lembah') else 'P'

    def _label_pl(self):
        if not hasattr(self, 'lbl_pl'):
            return
        b = self._pl_bernomor()
        urut = '  '.join(f'{j}{no}' for j, no, _, _ in b[-10:])
        awal = '… ' if len(b) > 10 else ''
        nama = 'PUNCAK' if self._pl_berikut() == 'P' else 'LEMBAH'
        self.lbl_pl.config(text=(f'Titik: {awal}{urut}\n' if b else 'Belum ada titik.\n')
                                + f'Klik berikutnya = {nama}.')

    def _ganti_mode_pl(self):
        aktif = self.var['edit_pl'].get()
        if aktif:
            if not self._cek_data():
                self.var['edit_pl'].set(False)
                return
            self._matikan_mode_klik(kecuali='edit_pl')
            self.puncak_tambah = []
            self._siapkan_puncak_manual()
            self.nb3.select(0)
            self._status('Klik berurutan pada grafik atas: puncak → lembah → puncak → lembah. '
                         'Klik kanan = urungkan.', '#1a4f8a')
        self._atur_kursor()
        self._label_pl()
        if self.hasil_pantulan_ada():
            self._analisis_pantulan(otomatis=True)

    def _klik_pl(self, ev):
        if ev.button == 3:
            self._urungkan_pl()
            return
        if ev.button != 1:
            return
        t = np.asarray(self.data['t'], float)
        s = np.asarray(self.data['s'], float)
        self._pl_valid()
        jenis = self._pl_berikut()
        hw = max(2, int(round(0.01 * len(t))))
        lo_min = self.pl_manual[-1][1] + 1 if self.pl_manual else 0
        idx = _snap_ekstrem(t, s, float(ev.xdata), jenis, hw, lo_min)
        nama = 'puncak' if jenis == 'P' else 'lembah'
        if idx is None:
            self._status(f'Klik {nama} di KANAN titik sebelumnya (urutan waktu harus maju).', '#a60')
            return
        if self.pl_manual:
            sebelum = float(s[self.pl_manual[-1][1]])
            if (jenis == 'P' and not s[idx] > sebelum) or (jenis == 'V' and not s[idx] < sebelum):
                self._status(f'Titik itu bukan {nama}: klik lebih dekat ke '
                             f'{"puncak" if jenis == "P" else "lembah"} berikutnya.', '#a60')
                return
        if not self.pl_manual:
            self._pl_n = len(t)
        self.pl_manual.append((jenis, int(idx), float(t[idx])))
        self._label_pl()
        self._analisis_pantulan(otomatis=True)

    def _urungkan_pl(self):
        if not self._pl_valid():
            self._status('Belum ada titik puncak/lembah untuk diurungkan.', '#a60')
            return
        self.pl_manual.pop()
        self._label_pl()
        if self.data.get('t') is not None:
            self._analisis_pantulan(otomatis=True)

    def _hapus_pl(self):
        self.pl_manual = []
        self._label_pl()
        if self.data.get('t') is not None and self.hasil_pantulan_ada():
            self._analisis_pantulan(otomatis=True)

    def _gambar_pl(self, ax, t, s):
        """Tandai puncak (lingkaran biru) & lembah (belah ketupat ungu) hasil klik."""
        b = self._pl_bernomor()
        if not b:
            return
        pp = [(tt, s[ix], f'P{no}') for j, no, tt, ix in b if j == 'P']
        vv = [(tt, s[ix], f'V{no}') for j, no, tt, ix in b if j == 'V']
        if pp:
            ax.plot([q[0] for q in pp], [q[1] for q in pp], 'o', mfc='none', mec='#1f4fd8',
                    ms=11, mew=1.6, label='puncak (klik)')
        if vv:
            ax.plot([q[0] for q in vv], [q[1] for q in vv], 'D', mfc='none', mec='#7b1fa2',
                    ms=9, mew=1.6, label='lembah (klik)')
        for tt, yy, lb in pp + vv:
            ax.annotate(lb, (tt, yy), textcoords='offset points',
                        xytext=(0, -14 if lb[0] == 'V' else 8), ha='center', fontsize=7,
                        color='#7b1fa2' if lb[0] == 'V' else '#1f4fd8')

    def _urungkan_puncak(self):
        if not self.puncak_tambah:
            self._status('Tidak ada puncak tambahan untuk diurungkan.', '#a60')
            return
        self.puncak_tambah.pop()
        if self.data.get('t') is not None:
            self._analisis_pantulan(otomatis=True)

    def _klik_grafik3(self, ev):
        if ev.button not in (1, 3) or ev.xdata is None:
            return
        if ev.inaxes is None or ev.inaxes is not self.ax3_utama:
            return
        if self._zoom_aktif(self.kanvas3):
            return
        if self.data.get('t') is None:
            return
        if self.var['edit_pl'].get():
            self._klik_pl(ev)
            return
        if not self.var['edit_puncak'].get():
            return
        if self._ubah_puncak_dari_klik(float(ev.xdata), hapus=(ev.button == 3)):
            self._analisis_pantulan(otomatis=True)

    def _klik_grafik4(self, ev):
        if not self.var['edit_puncak4'].get() or ev.button not in (1, 3) or ev.xdata is None:
            return
        if ev.inaxes is None or ev.inaxes is not self.ax4_utama:
            return
        if self._zoom_aktif(self.kanvas4) or self.data.get('t') is None:
            return
        if self._ubah_puncak_dari_klik(float(ev.xdata), hapus=(ev.button == 3)):
            self._analisis_pantulan(otomatis=True)
            self._validasi()

    def _kurva_model(self, T, Y, t0):
        """Potong kurva model mulai t0; bila model baru lepas sesudah t0, bola diam di y0."""
        m = T >= t0
        Tp, Yp = T[m], Y[m, 0]
        if len(T) and T[0] > t0:
            Tp = np.concatenate([[t0], Tp]); Yp = np.concatenate([[Y[0, 0]], Yp])
        return Tp, Yp

    def _analisis_pantulan(self, otomatis=False):
        """Analisis puncak tiap pantulan + tumbukan + koefisien restitusi e.
        Hitungan di core_fisika.analisis_pantulan; di sini hanya tampilan."""
        if otomatis and self._tanda_data() is None:
            return
        if not otomatis and not self._cek_data():
            return
        t = np.asarray(self.data['t'], float)
        s = np.asarray(self.data['s'], float)
        try:
            g = self._f('g', 9.81)
            yl_txt = str(self.var['lantai3'].get()).replace(',', '.').strip()
            if yl_txt:
                yl = float(yl_txt)
            elif (self.var['mode_lantai3'].get().startswith('y = 0')
                  and not self.var['referensi'].get().startswith(('mentah', 'rata-rata'))):
                yl = 0.0          # data Tab 2 sudah dirujuk ke lantai (kalibrasi / titik terendah)
            else:
                yl = None
            prom = self._f('prom3', 0.02)
            # puncak = puncak otomatis + puncak yang ditambahkan lewat klik (tak ada yang dibuang)
            manual = None
            if self._pl_valid():
                # puncak/lembah klik berurutan: HANYA puncak pantulan hasil klik yang dihitung
                # (P0 = titik lepas tidak dikirim; titik lepas diambil dari awal data seperti biasa)
                _b = [tt for j, no, tt, ix in self._pl_bernomor() if j == 'P' and no >= 1]
                if len(_b) >= 2:
                    manual = sorted(_b)
            if manual is None and self.puncak_tambah:
                r0 = cf.analisis_pantulan(t, s, g, y_lantai=yl, prominence_frac=prom,
                                          t_puncak_manual=None)
                ii0 = np.asarray(r0.get('idx_puncak', []), int)
                dasar = [float(v) for v in t[ii0]] if len(ii0) else []
                tol = self._tol_puncak(t)
                manual = sorted(dasar + [x for x in self.puncak_tambah
                                         if all(abs(x - d) > tol for d in dasar)])
            self.puncak_manual = manual
            r = cf.analisis_pantulan(t, s, g, y_lantai=yl, prominence_frac=prom,
                                     t_puncak_manual=manual)
            self._sig_analisis = self._tanda_data()
            self._r3 = r

            for tv in (self.tv_puncak, self.tv_tumb):
                tv.delete(*tv.get_children())
            self.txt_rincian.delete('1.0', 'end')
            self.fig3.clear()
            self.ax3_utama = None

            if len(r.get('idx_puncak', [])):
                self._t_idx3 = t[np.asarray(r['idx_puncak'], int)]
            else:
                self._t_idx3 = np.array([])
            if self.puncak_tambah and len(self._t_idx3):      # tempel ke puncak hasil analisis
                baru = []
                for x in self.puncak_tambah:
                    v = float(self._t_idx3[int(np.argmin(np.abs(self._t_idx3 - x)))])
                    if v not in baru:
                        baru.append(v)
                self.puncak_tambah = baru

            if not r['berhasil']:
                pesan = 'Analisis gagal: ' + r.get('pesan', '')
                pesan += ('  → Nyalakan "Tambah puncak manual" lalu klik puncak-puncaknya di grafik.'
                          if len(t) >= 10 else '')
                self.lbl3.config(text=pesan, foreground='#c00')
                self._tulis3('Analisis pantulan gagal: ' + r.get('pesan', ''), bersih=True)
                self.txt_rincian.insert('end', '\n'.join(cf.rincian_pantulan(r)))
                ax = self.fig3.add_subplot(1, 1, 1)
                ax.plot(t, s, '.-', ms=3, lw=0.9, color='#999')
                ax.axhline(0 if yl is None else yl, ls='--', color='k', lw=0.8)
                ax.set_xlabel('Waktu (s)'); ax.set_ylabel(LABEL_SINYAL); ax.grid(True)
                if self.puncak_manual:
                    ii = np.asarray(r['idx_puncak'], int)
                    ax.plot(t[ii], s[ii], 'o', mfc='none', mec='#1f4fd8', ms=10, label='puncak pilihan')
                self._gambar_pl(ax, t, s)
                ax.legend(fontsize=8)
                self.ax3_utama = ax
                self.fig3.tight_layout(); self.kanvas3.draw()
                return

            self.hasil['e_terkoreksi'] = r['e_regresi']
            self.hasil['n_pantulan'] = r['n_pantulan']

            # ---- ringkasan (panel kiri) ----
            self._tulis3('=== ANALISIS KOEFISIEN RESTITUSI (PANTULAN) ===', bersih=True)
            self._tulis3(f"Jumlah pantulan  : {r['n_pantulan']}"
                         f"{' (+ titik lepas)' if r['ada_lepas'] else ''}"
                         f"{f'   [+{len(self.puncak_tambah)} puncak ditambah manual]' if self.puncak_tambah else ''}")
            self._tulis3(f"y lantai         : {r['y_lantai']:.4f} m ({r['sumber_lantai']})")
            if r['ada_lepas']:
                self._tulis3(f"h0 (tinggi lepas): {r['tinggi_semua'][0]:.4f} m  "
                             f"= tinggi awal data (Tab 2) {r['y_awal']:.4f} m − lantai")
            self._tulis3(f"e (rata-rata)    : {r['e_rerata']:.4f} ± {r['sd_e']:.4f}")
            self._tulis3(f"e (regresi ln h) : {r['e_regresi']:.4f}   R²={r['r2']:.5f}")
            self._tulis3(f"e (kecepatan)    : {r['e_kec_rerata']:.4f}")
            self._tulis3(f"e (waktu terbang): {r['e_waktu_rerata']:.4f}")
            self._tulis3(f"Energi tersisa   : {r['energi_sisa'] * 100:.2f} % tiap pantulan")
            if 'g_estimasi' in r:
                self._tulis3(f"g estimasi       : {r['g_estimasi']:.3f} m/s²")
            self._tulis3('Tinggi puncak (m): ' +
                         ', '.join(f'{v:.4f}' for v in r['tinggi'][:8]))
            if self._pl_valid():
                _b = self._pl_bernomor()
                self._tulis3(f"Pilihan manual   : {sum(1 for q in _b if q[0] == 'P')} puncak, "
                             f"{sum(1 for q in _b if q[0] == 'V')} lembah (klik berurutan)")
            self._tulis3('\nCatatan: data TIDAK dihaluskan di tahap ini; puncak diperoleh '
                         'dari fit parabola pada titik-titik data.')
            if r.get('n_dibuang'):
                self._tulis3(f"{r['n_dibuang']} puncak terakhir tidak dipakai karena dinilai "
                             'tidak wajar (derau saat bola hampir diam).')
            for w in r.get('peringatan', []):
                self._tulis3('⚠ ' + w)

            # ---- label ringkas (atas panel kanan) ----
            self.lbl3.config(
                foreground='#1a4f8a',
                text=(f"Puncak pantulan: {r['n_pantulan']}"
                      f"{' (+ titik lepas)' if r['ada_lepas'] else ''}"
                      f"{f' [+{len(self.puncak_tambah)} manual]' if self.puncak_tambah else ''}  |  "
                      f"e tinggi = {r['e_rerata']:.4f}, kecepatan = {r['e_kec_rerata']:.4f}, "
                      f"waktu = {r['e_waktu_rerata']:.4f}"
                      + (f"  |  ⚠ {r['peringatan'][0][:110]}" if r.get('peringatan') else '')))

            # ---- tabel ----
            for b in cf.tabel_puncak(r):
                self.tv_puncak.insert('', 'end', values=(
                    b['no'], b['jenis'], f"{b['t']:.4f}", f"{b['h']:.5f}",
                    f"{b['h_sampel']:.5f}", b['n_fit'], f"{b['g']:.3f}"))
            for b in cf.tabel_tumbukan(r):
                self.tv_tumb.insert('', 'end', values=(
                    b['no'], f"{b['t']:.4f}", f"{b['h_sebelum']:.5f}", f"{b['h_sesudah']:.5f}",
                    f"{b['e_tinggi']:.4f}", f"{b['v_sebelum']:.3f}", f"{b['v_sesudah']:.3f}",
                    f"{b['e_kec']:.4f}", f"{b['e_waktu']:.4f}"))

            # ---- rincian perhitungan ----
            self.txt_rincian.insert('end', '\n'.join(cf.rincian_pantulan(r, g)))

            # ---- grafik ----
            self._gambar_pantulan(r, t, s)
            self._status(f"e (regresi) = {r['e_regresi']:.4f}")
        except Exception as exc:
            self._galat(exc)

    def _gambar_pantulan(self, r, t, s):
        """Grafik: (1) data + fit parabola + puncak + titik tumbukan,
        (2) peluruhan tinggi puncak (skala log) beserta garis regresi."""
        yl = r['y_lantai']
        ada = r['ada_lepas']
        h = r['tinggi_semua']
        self.fig3.clear()

        ax1 = self.fig3.add_subplot(2, 1, 1)
        self.ax3_utama = ax1
        ax1.plot(t, s, '.-', ms=3, lw=0.9, color='#999', label='data')
        ax1.axhline(yl, ls='--', color='k', lw=0.8, label=f'lantai = {yl:.4f} m')
        pertama = True
        for j, P in enumerate(r['parabola']):
            if P is None:
                continue
            a, b, c, tr = P
            akar = cf._akar_di_lantai(P, yl)
            tv = tr - b / (2 * a)
            if akar is None:
                t0, t1 = tv - 0.1, tv + 0.1
            else:
                t0 = tv if (ada and j == 0) else akar[0]
                t1 = akar[1]
            tt = np.linspace(t0, t1, 60)
            yy = a * (tt - tr) ** 2 + b * (tt - tr) + c
            yy = np.where(yy >= yl - 1e-9, yy, np.nan)       # kurva tidak melewati lantai
            ax1.plot(tt, yy, '-', lw=1.3,
                     color='#d2691e', label='fit parabola' if pertama else None)
            pertama = False
        t_all = ([r['t_lepas']] if ada else []) + list(r['t_puncak'])
        ax1.plot(t_all, np.asarray(h) + yl, 'rv', ms=6, label='puncak (fit)')
        banyak = len(h) > 12
        for j, (tp, hh) in enumerate(zip(t_all, h)):
            no = cf._label_puncak(r, j)
            teks = f'{no}' if banyak else f'{no}\n{hh:.3f}'
            ax1.annotate(teks, (tp, hh + yl), textcoords='offset points', xytext=(0, 6),
                         ha='center', fontsize=7)
        tk_ = np.asarray(r['t_kontak'], float)
        ax1.plot(tk_, np.full(len(tk_), yl), 'g^', ms=5, label='titik tumbukan')
        if r.get('puncak_manual') or self.var['edit_puncak'].get():
            ii = np.asarray(r['idx_puncak'], int)
            ax1.plot(t[ii], s[ii], 'o', mfc='none', mec='#1f4fd8', ms=11, mew=1.5,
                     label='puncak pilihan')
        if self.puncak_tambah:
            it = [int(np.argmin(np.abs(t - x))) for x in self.puncak_tambah]
            ax1.plot(t[it], s[it], 'o', mfc='none', mec='#d81b60', ms=11, mew=1.8,
                     label='puncak ditambah (klik)')
        self._gambar_pl(ax1, t, s)
        ax1.set_title(f"Puncak pantulan: {r['n_pantulan']}" + (' (+ lepas)' if ada else ''))
        ax1.set_ylabel(LABEL_SINYAL); ax1.set_xlabel('Waktu (s)')
        ax1.set_ylim(bottom=yl - 0.03 * max(float(np.max(s)) - yl, 1e-9))
        ax1.grid(True); ax1.legend(fontsize=7, ncol=4)

        ax2 = self.fig3.add_subplot(2, 1, 2)
        n = np.arange(len(h))
        ax2.semilogy(n, h, 'o-', label='h dari fit')
        if len(h) >= 2:
            k, b0 = np.polyfit(n, np.log(h), 1)
            ax2.semilogy(n, np.exp(b0 + k * n), 'r--', label=f"regresi: e = {r['e_regresi']:.4f}")
        ax2.set_xlabel('Indeks puncak' + (' (0 = titik lepas)' if ada else ''))
        ax2.set_ylabel('h (m, log)'); ax2.grid(True, which='both'); ax2.legend(fontsize=8)
        ax2.set_title('Peluruhan tinggi:  ln h_n = ln h_0 + 2n ln e')
        self.fig3.tight_layout(); self.kanvas3.draw()

    def _salin_rincian(self):
        self.clipboard_clear()
        self.clipboard_append(self.txt_rincian.get('1.0', 'end'))
        self._status('Rincian perhitungan disalin ke clipboard.')

    def _simpan_rincian(self):
        nama = filedialog.asksaveasfilename(defaultextension='.txt',
                                            filetypes=[('Teks', '*.txt')])
        if nama:
            with open(nama, 'w', encoding='utf-8') as f:
                f.write(self.txt_rincian.get('1.0', 'end'))
            self._status(f'Rincian disimpan: {os.path.basename(nama)}')

    def _terapkan_koreksi(self):
        if not self.hasil:
            messagebox.showinfo('Analisis', 'Jalankan "② Analisis Pantulan" terlebih dahulu.')
            return
        self._ringkas_koreksi()
        self.nb.select(self.tab4)
        self._status('Koefisien restitusi hasil analisis diterapkan ke Tab 4 (Validasi RK4).')

    def _salin_hasil(self):
        self.clipboard_clear()
        self.clipboard_append(self.txt3.get('1.0', 'end'))
        self._status('Hasil analisis disalin ke clipboard.')


    # =================================================================
    # TAB 4 — VALIDASI MODEL RK4 (model terkoreksi + validasi, digabung)
    # =================================================================
    def _ringkas_koreksi(self):
        for i in self.tv_koreksi.get_children():
            self.tv_koreksi.delete(i)
        pasangan = [('m (kg)', self._f('m'), self._f('m')),
                    ('e', self._f('e'), self.hasil.get('e_terkoreksi', float('nan'))),
                    ('c (N·s/m)', self._f('c'), self.hasil.get('c_terkoreksi', float('nan'))),
                    ('y₀ (m) — Tab 1 / dipakai', self._f('x0'),
                     self._parameter(pakai_koreksi=True)['x0']),
                    ('v₀ (m/s)', self._f('v0'), self.hasil.get('v0_data', float('nan')))]
        for nama, a, b in pasangan:
            fa = '-' if not np.isfinite(a) else f'{a:.5g}'
            fb = '-' if not np.isfinite(b) else f'{b:.5g}'
            self.tv_koreksi.insert('', 'end', values=(nama, fa, fb))

    def _model_terkoreksi(self):
        p = self._parameter(pakai_koreksi=True)
        return cf.buat_model(KUNCI_SISTEM, p), p

    def _optimasi(self):
        """Kalibrasi halus: cari parameter yang paling cocok dengan data eksperimen."""
        if not self._cek_data():
            return
        try:
            p = self._parameter(pakai_koreksi=True)
            if not np.isfinite(p.get('x0', np.nan)) or p['x0'] == 0:
                p['x0'], p['v0'] = cf.kondisi_awal(self.data['t'], self.data['s'])
            self._status('Mengoptimasi parameter… (beberapa detik)', '#a60')
            self.update_idletasks()
            # y0 dikunci pada nilai Tab 1 (tidak ikut dioptimasi)
            r = cf.optimasi_parameter(KUNCI_SISTEM, p, self.data['t'], self.data['s'],
                                      bebas=['e', 'c', 'v0'])
            if not r['berhasil']:
                messagebox.showwarning('Optimasi', r.get('pesan', 'gagal')); return
            po, mk = r['p'], r['metrik']
            self.hasil['e_terkoreksi'] = po['e']
            self.hasil['c_terkoreksi'] = po['c']
            self.hasil['v0_data'] = po['v0']
            self._ringkas_koreksi()

            self.txt4.delete('1.0', 'end')
            self.txt4.insert('end', '=== OPTIMASI PARAMETER (kuadrat terkecil + RK4) ===\n')
            self.txt4.insert('end', f"y₀ dikunci ("
                                    f"{'tinggi awal data, Tab 2' if self._samakan_y0() and self._y0_data() is not None else 'Tab 1'}"
                                    f"): {p['x0']:.5g} m\n")
            self.txt4.insert('end', f"Parameter bebas : {', '.join(r['bebas'])}\n")
            self.txt4.insert('end', f"Evaluasi model  : {r['n_eval']}\n\n")
            for nama in r['bebas']:
                self.txt4.insert('end', f'  {nama:<4} = {po[nama]:.6g}   (awal {p[nama]:.6g})\n')
            self.txt4.insert('end', f"\nRMSE = {mk['RMSE']:.5e}   R² = {mk['R2']:.6f}   "
                                    f"NRMSE = {mk['NRMSE_persen']:.3f} %\n")
            for w in r.get('peringatan', []):
                self.txt4.insert('end', f'\n⚠ {w}\n')
            self.fig4.clear()
            ax = self.fig4.add_subplot(2, 1, 1)
            ax.plot(self.data['t'], self.data['s'], '.', ms=2.5, color='#444', label='eksperimen')
            ax.plot(self.data['t'], r['prediksi'], '-', color='crimson', lw=1.2,
                    label='model teroptimasi')
            ax.set_ylabel(LABEL_SINYAL); ax.grid(True); ax.legend(fontsize=8)
            ax.set_title('Hasil Optimasi Parameter — Posisi terhadap Waktu')
            ax2 = self.fig4.add_subplot(2, 1, 2)
            ax2.plot(self.data['t'], mk['residual'], '-', lw=0.8)
            ax2.axhline(0, color='k', lw=0.8)
            ax2.set_xlabel('Waktu (s)'); ax2.set_ylabel('Residual'); ax2.grid(True)
            self.fig4.tight_layout(); self.kanvas4.draw()
            self._status(f"Optimasi selesai — R² = {mk['R2']:.5f}"
                         + ('  ⚠ lihat catatan di Tab 4' if r.get('peringatan') else ''),
                         '#a60' if r.get('peringatan') else '#0a5')
        except Exception as exc:
            self._galat(exc)

    def _uji_konvergensi(self):
        """Galat RK4 (maksimum) terhadap solusi rujukan RK4 dengan h sangat kecil."""
        try:
            model, _ = self._model_terkoreksi()
            t_akhir = (float(self.data['t'][-1]) if self.data['t'] is not None
                       else self._f('t_akhir', 6.0))
            h_ref = 1e-5
            Tr, Yr = cf.simulasi(model, t_akhir, h_ref)
            hs = np.array([0.05, 0.02, 0.01, 0.005, 0.002, 0.001])
            galat = []
            for h in hs:
                T, Y = cf.simulasi(model, t_akhir, float(h))
                ref = np.interp(T, Tr, Yr[:, 0])
                galat.append(float(np.max(np.abs(Y[:, 0] - ref))))
            g_arr = np.maximum(np.array(galat), 1e-18)
            self.fig4.clear()
            ax = self.fig4.add_subplot(1, 1, 1)
            ax.loglog(hs, g_arr, 'o-', label='RK4')
            amp = float(np.max(np.abs(Yr[:, 0])) + 1e-12)
            sah = (g_arr < 5 * amp) & (g_arr > 1e-12)
            hs_s, g_s = hs[sah], g_arr[sah]
            orde = (np.polyfit(np.log(hs_s[-4:]), np.log(g_s[-4:]), 1)[0]
                    if len(hs_s) >= 2 else float('nan'))
            self.txt4.delete('1.0', 'end')
            self.txt4.insert('end', '=== UJI KONVERGENSI RK4 (galat maksimum vs h) ===\n')
            self.txt4.insert('end', f'Rujukan: RK4 dengan h = {h_ref:g} s\n\n')
            for h, g_ in zip(hs, g_arr):
                self.txt4.insert('end', f'h = {h:<7g} galat maks = {g_:.3e}\n')
            if float(np.max(g_arr)) < 1e-9:
                self.txt4.insert('end', '\nGalat ≈ 1e-11 untuk semua h: tanpa hambatan udara (c = 0) '
                                        'percepatan konstan sehingga RK4 praktis EKSAK.\n'
                                        'Isi c > 0 di Tab 1 untuk melihat orde konvergensi.\n')
            else:
                self.txt4.insert('end', f'\norde empiris ≈ {orde:.2f}\n(RK4 murni berorde 4; tumbukan '
                                        'dengan lantai bisa menurunkan orde yang teramati.)\n')
            ax.set_xlabel('Ukuran langkah h (s)'); ax.set_ylabel('Galat maksimum')
            ax.set_title('Konvergensi RK4 (rujukan: RK4 h=1e-5)')
            ax.grid(True, which='both'); ax.legend(fontsize=8)
            self.fig4.tight_layout(); self.kanvas4.draw()
            self._status('Uji konvergensi selesai.')
        except Exception as exc:
            self._galat(exc)

    # =================================================================
    # TAB 4 (lanjutan) — validasi, ekspor, laporan
    # =================================================================
    def _tab_validasi(self):
        kiri = self._panel_gulir(self.tab4, lebar=440)
        kanan = ttk.Frame(self.tab4); kanan.pack(side='left', fill='both', expand=True)

        b = ttk.LabelFrame(kiri, text='Parameter: Ideal vs Terkoreksi', padding=6)
        b.pack(fill='x')
        self.tv_koreksi = ttk.Treeview(b, columns=('p', 'ideal', 'koreksi'),
                                       show='headings', height=5)
        for kol, teks, lebar in (('p', 'Parameter', 170), ('ideal', 'Ideal', 100),
                                 ('koreksi', 'Terkoreksi', 110)):
            self.tv_koreksi.heading(kol, text=teks)
            self.tv_koreksi.column(kol, width=lebar, anchor='center')
        self.tv_koreksi.pack(fill='x')

        n = ttk.LabelFrame(kiri, text='Pengaturan (metode RK4)', padding=6)
        n.pack(fill='x', pady=6)
        self._baris(n, 0, 'Langkah h', 'h4', 0.001, 's')
        self.var['tandai_puncak'] = tk.BooleanVar(value=True)
        ttk.Checkbutton(n, text='Tandai puncak model RK4 pada grafik',
                        variable=self.var['tandai_puncak']).grid(row=1, column=0,
                                                                 columnspan=3, sticky='w')

        sel = ttk.LabelFrame(kiri, text='Pilih puncak & y₀ model', padding=6)
        sel.pack(fill='x', pady=(0, 6))
        self.var['samakan_y0'] = tk.BooleanVar(value=True)
        ttk.Checkbutton(sel, text='Samakan y₀ model dengan tinggi awal data (Tab 2)',
                        variable=self.var['samakan_y0'],
                        command=self._ringkas_koreksi).pack(anchor='w')
        self.var['edit_puncak4'] = tk.BooleanVar(value=False)
        ttk.Checkbutton(sel, text='✎ Tambah puncak eksperimen manual (klik grafik atas)',
                        variable=self.var['edit_puncak4'],
                        command=self._ganti_mode_edit4).pack(anchor='w', pady=(4, 0))
        ttk.Button(sel, text='↺ Kembali ke puncak otomatis',
                   command=lambda: self._puncak_otomatis(dari_tab4=True)).pack(anchor='w', pady=2)

        jg_ = ttk.LabelFrame(kiri, text='Penyesuaian model ke data (puncak & lembah)', padding=6)
        jg_.pack(fill='x', pady=(0, 6))
        self.var['jangkar12'] = tk.BooleanVar(value=True)
        ttk.Checkbutton(jg_, text='Sesuaikan model RK4 dengan puncak ke-1 & ke-2 data (Tab 2)',
                        variable=self.var['jangkar12'],
                        command=lambda: self._validasi_tunda(0)).pack(anchor='w')
        ttk.Label(jg_, wraplength=400, foreground='#666', justify='left',
                  text='Tinggi puncak ke-1 & ke-2 data menentukan e dan tinggi lepas y₀ model; '
                       'waktu kedua puncak itu menentukan skala & geseran waktu model. Puncak '
                       'berikutnya dan sisa kurva tinggal mengikuti model RK4. Matikan untuk '
                       'model murni dari parameter.').pack(anchor='w', pady=(2, 0))
        self.var['kunci_pl4'] = tk.BooleanVar(value=True)
        ttk.Checkbutton(jg_, text='Kunci waktu model ke SEMUA puncak & lembah data',
                        variable=self.var['kunci_pl4'],
                        command=lambda: self._validasi_tunda(0)).pack(anchor='w', pady=(6, 0))
        ttk.Label(jg_, wraplength=400, foreground='#666', justify='left',
                  text='Waktu model RK4 dipetakan sepotong-sepotong sehingga melewati tiap puncak '
                       'DAN lembah (tumbukan) data — dari klik manual di Tab 3 bila ada, kalau tidak '
                       'dari deteksi otomatis. Di luar tonggak terakhir model mengikuti saja. '
                       'Tinggi puncak tetap ditentukan puncak ke-1 & ke-2.'
                  ).pack(anchor='w', pady=(2, 0))

        rg = ttk.LabelFrame(kiri, text='Renggang – rapat pantulan (manual)', padding=6)
        rg.pack(fill='x', pady=(0, 6))
        ttk.Label(rg, wraplength=400, foreground='#666', justify='left',
                  text='Geser slider ke kiri / kanan, atau ketik angka k. k > 1 = pantulan model '
                       'lebih RENGGANG (lebih lambat), k < 1 = lebih RAPAT (lebih cepat). Sumbu '
                       'waktu model RK4 diregangkan dari puncak ke-1 data (bila penyesuaian '
                       'puncak aktif) atau dari saat bola dilepas (bila tidak). Dipakai untuk '
                       'penyetelan halus sesudah penyesuaian otomatis.').pack(anchor='w')
        self.var['renggang4'] = tk.StringVar(value='1.000')
        self.var['renggang4_s'] = tk.DoubleVar(value=1.0)
        bar_r = ttk.Frame(rg); bar_r.pack(fill='x', pady=(4, 0))
        ttk.Label(bar_r, text='rapat').pack(side='left')
        ttk.Scale(bar_r, from_=0.5, to=1.5, orient='horizontal',
                  variable=self.var['renggang4_s'],
                  command=self._renggang_dari_slider).pack(side='left', fill='x', expand=True, padx=4)
        ttk.Label(bar_r, text='renggang').pack(side='left')
        bar_k = ttk.Frame(rg); bar_k.pack(fill='x', pady=(4, 0))
        ttk.Label(bar_k, text='Faktor waktu k =').pack(side='left')
        sp = ttk.Spinbox(bar_k, from_=0.1, to=5.0, increment=0.005, width=8,
                         textvariable=self.var['renggang4'], command=self._renggang_dari_teks)
        sp.pack(side='left', padx=4)
        sp.bind('<Return>', self._renggang_dari_teks)
        sp.bind('<FocusOut>', self._renggang_dari_teks)
        ttk.Button(bar_k, text='◀', width=3, command=lambda: self._renggang_geser(-1)).pack(side='left')
        ttk.Button(bar_k, text='▶', width=3,
                   command=lambda: self._renggang_geser(+1)).pack(side='left', padx=(2, 6))
        ttk.Button(bar_k, text='Reset (1)', command=self._renggang_reset).pack(side='left')

        ttk.Button(kiri, text='✔  Validasi Model RK4 vs Eksperimen', width=44,
                   command=self._validasi).pack(pady=3)
        ttk.Button(kiri, text='🎯  Optimasi Parameter (fit e, c, v₀ ke data)', width=44,
                   command=self._optimasi).pack(pady=2)
        ttk.Button(kiri, text='📉  Uji Konvergensi RK4 (galat vs h)', width=44,
                   command=self._uji_konvergensi).pack(pady=2)
        ttk.Button(kiri, text='💾 Ekspor data keseluruhan (CSV)', width=44,
                   command=self._ekspor_validasi).pack(pady=2)
        ttk.Button(kiri, text='📄 Simpan laporan (TXT)', width=44,
                   command=self._laporan).pack(pady=2)

        self.txt4 = tk.Text(kiri, width=52, height=17, font=('Consolas', 9))
        self.txt4.pack(fill='x', pady=6)

        ttk.Label(kiri, text='Data Keseluruhan', font=('Segoe UI', 9, 'bold')).pack(anchor='w')
        wadah = ttk.Frame(kiri); wadah.pack(fill='x')
        kolom = ('n', 't', 'eks', 'rk4', 'res')
        self.tv5 = ttk.Treeview(wadah, columns=kolom, show='headings', height=12)
        for kol, teks, lebar in (('n', '#', 45), ('t', 't (s)', 75), ('eks', 'eksperimen', 95),
                                 ('rk4', 'RK4', 95), ('res', 'residual', 95)):
            self.tv5.heading(kol, text=teks)
            self.tv5.column(kol, width=lebar, anchor='center')
        sb = ttk.Scrollbar(wadah, orient='vertical', command=self.tv5.yview)
        self.tv5.configure(yscroll=sb.set)
        self.tv5.pack(side='left', fill='x', expand=True)
        sb.pack(side='left', fill='y')

        self.fig4, self.kanvas4 = self._kanvas(kanan, ukuran=(7.6, 8.4))
        self.kanvas4.mpl_connect('button_press_event', self._klik_grafik4)

    # ---------- renggang - rapat pantulan (manual) ----------
    def _renggang(self):
        """Faktor k (> 0): sumbu waktu model diregangkan k kali dari saat lepas."""
        try:
            k = float(str(self.var['renggang4'].get()).replace(',', '.'))
        except Exception:
            k = 1.0
        return k if (np.isfinite(k) and k > 0) else 1.0

    def _renggang_dari_slider(self, nilai):
        k = round(float(nilai), 3)
        self.var['renggang4'].set(f'{k:.3f}')
        self._validasi_tunda()

    def _renggang_dari_teks(self, _ev=None):
        k = self._renggang()
        self.var['renggang4'].set(f'{k:.3f}')
        self.var['renggang4_s'].set(min(max(k, 0.5), 1.5))
        if self._k_validasi is None or abs(k - self._k_validasi) > 1e-9:
            self._validasi_tunda(0)

    def _renggang_geser(self, arah):
        k = round(self._renggang() + arah * 0.005, 3)
        self.var['renggang4'].set(f'{max(k, 0.01):.3f}')
        self._renggang_dari_teks()

    def _renggang_reset(self):
        self.var['renggang4'].set('1.000')
        self._renggang_dari_teks()

    def _validasi_tunda(self, ms=120):
        """Render ulang grafik Tab 4 sesudah slider berhenti bergerak (tidak tiap piksel)."""
        if self._id_validasi is not None:
            try:
                self.after_cancel(self._id_validasi)
            except Exception:
                pass
        self._id_validasi = self.after(ms, self._validasi_diam)

    def _validasi_diam(self):
        self._id_validasi = None
        if self.validasi and self.data.get('t') is not None and len(self.data['t']) >= 5:
            self._validasi()

    def _jangkar_puncak12(self, p, r3, h):
        """Sesuaikan model RK4 dengan puncak pantulan ke-1 & ke-2 DATA (Tab 2).
        Tinggi  -> e dan y0 model (ideal: e = sqrt(h2/h1), y0 = h1/e², lalu dikoreksi lewat RK4
                   bila ada hambatan udara).
        Waktu   -> peta waktu  t_data = a·T_model + b  (a = skala, b = geseran)."""
        try:
            t_d, y_d = cf.puncak_data(r3)
            t_d = np.asarray(t_d, float); y_d = np.asarray(y_d, float)
            off = 1 if r3.get('ada_lepas') else 0            # indeks 0 = titik lepas bila ada
            if len(t_d) < off + 2:
                return {'ok': False, 'pesan': 'puncak pantulan data kurang dari 2'}
            yl = float(r3.get('y_lantai', 0.0))
            t1, t2 = float(t_d[off]), float(t_d[off + 1])
            y1, y2 = float(y_d[off]), float(y_d[off + 1])
            h1, h2 = y1 - yl, y2 - yl
            if not (h1 > 0 and h2 > 0 and t2 > t1):
                return {'ok': False, 'pesan': 'tinggi / waktu puncak ke-1 & ke-2 data tidak valid'}
            g = float(p.get('g', 9.81))
            q = dict(p)
            e = float(np.sqrt(min(h2 / h1, 0.999999)))
            y0 = h1 / e ** 2
            hs = max(h, 1e-3)
            tm1 = tm2 = float('nan')
            for _ in range(8):
                q['e'], q['x0'] = e, y0
                model = cf.buat_model(KUNCI_SISTEM, q)
                tau0 = np.sqrt(2.0 * y0 / g)
                pb = []
                for faktor in (1.5, 4.0):                      # perluas horizon bila perlu
                    T_, Y_ = cf.simulasi(model, faktor * tau0 * (1.0 + e) ** 2 + 0.3, hs)
                    pb = [u for u in cf.puncak_simulasi(T_, Y_) if u[2] != 'lepas']
                    if len(pb) >= 2:
                        break
                if len(pb) < 2:
                    return {'ok': False, 'pesan': 'model RK4 tidak menghasilkan 2 puncak pantulan'}
                tm1, tm2 = float(pb[0][0]), float(pb[1][0])
                h1m = float(pb[0][1]) - model.y_lantai
                h2m = float(pb[1][1]) - model.y_lantai
                if not (h1m > 0 and h2m > 0):
                    return {'ok': False, 'pesan': 'tinggi puncak model tidak valid'}
                err = abs(h1m / h1 - 1.0) + abs(h2m / h2 - 1.0)
                if err < 1e-5:
                    break
                e = float(np.clip(e * np.sqrt((h2 / h1) / (h2m / h1m)), 0.01, 0.9999))
                y0 = y0 * h1 / h1m
            if not tm2 > tm1:
                return {'ok': False, 'pesan': 'waktu puncak model tidak valid'}
            a = (t2 - t1) / (tm2 - tm1)
            b = t1 - a * tm1
            return {'ok': True, 'e': e, 'x0': y0, 'a': a, 'b': b, 't1': t1, 't2': t2,
                    'h1': h1, 'h2': h2, 'y1': y1, 'y2': y2, 'tm1': tm1, 'tm2': tm2, 'err': err}
        except Exception as exc:
            return {'ok': False, 'pesan': f'{type(exc).__name__}: {exc}'}

    def _tonggak_data(self, r3):
        """Puncak & lembah data sebagai [(jenis, no, t)]. Puncak dari analisis (Tab 3); lembah dari
        klik manual (bila ada, dirapikan ke waktu tumbukan hasil fit bila dekat) atau otomatis."""
        ev = []
        ada = bool(r3.get('ada_lepas'))
        t_d, _ = cf.puncak_data(r3)
        for j, tt in enumerate(np.asarray(t_d, float)):
            no = j if ada else j + 1
            if no >= 1:
                ev.append(('P', no, float(tt)))
        kont = np.asarray(r3.get('t_kontak', []), float)
        t = np.asarray(self.data['t'], float)
        dt = float(np.median(np.diff(t))) if len(t) > 1 else 0.0
        vm = [(no, tt) for j, no, tt, ix in self._pl_bernomor() if j == 'V']
        if vm:
            for no, tt in vm:
                if len(kont) and float(np.min(np.abs(kont - tt))) <= 3.0 * dt:
                    tt = float(kont[int(np.argmin(np.abs(kont - tt)))])
                ev.append(('V', no, float(tt)))
        else:
            for k, tc in enumerate(kont):
                ev.append(('V', k + 1 + (0 if ada else 1), float(tc)))
        return ev

    def _peta_tonggak_model(self, model, r3, a, b, h):
        """Simulasikan model, cocokkan puncak/lembahnya dengan puncak/lembah data, lalu bangun
        peta waktu sepotong-sepotong. Mengembalikan (dict | None, pesan)."""
        try:
            ev = self._tonggak_data(r3)
            if len(ev) < 2:
                return None, 'puncak/lembah data kurang dari 2'
            t_akhir = max(q[2] for q in ev)
            hor = max((t_akhir - b) / a, 0.1) * 1.25 + 0.3
            hs = min(max(float(h), 1e-4), 1e-3)
            T_, Y_ = cf.simulasi(model, hor, hs)
            T_ = np.asarray(T_, float); Y_ = np.asarray(Y_, float)
            evm = _tonggak_model(T_, Y_, cf.puncak_simulasi(T_, Y_))
            pas = [(j, no, td, evm[(j, no)]) for j, no, td in ev if (j, no) in evm]
            return _peta_tonggak(pas)
        except Exception as exc:
            return None, f'{type(exc).__name__}: {exc}'

    def _validasi(self):
        if not self._cek_data():
            return
        try:
            self._ringkas_koreksi()
            model, p = self._model_terkoreksi()
            t = np.asarray(self.data['t'], dtype=float)
            s = np.asarray(self.data['s'], dtype=float)
            k = self._renggang()
            h = max(self._f('h4', 0.001), 1e-6)
            catatan = []

            if self._r3 is None or self._sig_analisis != self._tanda_data():
                self._analisis_pantulan(otomatis=True)
            r3 = self._r3 if (self._r3 and self._r3.get('berhasil')) else None

            # ---- penyesuaian ke puncak ke-1 & ke-2 data:  t_data = a·T_model + b
            a, b, tref = 1.0, 0.0, 0.0
            jg = None
            if self.var['jangkar12'].get():
                if r3 is None:
                    catatan.append('Penyesuaian puncak ke-1 & ke-2 tidak bisa (analisis puncak data '
                                   'gagal); dipakai model murni.')
                else:
                    jg = self._jangkar_puncak12(p, r3, h)
                    if jg['ok']:
                        p = dict(p); p['e'] = jg['e']; p['x0'] = jg['x0']
                        model = cf.buat_model(KUNCI_SISTEM, p)
                        a, b, tref = jg['a'], jg['b'], jg['t1']
                    else:
                        catatan.append('Penyesuaian puncak ke-1 & ke-2 gagal (' + jg['pesan'] +
                                       '); dipakai model murni.')

            # ---- model RK4 pada waktu data. Peregangan manual k terhadap tref:
            #      t_data = tref + k·(a·T + b − tref)  ->  T = ((t_data − tref)/k + tref − b)/a
            #      Dengan kunci puncak & lembah: F = peta sepotong-sepotong (lihat _peta_tonggak).
            warp = None
            if r3 is not None and self.var['kunci_pl4'].get():
                warp, pesan_w = self._peta_tonggak_model(model, r3, a, b, h)
                if warp is None:
                    catatan.append('Kunci waktu ke puncak & lembah data tidak bisa (' + pesan_w +
                                   '); dipakai penyesuaian linear.')
                else:
                    if jg is None or not jg['ok']:
                        tref = warp['t_ref']
                    if warp['n_buang']:
                        catatan.append(f"{warp['n_buang']} puncak/lembah terakhir tidak dipakai mengunci "
                                       'waktu (selang waktunya tidak wajar); sisanya mengikuti model.')
            fwd, inv, dfwd = ((warp['fwd'], warp['inv'], warp['dfwd']) if warp is not None
                              else _peta_linear(a, b))
            Tm = inv(tref + (t - tref) / k)
            msk = Tm >= 0.0
            if msk.sum() < 2:
                a, b, tref, jg, warp = 1.0, 0.0, 0.0, None, None
                fwd, inv, dfwd = _peta_linear(1.0, 0.0)
                catatan.append('Penyesuaian waktu menghasilkan waktu model negatif; dipakai model murni.')
                Tm = t / k
                msk = np.ones(len(t), dtype=bool)
            pos = np.full(len(t), float(p['x0']))            # sebelum model dilepas: diam di y0
            pos_m, vel_m, T, Y = cf.simulasi_pada_grid(model, Tm[msk], h=h)
            pos[msk] = pos_m
            T_model = np.asarray(T, dtype=float)
            T = tref + k * (fwd(T_model) - tref)
            Y = np.array(Y, dtype=float)
            Y[:, 1] = Y[:, 1] / (k * dfwd(T_model))
            mk = cf.metrik_validasi(s, pos)
            res = mk['residual']
            self._k_validasi = k

            if jg is not None and jg['ok']:
                if abs(a - 1.0) > 0.15:
                    catatan.append(f'skala waktu hasil penyesuaian a = {a:.3f} jauh dari 1: waktu data '
                                   '(FPS) atau skala px/m kemungkinan keliru.')
            elif r3 is not None and abs(k - 1.0) < 1e-9:
                g_fit = r3.get('g_estimasi', float('nan'))
                g0 = self._f('g', 9.81)
                if np.isfinite(g_fit) and g_fit > 0 and abs(g_fit / g0 - 1.0) > 0.15:
                    catatan.append(f'g hasil fit data = {g_fit:.2f} (acuan {g0:g}): waktu data (FPS) '
                                   f'atau px/m kemungkinan keliru. Coba geser "renggang–rapat" ke '
                                   f'k ≈ {np.sqrt(g0 / g_fit):.3f}.')
            y0d = self._y0_data()
            if (jg is None or not jg['ok']) and y0d is not None and not self._samakan_y0() \
                    and abs(p['x0'] - y0d) > 0.05 * max(y0d, 1e-9):
                catatan.append(f"y₀ Tab 1 = {p['x0']:.4f} m beda dari tinggi awal data {y0d:.4f} m: "
                               'centang "Samakan y₀ model dengan tinggi awal data".')

            self.validasi = {'t': t, 'eks': s, 'rk4': pos, 'res': res, 'metrik': mk,
                             'param': p, 'T': T, 'Y': Y, 'renggang': k, 'jangkar': jg,
                             'a': a, 'b': b, 'tonggak': warp is not None}

            self.txt4.delete('1.0', 'end')
            w = self.txt4.insert
            w('end', '=== VALIDASI MODEL RK4 vs EKSPERIMEN ===\n')
            w('end', f"Sistem      : {NAMA_SISTEM}\n")
            w('end', f"Sumber data : {self.data['sumber']}\n")
            w('end', f"Jumlah titik: {mk['n']}   h = {h}\n")
            w('end', f"Parameter   : m={p['m']:.5g} kg, e={p['e']:.5g}, c={p['c']:.5g}, "
                     f"y0={p['x0']:.5g} m, v0={p['v0']:.5g} m/s\n")
            if jg is not None and jg['ok']:
                w('end', 'Model       : RK4, DISESUAIKAN ke puncak ke-1 & ke-2 data\n')
                w('end', f"  data  P1: t = {jg['t1']:.4f} s, h = {jg['h1']:.4f} m\n")
                w('end', f"  data  P2: t = {jg['t2']:.4f} s, h = {jg['h2']:.4f} m\n")
                w('end', f"  hasil   : e = {jg['e']:.4f}, y₀ model = {jg['x0']:.4f} m "
                         f"(galat tinggi relatif {jg['err']:.1e})\n")
                w('end', f"  waktu   : t_data = {a:.4f}·T_model {b:+.4f} s\n")
            else:
                w('end', 'Model       : RK4 murni (t = 0 saat lepas), seperti simulasi Tab 1\n')
            if warp is not None:
                w('end', f"Kunci waktu  : {warp['nP']} puncak + {warp['nV']} lembah data "
                         f"(kemiringan global {warp['a']:.4f}); sesudahnya model mengikuti\n")
            w('end', f"Renggang-rapat: k = {k:.4f}"
                     + (' (pantulan lebih renggang)' if k > 1 else
                        ' (pantulan lebih rapat)' if k < 1 else ' (tanpa penyesuaian manual)') + '\n')
            w('end', '-' * 42 + '\n')
            for nama, nilai, sat in (('RMSE', mk['RMSE'], ''), ('MAE', mk['MAE'], ''),
                                     ('Galat maksimum', mk['maks'], ''),
                                     ('Bias (rata2 res)', mk['bias'], ''),
                                     ('SD residual', mk['SD_residual'], ''),
                                     ('NRMSE', mk['NRMSE_persen'], '%'),
                                     ('R²', mk['R2'], ''), ('Korelasi r', mk['korelasi'], '')):
                w('end', f'{nama:<18}: {nilai:>12.6g} {sat}\n')
            mutu = ('sangat baik' if mk['R2'] > 0.98 else 'baik' if mk['R2'] > 0.90 else
                    'cukup' if mk['R2'] > 0.75 else 'kurang — periksa kalibrasi/parameter')
            w('end', f'\nKesimpulan  : kecocokan {mutu}.\n')
            for c_ in catatan:
                w('end', f'⚠ {c_}\n')

            # --- perbandingan puncak data vs RK4 (urut: lepas, pantulan 1, 2, …)
            pk = cf.puncak_simulasi(T, Y)
            if r3 is not None:
                t_d, y_d = cf.puncak_data(r3)
                off = 0 if r3['ada_lepas'] else 1
                n_ban = min(len(t_d), max(len(pk) - off, 0), 10)
                if n_ban:
                    w('end', '\nPerbandingan puncak (data ↔ RK4):\n')
                    w('end', f"{'no':>3} {'t data':>8} {'t RK4':>8} {'Δt ms':>7} {'y data':>8} {'y RK4':>8} {'Δy mm':>7}\n")
                    for j in range(n_ban):
                        tr_, yr_, _ = pk[j + off]
                        no = j if r3['ada_lepas'] else j + 1
                        w('end', f"{no:>3} {t_d[j]:>8.4f} {tr_:>8.4f} {(tr_ - t_d[j]) * 1000:>7.1f} "
                                 f"{y_d[j]:>8.4f} {yr_:>8.4f} {(yr_ - y_d[j]) * 1000:>7.1f}\n")
            w('end', '\n' + model.info() + '\n')

            for i in self.tv5.get_children():
                self.tv5.delete(i)
            langkah = max(len(t) // 3000, 1)
            for i in range(0, len(t), langkah):
                self.tv5.insert('', 'end', values=(i, f'{t[i]:.4f}', f'{s[i]:.6f}',
                                                   f'{pos[i]:.6f}', f'{res[i]:.6f}'))

            # ---------------- grafik: posisi + kecepatan bergaya Tab 1, lalu residual ----------------
            self.fig4.clear()
            gs = self.fig4.add_gridspec(4, 2, height_ratios=[1.5, 1.1, 0.9, 1.0])
            ax1 = self.fig4.add_subplot(gs[0, :])
            ax2v = self.fig4.add_subplot(gs[1, :], sharex=ax1)
            self.ax4_utama = ax1

            Tp, Yp = self._kurva_model(T, Y, float(t[0]))
            ax1.plot(t, s, '.-', ms=2.5, lw=0.8, color='#444', alpha=0.7, label='eksperimen')
            ax1.plot(Tp, Yp, '-', lw=1.4, color='#1f77b4', label='posisi (RK4)')
            try:
                ana = model.solusi_analitik(T_model)
            except Exception:
                ana = None
            if ana is not None:
                ax1.plot(T, ana, '--', lw=1.0, color='crimson', label='analitik (ideal)')
            ax1.axhline(0, color='k', lw=0.7, ls='--', alpha=0.4, label='lantai y=0')
            tp = np.array([q[0] for q in pk]); yp = np.array([q[1] for q in pk])
            lab, n_p = [], 0
            for _, _, jenis in pk:
                if jenis == 'lepas':
                    lab.append('L')
                else:
                    n_p += 1
                    lab.append(f'P{n_p}')
            if self.var['tandai_puncak'].get() and len(pk):
                ax1.plot(tp, yp, 'rv', ms=7, label=f'puncak ({len(pk)})')
                for j, (tt, yy, lb) in enumerate(zip(tp, yp, lab)):
                    if j >= 12:
                        break
                    teks = f'{lb}\n{yy:.3f} m' if j < 6 else lb
                    ax1.annotate(teks, (tt, yy), textcoords='offset points', xytext=(0, 7),
                                 ha='center', fontsize=7, color='#8b0000')
            if len(pk):
                for tt in tp[:30]:                       # garis bantu: puncak sama di kedua grafik
                    ax1.axvline(tt, color='r', ls=':', lw=0.6, alpha=0.45)
                    ax2v.axvline(tt, color='r', ls=':', lw=0.6, alpha=0.45)
            if r3 is not None:
                t_d, y_d = cf.puncak_data(r3)
                ax1.plot(t_d, y_d, 'o', mfc='none', mec='#2e8b57', ms=10, mew=1.5,
                         label='puncak eksperimen')
            if jg is not None and jg['ok']:
                ax1.plot([jg['t1'], jg['t2']], [jg['y1'], jg['y2']], '*', ms=15, color='gold',
                         mec='k', mew=0.8, label='puncak ke-1 & ke-2 data (penyesuaian)')
            if self.puncak_tambah:
                it = [int(np.argmin(np.abs(t - x))) for x in self.puncak_tambah]
                ax1.plot(t[it], s[it], 'o', mfc='none', mec='#d81b60', ms=12, mew=1.8,
                         label='puncak ditambah (klik)')
            if warp is not None:
                yl_ = float(r3.get('y_lantai', 0.0))
                tv_ = [q[2] for q in warp['pasangan'] if q[0] == 'V']
                if tv_:
                    ax1.plot(tv_, np.full(len(tv_), yl_), 'D', mfc='none', mec='#7b1fa2', ms=9,
                             mew=1.6, label='lembah data (dikunci)')
            ax1.set_xlim(float(t[0]), float(t[-1]))
            judul = f"POSISI TERHADAP WAKTU — R² = {mk['R2']:.5f}, RMSE = {mk['RMSE']:.4e}"
            if abs(k - 1.0) > 1e-9:
                judul += f'   [k = {k:.3f}]'
            if self.var['edit_puncak4'].get():
                judul += '   [klik kiri = tambah puncak, klik kanan = hapus puncak tambahan]'
            ax1.set_title(judul, fontsize=10)
            ax1.set_ylabel(LABEL_SINYAL); ax1.grid(True); ax1.legend(fontsize=7, ncol=3)

            m_ = (T >= float(t[0])) & (T <= float(t[-1]))
            Tv, v = (T[m_], Y[m_, 1]) if m_.any() else (T, Y[:, 1])
            ax2v.axhline(0, color='k', lw=0.8)
            ax2v.fill_between(Tv, 0, v, where=v > 0, color='#2e8b57', alpha=0.18,
                              interpolate=True, label='bergerak naik (v > 0)')
            ax2v.fill_between(Tv, 0, v, where=v < 0, color='#d2691e', alpha=0.18,
                              interpolate=True, label='bergerak turun (v < 0)')
            ax2v.plot(Tv, v, '-', color='#d2691e', lw=1.2)
            if len(pk):
                ax2v.plot(tp, np.zeros_like(tp), 'rv', ms=7, label='puncak (v = 0)')
            tumb = cf.tumbukan_simulasi(T, Y, model.e)
            if tumb:
                tt_ = np.array([q[0] for q in tumb]); vb_ = np.array([q[1] for q in tumb])
                va_ = np.array([q[2] for q in tumb])
                ax2v.vlines(tt_, vb_, va_, color='gray', ls=':', lw=0.8)
                ax2v.plot(tt_, vb_, 'bv', ms=5, label='tepat sebelum tumbukan')
                ax2v.plot(tt_, va_, '^', color='green', ms=5, label='tepat sesudah tumbukan')
                for j in range(min(len(tumb), 5)):
                    ax2v.annotate(f'{vb_[j]:.2f}', (tt_[j], vb_[j]), textcoords='offset points',
                                  xytext=(-14, -3), ha='right', fontsize=7, color='blue')
                    ax2v.annotate(f'{va_[j]:.2f}', (tt_[j], va_[j]), textcoords='offset points',
                                  xytext=(14, 2), ha='left', fontsize=7, color='green')
            ax2v.set_title('KECEPATAN TERHADAP WAKTU — v = 0 di setiap puncak', fontsize=10)
            ax2v.set_ylim(top=max(float(np.max(v)), 0.1) * 1.5)
            ax2v.set_ylabel('Kecepatan (m/s)\n(+ naik, − turun)')
            ax2v.grid(True); ax2v.legend(fontsize=7, ncol=3, loc='upper right')

            ax2 = self.fig4.add_subplot(gs[2, :])
            ax2.plot(t, res, '-', lw=0.8, color='#1f77b4')
            ax2.axhline(0, color='k', lw=0.8)
            ax2.axhline(mk['SD_residual'] * 2, color='r', ls='--', lw=0.8, label='±2σ')
            ax2.axhline(-mk['SD_residual'] * 2, color='r', ls='--', lw=0.8)
            ax2.set_ylabel('Residual'); ax2.set_xlabel('Waktu (s)')
            ax2.grid(True); ax2.legend(fontsize=8)
            ax2.set_title('Residual (eksperimen − model) terhadap Waktu', fontsize=10)
            ax3 = self.fig4.add_subplot(gs[3, 0])
            ax3.hist(res, bins=30, color='#2e8b57', alpha=0.85)
            ax3.set_title('Distribusi Residual', fontsize=10); ax3.set_xlabel('Residual'); ax3.grid(True)
            ax4 = self.fig4.add_subplot(gs[3, 1])
            ax4.plot(s, pos, '.', ms=3, alpha=0.6)
            lim = [min(np.min(s), np.min(pos)), max(np.max(s), np.max(pos))]
            ax4.plot(lim, lim, 'r--', lw=1)
            ax4.set_xlabel('Eksperimen'); ax4.set_ylabel('Model')
            ax4.set_title('Korelasi 1:1', fontsize=10); ax4.grid(True)
            self.fig4.tight_layout(); self.kanvas4.draw()
            self._status(f"Validasi selesai — R² = {mk['R2']:.4f}, RMSE = {mk['RMSE']:.3e}"
                         + (' (disesuaikan ke puncak ke-1 & ke-2)' if jg is not None and jg['ok'] else '')
                         + (f'  k = {k:.3f}' if abs(k - 1.0) > 1e-9 else ''))
        except Exception as exc:
            self._galat(exc)

    def _ekspor_validasi(self):
        if not self.validasi:
            messagebox.showwarning('Validasi', 'Jalankan validasi terlebih dahulu.'); return
        nama = filedialog.asksaveasfilename(defaultextension='.csv', filetypes=[('CSV', '*.csv')])
        if not nama:
            return
        v = self.validasi
        with open(nama, 'w', newline='', encoding='utf-8') as f:
            w = csv.writer(f)
            w.writerow(['#', f'sistem={NAMA_SISTEM}', f"RMSE={v['metrik']['RMSE']}",
                        f"R2={v['metrik']['R2']}"])
            w.writerow(['n', 't_s', 'eksperimen', 'model_rk4', 'residual'])
            for i in range(len(v['t'])):
                w.writerow([i, f"{v['t'][i]:.6f}", f"{v['eks'][i]:.6f}",
                            f"{v['rk4'][i]:.6f}", f"{v['res'][i]:.6f}"])
        self._status(f'Data validasi diekspor: {os.path.basename(nama)}')

    def _laporan(self):
        if not self.validasi:
            messagebox.showwarning('Validasi', 'Jalankan validasi terlebih dahulu.'); return
        nama = filedialog.asksaveasfilename(defaultextension='.txt', filetypes=[('Teks', '*.txt')])
        if not nama:
            return
        with open(nama, 'w', encoding='utf-8') as f:
            f.write('LAPORAN ANALISIS PANTULAN BOLA\n')
            f.write(f'Tanggal : {datetime.datetime.now():%Y-%m-%d %H:%M}\n')
            f.write(f'Sistem  : {NAMA_SISTEM}\n')
            f.write(f"Sumber  : {self.data['sumber']}\n\n")
            f.write('--- HASIL ANALISIS (koefisien restitusi) ---\n')
            f.write(self.txt3.get('1.0', 'end'))
            f.write('\n--- VALIDASI MODEL RK4 ---\n')
            f.write(self.txt4.get('1.0', 'end'))
        self._status(f'Laporan disimpan: {os.path.basename(nama)}')

    # =================================================================
    # TAB 6 — BANTUAN
    # =================================================================
    def _tab_bantuan(self):
        t = tk.Text(self.tab5, wrap='word', font=('Segoe UI', 10), padx=14, pady=10)
        sb = ttk.Scrollbar(self.tab5, orient='vertical', command=t.yview)
        t.configure(yscroll=sb.set)
        sb.pack(side='right', fill='y'); t.pack(fill='both', expand=True)
        t.insert('end', """PANDUAN SINGKAT — PANTULAN BOLA

ALUR KERJA
1. Tab 1 — isi parameter ideal: massa bola m, koefisien restitusi e, ketinggian
   awal y₀, kecepatan awal v₀, konstanta redaman udara c, gravitasi g, dan
   jenis hambatan udara (linier/kuadratik). Tekan "Hitung & Plot Prediksi
   Ideal" untuk melihat grafik POSISI terhadap WAKTU dan KECEPATAN terhadap
   WAKTU dari simulasi murni sesuai isian "Waktu akhir" (isian ini SELALU
   dipakai apa adanya, tidak lagi otomatis mengikuti durasi data eksperimen).
   Simulasi memakai metode RK4. Grafik posisi otomatis menandai PUNCAK tiap
   pantulan (▼, diberi nomor L/P1/P2… beserta tinggi & waktunya). Grafik kecepatan
   diberi warna hijau saat bola naik (v > 0) dan oranye saat turun (v < 0), titik
   merah = puncak (v = 0), serta penanda kecepatan tepat sebelum/sesudah tumbukan.
   Perbandingan dengan data eksperimen ada di Tab 4.
2. Tab 2 — pilih sumber data:
   • Kamera        : indeks 0 untuk webcam bawaan; kamera HP bisa lewat DroidCam/IP
                     (isi URL pada indeks kamera, mis. http://192.168.1.5:4747/video).
                     Kamera adalah aliran LANGSUNG (tidak bisa ditelusur mundur), jadi
                     track MANUAL/semi-otomatis tidak bisa langsung dari kamera. Untuk
                     itu: buka Kamera & Pratinjau, tekan "🎥 Rekam Video dari Kamera"
                     saat momen mulai (mis. sesaat sebelum bola dijatuhkan), lakukan
                     gerakannya, lalu tekan "⏹🎥 Selesai Rekam & Buka untuk Track" —
                     video rekaman otomatis terbuka sebagai PENELUSUR FRAME sehingga
                     bisa ditrack MANUAL, semi-otomatis, atau otomatis.
   • Berkas video  : rekaman pantulan bola yang sudah ada.
   • Impor gambar  : pilih banyak gambar sekaligus (urutan frame); FPS menentukan Δt.
   • Berkas CSV    : data lama atau hasil alat lain.
   • Arduino       : Serial mengirim "waktu,nilai" atau "nilai" saja (mis. sensor jarak).
   Setelah "Buka & Pratinjau", video/urutan gambar langsung terbuka dalam PENELUSUR
   FRAME: bar di bawah pratinjau berisi ⏮ ◀◀ ◀ ▶ ▶▶, slider posisi, dan tombol
   ▶ Putar/⏸ Jeda. Tombol panah ←/→ pada keyboard juga berpindah frame.

   Bila pratinjau HITAM atau bertuliskan frame tidak terbaca, artinya OpenCV tidak
   dapat mendekode berkasnya — paling sering pada rekaman HP ber-codec H.265/HEVC.
   Solusi: konversikan ke H.264,
       ffmpeg -i masukan.mp4 -c:v libx264 -pix_fmt yuv420p keluaran.mp4
   atau ekspor video menjadi urutan gambar lalu pakai "Impor gambar (banyak)".
   Bila jendela pratinjau meminta Pillow: pip install pillow.

   Lalu pilih CARA PELACAKAN:
   a) OTOMATIS — klik objek pada pratinjau untuk mengambil warnanya (atau pilih mode
      terang/gelap/gerak), lalu "Mulai Rekam". Untuk video/gambar pakai
      "Proses Seluruh Berkas" agar seluruh frame dilacak sendiri.
   b) SEMI-OTOMATIS — set klik ke "Pilih ROI / kotak objek", seret kotak tepat pada
      bola, tekan "Kunci Objek", lalu Mulai Rekam / Proses Seluruh Berkas.
   c) MANUAL — klik sendiri posisi bola di setiap frame (seperti digitasi Tracker).
      Frame maju otomatis sebanyak "Lompat tiap N frame" setiap kali Anda menandai.
      Navigasi: tombol ◀ ▶, slider, atau tombol panah ←/→ pada keyboard.
      Ctrl+Z atau "Hapus titik terakhir" membatalkan tanda.
3. KALIBRASI PENGGARIS & SUMBU x–y (kunci agar grafik stabil dan mulai dari tinggi yang benar):
   • Kamera di tripod, jarak tidak berubah. Pasang penggaris/meteran di bidang yang sama
     dengan lintasan bola (boleh berdiri tegak atau tergeletak di lantai).
   • PENGGARIS (skala): pilih klik "Penggaris", lalu SERET pada pratinjau dari titik 0
     ke ujung penggaris (mis. tanda 30 cm). Isi "Panjang acuan" (cm) sesuai jarak itu —
     nilainya bebas divariasikan (5, 10, 20, 30, 50, 100 cm atau ketik sendiri) dan skala
     px/m langsung dihitung ulang. Ujung dan badan garis bisa diseret lagi untuk
     menyesuaikan; "Tukar titik 0 ↔ ujung" menukar kedua ujungnya.
   • SUMBU x–y: pilih klik "Sumbu x–y". SUMBU x ADALAH LANTAI (y = 0); sumbu y tegak lurus
     ke atas. Seret ⊕ (asal) untuk menggeser, seret ● di ujung panah x untuk memutar
     (seperti sumbu koordinat pada Tracker), atau tekan-seret sepanjang lantai untuk
     menggambar sumbu baru. Sudut juga bisa diketik ("Sudut sumbu x"). "Pasang sumbu
     mengikuti penggaris" menaruh asal di titik 0 penggaris dan meluruskan sumbu mengikutinya.
   • Mode OTOMATIS / SEMI-OTOMATIS: penggaris & sumbu HARUS diatur SEBELUM merekam atau
     memproses berkas; sesudah itu terkunci selama dan sesudah data diambil (tekan
     "Hapus data" bila ingin mengatur ulang).
   • Mode MANUAL: penggaris & sumbu boleh diatur/diubah KAPAN SAJA, juga sesudah semua titik
     ditandai; grafik dihitung ulang dari piksel mentah saat sumbu digeser/diputar.
   • Isi "Jari-jari bola" agar y = tinggi dasar bola dari lantai (y = 0 saat bola
     menyentuh lantai, sama seperti model).
   • "Referensi nol" = "kalibrasi (sumbu x)" memakai lantai hasil kalibrasi (bukan
     titik terendah data). Judul grafik menampilkan tinggi awal; bandingkan dengan tinggi lepas
     sebenarnya. Pangkas-lepas membuat t = 0 tepat saat bola dilepas.
4. Tab 3 — Analisis Data (satu halaman):
   Analisis berjalan OTOMATIS saat Tab 3 dibuka (bila ada data baru); tombol
   "▶ Analisis Pantulan" untuk mengulang setelah mengubah pengaturan. Hasil muncul di tiga
   sub-tab: Grafik (data + fit parabola + puncak + titik tumbukan, dan peluruhan tinggi),
   Tabel Puncak & Tumbukan, serta Perhitungan & Rumus (bisa disalin/disimpan).
   Lantai: bawaannya y = 0 (sesuai Tab 2), sehingga kurva berhenti tepat di lantai.
   PUNCAK & LEMBAH MANUAL (cara termudah bila deteksi otomatis meleset): nyalakan
   "Klik: puncak → lembah → puncak → lembah", lalu klik berurutan dari kiri ke kanan; jenis
   titik bergantian otomatis dan menempel ke puncak/lembah data terdekat (klik pertama di titik
   lepas = P0). Klik kanan = urungkan. Tab 4 memakai lembah itu untuk mengunci waktu model.
   Bila ada puncak yang terlewat: nyalakan "Tambah puncak manual", lalu klik kiri pada titik
   puncaknya di grafik atas — titik itu langsung dihitung sebagai puncak, puncak lain TIDAK
   berubah atau hilang. Klik kanan dekat lingkaran merah muda = hapus puncak tambahan itu;
   "Urungkan" membatalkan yang terakhir; "Kembali ke puncak otomatis" membatalkan semuanya.
   Tekan "Terapkan e ke Validasi Model RK4" untuk memakai e di Tab 4.
   Tinggi lepas h0 memakai tinggi awal data yang sama dengan Tab 2 (puncak parabola jatuh
   pertama dikunci di sana; yang dicari hanya saat mulai jatuh dan kelengkungannya).
5. Tab 4 — Validasi Model RK4 (gabungan model terkoreksi + validasi):
   Hasil analisis/optimasi hanya mengoreksi e, c, dan v₀; y₀ model = tinggi awal data Tab 2
   (centang "Samakan y₀"), atau y₀ Tab 1 bila tidak dicentang. Grafik model = simulasi RK4
   murni seperti Tab 1 (posisi + kecepatan, puncak, tumbukan) dengan data eksperimen di
   atasnya. Model otomatis DISESUAIKAN ke puncak pantulan ke-1 & ke-2 data (tinggi -> e dan
   y₀ model, waktu -> skala & geseran waktu); puncak berikutnya mengikuti model. "Renggang –
   rapat pantulan": geser slider kiri/kanan atau ketik angka k untuk penyetelan halus — k > 1
   membuat pantulan lebih renggang, k < 1 lebih rapat. Tersedia: Validasi RK4 vs eksperimen (grafik overlay,
   residual ±2σ, histogram, korelasi 1:1, RMSE/MAE/R²/NRMSE, tabel data keseluruhan), Optimasi Parameter (fit e, c, v₀ dengan
   batas fisis dan galat robust), Uji Konvergensi RK4, ekspor CSV, dan laporan TXT.

MODEL YANG DIPAKAI
  Pantulan : y″ = −g − (c/m)·v (atau −(c/m)·v|v| untuk hambatan kuadratik)
             saat y < 0 → v ← −e·v   (pantulan dengan koefisien restitusi e)

TIPS PRAKTIS
  • Rekam dengan latar kontras dan bola berwarna mencolok; kunci fokus & pencahayaan.
  • FPS tinggi (≥60) penting agar puncak & lembah pantulan tidak terlewat.
  • Bila objek tidak terdeteksi: naikkan toleransi H, ganti mode (terang/gelap/gerak),
    perkecil ROI, atau beralih ke semi-otomatis/manual.
  • "Buat data sintetis" berguna untuk mencoba seluruh alur tanpa alat.

KENDALI GEARBOX (PEMICU REKAM)
  Di Tab 2 terdapat panel "Kendali Gearbox (Pemicu Rekam)". Bila kotak centang
  "Putar gearbox (buka) otomatis saat 'Mulai Rekam'" aktif, begitu tombol ⏺ Mulai Rekam
  (atau 🎥 Rekam Video dari Kamera) ditekan, aplikasi mengirim karakter BUKA
  (bawaan: "1", TANPA newline) ke ESP32 lewat serial. ESP32 memutar motor
  gearbox lewat driver L298N selama waktu90 ms, mengerem, lalu membalas lewat
  serial — cocok untuk mekanisme pelepas bola yang harus terbuka tepat saat
  perekaman dimulai. Perintah TUTUP (bawaan: "0") memutar balik gearbox ke
  posisi semula. Sketch ESP32 yang dipakai (kendali motor DC + L298N;
  pin EN = GPIO25, IN1 = GPIO26, IN2 = GPIO27):

      const int EN  = 25;
      const int IN1 = 26;
      const int IN2 = 27;
      const unsigned long waktu90  = 300;  // lama putar ±90° (ms)
      const unsigned long waktuRem = 50;   // lama pengereman (ms)
      bool posisiBuka = false;

      void putarMaju()   { digitalWrite(IN1, HIGH); digitalWrite(IN2, LOW);  digitalWrite(EN, HIGH); }
      void putarMundur() { digitalWrite(IN1, LOW);  digitalWrite(IN2, HIGH); digitalWrite(EN, HIGH); }
      void remAktif() {
        digitalWrite(IN1, LOW); digitalWrite(IN2, LOW); digitalWrite(EN, HIGH);
        delay(waktuRem);
        digitalWrite(EN, LOW);
      }
      void setup() {
        // level LOW dulu agar motor tidak bergerak saat ESP32 boot
        digitalWrite(EN, LOW); digitalWrite(IN1, LOW); digitalWrite(IN2, LOW);
        pinMode(EN, OUTPUT); pinMode(IN1, OUTPUT); pinMode(IN2, OUTPUT);
        digitalWrite(EN, LOW); digitalWrite(IN1, LOW); digitalWrite(IN2, LOW);
        Serial.begin(9600);
        Serial.println("Ketik 1 untuk membuka 90 derajat");
        Serial.println("Ketik 0 untuk kembali ke posisi semula");
      }
      void loop() {
        if (Serial.available() > 0) {
          char perintah = Serial.read();
          if (perintah == '\r' || perintah == '\n') return;   // abaikan newline
          if (perintah == '1' && !posisiBuka) {
            putarMaju(); delay(waktu90); remAktif();
            posisiBuka = true;
            Serial.println("Motor berhenti pada posisi buka");
          } else if (perintah == '0' && posisiBuka) {
            putarMundur(); delay(waktu90); remAktif();
            posisiBuka = false;
            Serial.println("Motor kembali ke posisi semula");
          }
        }
      }

    • Pilih "Port gearbox" dari daftar COM (klik 🔄 untuk memindai ulang), lalu
      tekan "🔌 Sambung". Kosongkan "Port gearbox" HANYA bila gearbox memakai
      papan (ESP32/Arduino) yang SAMA dengan sensor pada sumber "Arduino (serial)".
    • Baud gearbox harus SAMA dengan Serial.begin(...) di sketsa (bawaan 9600).
    • "🔓 Uji Buka" / "🔒 Uji Tutup" mengirim perintah dan menampilkan balasan
      dari ESP32 pada baris di bawah tombol.
    • Sketch menyimpan state posisiBuka: perintah "1" saat sudah terbuka dan
      "0" saat sudah tertutup DIABAIKAN tanpa balasan.
    • Membuka port serial me-reset ESP32 (lewat DTR/RTS) sehingga state kembali
      "tertutup". Pastikan gearbox secara fisik berada di posisi awal sebelum "Sambung".
    • Saat boot ESP32 mencetak log pada 115200 baud sehingga tampak sebagai karakter
      acak pada 9600 — normal dan tidak mengganggu.
    • GND ESP32, GND L298N, dan GND catu daya motor harus disatukan.
    • Bila putaran kurang/lebih dari 90°, ubah waktu90 di sketch (kurangi
      bertahap 400 → 300 → 200) lalu unggah ulang; waktuRem mengatur lama rem.
    • Selama motor berputar (±350 ms) ESP32 memakai delay() dan tidak membaca
      serial; bila sensor dan gearbox berbagi papan yang sama, data sensor
      terhenti sebentar pada saat itu.
    • Membutuhkan pyserial: pip install pyserial

SIMPAN SEMUA (GRAFIK + EXCEL + DOKUMENTASI VIDEO)
  Tombol "💾 Simpan Semua (Grafik + Excel)" di bar atas menyimpan dalam SATU folder
  bertanda waktu: seluruh grafik yang sudah terisi (Tab 1–4) sebagai PNG, satu berkas
  Excel "LabGetaran_Data.xlsx" (Data Eksperimen, Prediksi Ideal, Validasi RK4, Catatan
  Teks, Grafik), dan DOKUMENTASI VIDEO bila data berasal dari video/gambar:
    • folder dokumentasi_video/ : foto frame (JPG) bertanda titik track + jejak_semua_titik.jpg
    • lembar Excel "Titik Track": frame, t, posisi piksel, posisi meter tiap titik track
    • lembar Excel "Foto Video" : daftar foto beserta cuplikannya
  Maksimum 60 foto (dipilih merata dari seluruh titik track). Bila data berasal dari
  kamera langsung/CSV/Arduino/sintetis, foto tidak dibuat (titik track tetap disimpan
  bila ada). Membutuhkan openpyxl: pip install openpyxl

KEBUTUHAN
  pip install numpy scipy matplotlib opencv-python pillow pyserial openpyxl
""")
        t.config(state='disabled')

    # =================================================================
    def _galat(self, exc):
        traceback.print_exc()
        messagebox.showerror('Terjadi kesalahan', f'{type(exc).__name__}: {exc}')
        self._status(f'Galat: {exc}', '#c00')

    def _reset(self):
        if messagebox.askyesno('Reset', 'Hapus semua data dan hasil analisis?'):
            self._hapus_data()
            self.prediksi = None; self.validasi = None
            for tw in (self.txt3, self.txt4):
                tw.delete('1.0', 'end')
            for fig, kanvas in ((self.fig1, self.kanvas1), (self.fig3, self.kanvas3),
                                (self.fig4, self.kanvas4)):
                fig.clear(); kanvas.draw()
            for i in self.tv5.get_children():
                self.tv5.delete(i)
            self.kal = ak.Kalibrasi(px_per_m=self._f('px_per_m', 1000.0))
            self._seret = None
            self.var['sudut_x'].set('0.0')
            self._sinkron_kal_dari_form()
            self._ringkas_kalibrasi()
            self.pel_auto = ak.PelacakObjek()
            self.pel_csrt = ak.PelacakCSRT()
            self.pelacak = self.pel_csrt if self.cara == 'semi' else self.pel_auto
            self.titik_manual = {}
            self.kal_frame = {}
            self.manual_i = 0
            self._ringkas_koreksi()
            self._status('Semua direset.')

    def _titik_track(self):
        """Titik track mentah dari video: daftar dict (frame, t, piksel, meter)."""
        b = self._buf
        n = len(b.get('t', []))
        if n == 0 or not self._buf_dari_piksel():
            return []
        idx = b.get('i', [])
        fps = max(self._f('fps', 60.0), 1.0)
        daftar = []
        for k in range(n):
            if len(idx) == n and idx[k] is not None and idx[k] >= 0:
                fr = int(idx[k])
            else:
                fr = int(round(float(b['t'][k]) * fps))
            daftar.append({'n': k, 'frame': fr, 't': float(b['t'][k]),
                           'cx': float(b['cx'][k]), 'cy': float(b['cy'][k]),
                           'x': float(b['x'][k]), 'y': float(b['y'][k])})
        daftar.sort(key=lambda d: (d['frame'], d['t']))
        return daftar

    def _simpan_foto_track(self, folder, titik, maks=60):
        """Simpan foto frame video bertanda titik track. -> (daftar_foto, path_jejak, catatan)."""
        if not titik:
            return [], None, 'Tidak ada titik track dari video (data dari CSV/Arduino/sintetis).'
        if not self._dok_sumber:
            return [], None, 'Sumber video/gambar tidak tersedia (kamera langsung): foto tidak dibuat.'
        try:
            import cv2
        except Exception as exc:
            return [], None, f'OpenCV tidak tersedia: {exc}'
        tipe, arg = self._dok_sumber
        try:
            sumber = ak.SumberVideo(tipe, arg if tipe == 'video' else list(arg),
                                    self._f('fps', 60.0))
        except Exception as exc:
            return [], None, f'Sumber video tidak dapat dibuka ulang: {exc}'
        os.makedirs(folder, exist_ok=True)
        daftar, path_jejak, frame_pertama = [], None, None
        try:
            n = len(titik)
            pilih = sorted(set(int(round(v)) for v in np.linspace(0, n - 1, min(n, maks))))
            for k in pilih:
                p = titik[k]
                ok, frame, _ = sumber.ambil_frame(p['frame'])
                if not ok or frame is None:
                    continue
                if frame_pertama is None:
                    frame_pertama = frame
                riwayat = [(q['cx'], q['cy']) for q in titik[:k]]
                foto = ak.gambar_penanda(frame, (p['cx'], p['cy']),
                                         f"Frame {p['frame']}  t={p['t']:.3f} s", riwayat)
                nama = f"foto_{len(daftar) + 1:03d}_frame{p['frame']}.jpg"
                path = os.path.join(folder, nama)
                if cv2.imwrite(path, foto):
                    daftar.append({'nama': nama, 'path': path, 'frame': p['frame'],
                                   't': p['t'], 'cx': p['cx'], 'cy': p['cy']})
            if frame_pertama is not None:
                jejak = ak.gambar_penanda(frame_pertama, None, f'Jejak track: {n} titik',
                                          [(q['cx'], q['cy']) for q in titik])
                pj = os.path.join(folder, 'jejak_semua_titik.jpg')
                if cv2.imwrite(pj, jejak):
                    path_jejak = pj
        finally:
            sumber.lepas()
        return daftar, path_jejak, ('' if daftar else 'Frame video tidak dapat dibaca ulang.')

    def _simpan_semua(self):
        """Simpan SEMUA grafik (PNG), SEMUA data (Excel), dan dokumentasi video
        (foto frame bertanda titik track + tabel titik track) ke satu folder."""
        try:
            import openpyxl
            from openpyxl.drawing.image import Image as XLImage
        except Exception as exc:
            messagebox.showerror('Simpan Semua',
                                 'Modul openpyxl (dan/atau Pillow) belum terpasang atau gagal '
                                 f'dimuat: {type(exc).__name__}: {exc}\n\n'
                                 'pip install openpyxl pillow')
            return
        folder = filedialog.askdirectory(title='Pilih folder untuk menyimpan grafik, data & dokumentasi')
        if not folder:
            return
        tmp = None
        try:
            cap = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            sub = os.path.join(folder, f'LabGetaran_{cap}')
            os.makedirs(sub, exist_ok=True)
            tmp = tempfile.mkdtemp(prefix='labgetaran_')
            self._status('Menyimpan grafik, data, dan dokumentasi video…', '#a60')

            # ---------- 1) grafik PNG ----------
            grafik = [('01_parameter_prediksi_rk4', getattr(self, 'fig1', None)),
                      ('02_data_eksperimen', getattr(self, 'fig2', None)),
                      ('03_analisis_data', getattr(self, 'fig3', None)),
                      ('04_validasi_model_rk4', getattr(self, 'fig4', None))]
            berkas_png = []
            for nama, fig in grafik:
                if fig is not None and fig.get_axes():
                    path_png = os.path.join(sub, f'{nama}.png')
                    try:
                        fig.savefig(path_png, dpi=150, bbox_inches='tight')
                        berkas_png.append((nama, path_png))
                    except Exception:
                        traceback.print_exc()

            # ---------- 2) dokumentasi video: titik track + foto ----------
            titik = self._titik_track()
            foto, path_jejak, cat_foto = self._simpan_foto_track(
                os.path.join(sub, 'dokumentasi_video'), titik)

            # ---------- 3) Excel ----------
            wb = openpyxl.Workbook()
            ws0 = wb.active
            ws0.title = 'Ringkasan'
            ws0.append(['LAPORAN LAB GETARAN & GERAK — PANTULAN BOLA (RK4)'])
            ws0.append(['Tanggal', datetime.datetime.now().strftime('%Y-%m-%d %H:%M')])
            ws0.append(['Sistem', NAMA_SISTEM])
            ws0.append(['Sumber data', self.data.get('sumber', '-')])
            ws0.append([])
            ws0.append(['Lembar', 'Isi'])
            ws0.append(['Data Eksperimen', 'Sinyal t, x, y hasil rekaman/pelacakan (Tab 2)'])
            ws0.append(['Prediksi Ideal', 'Hasil simulasi RK4 model ideal (Tab 1)'])
            ws0.append(['Validasi RK4', 'Perbandingan eksperimen vs model RK4 (Tab 4)'])
            ws0.append(['Titik Track', 'Titik hasil tracking pada video: frame, t, piksel, meter'])
            ws0.append(['Foto Video', 'Foto frame video bertanda titik track (cuplikan)'])
            ws0.append(['Catatan Teks', 'Ringkasan teks hasil analisis dari Tab 3 dan 4'])
            ws0.append(['Grafik', 'Cuplikan seluruh grafik (PNG) yang sudah dibuat'])
            for kolom, lebar in (('A', 22), ('B', 60)):
                ws0.column_dimensions[kolom].width = lebar

            if self.data.get('t') is not None:
                ws = wb.create_sheet('Data Eksperimen')
                ws.append(['t (s)', 'x (m)', 'y (m)', 's (sinyal)'])
                t, x, y, sg = self.data['t'], self.data['x'], self.data['y'], self.data['s']
                for i in range(len(t)):
                    ws.append(_baris_aman([float(t[i]), float(x[i]), float(y[i]), float(sg[i])]))

            if self.prediksi:
                T, Y, _m = self.prediksi
                ws = wb.create_sheet('Prediksi Ideal')
                ws.append(['t (s)', 'posisi', 'kecepatan'])
                for i in range(len(T)):
                    ws.append(_baris_aman([float(T[i]), float(Y[i, 0]), float(Y[i, 1])]))

            if self.validasi:
                v = self.validasi
                ws = wb.create_sheet('Validasi RK4')
                ws.append(['n', 't (s)', 'eksperimen', 'model RK4', 'residual'])
                for i in range(len(v['t'])):
                    ws.append(_baris_aman([i, float(v['t'][i]), float(v['eks'][i]),
                               float(v['rk4'][i]), float(v['res'][i])]))
                ws.append([])
                for k, val in v['metrik'].items():
                    if np.isscalar(val) or isinstance(val, (int, float, np.floating, np.integer)):
                        ws.append(_baris_aman([k, val]))

            if titik:
                ws = wb.create_sheet('Titik Track')
                ws.append(['no', 'frame', 't (s)', 'x (px)', 'y (px)', 'x (m)', 'y (m)'])
                for i, p in enumerate(titik):
                    ws.append(_baris_aman([i + 1, p['frame'], p['t'], p['cx'], p['cy'],
                                           p['x'], p['y']]))
                for kolom, lebar in zip('ABCDEFG', (8, 9, 11, 11, 11, 11, 11)):
                    ws.column_dimensions[kolom].width = lebar

            if foto or path_jejak or cat_foto:
                import cv2
                wsf = wb.create_sheet('Foto Video')
                wsf.append(['no', 'frame', 't (s)', 'berkas foto', 'cuplikan'])
                for kolom, lebar in zip('ABCDE', (7, 9, 10, 34, 48)):
                    wsf.column_dimensions[kolom].width = lebar
                if cat_foto:
                    wsf.append(['', '', '', cat_foto])

                def tempel(path, baris):
                    img = cv2.imread(path)
                    if img is None:
                        return
                    hh, ww = img.shape[:2]
                    lebar = 320
                    kecil = cv2.resize(img, (lebar, max(int(hh * lebar / ww), 1)))
                    pt = os.path.join(tmp, os.path.basename(path))
                    cv2.imwrite(pt, kecil)
                    wsf.add_image(XLImage(pt), f'E{baris}')
                    wsf.row_dimensions[baris].height = kecil.shape[0] * 0.75 + 4

                if path_jejak:
                    wsf.append(['-', '-', '-', os.path.basename(path_jejak) + ' (jejak seluruh titik)'])
                    tempel(path_jejak, wsf.max_row)
                for j, f in enumerate(foto):
                    wsf.append([j + 1, f['frame'], f['t'], f['nama']])
                    if j < 40:                                   # batasi cuplikan di Excel
                        tempel(f['path'], wsf.max_row)

            ws = wb.create_sheet('Catatan Teks')
            baris = 1
            for judul, widget in (('HASIL ANALISIS (Tab 3)', getattr(self, 'txt3', None)),
                                  ('VALIDASI MODEL RK4 (Tab 4)', getattr(self, 'txt4', None))):
                ws.cell(row=baris, column=1, value=judul); baris += 1
                isi = widget.get('1.0', 'end').strip() if widget is not None else ''
                for baris_teks in (isi.splitlines() if isi else ['(kosong — belum dijalankan)']):
                    ws.cell(row=baris, column=1, value=baris_teks); baris += 1
                baris += 1
            ws.column_dimensions['A'].width = 100

            if berkas_png:
                wsg = wb.create_sheet('Grafik')
                baris_img = 1
                for nama, path_png in berkas_png:
                    wsg.cell(row=baris_img, column=1, value=nama)
                    try:
                        img = XLImage(path_png)
                        img.width = int(img.width * 0.5)
                        img.height = int(img.height * 0.5)
                        wsg.add_image(img, f'A{baris_img + 1}')
                    except Exception:
                        traceback.print_exc()
                    baris_img += 26

            path_xlsx = os.path.join(sub, 'LabGetaran_Data.xlsx')
            try:
                wb.save(path_xlsx)
            except PermissionError:
                messagebox.showerror(
                    'Simpan Semua',
                    f'Tidak bisa menulis "{os.path.basename(path_xlsx)}" — kemungkinan berkas '
                    'sedang terbuka di Excel, atau folder tujuan tidak bisa ditulis. '
                    'Tutup berkas Excel yang mungkin masih terbuka lalu coba lagi.')
                return

            self._status(f'Semua grafik, data & dokumentasi tersimpan di: {sub}')
            dok = (f'• Dokumentasi video: {len(foto)} foto + {len(titik)} titik track'
                   if foto else
                   (f'• Titik track: {len(titik)} titik (tanpa foto: {cat_foto})' if titik
                    else '• Dokumentasi video: tidak ada (data bukan dari video)'))
            messagebox.showinfo('Simpan Semua — Selesai',
                                f'Berhasil disimpan di:\n{sub}\n\n'
                                f'• {len(berkas_png)} berkas grafik (PNG)\n'
                                f'• 1 berkas Excel: LabGetaran_Data.xlsx\n{dok}')
        except Exception as exc:
            self._galat(exc)
        finally:
            if tmp:
                shutil.rmtree(tmp, ignore_errors=True)

    def _tutup(self):
        try:
            self.merekam = False
            self.pratinjau_aktif = False
            if self._vid_writer is not None:
                try:
                    self._vid_writer.release()
                except Exception:
                    pass
                self._vid_writer = None
            if self.sumber:
                self.sumber.lepas()
            if self.serial:
                self.serial.lepas()
            if self.servo_ser is not None:
                try:
                    self.servo_ser.close()
                except Exception:
                    pass
            try:
                import cv2
                cv2.destroyAllWindows()
            except Exception:
                pass
        finally:
            self.destroy()


if __name__ == '__main__':
    app = Aplikasi()
    app.mainloop()
