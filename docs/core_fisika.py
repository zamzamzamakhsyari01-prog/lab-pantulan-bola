"""
core_fisika.py  (KHUSUS PANTULAN BOLA)
======================================
  1. Integrator numerik : HANYA RK4 (gaya run_kut4)
  2. Model              : ModelPantulan (bola jatuh + pantul, hambatan opsional)
  3. Analisis data      : analisis_pantulan (fit parabola per penerbangan),
                          metrik_validasi, optimasi_parameter, kondisi_awal

Tumbukan diterapkan tepat saat bola menyentuh lantai (v -> -e v); analisis
tidak menghaluskan/menyunting data (puncak, kecepatan tumbukan, waktu kontak
dari fit parabola). Estimasi e: dari tinggi, kecepatan tumbukan, dan waktu terbang.
"""

import numpy as np

try:
    from scipy.optimize import curve_fit
    from scipy.signal import find_peaks as _sp_find_peaks
    SCIPY_ADA = True
except Exception:            # pragma: no cover
    SCIPY_ADA = False


# =====================================================================
# 1. INTEGRATOR NUMERIK — HANYA RK4  (gaya module run_kut4)
# =====================================================================
# run_kut4(F,x,y,h) mengembalikan PENAMBAH y (delta y) untuk satu langkah h,
# sehingga pembaruannya:  y = y + run_kut4(F,x,y,h)

def run_kut4(F, x, y, h):
    """Runge-Kutta orde 4 klasik."""
    K0 = h*F(x, y)
    K1 = h*F(x + h/2.0, y + K0/2.0)
    K2 = h*F(x + h/2.0, y + K1/2.0)
    K3 = h*F(x + h, y + K2)
    return (K0 + 2.0*K1 + 2.0*K2 + K3)/6.0


def _waktu_kontak(F, x, y, h, gfun):
    """
    Cari sub-langkah s in (0, h] sehingga g(y(s)) = 0 (bola tepat di lantai),
    memakai RK4 yang sama dengan langkah utama (bisection).
    """
    lo, hi = 0.0, h
    for _ in range(100):
        mid = 0.5*(lo + hi)
        if gfun(y + run_kut4(F, x, y, mid)) >= 0.0:
            lo = mid
        else:
            hi = mid
        if hi - lo <= 1e-15*max(h, 1e-300):
            break
    return hi


def integrate(F, x, y, xStop, h, event=None, kontak=None):
    ''' X,Y = integrate(F,x,y,xStop,h).
    Menggunakan metode Runge-Kutta Orde-4 untuk mencari solusi persamaan
    diferensial biasa {y}' = {F(x,{y})}, dimana {y} = {y[0],y[1],...,y[n-1]}.
    x,y = Kondisi awal
    xStop = nilai akhir x
    h = increment (interval) x yang digunakan
    F = Fungsi yang ingin dicari solusinya
    array F(x,y) = {y'[0],y'[1],...,y'[n-1]}.
    event  = (opsional) event(x,y) -> y terkoreksi setelah tiap langkah
    kontak = (opsional) (g, aksi) untuk tumbukan dengan lantai:
             g(y) >= 0 berarti belum menembus lantai;
             aksi(y) -> (y_baru, diam) dipanggil tepat saat menyentuh lantai.
    '''
    run = run_kut4

    y = np.asarray(y, dtype=float).copy()
    gfun, aksi = kontak if kontak is not None else (None, None)
    diam = False

    X = []
    Y = []
    X.append(x)
    Y.append(y.copy())
    while x < xStop - 1e-12 and not diam:
        h = min(h, xStop - x)
        x_akhir = x + h
        h_sisa = h
        for _ in range(100):                       # tumbukan dalam satu langkah
            y_baru = y + run(F, x, y, h_sisa)
            if gfun is not None and gfun(y) >= 0.0 and gfun(y_baru) < 0.0:
                s = _waktu_kontak(F, x, y, h_sisa, gfun)
                y, diam = aksi(y + run(F, x, y, s))
                x = x + s
                h_sisa = h_sisa - s
                X.append(x)
                Y.append(y.copy())
                if diam or h_sisa <= 1e-14:
                    break
            else:
                y = y_baru
                break
        if diam:
            break
        x = x_akhir
        if event is not None:
            y = event(x, y)
        X.append(x)
        Y.append(y.copy())

    if diam and x < xStop - 1e-12:                 # bola sudah diam di lantai
        X.append(xStop)
        Y.append(y.copy())

    return np.array(X), np.array(Y)


# =====================================================================
# 2. MODEL FISIKA
# =====================================================================

class ModelDasar:
    kunci = 'dasar'
    nama = 'Model Dasar'
    label_sinyal = 'Simpangan (m)'
    event = None
    kontak = None

    def F(self, t, y):
        raise NotImplementedError

    def y0(self):
        return np.array([self.x0, self.v0], dtype=float)

    def solusi_analitik(self, t):
        return None

    def info(self):
        return ''


class ModelPantulan(ModelDasar):
    """
    Bola jatuh + pantul:  y'' = -g - (hambatan)/m ,  tumbukan: v_sesudah = -e * v_sebelum
    tepat saat y = y_lantai.

    y        : tinggi (pusat bola) terhadap acuan pengukuran
    y_lantai : tinggi pusat bola saat menyentuh lantai (default 0; untuk data
               kamera biasanya = jari-jari bola terhadap lantai)
    v_diam   : kecepatan pantul di bawah ini dianggap berhenti (bola 'menggelinding').
               Hanya penghenti numerik untuk deret pantulan tak hingga (Zeno);
               tidak mengubah tinggi/waktu pantulan yang terlihat.
    """
    kunci = 'pantulan'
    nama = 'Pantulan (Bola Jatuh Bebas)'
    label_sinyal = 'Ketinggian y (m)'

    def __init__(self, m=0.05, e=0.8, c=0.0, x0=1.0, v0=0.0, g=9.81,
                 hambatan='linier', y_lantai=0.0, v_diam=1e-3):
        self.m, self.e, self.c = float(m), float(e), float(c)
        self.x0, self.v0, self.g = float(x0), float(v0), float(g)
        self.hambatan = hambatan
        self.y_lantai = float(y_lantai)
        self.v_diam = float(v_diam)
        self.event = None
        self.kontak = (self._g_lantai, self._tabrak)

    def F(self, t, y):
        v = y[1]
        if self.hambatan == 'kuadratik':
            drag = -(self.c / self.m) * v * abs(v)
        else:
            drag = -(self.c / self.m) * v
        return np.array([v, -self.g + drag])

    def _g_lantai(self, y):
        """>= 0 selama bola di atas/pada lantai."""
        return y[0] - self.y_lantai

    def _tabrak(self, y):
        """Tumbukan di lantai: posisi dipaku di lantai, v -> -e v. Berlaku untuk
        hambatan apa pun karena dihitung pada saat sentuh yang tepat."""
        y = np.array(y, dtype=float)
        y[0] = self.y_lantai
        v_baru = -self.e * y[1]
        if v_baru < self.v_diam:
            y[1] = 0.0
            return y, True
        y[1] = v_baru
        return y, False

    def info(self):
        g, e = self.g, self.e
        h0 = self.x0 - self.y_lantai
        v_c = np.sqrt(self.v0 ** 2 + 2 * g * max(h0, 0.0))     # laju saat sentuh pertama
        t1 = (self.v0 + v_c) / g                                 # waktu jatuh pertama
        t_total = t1 + 2 * e * v_c / (g * (1 - e)) if e < 1 else float('inf')
        return (f"e = {e:.4f} | t_jatuh_pertama = {t1:.4f} s | "
                f"t_total_teoretis = {t_total:.4f} s (tanpa hambatan) | "
                f"tinggi pantulan ke-n = h0*e^(2n) (v0=0)")


def buat_model(kunci, p):
    """Pabrik model dari dictionary parameter (hanya sistem pantulan)."""
    if str(kunci).lower() != 'pantulan':
        raise ValueError("Modul ini khusus pantulan bola; kunci harus 'pantulan'.")
    return ModelPantulan(m=p['m'], e=p['e'], c=p['c'], x0=p['x0'], v0=p['v0'],
                         g=p.get('g', 9.81), hambatan=p.get('hambatan', 'linier'),
                         y_lantai=p.get('y_lantai', 0.0))


def simulasi(model, t_akhir, h, t_awal=0.0):
    """Jalankan integrasi RK4 untuk sebuah model -> (T, Y)."""
    return integrate(model.F, t_awal, model.y0(), t_akhir, h,
                     event=getattr(model, 'event', None),
                     kontak=getattr(model, 'kontak', None))


def simulasi_pada_grid(model, t_grid, h=None):
    """Simulasi RK4 lalu interpolasi ke grid waktu data eksperimen."""
    t_grid = np.asarray(t_grid, dtype=float)
    if len(t_grid) < 2:
        raise ValueError('Grid waktu terlalu pendek.')
    dt = np.median(np.diff(t_grid))
    if h is None:
        h = max(dt / 20.0, 1e-5)
    T, Y = simulasi(model, float(t_grid[-1] - t_grid[0]) + dt, h)
    T = T + t_grid[0]
    pos = np.interp(t_grid, T, Y[:, 0])
    vel = np.interp(t_grid, T, Y[:, 1])
    return pos, vel, T, Y


def _jatuh_ke_lantai(model, y_awal, h, t_maks=60.0):
    """RK4 dari kondisi awal sampai TEPAT menyentuh lantai (tanpa tumbukan).
    Mengembalikan (T, Y, kena); T dihitung dari 0."""
    F, gf = model.F, model._g_lantai
    y = np.array(y_awal, dtype=float).copy()
    t = 0.0
    T, Y = [0.0], [y.copy()]
    if gf(y) <= 0.0 and y[1] <= 0.0:
        return np.array(T), np.array(Y), True
    while t < t_maks:
        y_baru = y + run_kut4(F, t, y, h)
        if gf(y_baru) < 0.0:
            s = _waktu_kontak(F, t, y, h, gf)
            y = y + run_kut4(F, t, y, s)
            y[0] = model.y_lantai
            t += s
            T.append(t); Y.append(y.copy())
            return np.array(T), np.array(Y), True
        y = y_baru
        t += h
        T.append(t); Y.append(y.copy())
    return np.array(T), np.array(Y), False


def _durasi_terbang(model, u, h, t_maks):
    """Lama terbang (RK4) bola yang dilempar ke atas dari lantai dengan laju u
    sampai kembali ke lantai."""
    F, gf = model.F, model._g_lantai
    y = np.array([model.y_lantai, u], dtype=float)
    t = 0.0
    while t < t_maks:
        y_baru = y + run_kut4(F, t, y, h)
        if gf(y_baru) < 0.0:
            return t + _waktu_kontak(F, t, y, h, gf)
        y = y_baru
        t += h
    return t_maks


def _laju_untuk_durasi(model, D, h):
    """Laju lepas dari lantai agar bola (RK4) kembali ke lantai tepat setelah D detik.
    Tanpa hambatan: u = g D / 2 (eksak). Dengan hambatan: bisection pada RK4."""
    u0 = 0.5 * model.g * D
    if model.c <= 0.0:
        return u0
    h_e = min(h, D / 120.0)
    lo, hi = u0, 2.0 * u0
    for _ in range(10):                                  # pastikan lo terlalu lambat
        if _durasi_terbang(model, lo, h_e, 4.0 * D) < D:
            break
        lo *= 0.5
    for _ in range(30):                                  # pastikan hi terlalu cepat
        if _durasi_terbang(model, hi, h_e, 4.0 * D) >= D:
            break
        lo, hi = hi, hi * 2.0
    for _ in range(45):
        mid = 0.5 * (lo + hi)
        if _durasi_terbang(model, mid, h_e, 4.0 * D) < D:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-10 * hi:
            break
    return 0.5 * (lo + hi)


def simulasi_waktu_kontak(model, t_kontak, t_akhir, h):
    """RK4 dengan WAKTU TUMBUKAN DIATUR (mis. dari data eksperimen / input manual).

    Mengapa: pada model murni, waktu tumbukan ditentukan e dan y0; selisih kecil (FPS,
    stempel waktu, y0) membuat fase pantulan menjauh dan residual 'melesat'. Di sini
    waktu tumbukan dipatok, dan RK4 tetap dipakai untuk fisikanya:
      * jatuh pertama : RK4 dari (y0, v0) hingga lantai; saat lepas model digeser
                        (t_lepas = t_kontak[0] - waktu_jatuh), sebelumnya bola diam di y0;
      * tiap terbang  : laju lepas dicari (RK4 + bisection bila ada hambatan) agar bola
                        kembali ke lantai tepat di waktu tumbukan berikutnya;
                        e tiap tumbukan = laju lepas / laju datang (hasil, bukan input);
      * sesudah tumbukan terakhir : RK4 biasa dengan e = median e yang diperoleh.
    Mengembalikan (T, Y, info)."""
    tk = np.sort(np.asarray([v for v in t_kontak if np.isfinite(v)], dtype=float))
    if len(tk) > 1:
        tk = tk[np.insert(np.diff(tk) > 1e-6, 0, True)]
    if len(tk) == 0:
        raise ValueError('Waktu tumbukan kosong.')
    yl = model.y_lantai
    T0, Y0, kena = _jatuh_ke_lantai(model, model.y0(), h)
    if not kena:
        raise ValueError('Bola tidak mencapai lantai dari kondisi awal (periksa y0 dan v0).')
    tau1 = float(T0[-1])
    t_lepas = float(tk[0]) - tau1
    T_list, Y_list = [T0 + t_lepas], [Y0]
    v_sb = float(Y0[-1, 1])
    e_list, u_list, v_list = [], [], [v_sb]
    for k in range(len(tk) - 1):
        D = float(tk[k + 1] - tk[k])
        u = _laju_untuk_durasi(model, D, h)
        Tf, Yf = integrate(model.F, 0.0, np.array([yl, u]), D, min(h, D / 50.0))
        Tk = tk[k] + Tf
        Tk[0] += 1e-9
        T_list.append(Tk); Y_list.append(Yf)
        e_list.append(u / abs(v_sb) if v_sb != 0.0 else float('nan'))
        u_list.append(u)
        v_sb = float(Yf[-1, 1])
        v_list.append(v_sb)
    ok = [e for e in e_list if np.isfinite(e)]
    e_t = float(np.clip(np.median(ok), 0.05, 0.99)) if ok else float(np.clip(model.e, 0.05, 0.99))
    t_mulai = float(tk[-1])
    if t_akhir > t_mulai + 1e-9:
        u = e_t * abs(v_sb)
        if u > model.v_diam:
            m2 = ModelPantulan(m=model.m, e=e_t, c=model.c, x0=yl, v0=0.0, g=model.g,
                               hambatan=model.hambatan, y_lantai=yl, v_diam=model.v_diam)
            Tf, Yf = integrate(m2.F, t_mulai, np.array([yl, u]), t_akhir, h, kontak=m2.kontak)
            Tf = Tf.copy()
            Tf[0] += 1e-9
        else:
            Tf = np.array([t_mulai + 1e-9, t_akhir])
            Yf = np.array([[yl, 0.0], [yl, 0.0]])
        T_list.append(Tf); Y_list.append(Yf)
    T = np.concatenate(T_list)
    Y = np.vstack(Y_list)
    info = {'t_lepas': t_lepas, 'tau_jatuh': tau1, 't_kontak': tk,
            'e': np.array(e_list, dtype=float), 'u': np.array(u_list, dtype=float),
            'v_sebelum': np.array(v_list, dtype=float), 'e_lanjut': e_t}
    return T, Y, info


def simulasi_pada_grid_kontak(model, t_grid, t_kontak, h=0.001):
    """simulasi_waktu_kontak lalu interpolasi ke grid waktu data -> (pos, vel, T, Y, info)."""
    t_grid = np.asarray(t_grid, dtype=float)
    if len(t_grid) < 2:
        raise ValueError('Grid waktu terlalu pendek.')
    dt = float(np.median(np.diff(t_grid)))
    T, Y, info = simulasi_waktu_kontak(model, t_kontak, float(t_grid[-1]) + dt, h)
    pos = np.interp(t_grid, T, Y[:, 0])
    vel = np.interp(t_grid, T, Y[:, 1])
    return pos, vel, T, Y, info


def puncak_simulasi(T, Y):
    """Puncak (titik tertinggi tiap penerbangan) dari hasil simulasi RK4.
    Puncak = saat kecepatan berubah dari positif ke <= 0 (v = 0), waktunya
    diinterpolasi dan tingginya dihitung dengan percepatan konstan di dalam langkah.
    Bila v0 <= 0 (bola dilepas / dilempar ke bawah) titik awal dihitung sebagai
    puncak jenis 'lepas'. Mengembalikan daftar (t, y, jenis)."""
    T = np.asarray(T, dtype=float)
    y = Y[:, 0]
    v = Y[:, 1]
    hasil = []
    if len(T) and v[0] <= 0.0:
        hasil.append((float(T[0]), float(y[0]), 'lepas'))
    for i in range(1, len(T)):
        if v[i - 1] > 0.0 and v[i] <= 0.0:
            H = T[i] - T[i - 1]
            f = v[i - 1] / (v[i - 1] - v[i])
            dt = f * H
            yp = y[i - 1] + v[i - 1] * dt + (v[i] - v[i - 1]) / (2.0 * H) * dt * dt
            hasil.append((float(T[i - 1] + dt), float(yp), 'pantulan'))
    return hasil


def tumbukan_simulasi(T, Y, e):
    """Tumbukan dengan lantai pada hasil simulasi: (t, v_sebelum, v_sesudah).
    v_sesudah diambil dari data; v_sebelum = -v_sesudah / e (karena v -> -e v)."""
    T = np.asarray(T, dtype=float)
    v = Y[:, 1]
    hasil = []
    for i in range(1, len(T)):
        if v[i - 1] < 0.0 and v[i] > 0.0:
            va = float(v[i])
            vb = -va / e if e > 0 else float('nan')
            hasil.append((float(T[i]), vb, va))
    return hasil


# =====================================================================
# 3. ANALISIS DATA
# =====================================================================

def regresi_linear(x, y):
    """Kembalikan dict: slope, intercept, r2, galat slope & intercept."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)
    if n < 2:
        raise ValueError('Butuh minimal 2 titik data.')
    A = np.vstack([x, np.ones(n)]).T
    koef, *_ = np.linalg.lstsq(A, y, rcond=None)
    m_, b_ = koef
    y_fit = m_ * x + b_
    ss_res = float(np.sum((y - y_fit) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float('nan')
    if n > 2:
        s2 = ss_res / (n - 2)
        sxx = float(np.sum((x - x.mean()) ** 2))
        sm = np.sqrt(s2 / sxx) if sxx > 0 else float('nan')
        sb = np.sqrt(s2 * (1.0 / n + x.mean() ** 2 / sxx)) if sxx > 0 else float('nan')
    else:
        sm = sb = float('nan')
    return {'slope': float(m_), 'intercept': float(b_), 'r2': float(r2),
            'galat_slope': float(sm), 'galat_intercept': float(sb),
            'y_fit': y_fit}


def cari_puncak(t, x, jarak_min=None, prominence_frac=0.05):
    """Indeks puncak (maksimum lokal) sinyal. Pakai scipy bila ada."""
    t = np.asarray(t, dtype=float)
    x = np.asarray(x, dtype=float)
    rentang = float(np.max(x) - np.min(x))
    prom = max(prominence_frac * rentang, 1e-12)
    if SCIPY_ADA:
        kw = {'prominence': prom}
        if jarak_min:
            dt = np.median(np.diff(t))
            kw['distance'] = max(int(jarak_min / dt), 1)
        idx, _ = _sp_find_peaks(x, **kw)
        return np.asarray(idx, dtype=int)
    idx = [i for i in range(1, len(x) - 1)
           if x[i] > x[i - 1] and x[i] >= x[i + 1] and (x[i] - np.min(x)) > prom]
    return np.array(idx, dtype=int)


# ---------------------------------------------------------------------
# Analisis pantulan (berbasis fit parabola per penerbangan)
# ---------------------------------------------------------------------

def _rerata_aman(v):
    v = np.asarray(v, dtype=float)
    v = v[np.isfinite(v)]
    return float(np.mean(v)) if len(v) else float('nan')


def _akar_di_lantai(P, y_lantai):
    """
    Parabola P=(a,b,c,t_ref): y = a*tau^2 + b*tau + c, tau = t - t_ref.
    Kembalikan (t_naik, t_turun): waktu saat y = y_lantai pada cabang naik
    (sebelum puncak) dan cabang turun (sesudah puncak). None bila tak ada akar.
    """
    a, b, c, tr = P
    disc = b * b - 4.0 * a * (c - y_lantai)
    if disc < 0.0 or a == 0.0:
        return None
    s = np.sqrt(disc)
    tau_turun = (-b - s) / (2.0 * a)      # a<0 -> akar yang lebih besar
    tau_naik = (-b + s) / (2.0 * a)
    return tr + tau_naik, tr + tau_turun


def _potong_parabola(P1, P2, t_lo, t_hi):
    """Titik potong dua parabola di dalam [t_lo, t_hi] (mis. di titik tumbukan)."""
    a1, b1, c1, r1 = P1
    a2, b2, c2, r2 = P2
    d = r2 - r1
    b2p = b2 - 2.0 * a2 * d
    c2p = a2 * d * d - b2 * d + c2
    A, B, C = a1 - a2, b1 - b2p, c1 - c2p
    if abs(A) < 1e-6 * max(abs(a1), abs(a2), 1e-12):
        if abs(B) < 1e-12:
            return None
        akar = [-C / B]
    else:
        disc = B * B - 4.0 * A * C
        if disc < 0.0:
            return None
        s = np.sqrt(disc)
        akar = [(-B + s) / (2.0 * A), (-B - s) / (2.0 * A)]
    tengah = 0.5 * (t_lo + t_hi)
    t_akar = [r1 + tau for tau in akar]
    t_c = min(t_akar, key=lambda tt: abs(tt - tengah))
    if not (t_lo - 1e-9 <= t_c <= t_hi + 1e-9):
        return None
    tau = t_c - r1
    return float(t_c), float(a1 * tau * tau + b1 * tau + c1)


def _fit_parabola_puncak_tetap(ts, ys, y_puncak, tr, t_jatuh_perkiraan):
    """Fit parabola jatuh pertama dengan TINGGI PUNCAK DIKUNCI = y_puncak (tinggi awal
    data / Tab 2):  y = y_puncak + a (t - t_v)^2,  a < 0.  Bebas: a dan t_v (saat bola mulai
    jatuh). Untuk t_v tertentu a dicari eksak (kuadrat terkecil), t_v dicari dengan kisi
    yang dipersempit bertahap. Mengembalikan (a, b, c, tr) seperti parabola lain atau None."""
    ts = np.asarray(ts, dtype=float)
    d = np.asarray(ys, dtype=float) - float(y_puncak)

    def biaya(G):
        q = (ts[None, :] - G[:, None]) ** 2
        a = (q * d[None, :]).sum(1) / np.maximum((q * q).sum(1), 1e-300)
        c = ((d[None, :] - a[:, None] * q) ** 2).sum(1)
        return np.where(a < 0.0, c, np.inf), a

    hi = float(ts[0])
    lo = hi - max(0.6 * float(t_jatuh_perkiraan), 3.0 * float(np.median(np.diff(ts))) if len(ts) > 1 else 0.05)
    best_tv, best_a = None, None
    for _ in range(4):
        G = np.linspace(lo, hi, 801)
        c, a = biaya(G)
        i = int(np.argmin(c))
        if not np.isfinite(c[i]):
            return None
        best_tv, best_a = float(G[i]), float(a[i])
        lo, hi = float(G[max(i - 1, 0)]), float(G[min(i + 1, len(G) - 1)])
    sh = tr - best_tv
    return (best_a, 2.0 * best_a * sh, best_a * sh * sh + float(y_puncak), float(tr))


def analisis_pantulan(t, y, g=9.81, y_lantai=None, prominence_frac=0.02,
                      t_puncak_manual=None):
    """
    Analisis data ketinggian bola yang dipantulkan, tanpa menghaluskan data.

    Prinsip (teori): di antara dua tumbukan bola bergerak sebagai parabola
        y(t) = y_v - (g/2)(t - t_v)^2      (tanpa hambatan)
    sehingga puncak, kecepatan tumbukan, dan waktu kontak diperoleh dari fit
    parabola pada titik-titik data (bukan dari satu sampel puncak, karena
    kamera tidak selalu merekam tepat di puncak/lantai).

    Keluaran koefisien restitusi, tiga cara bebas (harus saling mendekati
    bila teori terpenuhi):
      e_tiap / e_rerata / e_regresi : dari tinggi puncak,  e = sqrt(h_{n+1}/h_n)
      e_kecepatan / e_kec_rerata    : definisi langsung,   e = -v_sesudah/v_sebelum
                                      di titik tumbukan (tetap benar bila ada
                                      hambatan udara)
      e_waktu / e_waktu_rerata      : dari waktu terbang, e = T_{n+1}/T_n
                                      (pantulan pertama: T_1/(2 t_jatuh))

    t_puncak_manual : daftar waktu puncak pilihan pengguna (menggantikan deteksi
               otomatis); tiap waktu menempel ke sampel tertinggi di sekitarnya.
    y_lantai : tinggi pusat bola saat menyentuh lantai. Bila None, diestimasi
               dari titik potong parabola-parabola berurutan (dan jika tidak
               memungkinkan, dari min(y)).
    Asumsi: bila rekaman dimulai dengan bola diam lalu dilepas, tinggi lepas h0
    otomatis ikut dihitung (diperoleh dari fit parabola jatuh pertama).
    """
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(t)
    hasil = {'berhasil': False, 'idx_puncak': np.array([], int)}
    if n < 8:
        hasil['pesan'] = 'Data terlalu sedikit.'
        return hasil

    dt = float(np.median(np.diff(t)))
    rentang = float(np.max(y) - np.min(y))
    y_min = float(np.min(y))
    y_awal = float(np.median(y[:3]))      # tinggi awal data = yang ditampilkan di Tab 2
    # derau diestimasi dari beda-kedua (kokoh terhadap lonjakan di titik tumbukan)
    d2 = y[2:] - 2.0 * y[1:-1] + y[:-2]
    sigma = float(1.4826 * np.median(np.abs(d2 - np.median(d2))) / np.sqrt(6.0))
    prom_eff = max(prominence_frac * rentang, 4.0 * sigma) / max(rentang, 1e-12)
    if t_puncak_manual is not None:
        # puncak ditentukan pengguna: tiap waktu 'menempel' ke sampel tertinggi di sekitarnya
        w = max(4.0 * dt, 0.04 * float(t[-1] - t[0]))
        pilih = []
        for tp in sorted(float(v) for v in t_puncak_manual):
            m = np.where(np.abs(t - tp) <= w)[0]
            if len(m):
                pilih.append(int(m[np.argmax(y[m])]))
        idx = np.array(sorted(set(pilih)), dtype=int)
    else:
        idx = cari_puncak(t, y, prominence_frac=prom_eff)
    hasil['idx_puncak'] = idx
    hasil['puncak_manual'] = t_puncak_manual is not None

    # --- apakah rekaman dimulai dari bola yang dilepas? ---
    ambang = 0.02 * rentang
    turun = np.where(y < y[0] - ambang)[0]
    n0 = int(turun[0]) if len(turun) else None
    ada_lepas = (n0 is not None) and (len(idx) >= 1) and (idx[0] > n0)

    ap_idx = ([0] if ada_lepas else []) + [int(i) for i in idx]
    if len(ap_idx) < 2:
        hasil['pesan'] = 'Butuh minimal 2 puncak (termasuk titik lepas).'
        return hasil

    valleys = []
    for k in range(len(ap_idx) - 1):
        a, b = ap_idx[k], ap_idx[k + 1]
        valleys.append(a + int(np.argmin(y[a:b + 1])))

    # --- fit parabola per penerbangan ---
    def _fit_semua(g_pakai):
        P_l, tv_l, yv_l, yr_l, nf_l = [], [], [], [], []
        for k, ap in enumerate(ap_idx):
            lepas_k = ada_lepas and k == 0
            lo = valleys[k - 1] if k > 0 else -1
            hi = valleys[k] if k < len(valleys) else n
            y_puncak = y_awal if lepas_k else y[ap]
            H = max(y_puncak - y_min, 1e-9)
            lantai_kasar = y_min + 0.05 * H           # buang titik zona kontak

            if lepas_k:
                sel = [i for i in range(0, hi) if y[i] < y[0] - ambang and y[i] > lantai_kasar]
                tr = t[sel[0]] if sel else t[0]
            else:
                w = max(0.8 * np.sqrt(2.0 * H / g_pakai), 1.5 * dt)
                sel = [i for i in range(lo + 1, hi)
                       if abs(t[i] - t[ap]) <= w and y[i] > lantai_kasar]
                if len(sel) < 3:
                    sel = list(range(max(ap - 1, lo + 1), min(ap + 2, hi)))
                tr = t[ap]

            P = None
            if len(sel) >= 3:
                if lepas_k:
                    # tinggi lepas DIKUNCI = tinggi awal data (sama dengan Tab 2); yang dicari
                    # hanya saat mulai jatuh dan kelengkungan dari titik-titik jatuh pertama
                    P = _fit_parabola_puncak_tetap(t[sel], y[sel], y_awal, tr,
                                                   np.sqrt(2.0 * H / g_pakai))
                if P is None:
                    a_, b_, c_ = np.polyfit(t[sel] - tr, y[sel], 2)
                    if np.isfinite(a_) and a_ < 0.0:
                        P = (float(a_), float(b_), float(c_), float(tr))
            P_l.append(P)
            nf_l.append(len(sel) if P is not None else 0)
            if P is not None:
                a_, b_, c_, _ = P
                tv_l.append(tr + (-b_ / (2.0 * a_)))
                yv_l.append(c_ - b_ * b_ / (4.0 * a_))
            else:                                       # cadangan: sampel mentah
                tv_l.append(float(t[ap]))
                yv_l.append(float(y_puncak))
            yr_l.append(float(y_puncak))
        return P_l, tv_l, yv_l, yr_l, nf_l

    def _g_median(P_l, nf_l):
        gs = [-2.0 * P[0] for k, P in enumerate(P_l)
              if P is not None and nf_l[k] >= 3 and not (ada_lepas and k == 0)]
        return float(np.median(gs)) if gs else float('nan')

    # Percepatan "g" dalam satuan data bisa berbeda dari 9,81 bila kalibrasi px/m belum
    # tepat. Karena itu lebar jendela fit diatur ulang memakai g hasil fit (2 putaran),
    # dan kewajaran puncak dinilai relatif terhadap g hasil fit, bukan terhadap 9,81.
    P_list, t_v, y_v, y_raw, n_fit = _fit_semua(g)
    g_data = _g_median(P_list, n_fit)
    if np.isfinite(g_data) and abs(g_data / g - 1.0) > 0.2:
        P_list, t_v, y_v, y_raw, n_fit = _fit_semua(g_data)
        g_data = _g_median(P_list, n_fit)
    g_ref = g_data if np.isfinite(g_data) else g

    # --- periksa kewajaran tiap puncak pantulan (derau saat bola hampir diam) ---
    diagnosa = []
    kcut = len(P_list)
    for k in range(len(P_list)):
        if ada_lepas and k == 0:
            continue
        P = P_list[k]
        g_k = -2.0 * P[0] if P is not None else float('nan')
        alasan = []
        if P is None:
            alasan.append('fit parabola gagal')
        if n_fit[k] < 3:
            alasan.append(f'titik fit {n_fit[k]} < 3 (data terlalu jarang)')
        if P is not None and not (0.5 * g_ref <= g_k <= 1.8 * g_ref):
            alasan.append(f'kelengkungan g_fit={g_k:.2f} jauh dari g tipikal {g_ref:.2f}')
        if (y_v[k] - y_min) <= 5.0 * sigma:
            alasan.append(f'tinggi {(y_v[k]-y_min):.4f} <= 5 sigma derau ({5*sigma:.4f})')
        diagnosa.append({'no': (k if ada_lepas else k + 1), 'g_fit': g_k,
                         'n_fit': int(n_fit[k]), 'h': float(y_v[k] - y_min),
                         'valid': not alasan, 'alasan': '; '.join(alasan)})
        if alasan and kcut == len(P_list):
            kcut = k
    hasil['diagnosa'] = diagnosa
    n_dibuang = 0
    peringatan = []
    if kcut < len(P_list):
        if kcut >= 2:
            n_dibuang = len(P_list) - kcut
            P_list, t_v, y_v, y_raw, n_fit = (P_list[:kcut], t_v[:kcut], y_v[:kcut],
                                               y_raw[:kcut], n_fit[:kcut])
            ap_idx = ap_idx[:kcut]
        else:
            # Pemeriksaan ketat akan menyisakan < 2 puncak. Jangan gagal total:
            # pakai semua puncak yang terdeteksi, tetapi beri peringatan.
            peringatan.append('Validasi dilonggarkan: puncak pertama dinilai tidak wajar '
                              f'({diagnosa[0]["alasan"] if diagnosa else "-"}). '
                              'Hasil e mungkin kurang andal.')
    if len(P_list) < 2:
        hasil['pesan'] = 'Puncak yang terdeteksi kurang dari 2 (data terlalu pendek / tidak ada pantulan).'
        return hasil
    if abs(g_ref / g - 1.0) > 0.15:
        peringatan.append(f'g hasil fit = {g_ref:.2f} (acuan {g:g}): skala data kemungkinan belum '
                          f'terkalibrasi. Kalikan px/m dengan {g_ref / g:.3f} agar g sesuai. '
                          '(Nilai e tidak terpengaruh skala.)')
    hasil['peringatan'] = peringatan
    hasil['idx_puncak'] = np.asarray(ap_idx[1 if ada_lepas else 0:], dtype=int)
    t_v, y_v = np.array(t_v), np.array(y_v)

    # --- lantai: dari titik potong parabola berurutan ---
    y_pot = []
    for k in range(len(P_list) - 1):
        if P_list[k] is not None and P_list[k + 1] is not None:
            r = _potong_parabola(P_list[k], P_list[k + 1], t_v[k], t_v[k + 1])
            if r is not None:
                y_pot.append(r[1])
    if y_lantai is not None:
        sumber = 'diberikan'
    elif y_pot:
        y_lantai, sumber = float(np.median(y_pot)), 'titik potong parabola'
    else:
        y_lantai, sumber = y_min, 'min(y)'
    y_lantai = float(y_lantai)

    # --- tinggi puncak & e dari tinggi ---
    h_semua = np.clip(y_v - y_lantai, 1e-12, None)
    rasio = h_semua[1:] / h_semua[:-1]
    e_tiap = np.sqrt(np.clip(rasio, 0, None))
    reg = regresi_linear(np.arange(len(h_semua)), np.log(h_semua))   # ln h_n = ln h0 + 2n ln e
    e_reg = float(np.exp(reg['slope'] / 2.0))

    # --- waktu kontak, kecepatan tumbukan ---
    m = len(P_list) - 1
    t_c = np.full(m, np.nan)
    sel_c = np.full(m, np.nan)
    v_sblm = np.full(m, np.nan)
    v_ssdh = np.full(m, np.nan)
    for k in range(m):
        if P_list[k] is None or P_list[k + 1] is None:
            continue
        r1 = _akar_di_lantai(P_list[k], y_lantai)
        r2 = _akar_di_lantai(P_list[k + 1], y_lantai)
        if r1 is None or r2 is None:
            continue
        t_turun, t_naik = r1[1], r2[0]
        a1, b1, _, tr1 = P_list[k]
        a2, b2, _, tr2 = P_list[k + 1]
        t_c[k] = 0.5 * (t_turun + t_naik)
        sel_c[k] = abs(t_turun - t_naik)
        v_sblm[k] = 2.0 * a1 * (t_turun - tr1) + b1
        v_ssdh[k] = 2.0 * a2 * (t_naik - tr2) + b2
    with np.errstate(divide='ignore', invalid='ignore'):
        e_kec = np.where(v_sblm < 0, -v_ssdh / v_sblm, np.nan)

    # --- e dari waktu terbang ---
    T = np.diff(t_c) if m >= 2 else np.array([])
    e_waktu = []
    if len(T) >= 1:
        if ada_lepas:
            if P_list[0] is not None and np.isfinite(t_c[0]) and t_c[0] > t_v[0]:
                e_waktu.append(T[0] / (2.0 * (t_c[0] - t_v[0])))
            else:
                e_waktu.append(np.nan)           # jaga keselarasan indeks dengan tumbukan
        if len(T) >= 2:
            e_waktu.extend(list(T[1:] / T[:-1]))
    e_waktu = np.array(e_waktu, dtype=float)

    # --- percepatan gravitasi dari kelengkungan parabola (a = -g/2) ---
    g_fit = [-2.0 * P[0] for P in P_list if P is not None]

    ofs = 1 if ada_lepas else 0          # elemen pertama = titik lepas
    hasil.update({
        'berhasil': True,
        'tinggi': h_semua[ofs:],                 # tinggi puncak tiap pantulan (selaras idx_puncak)
        'tinggi_semua': h_semua,                 # termasuk tinggi lepas bila ada
        't_puncak': t_v[ofs:],
        't_lepas': float(t_v[0]) if ada_lepas else float('nan'),
        'y_lantai': y_lantai, 'sumber_lantai': sumber,
        'e_tiap': e_tiap, 'e_rerata': float(np.mean(e_tiap)), 'e_regresi': e_reg,
        'sd_e': float(np.std(e_tiap, ddof=1)) if len(e_tiap) > 1 else 0.0,
        'r2': reg['r2'], 'n_pantulan': int(len(P_list) - (1 if ada_lepas else 0)),
        'e_kecepatan': e_kec, 'e_kec_rerata': _rerata_aman(e_kec),
        'e_waktu': e_waktu, 'e_waktu_rerata': _rerata_aman(e_waktu),
        't_kontak': t_c, 'selisih_kontak': sel_c,
        't_terbang': T,
        'v_sebelum': v_sblm, 'v_sesudah': v_ssdh,
        'g_estimasi': float(np.median(g_fit)) if g_fit else float('nan'),
        'energi_sisa': float(np.mean(rasio)),
        'parabola': P_list,
        # --- tambahan untuk rincian/penjelasan ---
        'ada_lepas': bool(ada_lepas), 'n_fit': n_fit,
        'g_tiap': np.array([-2.0 * P[0] if P is not None else np.nan for P in P_list]),
        'tinggi_sampel_semua': np.array(y_raw) - y_lantai,
        'y_potong': np.array(y_pot, dtype=float),
        'n_titik': int(n), 'dt': dt, 'durasi': float(t[-1] - t[0]),
        'prominence_frac': float(prominence_frac), 'sigma_derau': sigma,
        'n_dibuang': int(n_dibuang), 'g_acuan': float(g),
        'y_awal': y_awal,
    })
    return hasil


def metrik_validasi(y_eks, y_model):
    """RMSE, MAE, galat maksimum, R^2, korelasi, dan galat relatif."""
    y_eks = np.asarray(y_eks, dtype=float)
    y_model = np.asarray(y_model, dtype=float)
    n = min(len(y_eks), len(y_model))
    y_eks, y_model = y_eks[:n], y_model[:n]
    res = y_eks - y_model
    ss_res = float(np.sum(res ** 2))
    ss_tot = float(np.sum((y_eks - np.mean(y_eks)) ** 2))
    rentang = float(np.max(y_eks) - np.min(y_eks))
    korelasi = float(np.corrcoef(y_eks, y_model)[0, 1]) if n > 2 and np.std(y_model) > 0 else float('nan')
    return {'n': int(n), 'residual': res,
            'RMSE': float(np.sqrt(ss_res / n)),
            'MAE': float(np.mean(np.abs(res))),
            'maks': float(np.max(np.abs(res))),
            'bias': float(np.mean(res)),
            'SD_residual': float(np.std(res, ddof=1)) if n > 1 else 0.0,
            'R2': 1 - ss_res / ss_tot if ss_tot > 0 else float('nan'),
            'korelasi': korelasi,
            'NRMSE_persen': 100 * float(np.sqrt(ss_res / n)) / rentang if rentang > 0 else float('nan')}


def kondisi_awal(t, x, n_pakai=None):
    """
    Estimasi x(0) dan v(0) dengan fit kuadratik pada awal data
    (jauh lebih tahan derau dibanding beda hingga langsung).
    """
    t = np.asarray(t, dtype=float)
    x = np.asarray(x, dtype=float)
    n = len(t)
    if n_pakai is None:
        n_pakai = int(np.clip(round(0.05 * n), 5, 30))
    n_pakai = int(min(max(n_pakai, 3), n))
    tt = t[:n_pakai] - t[0]
    koef = np.polyfit(tt, x[:n_pakai], min(2, n_pakai - 1))
    if len(koef) == 3:
        a, b, c0 = koef
        return float(c0), float(b)
    if len(koef) == 2:
        b, c0 = koef
        return float(c0), float(b)
    return float(koef[0]), 0.0


def optimasi_parameter(kunci, p_awal, t, s, bebas=None, h=None, maks_eval=300):
    """
    Kalibrasi halus: cari parameter model yang paling cocok dengan data
    (kuadrat terkecil non-linear terhadap solusi RK4).

    kunci  : 'pantulan'
    p_awal : dict parameter awal
    bebas  : daftar nama parameter yang dioptimasi
             (pantulan juga boleh 'y_lantai')

    Dibuat TAHAN terhadap data berderau / salah-deteksi:
      * batas fisis: 0.05 <= e <= 0.99, 0 <= c <= 0.5*m (hambatan udara bola kecil
        selalu jauh di bawah ini), |v0| <= 1.5 m/s. Tanpa batas ini, optimasi bisa
        'menyulap' c besar (bola seolah jatuh pelan pada kecepatan batas) hanya
        untuk mengejar titik-titik yang salah.
      * fungsi galat robust (soft-L1): beberapa titik pencilan tidak mendominasi.
      * pencarian awal pada kisi nilai e, lalu penghalusan dari beberapa kandidat
        terbaik; hasil dengan galat robust terkecil yang dipakai (fungsi galat
        pantulan punya banyak minimum lokal).
    Hasil memuat 'peringatan' (daftar teks) bila parameter menempel di batas atau
    kecocokan buruk — tanda data/FPS/skala yang bermasalah, bukan modelnya.
    """
    if not SCIPY_ADA:
        return {'berhasil': False, 'pesan': 'scipy tidak tersedia.'}
    from scipy.optimize import least_squares

    t = np.asarray(t, dtype=float)
    s = np.asarray(s, dtype=float)
    if bebas is None:
        bebas = ['e', 'c', 'x0', 'v0']
    p = dict(p_awal)
    dt = float(np.median(np.diff(t)))
    if h is None:
        h = max(dt / 4.0, 1e-5)
    h_kasar = max(dt / 2.0, 1e-5)
    c_maks = 0.5 * max(float(p.get('m', 0.05)), 1e-6)

    batas_bawah, batas_atas, awal = [], [], []
    for nama in bebas:
        nilai = float(p.get(nama, 0.0))
        if nama == 'e':
            lo, hi = 0.05, 0.99
        elif nama == 'c':
            lo, hi = 0.0, c_maks
        elif nama == 'v0':
            lo, hi = -1.5, 1.5
        else:                                   # x0, y_lantai
            rentang = max(np.max(np.abs(s)) * 5, 1e-6)
            lo, hi = -rentang, rentang
        awal.append(float(np.clip(nilai, lo, hi)))
        batas_bawah.append(lo); batas_atas.append(hi)
    awal = np.array(awal, dtype=float)
    skala = np.maximum((np.array(batas_atas) - np.array(batas_bawah)) / 10.0, 1e-3)

    # skala galat robust dari derau data (beda-kedua, kokoh terhadap pencilan)
    if len(s) >= 5:
        d2 = s[2:] - 2.0 * s[1:-1] + s[:-2]
        sigma = 1.4826 * np.median(np.abs(d2 - np.median(d2))) / np.sqrt(6.0)
    else:
        sigma = 0.0
    f_skala = float(max(3.0 * sigma, 0.01))

    def residual(vek, hh):
        q = dict(p)
        for nama, nilai in zip(bebas, vek):
            q[nama] = float(nilai)
        try:
            model = buat_model(kunci, q)
            pos, _, _, _ = simulasi_pada_grid(model, t, h=hh)
        except Exception:
            return np.full(len(s), 1e3)
        r = pos - s
        return np.where(np.isfinite(r), r, 1e3)

    def biaya(r):                                # sama dengan loss soft_l1 milik scipy
        z = (r / f_skala) ** 2
        return float(np.sum(f_skala ** 2 * (np.sqrt(1.0 + z) - 1.0)))

    # --- 1) kisi nilai e (parameter paling sulit / banyak minimum lokal) ---
    titik_awal = [awal.copy()]
    if 'e' in bebas:
        ie = bebas.index('e')
        kisi = list(np.linspace(0.2, 0.98, 14)) + [float(awal[ie])]
        skor = []
        for ev in kisi:
            v = awal.copy()
            v[ie] = ev
            skor.append((biaya(residual(v, h_kasar)), ev))
        skor.sort()
        titik_awal = []
        for _, ev in skor[:3]:
            v = awal.copy()
            v[ie] = ev
            titik_awal.append(v)

    # --- 2) penghalusan dari tiap kandidat, ambil yang galat robust terkecil ---
    terbaik = None
    n_eval = 0
    jatah = max(int(maks_eval) // max(len(titik_awal), 1), 30)
    for v0_awal in titik_awal:
        try:
            hs = least_squares(lambda v: residual(v, h_kasar), v0_awal,
                               bounds=(batas_bawah, batas_atas), x_scale=skala,
                               loss='soft_l1', f_scale=f_skala, diff_step=1e-4,
                               max_nfev=jatah, xtol=1e-8, ftol=1e-8)
        except Exception as exc:
            continue
        n_eval += int(hs.nfev)
        if terbaik is None or hs.cost < terbaik.cost:
            terbaik = hs
    if terbaik is None:
        return {'berhasil': False, 'pesan': 'Optimasi gagal.'}
    try:                                           # polesan akhir dengan langkah h halus
        hs = least_squares(lambda v: residual(v, h), terbaik.x,
                           bounds=(batas_bawah, batas_atas), x_scale=skala,
                           loss='soft_l1', f_scale=f_skala, diff_step=1e-4,
                           max_nfev=max(int(maks_eval) // 3, 30), xtol=1e-9, ftol=1e-9)
        n_eval += int(hs.nfev)
        terbaik = hs
    except Exception:
        pass

    p_opt = dict(p)
    for nama, nilai in zip(bebas, terbaik.x):
        p_opt[nama] = float(nilai)
    model = buat_model(kunci, p_opt)
    pos, _, _, _ = simulasi_pada_grid(model, t, h=h)
    metrik = metrik_validasi(s, pos)

    peringatan = []
    for nama, nilai, lo, hi in zip(bebas, terbaik.x, batas_bawah, batas_atas):
        lebar = hi - lo
        if nama == 'c' and nilai > hi - 0.02 * lebar:
            peringatan.append('c menempel di batas atas: pola jatuh data lebih LAMBAT dari jatuh bebas. '
                              'Periksa FPS / skala waktu, skala penggaris, dan tinggi lepas (y0).')
        elif nama in ('e', 'v0') and (nilai < lo + 0.01 * lebar or nilai > hi - 0.01 * lebar):
            peringatan.append(f'{nama} menempel di batas ({nilai:.4g}): data tidak cocok dengan model.')
    if np.isfinite(metrik['R2']) and metrik['R2'] < 0.5:
        peringatan.append(f"R2 rendah ({metrik['R2']:.2f}): periksa titik salah-deteksi, FPS, dan kalibrasi.")

    return {'berhasil': True, 'p': p_opt, 'bebas': list(bebas), 'model': model,
            'prediksi': pos, 'metrik': metrik, 'peringatan': peringatan,
            'n_eval': int(n_eval), 'pesan': terbaik.message}


def turunan_numerik(t, x, halus=5):
    """Kecepatan dari data posisi (beda pusat) dengan perataan bergerak opsional."""
    t = np.asarray(t, dtype=float)
    x = np.asarray(x, dtype=float)
    if halus and halus > 1 and len(x) > halus:
        kernel = np.ones(int(halus)) / float(halus)
        x = np.convolve(x, kernel, mode='same')
    return np.gradient(x, t)


def haluskan(x, jendela=5):
    """Perataan bergerak sederhana (moving average).

    Tepi data diperpanjang dengan nilai ujung (bukan nol) agar titik awal/akhir
    tidak ikut tertarik turun ke 0 — dulu ini membuat tinggi awal tampak lebih
    rendah dari yang sebenarnya."""
    x = np.asarray(x, dtype=float)
    if jendela is None or jendela < 2 or len(x) < jendela:
        return x
    j = int(jendela)
    k = np.ones(j) / float(j)
    kiri = j // 2
    xp = np.pad(x, (kiri, j - 1 - kiri), mode='edge')
    return np.convolve(xp, k, mode='valid')


def buang_lonjakan(t, s, ambang=None, zona_lantai=0.01, g=9.81):
    """Mask boolean (True = pertahankan) untuk membuang lonjakan TERISOLASI.

    (1) Satu sampel yang salah (tracker sesaat melompat) meninggalkan pola beda-kedua
        yang khas: (+D/2, -D, +D/2) di sekitarnya, dengan tanda berlawanan pada kedua
        tetangga. Tumbukan lantai berpola lain (satu sisi saja), parabola mulus
        beda-keduanya hanya g*dt^2 (kecil), sehingga keduanya tidak ikut terbuang.
        Sampel yang sendiri sudah di dasar lantai (< zona_lantai) tidak disentuh
        oleh aturan ini.
    (2) 'Lantai palsu': satu sampai tiga sampel berturut-turut di dasar lantai yang
        dikapit sampel jauh di atasnya pada KEDUA sisi. Bola tidak mungkin turun lalu
        naik lebih dari ~1,5 * v_maks * dt dalam satu frame, jadi ini pasti salah
        deteksi (biasanya y negatif yang kemudian dipotong menjadi 0).
    """
    s = np.asarray(s, dtype=float)
    t = np.asarray(t, dtype=float)
    n = len(s)
    ok = np.ones(n, dtype=bool)
    if n < 7:
        return ok
    d2 = s[2:] - 2.0 * s[1:-1] + s[:-2]                  # d2[j] milik titik j+1
    sigma = 1.4826 * np.median(np.abs(d2 - np.median(d2))) / np.sqrt(6.0)
    amp = float(ambang) if ambang else max(6.0 * sigma, 0.012)
    D = d2[1:-1]; Dk = d2[:-2]; Dn = d2[2:]               # titik pusat, kiri, kanan
    pola = ((np.abs(D) > 2.0 * amp)
            & (np.sign(Dk) == -np.sign(D)) & (np.sign(Dn) == -np.sign(D))
            & (np.abs(Dk) > 0.25 * np.abs(D)) & (np.abs(Dn) > 0.25 * np.abs(D)))
    titik = np.arange(2, n - 2)                           # indeks titik untuk D
    buruk = pola & (s[titik] > zona_lantai)
    ok[titik[buruk]] = False

    # (2) lantai palsu
    dt = float(np.median(np.diff(t))) if n > 1 else 0.0
    v_maks = np.sqrt(2.0 * g * max(float(np.max(s)), 0.0))
    d_maks = 1.5 * v_maks * dt + 0.02
    di_lantai = (s <= zona_lantai) & ok
    i = 0
    while i < n:
        if not di_lantai[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and di_lantai[j + 1]:
            j += 1
        if i >= 1 and j <= n - 2 and (j - i + 1) <= 3:
            dasar = float(np.max(s[i:j + 1]))
            if s[i - 1] - dasar > d_maks and s[j + 1] - dasar > d_maks:
                ok[i:j + 1] = False
        i = j + 1
    return ok


def indeks_lepas(s):
    """Indeks sampel saat bola DILEPAS (perkiraan konservatif): sampel terakhir di
    'puncak awal' sebelum sentuhan lantai pertama, mundur satu sampel. Data sebelum
    indeks ini (tangan masih memegang bola, tracker belum mengunci) tidak mewakili
    jatuh bebas dan sebaiknya dipangkas agar t = 0 bertepatan dengan saat dilepas."""
    s = np.asarray(s, dtype=float)
    n = len(s)
    if n < 8:
        return 0
    sm = s.copy()                                   # median-3: satu lonjakan bukan 'puncak'
    sm[1:-1] = np.median(np.vstack([s[:-2], s[1:-1], s[2:]]), axis=0)
    smax = float(np.max(sm))
    if smax <= 1e-9:
        return 0
    tinggi = np.where(sm >= 0.5 * smax)[0]
    i_a = int(tinggi[0])
    rendah = np.where((sm < 0.1 * smax) & (np.arange(n) > i_a))[0]
    i_k = int(rendah[0]) if len(rendah) else n - 1
    d2 = s[2:] - 2.0 * s[1:-1] + s[:-2]
    sigma = 1.4826 * np.median(np.abs(d2 - np.median(d2))) / np.sqrt(6.0)
    tol = min(max(3.0 * sigma, 0.004), 0.03 * smax)
    dalam = np.where(sm[i_a:i_k + 1] >= smax - tol)[0]
    return int(max(i_a + dalam[-1] - 1, i_a))


# ---------------------------------------------------------------------
# Penyajian hasil analisis pantulan (tabel & rincian perhitungan)
# Semua hitungan tetap di sini; GUI hanya menampilkan.
# ---------------------------------------------------------------------

def _f(v, d=4):
    try:
        if not np.isfinite(v):
            return 'n/a'
        out = f'{float(v):.{d}f}'
        if out.startswith('-') and float(out) == 0.0:      # hindari "-0.0000"
            out = out[1:]
        return out
    except Exception:
        return 'n/a'


def _semua_waktu_puncak(hasil):
    t = list(hasil['t_puncak'])
    return ([hasil['t_lepas']] if hasil['ada_lepas'] else []) + t


def _label_puncak(hasil, j):
    """Nama puncak ke-j: 0 = titik lepas (bila ada), 1.. = pantulan ke-n."""
    return j if hasil['ada_lepas'] else j + 1


def puncak_data(hasil):
    """(waktu, ketinggian mutlak) semua puncak data, termasuk titik lepas bila ada."""
    t_all = np.array(_semua_waktu_puncak(hasil), dtype=float)
    return t_all, np.asarray(hasil['tinggi_semua'], dtype=float) + hasil['y_lantai']


def waktu_kontak_dari_hasil(hasil, g=9.81):
    """Waktu tumbukan lantai dari hasil analisis_pantulan (titik potong parabola dengan
    lantai). Yang NaN diisi: titik tengah dua puncak berurutan, atau - untuk jatuh pertama
    - waktu lepas + sqrt(2 h0 / g)."""
    if not hasil.get('berhasil'):
        return np.array([])
    tc = np.array(hasil['t_kontak'], dtype=float).copy()
    t_all = np.array(_semua_waktu_puncak(hasil), dtype=float)
    h = np.asarray(hasil['tinggi_semua'], dtype=float)
    g_pakai = hasil.get('g_estimasi', float('nan'))
    if not np.isfinite(g_pakai) or g_pakai <= 0:
        g_pakai = g
    for k in range(len(tc)):
        if np.isfinite(tc[k]) or k + 1 >= len(t_all):
            continue
        if k == 0 and hasil['ada_lepas']:
            tc[k] = t_all[0] + np.sqrt(2.0 * max(h[0], 0.0) / g_pakai)
        else:
            tc[k] = 0.5 * (t_all[k] + t_all[k + 1])
    tc = np.sort(tc[np.isfinite(tc)])
    if len(tc) > 1:
        tc = tc[np.insert(np.diff(tc) > 1e-6, 0, True)]
    return tc


def tabel_puncak(hasil):
    """Satu baris per puncak (termasuk titik lepas): waktu, tinggi, jumlah titik fit, g."""
    if not hasil.get('berhasil'):
        return []
    t_all = _semua_waktu_puncak(hasil)
    baris = []
    for j, h in enumerate(hasil['tinggi_semua']):
        lepas = hasil['ada_lepas'] and j == 0
        baris.append({'no': _label_puncak(hasil, j),
                      'jenis': 'lepas (h0)' if lepas else 'pantulan',
                      't': float(t_all[j]), 'h': float(h),
                      'h_sampel': float(hasil['tinggi_sampel_semua'][j]),
                      'n_fit': int(hasil['n_fit'][j]),
                      'g': float(hasil['g_tiap'][j])})
    return baris


def tabel_tumbukan(hasil):
    """Satu baris per tumbukan dengan lantai: e dari tinggi, kecepatan, dan waktu."""
    if not hasil.get('berhasil'):
        return []
    h = hasil['tinggi_semua']
    ew = hasil['e_waktu']
    baris = []
    for k in range(len(hasil['t_kontak'])):
        if hasil['ada_lepas']:
            e_w = ew[k] if k < len(ew) else np.nan
        else:
            e_w = ew[k - 1] if 1 <= k <= len(ew) else np.nan
        baris.append({'no': k + 1, 't': float(hasil['t_kontak'][k]),
                      'h_sebelum': float(h[k]), 'h_sesudah': float(h[k + 1]),
                      'e_tinggi': float(hasil['e_tiap'][k]),
                      'v_sebelum': float(hasil['v_sebelum'][k]),
                      'v_sesudah': float(hasil['v_sesudah'][k]),
                      'e_kec': float(hasil['e_kecepatan'][k]),
                      'e_waktu': float(e_w)})
    return baris


def rincian_pantulan(hasil, g=9.81):
    """
    Teks langkah-langkah perhitungan (rumus + substitusi angka) dari hasil
    analisis_pantulan(). Angka di sini identik dengan yang ada di tabel.
    """
    if not hasil.get('berhasil'):
        out = ['Analisis belum berhasil: ' + str(hasil.get('pesan', '-'))]
        for d in hasil.get('diagnosa', []):
            out.append(f"  puncak {d['no']}: titik fit={d['n_fit']}, g_fit={_f(d['g_fit'], 2)}, "
                       f"h={_f(d['h'])} -> {'valid' if d['valid'] else d['alasan']}")
        return out
    L = []
    add = L.append
    ada = hasil['ada_lepas']
    h = hasil['tinggi_semua']
    P = hasil['parabola']
    yl = hasil['y_lantai']
    t_all = _semua_waktu_puncak(hasil)
    n_pant = hasil['n_pantulan']

    add('RINCIAN PERHITUNGAN ANALISIS PANTULAN')
    add('=' * 64)
    add(f"Data: {hasil['n_titik']} titik, dt = {hasil['dt']*1000:.2f} ms, "
        f"durasi = {hasil['durasi']:.3f} s. Data TIDAK dihaluskan di tahap ini.")
    add('')

    # [1] jumlah puncak
    add('[1] JUMLAH PUNCAK')
    add(f"    Puncak pantulan terdeteksi : {n_pant}")
    add(f"    Titik lepas (h0) ikut      : {'ya' if ada else 'tidak'}  "
        f"-> total puncak dipakai = {len(h)}")
    add(f"    Kriteria: maksimum lokal dengan prominence >= "
        f"{hasil['prominence_frac']*100:.1f}% rentang data dan >= 4 sigma derau "
        f"(sigma derau ~ {hasil['sigma_derau']*1000:.2f} mm).")
    add('    Puncak dianggap valid bila fit parabolanya wajar: -2a antara 0,5 dan 1,8 kali g tipikal data,')
    add('    (g tipikal = median -2a semua puncak, jadi tidak bergantung pada kalibrasi skala), >= 3 titik,')
    add('    dan tinggi > 5 sigma. Deret dipotong di puncak pertama yang tidak valid')
    add(f"    (puncak dibuang karena derau/terlalu kecil: {hasil['n_dibuang']}).")
    for p in hasil.get('peringatan', []):
        add('    PERINGATAN: ' + p)
    for d in hasil.get('diagnosa', []):
        if not d['valid']:
            add(f"    puncak {d['no']} tidak wajar: {d['alasan']}")
    add('')

    # [2] fit parabola
    add('[2] PUNCAK DARI FIT PARABOLA (bukan dari satu sampel mentah)')
    add('    Tanpa hambatan, di antara dua tumbukan:  y(t) = y_v - (g/2)(t - t_v)^2')
    add('    Fit kuadrat terkecil:  y = a*tau^2 + b*tau + c,   tau = t - t_ref')
    add('    Puncak : tau_v = -b/(2a),   y_v = c - b^2/(4a),   t_v = t_ref + tau_v')
    add('    Percepatan gravitasi dari kelengkungan:  g = -2a')
    for j, p in enumerate(P):
        nm = 'lepas' if (ada and j == 0) else f'pantulan {_label_puncak(hasil, j)}'
        if p is None:
            add(f"    - {nm}: fit gagal -> dipakai sampel mentah y = {_f(h[j] + yl)} m")
            continue
        a, b, c, tr = p
        tv = -b / (2 * a)
        add(f"    - {nm}: {hasil['n_fit'][j]} titik; a={_f(a)} b={_f(b)} c={_f(c)} (t_ref={_f(tr)})")
        add(f"        tau_v = -({_f(b)})/(2*{_f(a)}) = {_f(tv)} s  ->  t_v = {_f(tr + tv)} s")
        add(f"        y_v = {_f(c)} - ({_f(b)})^2/(4*{_f(a)}) = {_f(c - b*b/(4*a))} m ;  g = {_f(-2*a, 3)} m/s^2")
    add('')

    # [3] lantai
    add('[3] LANTAI (y_lantai)')
    add(f"    Sumber: {hasil['sumber_lantai']}")
    if len(hasil['y_potong']):
        add('    Parabola berurutan berpotongan di titik tumbukan, jadi tinggi titik potongnya = lantai.')
        add('    y_lantai = median(' + ', '.join(_f(v) for v in hasil['y_potong'][:8]) +
            (', ...' if len(hasil['y_potong']) > 8 else '') + ')')
    add(f"    y_lantai = {_f(yl)} m")
    add('')

    # [4] tinggi
    add('[4] TINGGI PUNCAK DI ATAS LANTAI:  h = y_v - y_lantai')
    for j in range(len(h)):
        nm = 'h0 (lepas)' if (ada and j == 0) else f'h{_label_puncak(hasil, j)}'
        add(f"    {nm:<11}= {_f(h[j] + yl)} - ({_f(yl)}) = {_f(h[j])} m"
            f"   (sampel mentah: {_f(hasil['tinggi_sampel_semua'][j])} m)")
    add('')

    # [5] e dari tinggi
    add('[5] KOEFISIEN RESTITUSI DARI TINGGI')
    add('    Kecepatan pantul = e * kecepatan datang, dan h = v^2/(2g)  ->  e = sqrt(h_sesudah / h_sebelum)')
    for k, e in enumerate(hasil['e_tiap']):
        add(f"    tumbukan {k+1}: e = sqrt({_f(h[k+1])} / {_f(h[k])}) = {_f(e)}")
    add(f"    rata-rata = {_f(hasil['e_rerata'])}   (SD = {_f(hasil['sd_e'])})")
    add('    Regresi:  ln h_n = ln h_0 + 2n ln e   ->  e = exp(kemiringan / 2)')
    add(f"    e (regresi) = {_f(hasil['e_regresi'])}   R^2 = {_f(hasil['r2'], 5)}")
    add(f"    Energi tersisa tiap pantulan = e^2 = {_f(hasil['energi_sisa']*100, 2)} %")
    add('')

    # [6] kecepatan
    add('[6] KOEFISIEN RESTITUSI DARI KECEPATAN TUMBUKAN (definisi langsung)')
    add('    Waktu sentuh: akar y(t) = y_lantai pada cabang turun (parabola sebelum) dan cabang naik (sesudah).')
    add('    Kecepatan: v = dy/dt = 2a*tau + b ;   e = -v_sesudah / v_sebelum')
    for r in tabel_tumbukan(hasil):
        add(f"    tumbukan {r['no']}: t = {_f(r['t'])} s ; v- = {_f(r['v_sebelum'], 3)} m/s ; "
            f"v+ = {_f(r['v_sesudah'], 3)} m/s ; e = {_f(r['e_kec'])}")
    add(f"    rata-rata e = {_f(hasil['e_kec_rerata'])}")
    add('')

    # [7] waktu terbang
    add('[7] KOEFISIEN RESTITUSI DARI WAKTU TERBANG')
    add('    Waktu terbang antar tumbukan T = 2v/g, jadi  e = T_berikut / T_sebelum')
    if ada:
        add('    Pantulan pertama: e = T_1 / (2 * t_jatuh), t_jatuh = waktu dari lepas hingga tumbukan 1.')
    for j, T in enumerate(hasil['t_terbang']):
        add(f"    T{j+1} = {_f(T)} s")
    add(f"    rata-rata e = {_f(hasil['e_waktu_rerata'])}")
    add('')

    # [8] g
    add('[8] GRAVITASI DARI KELENGKUNGAN PARABOLA')
    add(f"    g = median(-2a) = {_f(hasil['g_estimasi'], 3)} m/s^2   (acuan input: {g:g})")
    add('')

    # [9] ringkasan
    add('[9] RINGKASAN & KONSISTENSI')
    add(f"    e (tinggi)    = {_f(hasil['e_rerata'])}")
    add(f"    e (regresi)   = {_f(hasil['e_regresi'])}")
    add(f"    e (kecepatan) = {_f(hasil['e_kec_rerata'])}")
    add(f"    e (waktu)     = {_f(hasil['e_waktu_rerata'])}")
    es = np.array([hasil['e_rerata'], hasil['e_kec_rerata'], hasil['e_waktu_rerata']], dtype=float)
    es = es[np.isfinite(es)]
    if len(es) >= 2:
        sel = (es.max() - es.min()) / np.mean(es) * 100
        add(f"    Selisih terbesar antar cara = {sel:.2f} %  -> " +
            ('sesuai teori (tanpa hambatan berarti).' if sel < 3 else
             'perlu dicek: hambatan udara, posisi lantai, atau data kurang rapat.'))
    return L
