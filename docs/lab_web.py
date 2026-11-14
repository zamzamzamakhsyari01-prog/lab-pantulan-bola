"""Jembatan Python <-> JavaScript untuk versi web (dijalankan di Pyodide).

Semua fungsi menerima/mengembalikan TEKS (JSON / CSV) supaya mudah dipanggil dari JS.
Perhitungan fisika memakai core_fisika.py yang sama dengan aplikasi desktop.
"""

import json
import numpy as np
import core_fisika as cf


def _bersih(o):
    """numpy / NaN / Inf -> tipe JSON yang aman (NaN -> null)."""
    if isinstance(o, dict):
        return {str(k): _bersih(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_bersih(v) for v in o]
    if isinstance(o, np.ndarray):
        return _bersih(o.tolist())
    if isinstance(o, (np.floating, float)):
        v = float(o)
        return v if np.isfinite(v) else None
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def _param(p):
    return {'m': float(p.get('m', 0.06)), 'e': float(p.get('e', 0.8)),
            'c': float(p.get('c', 0.0)), 'x0': float(p.get('x0', 1.0)),
            'v0': float(p.get('v0', 0.0)), 'g': float(p.get('g', 9.81)),
            'hambatan': p.get('hambatan', 'linier')}


def _jarang(T, Y, maks=4000):
    n = len(T)
    langkah = max(n // maks, 1)
    return T[::langkah], Y[::langkah]


# --------------------------------------------------------------------
# 1. Simulasi RK4
# --------------------------------------------------------------------
def simulasi(json_param):
    p = json.loads(json_param)
    t_akhir = max(float(p.get('t_akhir', 6.0)), 1e-3)
    h = min(max(float(p.get('h', 0.005)), 1e-5), 0.05)
    model = cf.buat_model('pantulan', _param(p))
    T, Y = cf.simulasi(model, t_akhir, h)
    pk = cf.puncak_simulasi(T, Y)
    tb = cf.tumbukan_simulasi(T, Y, model.e)
    Tj, Yj = _jarang(T, Y)
    return json.dumps(_bersih({
        't': Tj, 'y': Yj[:, 0], 'v': Yj[:, 1],
        'puncak': [{'t': a, 'y': b, 'jenis': c} for a, b, c in pk],
        'tumbukan': [{'t': a, 'v_sebelum': b, 'v_sesudah': c} for a, b, c in tb],
        'info': model.info(),
    }))


# --------------------------------------------------------------------
# 2. Data sintetis (untuk mencoba tanpa data nyata)
# --------------------------------------------------------------------
def sintetis(json_param):
    p = json.loads(json_param)
    fps = max(float(p.get('fps', 60.0)), 5.0)
    derau = max(float(p.get('derau', 0.01)), 0.0)
    model = cf.buat_model('pantulan', _param(p))
    T, Y = cf.simulasi(model, float(p.get('t_akhir', 4.0)), 0.001)
    t = np.arange(0, T[-1], 1.0 / fps)
    s = np.interp(t, T, Y[:, 0])
    rng = np.random.default_rng(7)
    s = s + rng.normal(0, derau * (np.max(s) - np.min(s) + 1e-9), len(s))
    baris = ['t_s,y_m'] + [f'{a:.6f},{b:.6f}' for a, b in zip(t, s)]
    return '\n'.join(baris)


# --------------------------------------------------------------------
# 3. Baca CSV (teks)
# --------------------------------------------------------------------
def parse_csv(teks):
    """Kolom 1 = waktu. 2 kolom -> kolom ke-2 = ketinggian. >=3 kolom (t,x,y) -> kolom ke-3."""
    t, kol = [], []
    for baris in str(teks).splitlines():
        baris = baris.strip()
        if not baris or baris.startswith('#'):
            continue
        bagian = [b for b in baris.replace(';', ',').replace('\t', ',').split(',') if b != '']
        try:
            angka = [float(b) for b in bagian]
        except ValueError:
            continue                              # baris header
        if len(angka) < 2:
            continue
        t.append(angka[0])
        kol.append(angka[1:])
    if len(t) < 10:
        raise ValueError('CSV berisi kurang dari 10 baris data numerik.')
    lebar = min(len(k) for k in kol)
    A = np.array([k[:lebar] for k in kol], dtype=float)
    t = np.array(t, dtype=float)
    s = A[:, 1] if lebar >= 2 else A[:, 0]
    urut = np.argsort(t, kind='stable')
    t, s = t[urut], s[urut]
    t, iu = np.unique(t, return_index=True)
    return t, s[iu]


# --------------------------------------------------------------------
# 4. Analisis pantulan
# --------------------------------------------------------------------
def _kurva_parabola(r):
    yl = r['y_lantai']
    ada = r['ada_lepas']
    kurva = []
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
        yy = np.where(yy >= yl - 1e-9, yy, np.nan)
        kurva.append({'t': tt, 'y': yy})
    return kurva


def analisis(csv_teks, json_opsi):
    o = json.loads(json_opsi)
    t, s = parse_csv(csv_teks)
    if o.get('satuan') == 'cm':
        s = s * 0.01
    if o.get('geser_min', True):
        s = s - float(np.min(s))
    yl = o.get('y_lantai')
    if yl in (None, ''):
        yl = 0.0 if o.get('geser_min', True) else None
    else:
        yl = float(yl)
    g = float(o.get('g', 9.81))
    r = cf.analisis_pantulan(t, s, g, y_lantai=yl,
                             prominence_frac=float(o.get('prom', 0.02)))
    out = {'t': t, 'y': s, 'berhasil': bool(r['berhasil']),
           'pesan': r.get('pesan', ''), 'n_titik': int(len(t))}
    if not r['berhasil']:
        out['rincian'] = '\n'.join(cf.rincian_pantulan(r, g))
        return json.dumps(_bersih(out))
    t_all, y_all = cf.puncak_data(r)
    out.update({
        'y_lantai': r['y_lantai'], 'sumber_lantai': r['sumber_lantai'],
        'n_pantulan': r['n_pantulan'], 'ada_lepas': r['ada_lepas'],
        'e_tinggi': r['e_rerata'], 'sd_e': r['sd_e'], 'e_regresi': r['e_regresi'],
        'r2': r['r2'], 'e_kecepatan': r['e_kec_rerata'], 'e_waktu': r['e_waktu_rerata'],
        'g_estimasi': r['g_estimasi'], 'energi_sisa': r['energi_sisa'],
        'tinggi': r['tinggi_semua'], 'peringatan': r.get('peringatan', []),
        'puncak_t': t_all, 'puncak_y': y_all,
        'kontak_t': r['t_kontak'],
        'tabel_puncak': cf.tabel_puncak(r), 'tabel_tumbukan': cf.tabel_tumbukan(r),
        'kurva': _kurva_parabola(r),
        'rincian': '\n'.join(cf.rincian_pantulan(r, g)),
    })
    return json.dumps(_bersih(out))
