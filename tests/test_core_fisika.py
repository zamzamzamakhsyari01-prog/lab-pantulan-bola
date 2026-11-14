import numpy as np
import pytest

from lab_pantulan import core_fisika as cf


def _model(e=0.8, c=0.0):
    return cf.buat_model("pantulan", dict(m=0.06, e=e, c=c, x0=1.0, v0=0.0, g=9.81))


def test_jatuh_bebas_waktu_tumbukan_pertama():
    m = _model()
    T, Y = cf.simulasi(m, 1.0, 0.001)
    t_teori = np.sqrt(2 * 1.0 / 9.81)
    tumb = cf.tumbukan_simulasi(T, Y, m.e)
    assert tumb, "harus ada tumbukan"
    assert tumb[0][0] == pytest.approx(t_teori, abs=2e-3)


def test_tinggi_pantulan_mengikuti_e_kuadrat():
    m = _model(e=0.8)
    T, Y = cf.simulasi(m, 4.0, 0.001)
    pk = [p for p in cf.puncak_simulasi(T, Y) if p[2] == "pantulan"]
    assert pk[0][1] == pytest.approx(1.0 * 0.8 ** 2, rel=1e-3)
    assert pk[1][1] == pytest.approx(1.0 * 0.8 ** 4, rel=1e-3)


def test_analisis_memulihkan_e_dari_data_sintetis():
    e_asli = 0.75
    m = _model(e=e_asli)
    T, Y = cf.simulasi(m, 3.0, 0.0005)
    t = np.arange(0, T[-1], 1 / 120.0)
    s = np.interp(t, T, Y[:, 0])
    r = cf.analisis_pantulan(t, s, g=9.81, y_lantai=0.0)
    assert r["berhasil"]
    assert r["e_regresi"] == pytest.approx(e_asli, abs=0.03)
    assert r["g_estimasi"] == pytest.approx(9.81, rel=0.05)


def test_metrik_validasi_sempurna():
    y = np.linspace(0, 1, 50)
    mk = cf.metrik_validasi(y, y)
    assert mk["RMSE"] == pytest.approx(0.0, abs=1e-12)
    assert mk["R2"] == pytest.approx(1.0)


def test_regresi_linear():
    x = np.arange(10, dtype=float)
    r = cf.regresi_linear(x, 2 * x + 1)
    assert r["slope"] == pytest.approx(2.0)
    assert r["intercept"] == pytest.approx(1.0)


def test_buat_model_menolak_kunci_lain():
    with pytest.raises(ValueError):
        cf.buat_model("pegas", dict(m=1, e=1, c=0, x0=0, v0=0))
