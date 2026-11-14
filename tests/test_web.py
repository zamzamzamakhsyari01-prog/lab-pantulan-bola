"""Tes jembatan web (docs/lab_web.py) dan sinkronisasi core_fisika untuk GitHub Pages."""
import json
import pathlib

import pytest

import lab_web  # dari docs/ (lihat pythonpath di pyproject.toml)

RAIZ = pathlib.Path(__file__).resolve().parents[1]


def test_core_fisika_web_sama_dengan_sumber():
    a = (RAIZ / "src" / "lab_pantulan" / "core_fisika.py").read_text(encoding="utf-8")
    b = (RAIZ / "docs" / "core_fisika.py").read_text(encoding="utf-8")
    assert a == b, "Jalankan: python scripts/sync_web.py"


def test_simulasi_json_valid():
    r = json.loads(lab_web.simulasi(json.dumps({"e": 0.8, "x0": 1.0, "t_akhir": 3, "h": 0.005})))
    assert len(r["t"]) == len(r["y"]) == len(r["v"]) > 100
    assert r["puncak"][0]["jenis"] == "lepas"


def test_analisis_dari_data_sintetis():
    csv = lab_web.sintetis(json.dumps({"e": 0.75, "x0": 1.0, "t_akhir": 3.5, "fps": 120, "derau": 0.005}))
    r = json.loads(lab_web.analisis(csv, json.dumps({"geser_min": True})))
    assert r["berhasil"]
    assert r["e_regresi"] == pytest.approx(0.75, abs=0.05)
    assert r["kurva"] and r["tabel_tumbukan"]


def test_csv_terlalu_pendek_ditolak():
    with pytest.raises(ValueError):
        lab_web.parse_csv("t,y\n0,1\n1,2\n")
