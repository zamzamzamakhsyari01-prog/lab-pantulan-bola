"""Salin core_fisika.py ke docs/ agar situs GitHub Pages memakai kode yang sama.

Jalankan setiap kali core_fisika.py diubah:  python scripts/sync_web.py
"""
import pathlib
import shutil

raiz = pathlib.Path(__file__).resolve().parents[1]
shutil.copyfile(raiz / "src" / "lab_pantulan" / "core_fisika.py", raiz / "docs" / "core_fisika.py")
print("docs/core_fisika.py diperbarui.")
