# Cara mengunggah ke GitHub (publik)

Ganti `USERNAME` dengan nama akun GitHub Anda (di README ada 3 tempat).

## Cara 1 — GitHub CLI (paling cepat)
```bash
cd lab-pantulan-bola
sed -i 's/USERNAME/NAMA_ANDA/g' README.md        # macOS: sed -i '' ...
git init -b main && git add . && git commit -m "Versi awal Lab Pantulan Bola"
gh auth login
gh repo create lab-pantulan-bola --public --source=. --push
```

## Cara 2 — lewat web
1. github.com → **New repository** → nama `lab-pantulan-bola` → **Public** → jangan centang README/.gitignore/license → Create.
2. Di folder proyek:
```bash
git init -b main && git add . && git commit -m "Versi awal"
git remote add origin https://github.com/NAMA_ANDA/lab-pantulan-bola.git
git push -u origin main
```

Link yang bisa dibuka semua orang: `https://github.com/NAMA_ANDA/lab-pantulan-bola`

Tips: tambahkan topik (python, physics, opencv, rk4) dan satu screenshot aplikasi di README.
