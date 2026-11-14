"""Jalankan aplikasi:  python -m lab_pantulan   (atau perintah `lab-pantulan`)."""


def main():
    from .app import Aplikasi
    Aplikasi().mainloop()


if __name__ == "__main__":
    main()
