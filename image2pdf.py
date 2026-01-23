import os
import sys
import subprocess

if len(sys.argv) < 2:
    print("Uso: python images2pdf.py /ruta/al/directorio")
    sys.exit(1)

src_dir = sys.argv[1]
pdf_dir = os.path.join(src_dir, 'pdfs')
os.makedirs(pdf_dir, exist_ok=True)

exts = ('.png', '.jpg', '.jpeg')

for fname in os.listdir(src_dir):
    if fname.lower().endswith(exts):
        src_path = os.path.join(src_dir, fname)
        base, _ = os.path.splitext(fname)
        pdf_path = os.path.join(pdf_dir, base + '.pdf')
        subprocess.run(['convert', src_path, pdf_path])
        print(f'Convertido: {src_path} -> {pdf_path}')