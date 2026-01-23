import subprocess
from pathlib import Path
import pandas as pd

A = pd.read_csv('casos_2D_lista_archivos.csv')
dir_datos = '/home/juan/data/balseiro/'
def crear_video_desde_tiff(carpeta_imagenes, output_video="video_salida_60b.mp4", fps=60, resolution="1080x1920"):
    """
    Crea un video a partir de imágenes .tiff en una carpeta específica.
    
    Args:
        carpeta_imagenes (str): Ruta de la carpeta con imágenes.
        output_video (str): Nombre del archivo de salida.
        fps (int): Fotogramas por segundo.
        resolution (str): Resolución en formato "ANCHOxALTO".
    """
    # Asegurar que la carpeta exista
    carpeta = Path(carpeta_imagenes)
    if not carpeta.is_dir():
        raise FileNotFoundError(f"Carpeta no encontrada: {carpeta}")

    # Renombrar imágenes a formato secuencial (frame_001.tiff, frame_002.tiff, ...)
    tiff_files = sorted(carpeta.glob("*.tiff"))

    # Comando FFmpeg
    cmd = [
        "ffmpeg",
        "-framerate", str(fps),
        "-i", str(carpeta / "frame_%06d.tiff"),  # Patrón de nombres
        "-s", resolution,
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-vf", "format=yuv420p",
        "-vf", "negate",
        str(carpeta / output_video)
    ]

    # Ejecutar
    try:
        subprocess.run(cmd, check=True)
        print(f"✅ Video creado en {carpeta / output_video}")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error en {carpeta}: {e}")

# --- Ejemplo de uso ---
if __name__ == "__main__":
    # Lista de carpetas a procesar (ajusta tus rutas)
    carpetas = A['nombre carpeta'].tolist()

    # Procesar todas las carpetas
    for carpeta in carpetas:
        print(f"\nProcesando: {carpeta}")
        crear_video_desde_tiff(dir_datos+carpeta)