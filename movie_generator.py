
import serial,socket,os,glob,sys
#import atexit
import numpy as np
import pandas as pd
import time, threading,sys,glob

import tifffile as tif

import subprocess
A = pd.read_csv('casos_2D_lista_archivos.csv')


# Configuración
input_pattern = "frame_%06d.tiff"  # Nombre de los frames (001, 002...)
output_video = "video_salida_60.mp4"
fps = 60  # Fotogramas por segundo
resolution = "1080x1920"  # Vertical (ancho x alto)

for i, Ai in enumerate(A['nombre carpeta']):
    lista_i = np.sort(glob.glob('/home/juan/data/balseiro/'+Ai+'/*.tiff'))
    # Comando FFmpeg
    cmd = [
        "ffmpeg",
        "-framerate", str(fps),
        "-i", input_pattern,
        "-s", resolution,
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-vf", "format=yuv420p",  # Compatibilidad con reproductores
        output_video
    ]
    # Ejecutar
    try:
        subprocess.run(cmd, check=True)
        print("✅ Video creado exitosamente!")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error al generar el video: {e}")
    raise ValueError()






