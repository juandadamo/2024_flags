import subprocess
from pathlib import Path
import pandas as pd
from funciones_flag import *

A = pd.read_csv('casos_2D_lista_archivos.csv')
dir_datos = '/home/juan/data/balseiro/'

for i, Ai in enumerate(A['nombre carpeta']):
    frec_i = A['freq motor'][i]
    caso = A['medida'][i]
    Velocidad =  veloc_tunel_ib(frec_i)
    print(dir_datos + Ai + '/video_salida.mp4')
    file_video = dir_datos + Ai + '/video_salida_60b.mp4'
    output_file = 'figures/video_amp/' +caso+ f'Veloc_{Velocidad:.2f}_60fps_rv.mp4'
    subprocess.run(['cp', file_video, output_file])