import matplotlib.pyplot as plt
import numpy as np
import glob,socket
from scipy.signal import find_peaks
from skimage import feature, morphology
from funciones_flag import *

# Set up directory for saving figures
if socket.gethostname() == 'CNRS304952':
    dirw = 'C:/Users/IRL2027 2/Documents/Juan/GitHub/2024_flags/figures/'
else:
    dirw = '/home/juan/Documents/Publicaciones/2025_euromech/flag/article/figures/'

plt.close('all')

# Constants
rhoa = 1.2
rhoa_b = 1.0888  # Density of air in Bariloche
nu = 1.5e-5 * rhoa_b / rhoa
Uinf = 12
delta_cl = 18e-3  # Boundary layer thickness for 12 m/s

# Characteristic length of the flat plate (tunnel) based on measurements
x_carac = longitud_equivalente_capa_limite_turbulenta(delta_cl, Uinf, nu)
fsampling = 1000  # Hz
escalax = 1 / 0.138  # px/mm
Lbandera = 138.5  # mm

Papel_80.L = Lbandera * 1e-3  # Convert to meters
Papel_80.freq_nat()

# Prepare for analysis
cases = ['rect', 'triang']
results = {}

for caso in cases:
    lista_caso_2d = np.sort(glob.glob('data_out/' + caso + '_freq*'))
    Velocidad, Amplitud, Frecuencia = np.zeros((3, len(lista_caso_2d)))

    for j, filej in enumerate(lista_caso_2d):
        A1 = np.load(filej)
        Asum = A1['Imagen_sum']
        image = Asum ** 0.12
        YT = A1['A_curva_i']
        frec_j = float(filej.split('freq_')[-1].split('.npz')[0])
        Velocidad[j] = veloc_tunel_ib(frec_j)

        # Edge detection
        edges = feature.canny(image, sigma=4)
        closed_edges = morphology.closing(edges, morphology.disk(radius=5))
        image[closed_edges] = 1
        image[np.logical_not(closed_edges)] = 0

        lim_superior = np.nonzero(image == 1)[0].max()
        lim_inferior = np.nonzero(image == 1)[0].min()
        delta_coord = lim_superior - lim_inferior
        Amplitud[j] = delta_coord * 1.0 / escalax  # mm

        # Fourier Transform
        Fourier_YT = np.fft.fft(YT.T, axis=1)
        FYT = np.abs(Fourier_YT).sum(axis=0)
        freq_YT = np.fft.fftfreq(len(YT), d=1 / fsampling)
        peak_freqs, _ = find_peaks(FYT, height=0.1 * np.max(FYT))
        peak_freqs = peak_freqs[freq_YT[peak_freqs] > 0]
        Frecuencia[j] = freq_YT[peak_freqs][FYT[peak_freqs].argmax()]

        print(f"Frecuencia de la señal: {Frecuencia[j]:.2f} Hz")

    Amplitud = Amplitud / Lbandera
    results[caso] = (Velocidad, Amplitud, Frecuencia)

# Plotting results for comparison
fig, ax = plt.subplots()
for caso, (Velocidad, Amplitud, Frecuencia) in results.items():
    Uc = veloc_tunel_ib(12.7 if caso == 'rect' else 11.4)
    U = Velocidad - Uc
    Velocidad_m = Velocidad / 2
    deltaw = delta_turb(x_carac, Velocidad, nu)

    ax.plot(Frecuencia * deltaw / Velocidad_m, Amplitud, 'o-', label=caso)

ax.set_ylabel(r'$A/L$')
ax.set_xlabel(r'$f_{foil}\delta_w/U$')
ax.grid()
ax.legend()
plt.tight_layout()
plt.savefig(dirw + 'Freq_Amp_comparativo.png')

plt.close('All')