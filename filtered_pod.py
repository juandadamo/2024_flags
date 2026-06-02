import numpy as np
import matplotlib.pyplot as plt

fs = 2000
f_senal = 12
senal_ruidosa = U[:,0]
t = np.arange(len(U[:,0]))/fs
# --- FILTRADO POR FFT ---
# 1. Transformada de Fourier
fft_senal = np.fft.fft(senal_ruidosa)
frecuencias = np.fft.fftfreq(len(t), 1/fs)

# 2. Visualizar espectro
plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.plot(t, senal_ruidosa, 'b-', alpha=0.7, linewidth=0.8)
plt.title('Señal Original Ruidosa')
plt.xlabel('Tiempo (s)')
plt.ylabel('Amplitud')
plt.grid(True, alpha=0.3)

plt.subplot(2, 2, 2)
plt.plot(frecuencias[:len(frecuencias)//2],
         np.abs(fft_senal[:len(fft_senal)//2]))
plt.title('Espectro de Frecuencia')
plt.xlabel('Frecuencia (Hz)')
plt.ylabel('Magnitud')
plt.grid(True, alpha=0.3)
plt.axvline(f_senal, color='r', linestyle='--', label=f'Frecuencia señal ({f_senal} Hz)')
plt.legend()

# 3. Crear máscara (solo pasar frecuencias cercanas a la señal)
ancho_banda = 5  # Hz a cada lado
mascara = np.zeros_like(fft_senal)
for i, f in enumerate(frecuencias):
    if abs(abs(f) - f_senal) < ancho_banda:
        mascara[i] = 1

# Aplicar máscara
fft_filtrada = fft_senal * mascara

# 4. Transformada inversa
senal_filtrada = np.fft.ifft(fft_filtrada).real

plt.subplot(2, 2, 3)
plt.plot(t, senal_ruidosa, 'g-', label='Señal original limpia', alpha=0.7)
plt.plot(t, senal_filtrada, 'r--', label='Señal filtrada', alpha=0.7)
plt.title('Comparación: Señal Filtrada vs Original')
plt.xlabel('Tiempo (s)')
plt.ylabel('Amplitud')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(2, 2, 4)
plt.plot(t, senal_ruidosa - senal_filtrada, 'k-', alpha=0.5)
plt.title('Ruido Removido')
plt.xlabel('Tiempo (s)')
plt.ylabel('Amplitud')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
