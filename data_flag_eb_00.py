import numpy as np
import matplotlib.pyplot as plt
dirw = '/home/juan/Documents/Estudiantes/Alice/2026_fluttering_energy/sandbox/V3/'
L_flag = 0.12
B_flag = 5e-5
rho_50 = 1305
m_flag = rho_50 * 50e-6  #masse surfacique

#racines théoriques / typiques (beta * L) pour une poutre d'Euler-Bernoulli Encastrée-Libre
beta_L = np.array([1.875104, 4.694091, 7.854757, 10.995541, 14.137168])

# projection de Galerkin
print("\n" + "="*50)
print("Spatial : forcer la forme Euler-Bernoulli")
print("="*50)

#fonction mathématique exacte des modes de poutre
def mode_euler_bernoulli(x, beta_L_val, L):
    beta = beta_L_val / L
    #ratio sigma pour annuler le moment et l'effort tranchant à l'extrémité libre
    sigma = (np.cosh(beta_L_val) + np.cos(beta_L_val)) / (np.sinh(beta_L_val) + np.sin(beta_L_val))
    return np.cosh(beta*x) - np.cos(beta*x) - sigma * (np.sinh(beta*x) - np.sin(beta*x))


data = np.load(dirw+'full_V_44.2_SINDy_ready.npz')
Y_SINDy = data['Y']
S_uniforme = data['S'][0]

#on construit une matrice avec les 4 premiers modes d'Euler-Bernoulli (au lieu de la SVD, qui avait eu lieu pour Gappy et générer le npz)
Phi_EB = np.zeros((len(S_uniforme), 4))
for i in range(4):
    Phi_EB[:, i] = mode_euler_bernoulli(S_uniforme / 1000.0, beta_L[i], L_flag) # S en mètres

#régression linéaire pour fit nos données et ces formes de Euler-Bernoulli
Y_reconstruit_EB = np.zeros_like(Y_SINDy)
amplitudes_temporelles = np.zeros((len(Y_SINDy), 4))

for t in range(len(Y_SINDy)):
    #trouver la combinaison de modes EB qui ressemble le plus à l'image t
    coefs, _, _, _ = np.linalg.lstsq(Phi_EB, Y_SINDy[t], rcond=None)
    amplitudes_temporelles[t] = coefs
    Y_reconstruit_EB[t] = np.dot(Phi_EB, coefs)


t_test = 12 #on regarde la frame n°

plt.figure(figsize=(14, 6))

#comparaison Spatiale
plt.subplot(1, 2, 1)
plt.plot(S_uniforme, Y_SINDy[t_test], 'go', markersize=4, label="Vraie courbe (Caméra)")
plt.plot(S_uniforme, Y_reconstruit_EB[t_test], 'r-', linewidth=3, label="Fit Euler-Bernoulli (4 modes)")
plt.axhline(0, color='black', linewidth=0.5)
plt.xlabel("Position le long du drapeau s (mm)")
plt.ylabel("Déflexion Y (mm)")
plt.title(f"Ajustement Spatial (Frame t={t_test})")
plt.legend()

#évolution Temporelle du Mode 1
plt.subplot(1, 2, 2)

#200 frames pour voir l'onde
plt.plot(amplitudes_temporelles[50:250, 0], 'b-', linewidth=2, label="Amplitude Mode 1 EB")
plt.xlabel("Temps (Frames)")
plt.ylabel("Amplitude")
plt.title("Vibration Stationnaire d'Euler-Bernoulli")
plt.legend()

plt.tight_layout()
plt.show()
