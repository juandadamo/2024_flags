import numpy as np
import matplotlib.pyplot as plt
import glob
import skimage as sk
import tifffile as tif
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d
import os

plt.close('all')
dir_w = '/home/juan/data/full/41_0_full/'
files_list = np.sort(glob.glob(dir_w+'/*.tif'))
# nfile = 1845


scalax = 1280 / 150 #en mesurant sur mon écran
L_flag_mm = 120.0
L_flag_pixels = int(L_flag_mm * scalax)







# files_list = np.sort(glob.glob('prises/45.5/C001H001S0001/*.tif'))

nsnapshots = len(files_list)
if nsnapshots == 0:
    raise ValueError("Aucune image trouvée.")

#courbe qu'on voudra utiliser comme illustration du bon processus à la fin
instant_t = [50, 100, 150, 300, 400, 800, 1200, 2100]
traces = {}

#matrice remplie de trous par défaut

print("1. Extraction des pixels visibles...")
nymin , nymax, nxmin, nxmax = [323,753,273,1150]
#nymin , nymax, nxmin, nxmax = [323,753,300,600]
nyorigin = 559
nyorigin = 194
# nsnapshots = 100
for i, filei in enumerate(files_list[:nsnapshots]):
    A = tif.imread(filei)
    A = A[nymin : nymax, nxmin: nxmax ][::-1,::-1]
    N_pixels = len(A.T)
    if i==0:
        Y_gappy = np.full((nsnapshots, N_pixels), np.nan)
    A_max_j = np.zeros(len(A.T))
    Int_j = np.zeros(len(A.T))
    for j, Aj in enumerate(A.T):
        A_max_j[j] = Aj.argmax()
        Int_j[j] = Aj.max()

    umbral = sk.filters.threshold_otsu(A) / 3
    masque = (Int_j > umbral)
    Y_gappy[i][masque] = A_max_j[masque]-nyorigin


# fig1,ax1 = plt.subplots()
# for yi in Y_gappy:
#     ax1.plot(yi,'o')

# Gappy POD
print(f"2. Lancement de Gappy POD sur {nsnapshots} images...")

#initialisation des trous avec la moyenne temporelle de chaque pixel
mean_col = np.nanmean(Y_gappy, axis=0)
#aligner sur le mât un pixel qui n'aurait jamais été éclairé (par sécurité)
mean_col[np.isnan(mean_col)] = 0

Y_filled = np.where(np.isnan(Y_gappy), mean_col, Y_gappy)
masque_trous = np.isnan(Y_gappy)

K_modes = 5
iterations = 15

for it in range(iterations):
    # SVD
    U, Sigma, Vt = np.linalg.svd(Y_filled - mean_col, full_matrices=False)

    #reconstruction
    Y_recon = np.dot(U[:, :K_modes], np.dot(np.diag(Sigma[:K_modes]), Vt[:K_modes, :])) + mean_col

    #combler les trous (le laser reste intouché)
    Y_filled[masque_trous] = Y_recon[masque_trous]

    print(f"   -> Itération {it+1}/{iterations} terminée.")
#
ni = 189
fig,ax = plt.subplots()
ax.plot(Y_gappy[ni],'o',fillstyle='none')
ax.plot(Y_filled[ni],'.')
# # Découpe, lissage et arc paramétré (car sinon le lissage entraîne un surplux de longueur pour s)
# print("3. Découpe sur le vrai drapeau, Lissage et Reparamétrisation S...")
#
# longueurs_s = []
# courbes_s = []
# courbes_y = []
#
# #je continue d'appeler SINDy le machine learning, même si ce n'est plus vraiment la méthode utilisée
# N_points_SINDy = 200
# Y_SINDy = np.zeros((nsnapshots, N_points_SINDy))
#
# #trouver la vraie longueur du drapeau éclairée en continu pour chaque image, car la queue est toujours visible
# for i in range(nsnapshots):
#     #recherche du pixel le plus à gauche n'étant pas un NaN dans Y_gappy
#     vrais_pixels_x = x_pixels_continus[~np.isnan(Y_gappy[i])]
#     x_tail = np.min(vrais_pixels_x) if len(vrais_pixels_x) > 0 else nx_min
#
#     #la courbe comblée par Gappy s'arrête exactement à ce moment
#     masque_physique = x_pixels_continus >= x_tail
#     x_coupe = x_pixels_continus[masque_physique]
#     y_coupe = Y_filled[i, masque_physique]
#
#     #lissage local, on reprend le filtre Savitzky-Golay (sur la courbe coupée)
#     #wl = min(51, len(y_coupe) // 2 * 2 + 1) #NB : le paramètre doit être un nombre impair
#     #if wl < 3: wl = 3
#     #y_smooth_local = savgol_filter(y_coupe, window_length=wl, polyorder=3)
#     y_smooth_local = savgol_filter(y_coupe, 21, polyorder=3)
#
#     #inverser les repères et mm
#     x_rev = x_coupe[::-1]
#     y_rev = y_smooth_local[::-1]
#
#     x_mm = (nx_mat - x_rev) / scalax
#     y_mm = (ny_mat - y_rev) / scalax
#
#     #calcul de s (avec Pythagore)
#     dx = np.diff(x_mm)
#     dy = np.diff(y_mm)
#     ds = np.sqrt(dx**2 + dy**2)
#     s_curve = np.zeros(len(x_mm))
#     s_curve[1:] = np.cumsum(ds)
#
#     longueurs_s.append(s_curve[-1])
#     courbes_s.append(s_curve)
#     courbes_y.append(y_mm)
#
#     #ici on met juste la courbe, choisie au début, en mémoire pour faire une illustration
#     if i in instant_t :
#         traces[i] = [x_coupe, y_smooth_local]
#
# #pour être sûrs que les positions des s seront alignées
# L_universelle = np.min(longueurs_s)
# print(f"   -> Arc physique sur toute la vidéo : {L_universelle:.2f} mm")
#
# S_uniforme = np.linspace(0, L_universelle, N_points_SINDy)
#
# for i in range(nsnapshots):
#     f_interp = interp1d(courbes_s[i], courbes_y[i], kind='cubic')
#     Y_SINDy[i, :] = f_interp(S_uniforme)
#
# os.makedirs('new_data_out', exist_ok=True)
# np.savez('new_data_out/full_V_45.5_SINDy_ready.npz',
#          Y=Y_SINDy,
#          S=np.tile(S_uniforme, (nsnapshots, 1)))
#
#
# #diagnostic visuel pour nous (instant_t choisi initialement sera affiché)
# print("4. Matrice SINDy prête. Affichage du diagnostic...")
#
#
# c = 0
# for i in instant_t :
#
#     # Nous allons considérer une grille de quatre lignes et d'une seule colonne.
#     # Première ligne
#     #plt.subplot(len, 1, 1 + c)
#
#     A_test = tif.imread(files_list[i])
#
#     plt.figure(figsize=(12, 6))
#     plt.imshow(A_test, cmap='gray')
#
#     plt.plot(x_pixels_continus, Y_gappy[i], 'go', markersize=3, label="Points bruts")
#
#     plt.plot(traces[i][0], traces[i][1], 'r-', linewidth=3, label="Courbe lissée (arrêt net)")
#     plt.scatter(nx_mat, ny_mat, color='blue', s=100, marker='X', label="Mât")
#
#     plt.title(f"Diagnostic (t={i}) - Lissage localisé")
#     plt.legend()
#
#     c += 1
#
#
# plt.tight_layout()      # Sans cette ligne, il y a des chevauchements dans les étiquettes
# plt.show()
