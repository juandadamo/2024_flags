# Proposition du 31/03/2026 pour Juan

## Objectif
Machine learning pour trouver les coefficients des équations de flag fluttering :

$$
R_1 X_{tt} = \partial_s(T\mathbf{s}) - R_2 \partial_{ss}(\kappa \hat{n}) - [p] \hat{n}.
$$

**NB :** on connaît la valeur théorique des coefficients dans des conditions parfaites. L'article de 2005 d'Argentina et Mahadevan donne notamment ces coefficients pour une plaque 2D dans un écoulement potentiel.

---

## Méthodologie

La méthode classique de sciences de l’ingénieur (vue en prépa) de caractérisation de la fonction de transfert d’un système ne fonctionne pas pour la situation du drapeau car on s’intéresse à quelque chose de non linéaire avec des cycles limites.

> « L'instabilité du drapeau est une bifurcation (souvent de type Hopf). En dessous de la vitesse critique, le drapeau est parfaitement stable (l'amortissement est dominant). Au-dessus de la vitesse critique, il devient instable, mais son amplitude ne part pas à l'infini : elle se stabilise à une valeur maximale à cause des forces de tension internes du matériau. Il entre dans ce qu'on appelle un cycle limite (un oscillateur non linéaire auto-entretenu). »  
> — *Gemini*

### Solution proposée : Machine learning avec SINDy

**Pourquoi SINDy plutôt qu’un réseau de neurones classique ?**

> « Un réseau de neurones profond (Deep Learning) est une "boîte noire" : il va prédire le mouvement du drapeau, mais sans vous dire comment il a fait. SINDy fait du Machine Learning "découvreur d'équations" : il va fouiller dans les données et vous recracher l'équation différentielle exacte, de manière lisible pour un humain (par exemple, il vous écrira textuellement $\ddot{y} = -a\,y - b\,y^3$). »  
> — *Gemini*

---

## Contraintes expérimentales

- Si l’apprentissage est fait avec des courbes 2D en entrée, il vaut mieux utiliser un flux uniforme.  
- En présence de repliement ou torsion dans un flux cisaillé (shear flow), les longueurs projetées sont biaisées.  
  → Il faut utiliser la méthode FTP Python 3D TIFF pour ne pas fausser l’entraînement.

> « Parce qu’on s’intéresse *in fine* à la récupération d’énergie marine, il faut prendre en compte le fait que les fonds marins frottent et créent un cisaillement dans le courant. »

---

## Pipeline : de la caméra à SINDy

SINDy ne peut pas traiter directement des images (courbes 2D), seulement des valeurs de position, vitesse, accélération.  
Il faut donc un programme de traitement d’image en amont.

### Étapes

1. **Tracking avec OpenCV**  
   - Détection du bord du drapeau sur chaque frame.  
   - Sortie : tableau des positions spatiales en fonction du temps.

2. **Filtrage et dérivation**  
   - Utilisation du filtre de Savitzky–Golay (`savgol_filter` de SciPy) pour lisser et calculer vitesse et accélération robustes.

3. **Entraînement avec PySINDy**  
   - Entrées : position, vitesse, accélération.  
   - SINDy trouve l’équation différentielle non linéaire reliant ces trois grandeurs.

---

## Cas de la 3D (Fourier Transform Profilometry)

> « La méthode de votre laboratoire (Fourier Transform Profilometry) devient obligatoire. La projection des franges, capturée par la caméra et traitée par votre code Python (déroulement de phase), va vous donner une véritable topographie 3D de la feuille. Vous pourrez voir exactement comment la feuille se tord sous l’effet de la différence de vitesse du vent. »

---

## Choix des entrées pour le Machine Learning

**Remarque importante :**  
> « Faire varier $L$, $H$, $\rho$, $U$ a déjà été fait intensivement par Raynaud. Ne perdez pas 2 mois à refaire ses courbes. Concentrez-vous sur ce qu'il n'a pas fait : ajoutez le flot cisaillé (shear flow) et les franges. »

### 1. Entrées physiques et fluides

- **Vitesse du fluide ($U$)** : en écoulement cisaillé, ce n’est pas une valeur unique mais un profil (vitesse en fonction de la hauteur $y$).  
- **Ne donner que la vitesse d’entrée** (amont). La vitesse aval est une conséquence du battement, elle ne doit pas être utilisée comme entrée.  
- **Masse volumique de l’air ($\rho_f$)** : à calculer à partir de la température et pression mesurées le jour de l’essai.  
- **Dimensions de la soufflerie** : utiles pour le taux de blocage, mais constantes dans une même soufflerie → non pertinentes pour l’apprentissage.

### 2. Entrées solides (drapeau)

- **Hauteur ($H$), Longueur ($L$), Grammage ($m$)** : variables de base.  
- **Vitesse des ondes élastiques ($U_B$)** : donnée par l’article de 2005 :  
  $$
  U_B = \frac{1}{L} \sqrt{\frac{B}{m}}
  $$
  avec $B = \dfrac{E h^3}{12(1 - \sigma^2)}$ (rigidité à la flexion, $E$ : module de Young, $h$ : épaisseur, $\sigma$ : coefficient de Poisson).  
  → Un test de flexion/traction sur le papier permettra d’obtenir $E$.

### 3. Modélisation des franges

L’idée d’utiliser le PDF/PNG du patron de découpe est brillante mais dépend de l’algorithme :

- **Avec SINDy** : il faut des variables numériques.  
  → Créer des entrées paramétriques :
  - `type_frange` (0 = plat, 1 = triangle, 2 = créneau)
  - `amplitude_frange` (mm)
  - `frequence_frange` (dents/m)

> « Commencez par la méthode paramétrique (SINDy avec amplitude et fréquence). C’est plus robuste avec un jeu de données limité (quelques dizaines de tests). L’approche par image nécessiterait des milliers de tests. »

### 4. Cisaillement (shear flow)

Le cisaillement nécessite la création d’une grille 3D en amont (OpenSCAD, SolidWorks, etc.) et la vérification des vitesses.

### 5. Données issues du traitement vidéo

Les positions, vitesses, accélérations du drapeau sont les entrées principales de SINDy après traitement d’image.

> **NB :** Comme c’est l’onde de flexion qui nous intéresse, on ne pourra pas utiliser seulement les battements de la queue pour faire du machine learning. Il faudra des courbes sur l’entièreté du drapeau.

---

## Synthèse des entrées pour SINDy

| Catégorie               | Variables proposées |
|-------------------------|---------------------|
| Fluide                  | Profil de vitesse $U(y)$, $\rho_f$ |
| Solide (drapeau)        | $L$, $H$, $m$, $U_B$ |
| Franges (paramètres)    | Type, amplitude, fréquence |
| Cisaillement            | Taux ou profil associé |
| Cinématique (issue vidéo) | Positions, vitesses, accélérations (sur l’ensemble du drapeau) |
