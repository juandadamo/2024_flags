Por el tamaño de los archivos, comparto las rutas/nombres de los datos

## [Data for profilometry 3D](casos_3D_lista_archivos.csv)

Datos para profilometría. Algunas series características para la bandera (full), recorte rectangular (segmented rect), recorte triangular (segmented triang). Las series corresponden a dinámica del modo 1, bandera ondeando(fluttering flag) y también al modo 0, cuasiestatico, de muy pequeñas oscilaciones. Tambien hay imagenes de referencia, estáticas. 


## [Data for stability analysis. 2D amplitude](casos_2D_lista_archivos.csv) 

Datos para determinar la evolución de la amplitud de la bandera ondeando hasta poder determinar el umbral inferior, donde se amortigua completamente el movimiento.


## [Variable Length analysis](Estabilidad_L_variable.csv)

Lista de valores  que indican los regímenes de estabilidad al variar la longitud de la bandera (full). Son dos valores de velocidad de tunel respecto a cada longitud de bandera.

## [Stability velocity range](Intervalos estabilidad.csv)

Análisis de lo anterior.

La velocidad del tunel se puede asimilar a un valor de frecuencia de inestabilidad de Kelvin Helmoltz de acuerdo a Ho, C. M., & Huerre, P. (1984). Perturbed free shear layers. Annual review of fluid mechanics, 16, 365-424.

$St_0 = f_{KH}\cdot\theta_0/U_m = 0.032$

$\theta_0$: momentum thickness $\sim$ boundary layer thickness

$U_m=(U_1+U_2)/2$: mean velocity

$f_{kh}$ Kelvin Helmoltz shear later non forced frequency
