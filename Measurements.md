Por el tamaño de los archivos, comparto las rutas/nombres de los datos

## 1) [Data for profilometry 3D](casos_3D_lista_archivos.csv)

Datos para profilometría. Algunas series características para la bandera (full), recorte rectangular (segmented rect), recorte triangular (segmented triang). Las series corresponden a dinámica del modo 1, bandera ondeando(fluttering flag) y también al modo 0, cuasiestatico, de muy pequeñas oscilaciones. Tambien hay imagenes de referencia, estáticas. 


## 2) [Data for stability analysis. 2D amplitude](casos_2D_lista_archivos.csv) 

Datos para determinar la evolución de la amplitud de la bandera ondeando hasta poder determinar el umbral inferior, donde se amortigua completamente el movimiento.


## 3) [Variable Length analysis](Estabilidad_L_variable.csv)

Lista de valores  que indican los regímenes de estabilidad al variar la longitud de la bandera (full). Son dos valores de velocidad de tunel respecto a cada longitud de bandera.

## 4) [Stability velocity range](Intervalos_estabilidad.csv)

Breve estudio de 3) a partir de relaciones de elasticidad y mecánica de fluidos.

Por un lado, señalo las frecuencias naturales de las banderas, modos 1 y modo 2. Resultan de aplicar la teoría de Euler-Bernoulli. Dan valores cercanos a los de las frecuencia de la capa de corte.

Por otro lado, las frecuencia de la Capa de Corte a la salida del Túnel.

La velocidad del tunel se puede asimilar a un valor de frecuencia de inestabilidad de Kelvin Helmoltz de acuerdo a Ho, C. M., & Huerre, P. (1984). Perturbed free shear layers. Annual review of fluid mechanics, 16, 365-424.

$St_0 = f_{KH}\cdot\theta_0/U_m = 0.032$ (Una especie de Strouahl de SL)

$\theta_0$: momentum thickness $\sim$ boundary layer thickness (Valor relevado por Nicolás)

$U_m=(U_1+U_2)/2$: mean velocity

$f_{kh}$ Kelvin Helmoltz shear layer non forced frequency

Stability Range 
<p align="left">
 <img src="/figures/figures/stability_ranges_1.png" alt="plot stability" width="250"/> 
</p>
<p align="right">
 <img src="/figures/figures/stability_ranges_2.png" alt="plot stability" width="250"/> 
</p>
