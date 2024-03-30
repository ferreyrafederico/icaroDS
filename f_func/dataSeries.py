import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KernelDensity


def rkde(sr):

    # Cálculo de la funcion kde, 'bw_method' es el ancho de banda.
    kde = gaussian_kde(sr.dropna(), bw_method=0.5)

    # Cálculo del numero de valores necesarios y el minimo y máximo de la serie.
    nv=sr.isna().sum()
    minim=sr.min()
    maxim=sr.max()

    # Inicializa los valores aleatorios y el índice
    valores = []
    i = 0

    # Genera valores aleatorios hasta que todos estén dentro del rango
    while len(valores) < nv:
        vs = kde.resample(size=nv)[0]
        vs = vs[(vs >= minim) & (vs <= maxim)]
        valores.extend(vs)
        i += 1
    valores=[int(round(x)) for x in valores]

    # Recorta la lista para que coincida con la cantidad de NaNs
    valores = valores[:nv]

    return valores

def rkdefit(sr,minim=None,maxim=None,val_interger=True):
    """
    Función para imputar valores faltantes (NaN) en una serie de datos utilizando Kernel Density Estimation (KDE).

    Parámetros:
    - sr: Serie de datos con valores faltantes (NaN).
    - minim: Valor mínimo permitido en la serie (si no se proporciona, se utiliza el mínimo de la serie).
    - maxim: Valor máximo permitido en la serie (si no se proporciona, se utiliza el máximo de la serie).
    - val_interger: Si es True, los valores generados se redondean a números enteros.

    Retorna:
    Lista de valores generados para reemplazar los NaNs en la serie original.
    """

    # grid.fit admite como entrada siempre solamente un array bidimensional porque sklearn esta preparado matricialmente.
    # En este caso .reshape(-1,1) le agrega una dimension al array unidimensional
    # Ejemplo pasa de [1,2,3,4,5] a esto [[1],[2],[3],[4],[5]] ya que en este caso es una sola variable. 
    # Se sacan de la serie original los NaNs 
    srd=sr.dropna().values.reshape(-1, 1)
    
    # Cálculo del número de valores necesarios para el reemplazo y el minimo y máximo de la serie.
    nv=sr.isna().sum() # nv numero de valores
    if not minim:
        minim=srd.min() # minimo por defecto es el minimo de la serie.
    if not maxim:
        maxim=srd.max() # máximo por defecto es el máximo de la serie.

    # Regla de Silverman para obtener una estimación inicial del ancho de banda
    silverman_bandwidth = 0.9 * min(np.std(srd), np.percentile(srd, 75) - np.percentile(srd, 25)) * len(srd)**(-1/5)

    # Regla de Scott para obtener otra estimación inicial del ancho de banda
    scott_bandwidth = (4 * np.std(srd)**5 / (3 * len(srd)))**(1/5)

    # Definir intervalos basados en las estimaciones de Silverman y Scott
    lower_bound = min(silverman_bandwidth, scott_bandwidth) / 2
    upper_bound = max(silverman_bandwidth, scott_bandwidth) * 2

    # Definir el espacio de búsqueda de valores de bw
    bws = np.logspace(np.log10(lower_bound), np.log10(upper_bound), 20)



    # Configurar la validación cruzada. Esto es para averiguar que ancho de banda es el que ajusta mejor KDE a los datos
    # de la serie sin los NaNs.
    # GridDearchCV toma el modelo KernelDensity y los anchos de banda posibles bws (son 20) e itera y ve con cual da mejor
    # resultado y mas abajo luego de hacer fit, se coloca en best_bw
    grid = GridSearchCV(KernelDensity(), {'bandwidth': bws}, cv=5)
    grid.fit(srd)

    # Obtener el mejor valor de bw
    best_bw = grid.best_params_['bandwidth']

    # Crear una nueva instancia de KDE con el mejor bw
    kde = KernelDensity(bandwidth=best_bw)

    # Ajustar el modelo a los datos sin NaN
    kde.fit(srd) # Aqui ajusta el KDE a sr sin nans (srd)

    # Inicializa los valores aleatorios y el índice
    valores = []
    i = 0

    # Genera valores aleatorios hasta que todos estén dentro del rango ya que kde genera valores entre -infinito y +infinito
    # y lo que se quiere es valores en un intervalo dado.
    while len(valores) < nv:        # Mientras no se haya completado el numero necesario de valores para reemplazar los NaNs sigue.
        vs = kde.sample(nv)[0]      # Aqui da nv valores segun KDE 
        vs = vs[(vs >= minim) & (vs <= maxim)] # Aqui los "recorta" para que queden solos los comprendidos en el intervalo establecido.
        valores.extend(vs) # Aqui los agrega 
        i += 1

    # Recorta la lista para que coincida con la cantidad necesaria para reemplazar los NaNs.
    valores = valores[:nv]

    # Si val_interger es True devuelve enteros.    
    if val_interger:
        valores=[int(round(x)) for x in valores] # Redondea a enteros los valores.


    return valores

