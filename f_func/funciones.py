# f_func/funciones.py
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold,KFold
from sklearn.metrics import accuracy_score, f1_score,mean_squared_error
from matplotlib.lines import Line2D

def malla(limits,nps,d=0,n=0):

    """
    Genera una malla para n variables especificando los límites o el numero de variables.

    Parameters:
    - limits (list of tuples or lists): Lista de tuplas o listas que representan los límites para cada variable.
    - nps (list of int): Lista de enteros que representa el número de puntos o la distancia entre ellos para cada variable.
    - d (int, optional): Si es 1, utiliza np.arange para generar la malla indicando que nps es la separación entre puntos
                        ;si es 0 (por defecto), utiliza np.linspace, y nps será el número de puntos.
    - n (int, optional): Si es mayor que 0, repite los límites y el número de puntos n veces para generar una malla n-dimensional.

    Returns:
    - X (numpy.ndarray): Matriz que representa la malla generada en formato flatten o scatter (x1,x2,...,xn) X[:,i].
    - Xc (numpy.ndarray): Matriz que representa la malla generada en formato contourf.

    Raises:
    - ValueError: Si el número de condiciones de límites y puntos no coincide.

    Example:
    >>> limits = [(0, 1), (0, 2), (0, 3)]
    >>> nps = [3, 4, 2]
    >>> X, v = malla(limits, nps)
    """
    n_lim=len(limits)
    n_nps=len(nps)
    if n:
        limits,nps=[limits[0]]*n,[nps[0]]*n
    else:
        if (n_lim!=n_nps):
            raise ValueError("El número de condiciones de límites y puntos no coincide.")
        n=len(limits)
    if d:
        variables = [np.arange(l[0], l[1], nps[i]) for i,l in enumerate(limits)]
    else:
        variables = [np.linspace(l[0], l[1], nps[i]) for i,l in enumerate(limits)]
    Xc = np.meshgrid(*variables, indexing='ij')
    v=[axis.ravel() for axis in Xc[:n]]
    
    return np.column_stack(v),Xc

def bestDepth(X,y,dm=1,dx=10,cv=1,g=0,fig=(10,6),colores=['green','red','blue']):
    depths=list(range(dm,dx+1))
    ac_values=[[],[],[]]
    sd_values=[[],[]]
    depths_values=[]
    acm=[0,0,0]
    sdm=[0,0,0]
    depthm=[0,0,0]

    # Crea una instancia de StratifiedKFold y crea la particion del dataframe en cv partes iguales.
    # Verifica si y es discreta (categórica) o continua
    if y.dtypes.eq('int64').all():
        # y es discreta, utiliza StratifiedKFold
        kf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
        particiones = list(kf.split(X, y))
    else:
        # y es continua, utiliza KFold
        kf = KFold(n_splits=cv, shuffle=True, random_state=42)
        particiones = list(kf.split(X, y))

    # Se itera en el rango de profundidad depth -> [dm,dx]
    for d in depths:
        ac=[[],[]]
        # Se itera en particiones estableciendo en cada iteración una particion distinta como prueba.
        for train_i, test_i in particiones:
            # Definición de valores de entrenamiento Xe ye y valores de prueba Xt yt
            Xt,yt=X.iloc[test_i],y.iloc[test_i]
            Xe,ye=X.iloc[train_i],y.iloc[train_i]
            model = DecisionTreeClassifier(max_depth=d)
            model.fit(Xe,ye)
            ypt=model.predict(Xt)
            ac[0].append(accuracy_score(yt,ypt))
            ype=model.predict(Xe)
            ac[1].append(accuracy_score(ye,ype))
        # Se obtiene el promedio y la desviación estandar para cada depth 
        for i in range(2):    
            ac_values[i].append(np.mean(ac[i]))
            sd_values[i].append(np.std(ac[i],ddof=1))

        model.fit(X,y)
        yp=model.predict(X)
        ac_values[2].append(accuracy_score(y,yp))

        depths_values.append(d)

    for i in range(2):
        im,acm[i] = max(enumerate(ac_values[i]), key=lambda x: x[1])    
        sdm[i]=sd_values[i][im]
        depthm[i]=depths_values[im]
    im,acm[2] = max(enumerate(ac_values[2]), key=lambda x: x[1])    
    depthm[2]=depths_values[im]
    ymin_acm = min(acm)
    ac_values=np.array(ac_values)
    sd_values=np.array(sd_values)
    if g:
        plt.figure(figsize=fig)
        # plt.scatter(depths_values,ac_values,s=5)
        lineas=['Validación Cruzada', 'Entrenamiento','Original sin prueba']
        for i in range(2):
            plt.plot(depths_values, ac_values[i], color=colores[i], marker='o',markersize=4, label=lineas[i])
            plt.fill_between(depths_values, ac_values[i] - sd_values[i], ac_values[i] + sd_values[i], alpha=0.15, color=colores[i])
        plt.plot(depths_values, ac_values[2], color=colores[2], marker='o',markersize=4, label=lineas[2])
        # Obtener los límites actuales del eje x
        xlim = plt.gca().get_xlim()
        ylim = plt.gca().get_ylim()
        xmin,xmax = xlim[0],xlim[1]
        ymin,ymax = ylim[0],ylim[1]
        for i in range(2):
            plt.hlines(y=acm[i],xmin=xmin,xmax=depthm[i],color=colores[i], linestyle='--', label=f'Accuracy Score Máximo = {np.round(acm[i],4)}',alpha=0.5,linewidth=0.8)
            plt.vlines(x=depthm[i],ymin=ymin,ymax=acm[i],color=colores[i], linestyle='--', label=f'depth = {depthm[i]}', alpha=0.5,linewidth=0.8)

        plt.hlines(y=acm[2],xmin=xmin,xmax=depthm[2], color=colores[2], linestyle='--', label=f'Accuracy Score Máximo = {np.round(acm[2],4)}',alpha=0.5,linewidth=0.8)
        plt.vlines(x=depthm[2],ymin=ymin,ymax=acm[2], color=colores[2], linestyle='--', label=f'depth = {depthm[2]}',alpha=0.5,linewidth=0.8)

        plt.xlim([xmin, xmax]) 
        plt.ylim([ymin, ymax]) 
        
        plt.title("Curva de Validación")
        plt.xlabel("max_depth")
        plt.ylabel("Accuracy Score")
        plt.legend(loc='center right')
        plt.show()
    return depthm,acm,sdm,depths_values,ac_values,sd_values

def bestParam(X,y,modelname,param,scorename,maxim=1,cv=1,g=0,fig=(10,6),colores=['green','red','blue']):
    if type(y) != pd.DataFrame:
        y= y.to_frame()
    nameP=param[0]
    pm=param[1]
    px=param[2]
    valuesP=list(range(pm,px+1))
    ac_values=[[],[],[]]
    sd_values=[[],[]]
    param_values=[]
    acm=[0,0,0]
    sdm=[0,0,0]
    paramM=[0,0,0]

    # Crea una instancia de StratifiedKFold y crea la particion del dataframe en cv partes iguales.
    # Verifica si "y" es discreta (categórica) o continua
    if y.dtypes.eq('int64').all():
        # "y" es discreta, utiliza StratifiedKFold
        kf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
        particiones = list(kf.split(X, y))
    else:
        # y es continua, utiliza KFold
        kf = KFold(n_splits=cv, shuffle=True, random_state=42)
        particiones = list(kf.split(X, y))

    # Se itera en el rango de profundidad depth -> [dm,dx]
    for p in valuesP:
        ac=[[],[]]
        # Se itera en particiones estableciendo en cada iteración una particion distinta como prueba.
        for train_i, test_i in particiones:
            # Definición de valores de entrenamiento Xe ye y valores de prueba Xt yt
            Xt,yt=X.iloc[test_i],y.iloc[test_i]
            Xe,ye=X.iloc[train_i],y.iloc[train_i]
            model = modelname.__class__(**modelname.get_params())
            model.set_params(**{nameP: p})
            model.fit(Xe,ye.values.ravel())
            ypt=model.predict(Xt)
            ac[0].append(scorename(yt,ypt))
            ype=model.predict(Xe)
            ac[1].append(scorename(ye,ype))
        # Se obtiene el promedio y la desviación estandar para cada depth 
        for i in range(2):    
            ac_values[i].append(np.mean(ac[i]))
            sd_values[i].append(np.std(ac[i],ddof=1))

        model.fit(X,y.values.ravel())
        yp=model.predict(X)
        ac_values[2].append(scorename(y,yp))

        param_values.append(p)

    for i in range(2):
        if maxim:
            im,acm[i] = max(enumerate(ac_values[i]), key=lambda x: x[1])
            strmaxim='Máximo'
        else:
            im,acm[i] = min(enumerate(ac_values[i]), key=lambda x: x[1])
            strmaxim='Mínimmo'
        sdm[i]=sd_values[i][im]
        paramM[i]=param_values[im]
    if maxim:
        im,acm[2] = max(enumerate(ac_values[2]), key=lambda x: x[1])
        strmaxim='máximo'
    else:
        im,acm[2] = min(enumerate(ac_values[2]), key=lambda x: x[1])    
        strmaxim='mínimo'
    paramM[2]=param_values[im]
    # ymin_acm = min(acm)
    ac_values=np.array(ac_values)
    sd_values=np.array(sd_values)
    if g:
        plt.figure(figsize=fig)
        plt.xticks(range(1,px+1))
        # plt.scatter(depths_values,ac_values,s=5)
        lineas=['Validación Cruzada', 'Entrenamiento','Original sin prueba']
        for i in range(2):
            plt.plot(param_values, ac_values[i], color=colores[i], linewidth=0.7, marker='o',markersize=4, label=lineas[i])
            plt.fill_between(param_values, ac_values[i] - sd_values[i], ac_values[i] + sd_values[i], alpha=0.15, color=colores[i])
        plt.plot(param_values, ac_values[2], color=colores[2], linewidth=0.7, marker='o',markersize=4, label=lineas[2])
        # Obtener los límites actuales del eje x
        xlim = plt.gca().get_xlim()
        ylim = plt.gca().get_ylim()
        xmin,xmax = xlim[0],xlim[1]
        ymin,ymax = ylim[0],ylim[1]
        for i in range(2):
            plt.hlines(y=acm[i],xmin=xmin,xmax=paramM[i],color=colores[i], linestyle='--', label=f'{scorename.__name__} {strmaxim} = {np.round(acm[i],4)}',alpha=0.5,linewidth=0.8)
            plt.vlines(x=paramM[i],ymin=ymin,ymax=acm[i],color=colores[i], linestyle='--', label=f'{nameP} = {paramM[i]}', alpha=0.5,linewidth=0.8)

        plt.hlines(y=acm[2],xmin=xmin,xmax=paramM[2], color=colores[2], linestyle='--', label=f'{scorename.__name__} {strmaxim} = {np.round(acm[2],4)}',alpha=0.5,linewidth=0.8)
        plt.vlines(x=paramM[2],ymin=ymin,ymax=acm[2], color=colores[2], linestyle='--', label=f'{nameP} = {paramM[2]}',alpha=0.5,linewidth=0.8)

        plt.xlim([xmin, xmax]) 
        plt.ylim([ymin, ymax]) 
        
        plt.title("Curva de Validación")
        plt.xlabel(nameP)
        plt.ylabel(f'{scorename.__name__}')
        plt.legend(loc='best')
        plt.show()
    return paramM,acm,sdm,param_values,ac_values,sd_values