# f_func/graficos.py
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay,classification_report,auc,accuracy_score

def histog(df,col,colg,gs=0,bins=[20],ncols=1,fsize=[15,6],colores=0,kde=0):
    """
    Crea subgráficos de histogramas para diferentes categorías de una variable categórica.

    Parameters:
    - df (DataFrame): DataFrame que contiene los datos.
    - col (str): Nombre de la variable numérica para la cual se desea hacer el histograma.
    - colg (str): Nombre de la variable categórica que define las categorías para las cuales se crearán los histogramas.
    - gs (list or int): Lista de categorías específicas para las cuales se crearán histogramas. Si es 0, se usan todas las categorías únicas de 'colg'.
    - bins (list or int): Número de divisiones o lista de divisiones para el histograma. Si es una lista, debe tener la misma longitud que 'gs'.
    - ncols (int): Número de columnas en la disposición de subgráficos.
    - fsize (list): Tamaño de la figura en pulgadas.
    - colores (list or int): Lista de colores para los histogramas. Si es 0, se utiliza una paleta de colores predeterminada.
    - kde (bool): Si se debe trazar la estimación de densidad de kernel (KDE) junto con el histograma.

    Returns:
    None

    Esta función crea subgráficos de histogramas para diferentes categorías de una variable categórica.
    Puedes especificar las categorías específicas (gs), el número de divisiones (bins), el número de columnas
    en la disposición de subgráficos (ncols), el tamaño de la figura (fsize), colores personalizados para
    los histogramas (colores) y si se debe trazar la estimación de densidad de kernel (kde).

    Ejemplo de uso:
    >>> import seaborn as sns
    >>> import pandas as pd
    >>> import matplotlib.pyplot as plt
    >>> data = {'col': [1, 2, 3, 4, 5, 6], 'colg': ['A', 'B', 'A', 'B', 'A', 'B']}
    >>> df = pd.DataFrame(data)
    >>> histog(df, 'col', 'colg', gs=0, bins=[10, 15], ncols=2, fsize=[12, 6], colores=sns.color_palette('pastel'), kde=True)
    """
    if not gs:
        gs=sorted(df[colg].unique())
    ng=len(gs)
    if len(bins)==1:
        bins=bins*ng
    nrows=-(-ng//ncols)
    if not colores:
        colores = sns.color_palette('deep')
    colores = colores* ((ng // len(colores)) + 1)
    colores=colores[:ng]

    # Crear subplots
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(fsize[0], fsize[1]))
    axes=np.ravel(axes)

    # Iterar sobre las clases y crear un histograma en cada subplot
    for i, g in enumerate(gs):
        subset = df[df[colg] == g]
        sns.histplot(data=subset, x=col, bins=bins[i], kde=kde, ax=axes[i], color=colores[i],alpha=0.3)
        axes[i].set_title(f'Distribución de {col} - {colg} {g}')
        axes[i].set_xlabel(col)
        axes[i].set_ylabel('Frecuencia')
        axes[i].legend(labels=[f'{colg} {g}'])

    # Mostrar el gráfico
    plt.tight_layout()
    plt.show()

def fmatcon(yt,yp,model,title='Title',label_celd=1,size=(7,5),colormap='Pastel2',num_celd=20,text_celd=10,num_celd_col='black',text_celd_col='DarkSlateGray'):
    """
    Visualiza la matriz de confusión y muestra un informe de clasificación.

    Parámetros:
    - yt (array-like): Etiquetas verdaderas.
    - yp (array-like): Etiquetas predichas.
    - model: Estimador del modelo que tiene un atributo 'classes_'.
    - title (str): Título del gráfico. Por defecto es 'Título'.
    - etiqueta_celda (int o lista de listas): Formato de etiqueta para las celdas en la matriz de confusión.
                                              Por defecto es 1 (formato de etiqueta: 'clase_verdadera-clase_predicha').
                                              Si es 0, se utilizan etiquetas predeterminadas: 'Verdadero Negativo', 'Falso Positivo',
                                              'Falso Negativo', 'Verdadero Positivo'.
                                              Si es una lista de listas, proporciona etiquetas para cada celda.
    - tamaño (tuple): Tamaño del gráfico (ancho, alto) en pulgadas. Por defecto es (7, 5).
    - mapa_colores (str): Mapa de colores reconocido por Matplotlib. Por defecto es 'Pastel2'.
    - tamaño_num_celda (int): Tamaño de fuente para los números en las celdas de la matriz de confusión. Por defecto es 20.
    - tamaño_texto_celda (int): Tamaño de fuente para las etiquetas de texto en las celdas de la matriz de confusión. Por defecto es 10.
    - color_num_celda (str): Color para los números en las celdas de la matriz de confusión. Por defecto es 'black'.
    - color_texto_celda (str): Color para las etiquetas de texto en las celdas de la matriz de confusión. Por defecto es 'DarkSlateGray'.

    Retorna:
    - float: Tasa de falsos positivos (fpr).
    - float: Tasa de verdaderos positivos (tpr).

    Esta función toma las etiquetas verdaderas (yt) y las etiquetas predichas (yp),
    calcula la matriz de confusión y muestra visualmente la matriz usando
    ConfusionMatrixDisplay de scikit-learn. Luego, imprime un informe de clasificación
    que incluye precision, recall, f1-score y soporte para cada clase.

    Finalmente, calcula y retorna la tasa de falsos positivos (fpr) y la tasa de
    verdaderos positivos (tpr) utilizando la matriz de confusión.

    Ejemplo de uso:
    >>> yt = [1, 0, 1, 1, 0, 1]
    >>> yp = [1, 0, 1, 0, 0, 1]
    >>> modelo = ...  # tu modelo aquí
    >>> fpr, tpr = fmatcon(yt, yp, modelo)
    """
    
    if label_celd==0:
        lb=0
        label_celd=[['Verdadero Negativo', 'Falso Positivo'],['Falso Negativo', 'Verdadero Positivo']]
    if label_celd==1:
        lb=1
        label_celd=[[f'{t}-{p}'for p in model.classes_] for t in model.classes_]
    cm=confusion_matrix(yt, yp)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=model.classes_)
    disp.plot(cmap=colormap)
    disp.ax_.set_title(title,fontsize=20)
    fig = disp.figure_
    fig.set_figwidth(size[0])
    fig.set_figheight(size[1]) 
    # fig.suptitle('Plot of confusion matrix')
    
    # Cambiar tamaño de letra del contenido de la matriz
    for text in disp.ax_.texts:
        text.set_fontsize(num_celd)  # Establece el tamaño de letra a 12 puntos
        text.set_color(num_celd_col)
    for i in range(len(model.classes_)):
        for j in range(len(model.classes_)):
            disp.ax_.text(j, i-0.3, f'{label_celd[i][j]}',ha="center", va='top',color=text_celd_col, fontsize=text_celd)
        
    plt.show()
    if lb != 1:
        print(classification_report(yt,yp))
        fpr=cm[0,1]/(cm[0, 1]+cm[0,0])
        tpr=cm[1,1]/(cm[1,1]+cm[1,0])
        return fpr,tpr
    return None

def fplotROC(fpr,tpr,name,color,fig=(10,6),marksize=7):
    roc_auc = 'ROC curve (area = %0.2f)' % auc(fpr, tpr)
    plt.figure(figsize=fig)
    plt.plot(fpr, tpr,'o-', label=roc_auc ,color=color,markersize=marksize)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Falsos Positivos tasa (fpr)')
    plt.ylabel('Verdaderos Positivos tasa (tpr)')
    plt.title(f'Curva ROC: {name}')
    plt.legend(loc="lower right")
    plt.show()

def fimpovar(X_train,model,fig=(5,15)):
    fi = pd.DataFrame(columns=["FEATURE", "IMPORTANCE"])
    fi["FEATURE"] = X_train.columns
    fi["IMPORTANCE"] = model.feature_importances_
    fi = fi.sort_values("IMPORTANCE", ascending=False)
    plt.figure(figsize=fig)
    # Crear una paleta de colores basada en 'viridis'
    colors = sns.color_palette("viridis", n_colors=fi.IMPORTANCE.nunique())
    sns.barplot(y=fi.FEATURE, x=fi.IMPORTANCE, hue=fi.IMPORTANCE, palette=colors)
    # plt.legend().set_visible(False)
    plt.show()
    return fi
