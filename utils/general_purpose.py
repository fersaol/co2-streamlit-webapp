import os
from pathlib import Path
from math import ceil
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.model_selection import cross_val_score,RepeatedKFold,train_test_split
from sklearn.compose import make_column_transformer
from sklearn.feature_selection import RFECV
import joblib

print("Módulo General Listo Para Usarse \U0001F4BB")



def varios_plot(df,wspace=0.3,hspace=0.3,figsize=(20,8),columnas=4,color="darkturquoise"):

    """Función que pinta en un grid con subplots los gráficos de las 
    variables del dataframe. Se deben usar los gráficos de Matplotlib
    ------------------------------------------
    Args:
    df = pd.DataFrame
    wspace = (float) el espacio ancho entre subplots
    hspace = (float) el espacio alto entre subplots
    figsize = (tuple) el tamaño de la figura
    columnas = (int) número de columnas del grid
    color = (str) el color del gráfico, de los disponibles para matplotlib, rgb o html"""

    columnas_df = df.columns
    fig = plt.figure(figsize=figsize)
    fig.subplots_adjust(hspace=hspace,wspace=wspace)
    filas = ceil(len(columnas_df)/columnas)
    for i in range(len(columnas_df)):
        pos = columnas_df[i]
        ax = fig.add_subplot(filas,columnas,i+1)
        plt.hist(df[pos],bins=30,color=color)
        plt.title(pos)
    plt.show()


def escalado(scaler,df,output="np.array"):

    """Función que escala los datos y te los devuelve como array de numpy o
        como pandas dataframe, según se elija
    ----------------------------------------------
    Args:
    scaler = (class sklearn.preprocesing scalers)
    df = (pd.DataFrame) datos a escalar
    output = (str, default: np.array), para obtener un dataframe poner 'df'"""
    
    scaler = scaler.fit(df)
    scaled_vars = scaler.transform(df)
    if output == "df":
        if type(scaled_vars) != np.ndarray:
            new_df = pd.DataFrame(scaled_vars.toarray(),columns=scaler.get_feature_names_out())
        else:
            new_df = pd.DataFrame(scaled_vars,columns=scaler.get_feature_names_out())
    else:
        return scaled_vars
    return new_df


def crossval_models(x,y,folds,estimadores,score,limit_chr=40):
    
    """Función que realiza el cross_val_score de diferentes algoritmos y 
    representa el score medio en un gráfico
    ---------------------------------------------
    # Args:
        x = (pd.DataFrame or np.array) features\n
        y = (pd.DataFrame or np.array) target\n
        folds = (int o kfold de sklearn.model_selection)\n
        estimadores = (list) lista de algoritmos\n
        score = (str) métrica para medir el resultado\n
        limit_chr = (int) límite de caracteres del nombre del algoritmo a mostrar

    ---------------------------------------------
    # Returns:
        gráfico de plotly.express con la media de los socores por estimador"""

    models = [] # guardamos todos los modelos a realizar
    cv_results = [] # guardamos los resultados del cross validation
    cv_means = [] # guardamos la media de los scores del cross validation
    cv_std = [] # guardamos la desviación típica media del cross validation

    # almacenamos los algoritmos en su lista
    for estimador in estimadores:
        models.append(estimador)

    # realizamos el cross validation y lo guardamos en su lista
    for model in models :
        cv_results.append(cross_val_score(model, x, y, 
                                        scoring = score, cv = folds, n_jobs=-1))
    # cogemos cada uno de los resultados obtenidos en el cv, lo promediamos y almacenamos
    for cv_result in cv_results:
        cv_means.append(abs(cv_result.mean()))
        cv_std.append(abs(cv_result.std()))
    # creamos un dataframe con los resultados
    cv_frame = pd.DataFrame(
        {
            "Algorithms":[str(x)[:limit_chr] for x in estimadores]
        })
    cv_frame["CrossValMeans"] = cv_means
    cv_frame["CrossValErrors"] = cv_std
    cv_frame = cv_frame.sort_values("CrossValMeans",ascending=False)
    
    # mostramos los resultados en un gráfico
    fig = px.bar(data_frame=cv_frame,x="CrossValMeans",y="Algorithms",
            title="CV Scores Base Line",orientation="h",
            labels={"CrossValMeans":f"Mean {score}"},
            color="Algorithms",error_x=cv_std).update_layout(showlegend=False)

    return fig


def varios_plot_regresion(estimadores,x_t,y_t,x_te,y_te,wspace=0.3,hspace=0.3,figsize=(20,8),columnas=4):

    """Función que pinta en un grid con subplots los gráficos de las 
    predicciones de cada estimador junto con los valores reales. 
    Se deben usar los gráficos de Matplotlib
    ------------------------------------------
    Args:
    estimadores = (list) lista de estimadores
    x_t = x_train
    y_t = y_train
    x_te = x_test
    y_te = y_test
    wspace = (float) el espacio ancho entre subplots
    hspace = (float) el espacio alto entre subplots
    figsize = (tuple) el tamaño de la figura
    columnas = (int) número de columnas del grid"""

    fig = plt.figure(figsize=figsize)
    fig.subplots_adjust(hspace=hspace,wspace=wspace)
    filas = ceil(len(estimadores)/columnas)

    for i in range(len(estimadores)):
        model = estimadores[i].fit(x_t,y_t)
        predic = model.predict(x_te)
        ax = fig.add_subplot(filas,columnas,i+1)
        plt.plot(y_te,y_te,label="realidad")
        plt.scatter(x=y_te,y=predic,label="predicción",color="orange")
        plt.legend()
        plt.title(str(estimadores[i]).split("(")[0])
    plt.show()


def data_transform(df,target,transformer,test_size=0.2,skip_t=[None],skip_x=None,remain="passthrough"):
    
    """Función que transforma los datos según el transformador elegido y 
    realiza el train split de los datos
    -----------------------------------------
    # Args:

        df = (pd.DataFrame) dataframe a transformar con las features (X)\n
        transformer = (class sklearn.preprocesing) transformador a aplicar\n
        target = (pd.DataFrame) variable objetivo\n
        test_size = (float) tamaño del testing set en tanto por uno\n
        skip_t = (str o list,default:None) variables que no deben ser transformadas\n
        skip_x = (list,default:None) variables que quieres eliminar de las features\n
        remain = (str, default:"passthrough") pasa las variables en skip_t sin transformar\n
                        pueden ser eliminadas también si se cambia a drop

    -----------------------------------------
    # Return:

        (x:pd.DataFrame, y:pd.Series) x_train,x_test,y_train,y_test"""

    x = df.drop(columns=skip_x)
    y = df[target]
    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=test_size,
                                                    random_state=1234)

    #Transformación
    transformer = make_column_transformer((transformer,
                            [i for i in x.columns if i not in skip_t]),
                            remainder=remain)

    train = transformer.fit_transform(x_train)
    test = transformer.fit_transform(x_test)

    nombres = x_train.columns
    nombres = [nombres[i].lower() for i in range(len(nombres))]

    train_df = pd.DataFrame(train,columns=nombres)
    test_df = pd.DataFrame(test,columns=nombres) 

    
    return (train_df,test_df,y_train,y_test)

def seleccion_variables(x,y,lista_estimadores,score="r2",n_cv=10):
    lista_variables = []

    for i in range(len(lista_estimadores)):
        rfecv = RFECV(estimator=lista_estimadores[i],scoring=score,cv=n_cv,n_jobs=-1).fit(x,y)
        for j in list(x.columns[rfecv.support_]):
            lista_variables.append(j)
        print(f"=======para el estimador {rfecv.estimator_} los datos han sido=======")
        print(f"el número de variables seleccionadas ha sido: {rfecv.n_features_}")
        print(f"el ranking de las variables vistas ha sido\n {rfecv.ranking_}")
        print(f"las variables elegidas han sido:\n {list(x.columns[rfecv.support_])}")
        print("\n")
    array_variables = np.array(lista_variables)
    var,veces = np.unique(array_variables,return_counts=True)
    df = pd.DataFrame({"variables":var,"veces":veces})\
            .sort_values("veces",ascending=False)
    print("="*40,"\nLas variables más elegidas han sido")
    print("="*40)
    return df

def cross_validation_report(estimador,x,y,kfold,lista_score):

    """Función que devuelve la media de las métricas elegidas para el cross_val_score
    --------------------------------------
    # Args:\n
        estimador: algoritmo de clasificación de sklearn para el cross_val_score\n
        x: (pd.DataFrame) las features\n
        y: (pd.DataFrame) la target\n
        kfold: (int or spliter) número de folds o método de división del dataset\n
        lista_score: (list) lista con las métricas a mostrar\n
    --------------------------------------
    # Return:\n
        la media de los scorings obtenidos en cada k división realizada"""

    mean_val = []
    mean_std = []

    print("----------(Cross Validation Metrics Report)----------")
    print( "GLOBAL VIEW:")

    for i in range(len(lista_score)):
        val = cross_val_score(estimador,x,y,cv=kfold,scoring=lista_score[i],n_jobs=-1)
        mean_val.append(round(val.mean(),3))
        mean_std.append(round(np.std(val),3))
        print(f"  - {lista_score[i]}_medio: {mean_val[i]} (+/- {mean_std[i]} std)")

def dataframes_charger(filename):
    """Función que importa el csv deseado desde el directorio
    
    -------------------------------------
    # Args:
       filename: (str)

    -------------------------------------
    # Return:
        pd.DataFrame"""

    current_path = Path(os.getcwd() + "\\data\\processed\\")
    data = pd.read_csv(current_path + filename)
    return data

def models_saver(object,filename):

    """Función para guardar los modelos de machine learning elegidos en .pkl
    
    -----------------------------------
    # Args:
        object: objeto con el modelo de machine learning entrenado
        filename: (str) nombre del archivo pickle a guardar
        
    -----------------------------------
    # Return
        Guarda el modelo en formato pickle (.pkl)"""

    destino = Path(os.getcwd().replace("notebooks","model"))
    joblib.dump(value=object,filename=destino/f"{filename}.pkl")
    print("Modelo guardado correctamente")