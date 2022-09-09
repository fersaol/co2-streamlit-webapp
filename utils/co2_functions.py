import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans,DBSCAN
from sklearn.metrics import silhouette_samples,silhouette_score,confusion_matrix,\
                            classification_report,precision_score,recall_score,\
                            f1_score
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import cross_val_score,KFold
import seaborn as sns
from sklearn.feature_selection import RFECV

print("Hey!, el módulo co2 ha sido importado correctamente \U0001F973")


class Clustering:

    """Contiene los métodos para la fase de clustering del proyecto co2"""

    def __init__(self):
        pass

    def grafico_silueta(df,estimador="k",radio=None,minpts=None,n_clusters=[2,3,4]):
        """Realiza el gráfico de silueta para el número de clusters elegidos con KMeans o DBSCAN
            
        --------------------------------------
        Argumentos:
        
        df = pd.DataFrame\n
        estimador = (str) k=kmeans o d= DBSCAN\n
        radio = (float or int) es el eps de DBSCAN\n
        minpts = (int) min_samples para DBSCAN\n
        n_clusters = (list) lista con los nº de clusters a mostrar para KMeans"""
        
        if estimador == "k":
            for k in n_clusters:
                fig,ax= plt.subplots(1)
                fig.set_size_inches(25, 7)
                km = KMeans(n_clusters=k)
                labels = km.fit_predict(df)
                centroids = km.cluster_centers_


                silhouette_vals = silhouette_samples(df, labels)

                y_lower, y_upper = 0, 0
                for i, cluster in enumerate(np.unique(labels)):
                    cluster_silhouette_vals = silhouette_vals[labels == cluster]
                    cluster_silhouette_vals.sort()
                    y_upper += len(cluster_silhouette_vals)

                    plt.barh(range(y_lower, y_upper), cluster_silhouette_vals, edgecolor='none', height=1,alpha=0.6)
                    plt.text(-0.03, (y_lower + y_upper) / 2, str(i + 1),weight="bold",fontsize="medium")
                    y_lower += len(cluster_silhouette_vals)

                
                avg_score = np.mean(silhouette_vals)
                plt.axvline(avg_score, linestyle='--', linewidth=5, color='red')
                plt.annotate(f"Silhouette Score:{round(avg_score,3)}",xy=(avg_score*1.1,y_upper/10),
                            fontsize="x-large",fontfamily="fantasy")
                plt.xlim([-0.06, 1])
                plt.xlabel('Silhouette coefficient values')
                plt.ylabel('Cluster labels')
                plt.title('--Silhouette plot for {} clusters--'.format(k), y=1.02,
                            fontsize="x-large",weight="bold",fontfamily="monospace",
                            bbox=dict(boxstyle="Round4",alpha=0.3,color="grey"))
            plt.show()

        if estimador == "d":
            
            fig,ax= plt.subplots(1)
            fig.set_size_inches(25, 7)
            km = DBSCAN(eps=radio,min_samples=minpts)
            labels = km.fit_predict(df)
            labels_1 = np.array([x for x in labels if x != -1])
            df2 = df.drop(index=np.where(labels==-1)[0])
            silhouette_vals = silhouette_samples(df2, labels_1)

            y_lower, y_upper = 0, 0
            for i, cluster in enumerate(np.unique(labels_1)):
                cluster_silhouette_vals = silhouette_vals[labels_1 == cluster]
                cluster_silhouette_vals.sort()
                y_upper += len(cluster_silhouette_vals)

                plt.barh(range(y_lower, y_upper), cluster_silhouette_vals, edgecolor='none', height=1,alpha=0.6)
                plt.text(-0.03, (y_lower + y_upper) / 2, str(i + 1),weight="bold",fontsize="medium")
                y_lower += len(cluster_silhouette_vals)

        
        avg_score = np.mean(silhouette_vals)
        plt.axvline(avg_score, linestyle='--', linewidth=5, color='red')
        plt.annotate(f"Silhouette Score:{round(avg_score,3)}",xy=(avg_score*1.1,y_upper/10),
                    fontsize="x-large",fontfamily="fantasy")
        plt.xlim([-0.06, 1])
        plt.xlabel('Silhouette coefficient values')
        plt.ylabel('Cluster labels')
        plt.title('--Silhouette plot for {} clusters--'.format(len(np.unique(km.labels_))-1), y=1.02,
                    fontsize="x-large",weight="bold",fontfamily="monospace",
                    bbox=dict(boxstyle="Round4",alpha=0.3,color="grey"))
        plt.show()

    def grafico_codo(df,max_clusters=9,semilla=None,optimo=None):
        """Funcion que grafica el codo en KMeans para detectar los clusters óptimos
        ---------------------------------------
        Args:
        df = pd.DataFrame\n
        max_clusters = (int) número máximo de clusters ha representar\n
        semilla = (int)
        optimo = (int) se puede poner despues de ver el gráfico para pintar una línea
        """
        inertia = []
        for i in np.arange(2,max_clusters):
            km = KMeans(n_clusters=i,random_state=semilla).fit(df)
            inertia.append(km.inertia_)
        # ahora dibujamos las difrentes distorsiones o inercias:
        fig = px.line(x= np.arange(2,max_clusters), y= inertia,markers=True,title="|| K-Means Inertia ||",\
                labels=dict(x="clusters",y="inertia")).add_vline(x=4,line_color="green")
        fig.show()

    def aplicacion_dbscan(df):

        """Aplica el algoritmo DBSCAN y nos devuelve el número de clusters,
        los puntos de ruido y el coeficiente de silueta
        ------------------------------------
        # Args:
            df=(pd.DataFrame)

        ------------------------------------
        # Return:
            (Print) número clusters, puntos de ruido, Silhouette Coefficient
        """
    

        db = DBSCAN().fit(df)
        labels = db.labels_

        # Número de clusters en labels, ignorando el ruido si está presente.
        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise_ = list(labels).count(-1)

        print("Número estimado de clusters: %d" % n_clusters_)
        print("Número estimado de puntos como ruido: %d" % n_noise_)
        print("Silhouette Coefficient: %0.3f" % silhouette_score(df, labels))


class Predicting:
    """Contiene los métodos de la fase de Regresión del proyecto co2"""
    def __init__(self):
        pass

    def compute_vif(df,considered_features):
        
        X = df.loc[:,considered_features]
        X['intercept'] = 1
        
        vif = pd.DataFrame()
        vif["Variable"] = X.columns
        vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
        vif = vif[vif['Variable']!='intercept']
        return vif.sort_values(by="VIF",ascending=False).reset_index().drop(columns="index").round(2)


    def cross_val_regression(estimador,xtrain,ytrain,xtest,ytest,pred="no",grafico="si"):
        lr = estimador
        lr.fit(xtrain,ytrain)
        predic = lr.predict(xtest)

        r2 = cross_val_score(lr,xtrain,ytrain,scoring="r2",cv=10).mean()
        mae = abs(cross_val_score(lr,xtrain,ytrain,scoring="neg_mean_absolute_error",cv=10).mean())
        mse = abs(cross_val_score(lr,xtrain,ytrain,scoring="neg_mean_squared_error",cv=10).mean())

        texto = "r2: {}  mae: {}  mse: {}  rmse: {}".format(round(r2,3),round(mae,3),
                                                        round(mse,3),
                                                        round(mse**(1/2),3))

        box = {"facecolor":"grey", "alpha":0.2}

        if grafico == "si":
            plt.figure(figsize=(10,5))
            sns.scatterplot(x=ytest,y=ytest,label="realidad",color="red")
            sns.scatterplot(x=ytest,y=predic,label="predicción",color="green")
            plt.title("Realidad VS Predicción")
            plt.figtext(s=texto,x=0.1287,y=-0.01,va="baseline",
            bbox=box,weight="bold",fontsize="x-large")

            plt.show()

        if pred == "si":
            return predic

    def sin_multico_unoauno(df,variables):
        """Función que elimina de una en una las variables con vif superior a 5
        -----------------------------------
        Args:
        df = (pd.DataFrame)
        variables = (list) variables a tener en cuenta"""
        
        try:
            df_vif = Predicting.compute_vif(df,variables).round(2)
            for vif in range(len(df_vif)):
                if (df_vif.loc[0,"VIF"] >= 5) or (df_vif.loc[0,"VIF"] == np.inf):
                    variables.remove(df_vif.loc[0,"Variable"])
                    df_vif = Predicting.compute_vif(df,variables).round(2)
                    df_vif = df_vif.reset_index().drop(columns="index").round(2)
            return df_vif.round(2)
        except KeyError:
            print("todas las variables han sido eliminadas al estar todas por encima de vif 5")



class Classification:
    def __init__(self):
        pass

    def new_classification_report(realidad,prediccion):
        """Función que le añade al classification report la confussion matrix
        -----------------------------------
        # Args:
            realidad: (y_test:np.array | pd.Series) (los datos reales de set de test)\n
            prediccion: (pred:np.array | pd.Series) (predicción del estimador)

        ------------------------------------
        # Return:
        Resúmen de las métricas Accuracy,Precision,Recall,F1,support, 
        confussion matrix"""
        
        sns.heatmap(confusion_matrix(realidad,prediccion),annot=True,
                                    fmt="g",cmap="Greys_r")
        plt.title("Confussion Matrix")
        plt.xlabel("Predicción")
        plt.ylabel("Realidad")
        plt.show()

        print("="*53)
        print(classification_report(realidad,prediccion))
        

    def multiclass_report_bycluster(df,target,l_vars,l_estim,metrica,splits,
                                    shuffle=False,seed=None):

        """Función que calcula la métrica elegida (Precision,Recall o F1) para cada
        uno de los estimadores seleccionados junto con sus variables elegidas de manera
        individualizada para cada cluster en modelos de clasificación multiclase
        ----------------------------------
        # Args:
            df: (pd.DataFrame) dataframe completo
            target: (str) variable objetivo
            l_vars: (list) lista de listas con las variables para cada estimador
            l_estim: (list) lista con los estimadores a usar, len(l_estim) = len(l_vars)
            metrica: (str) Precision, Recall o F1
            splits: (int) número de splits a realizar por KFold
            shuffle: (bool) si True aleatoriza las muestras
            seed: (int) para obtener resultados entre pruebas

        ----------------------------------
        # Return
            pd.DataFrame y plotly.express plot"""

        kf = KFold(n_splits= splits,shuffle=shuffle,random_state=seed)
        df_compar = pd.DataFrame()

        for i,vars in enumerate(l_vars):
            df_x = df[vars]
            s_y = df[target]
            for train_index,test_index in kf.split(df_x):
                x_train,x_test = df_x.iloc[train_index,:], df_x.iloc[test_index,:]
                y_train,y_test = s_y[train_index], s_y[test_index]

                estimator = l_estim[i].fit(x_train,y_train)
                predic = estimator.predict(x_test)
                dic_metrics = {
                "Precision":precision_score(y_test,predic,average=None),
                "Recall": recall_score(y_test,predic,average=None),
                "F1":f1_score(y_test,predic,average=None)
                        }
                metric = dic_metrics[metrica]
                df_compar[str(l_estim[i])] = metric

        df_compar = df_compar.T

        fig = make_subplots(rows=1, cols=len(df_compar.columns),shared_yaxes=True,
                        subplot_titles=["Cluster " + str(x) for x in df_compar.columns])

        for i in range(len(df_compar.columns)):
        
            fig.add_trace(go.Bar(y=df_compar[i],x=df_compar.index,text=round(df_compar[i],3)
                ,marker=dict(color = df_compar[df_compar.columns[0]],colorscale='blugrn',showscale=True)),
                row=1,
                col= i+1)

            fig.update_layout(showlegend=False,template="plotly_dark",
                title_text=f"Mean {metrica} Score for {splits } folds",height=500)

        fig.show()

        return df_compar