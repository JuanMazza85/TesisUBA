# -*- coding: utf-8 -*-
"""
Created on Tue Sep 12 11:02:48 2017

@author: waldo

Función trainSOM
-------------
Parámetros:
       P: es una matriz con los datos de los patrones con los cuales
           entrenar la red neuronal. 
       filas: la cantidad de filas del mapa SOM
       columnas: la cantidad de columnas del mapa SOM
       alfa_inicial: velocidad de aprendizaje inicial
       vecindad: vecindad inicial
       fun_vecindad: función para determinar la vecindad (1: lineal, 2: sigmoide)
       sigma: ancho de la campana (solo para vecindad sigmoide)
       ite_reduce: la cantidad de iteraciones por cada tamaño de vecindad
           (la cantidad de iteraciones total sera: total = ite_reduce * (vecindad+1))
       dibujar: si vale True (y los datos son en dos dimensiones) dibuja los
           ejemplos y el mapa SOM.

Devuelve:
       w_O: la matriz de pesos de las neuronas competitivas

Ejemplo de uso:
       (w_O) = trainSOM(P, filas, columnas, alfa, vecindad_inicial, fvecindario, sigma, reduce, True);

-------------------------------------------------------------------------------

Función trainOL
-------------
Parámetros:
       P: es una matriz con los datos de los patrones con los cuales
           entrenar la red neuronal. 
       T: es una matriz con la salida esperada para cada ejemplo. Esta matriz 
           debe tener tantas filas como neuronas de salida tenga la red
       T_O: clases con su valor original (0 .. n-1) (Solo es utilizado para graficar)
       w: es la matriz de pesos devuelta por la función trainSOM
       filas: la cantidad de filas del mapa SOM
       columnas: la cantidad de columnas del mapa SOM
       alfa: velocidad de aprendizaje 
       max_ite: la cantidad de iteraciones del entrenamiento
       dibujar: si vale True (y los datos son en dos dimensiones) dibuja los
           ejemplos y el mapa SOM.

Devuelve:
       w_S: la matriz de pesos de las neuronas de la capa de salida

Ejemplo de uso:
       (w_S) = trainOL(P, T_matriz.T, T, w, filas, columnas, alfa, 100, True);
       
-------------------------------------------------------------------------------

Función umatrix
-------------
Parámetros:
    def umatrix(w, filas, columnas):
       w: es la matriz de pesos devuelta por la función trainSOM
       filas: la cantidad de filas del mapa SOM
       columnas: la cantidad de columnas del mapa SOM

Devuelve:
       umatrix: la matriz de distancias del SOM

Ejemplo de uso:
       umatrix = umatrix(w, filas, columnas)
       
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as col
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform
from mpl_toolkits.mplot3d import Axes3D                   #Para poder plottear en 3D

marcadores = {0:('.','b'), 1:('.','g'), 2:('x', 'y'), 3:('*', 'm'), 4:('.', 'r'), 5:('+', 'k')}


# -*- coding: utf-8 -*-
"""
Created on Tue Aug 22 15:28:31 2017

@author: auvimo
"""

import numpy as np
import scipy.misc as sp
#En 2.7 hay que hacer otro import que ofrece imread (quizas scipy.image)

def OpenImage(archivo):
    datos = np.array(sp.imread(archivo, mode='P'))   # en mac es mode='I'
    maximo = len(datos)
    X = np.array([], dtype=np.int64).reshape(0,3)
    colores = np.array([0, 9, 12, 10, 6, 11]) # negro rojo azul verde
    for color in colores:
        filas, columnas = np.where(datos == color)
        clase = np.where(colores == color)[0][0] 
        clases = [clase] * len(filas)
        X = np.vstack([X, np.column_stack((columnas+1, maximo-(filas+1), clases))])
    return X


def dendograma(matriz, T):
    (filas, columnas) = matriz.shape
    labels=[]
    for f in range(filas):
        labels.append(str(int(T[f])))
        for c in range(columnas):
            if(f==c):
                matriz[f,c] = 0
            else:
                if(matriz[f,c] == 0):
                    matriz[f,c] = 2
                else:
                    matriz[f,c] = 1 / matriz[f,c]
                    
    
    
    dists = squareform(matriz)
    linkage_matrix = linkage(dists, "single")
    dendrogram(linkage_matrix, labels=labels)
    plt.title("Dendograma")
    plt.show()

def plot(P, T, W, filas, columnas, pasos, title):
    plt.figure(0)
    plt.clf()
    
    #Ejemplos
    x = []
    y = []
    if(T is None):
        for i in range(P.shape[0]):
            x.append(P[i, 0])
            y.append(P[i, 1])
        plt.scatter(x, y, marker='o', color='b', s=100)
    else:
        colores = len(marcadores)
        for class_value in np.unique(T):
            x = []
            y = []
            for i in range(len(T)):
                if T[i] == class_value:
                    x.append(P[i, 0])
                    y.append(P[i, 1])
            plt.scatter(x, y, marker=marcadores[class_value % colores][0], color=marcadores[class_value % colores][1])
    
    #ejes
    minimos = np.min(P, axis=0)
    maximos = np.max(P, axis=0)
    diferencias = maximos - minimos
    minimos = minimos - diferencias * 0.1
    maximos = maximos + diferencias * 0.1
    plt.axis([minimos[0], maximos[0], minimos[-1], maximos[1]])
    
    #centroides
    (neuronas, patr) = W.shape
    for neu in range(neuronas):
        plt.scatter(W[neu,0], W[neu,1], marker='o', color='r')
        
    #conexiones
    if(pasos is not None):
    #if(pasos != None):
        for f in range(filas):
            for c in range(columnas):
                n1 = f*columnas + c
                for ff in range(filas):
                    for cc in range(columnas):
                        if(pasos[f, c, ff, cc] == 1):
                            n2 = ff*columnas + cc
                            plt.plot([W[n1, 0], W[n2, 0]], [W[n1, 1], W[n2, 1]], color='r')                   
    
    plt.title(title)
    plt.draw()
    plt.pause(0.00001)
    
    
def plot3D(P, T, W, filas, columnas, pasos, title):
    
    
    fig = plt.figure(0)
    plt.clf()
    ax= fig.add_subplot(111, projection='3d')
    c=[]
    for i in range(P.shape[0]):
        c.append(col.rgb2hex((P[i,0]/255,P[i,1]/255,P[i,2]/255)))
    
    
    ax.scatter(P[:,0], P[:,1], P[:,2], c = c, marker = 'o', linewidths=.05, s=100)
    

    ax.set_xlim3d(0,np.max(P[:,0]))
    ax.set_ylim3d(0,np.max(P[:,1]))
    ax.set_zlim3d(0,np.max(P[:,2]))
    ax.set_xlabel('R')
    ax.set_ylabel('G')
    ax.set_zlabel('B')
    
    
    #centroides
    (neuronas, patr) = W.shape
    for neu in range(neuronas):
        ax.scatter(W[neu,0], W[neu,1], W[neu,2],marker='o', c='r')
        
    #conexiones
    if(pasos is not None):
    #if(pasos != None):
        for f in range(filas):
            for c in range(columnas):
                n1 = f*columnas + c
                for ff in range(filas):
                    for cc in range(columnas):
                        if(pasos[f, c, ff, cc] == 1):
                            n2 = ff*columnas + cc
                            ax.plot([W[n1, 0], W[n2, 0]], [W[n1, 1], W[n2, 1]], [W[n1, 2], W[n2, 2]], c='r')   
                           
    
    plt.title(title)
    plt.draw()
    plt.pause(0.00001)
    
def linkdist(filas, columnas):
    pasos = np.zeros((filas, columnas, filas, columnas))
    for f in range(filas):
        for c in range(columnas):
            for ff in range(filas):
                for cc in range(columnas):
                    pasos[f, c, ff, cc] = abs(f-ff) + abs(c-cc)
    return pasos
    
def trainSOM(P, filas, columnas, alfa_inicial, vecindad, fun_vecindad, sigma, ite_reduce, dibujar):
    (cant_patrones, cant_atrib) = P.shape

    ocultas = filas * columnas    
    w_O = np.random.rand(ocultas, cant_atrib) - 0.5
    
    #w_O = np.ones((ocultas, cant_atrib)) * 0
    
    pasos = linkdist(filas, columnas)
    
    max_ite = ite_reduce * (vecindad + 1)
    ite = 0;
    
    while (ite < max_ite):
        alfa = alfa_inicial * (1 - ite / max_ite)
        for p in range(cant_patrones): 
            distancias = -np.sqrt(np.sum(np.power(w_O-P[p,:],2),1))
            ganadora = np.argmax(distancias)
            fila_g = int(np.floor(ganadora / columnas))
            columna_g = int(ganadora % columnas)

            for f in range(filas):
               for c in range(columnas):
                   if(pasos[fila_g, columna_g, f, c] <= vecindad):
                       if fun_vecindad == 1:
                           gamma = 1
                       else:
                           gamma = np.exp(- pasos[fila_g, columna_g, f, c] / (2*sigma))
              
                       n = f * columnas + c
                       w_O[n,:] = w_O[n,:] + alfa * (P[p,:] - w_O[n,:]) * gamma
            
        ite = ite + 1
        
        if (vecindad >= 1) and ((ite % ite_reduce)==0):
            vecindad = vecindad - 1;
        
        if dibujar and (cant_atrib == 2):
            plot(P, None, w_O, filas, columnas, pasos, 'Iteración: ' + str(ite))
        if dibujar and (cant_atrib == 3):
            plot3D(P, None, w_O, filas, columnas, pasos, 'Iteración: ' + str(ite))
    
    if ite % 10 == 0:      
        if dibujar and (cant_atrib == 2):
            plot(P, None, w_O, filas, columnas, pasos, 'Iteración: ' + str(ite))
        if dibujar and (cant_atrib == 3):
            plot3D(P, None, w_O, filas, columnas, pasos, 'Iteración: ' + str(ite))    
            
    return (w_O)

def trainOL(P, T, T_O, w, filas, columnas, alfa, max_ite, dibujar):
    (cant_patrones, cant_atrib) = P.shape
    (cant_patrones, salidas) = T.shape   
    ocultas = filas * columnas
    
    pasos = linkdist(filas, columnas)
    w_S = np.random.rand(salidas, ocultas) - 0.5
    
    ite = 0;
    while ( ite <= max_ite ):
        for p in range(cant_patrones): 
            distancias = -np.sqrt(np.sum((w-(P[p,:])*np.ones((ocultas,1)))**2,1))
            ganadora = np.argmax(distancias)
       
            w_S[:, ganadora] = w_S[:, ganadora] + alfa * (T[p, :] - w_S[:, ganadora])
    
        ite = ite + 1
        
    if dibujar and (cant_atrib == 2):
        plot(P, T_O, w, filas, columnas, pasos, 'Fin')
    if dibujar and (cant_atrib == 3):
        plot3D(P, T_O, w, filas, columnas, pasos, 'Fin')    
        
    return (w_S)

def umatrix(w, filas, columnas):
    (ncen, atributos) = w.shape 
    umat = np.zeros((filas*2-1, columnas*2-1))
    
    for f in range(filas):
        for c in range(columnas):
            ff = f*2
            cc= c*2
            n1 = f * columnas + c 
            suma = 0
            n=0
            
            n2 = f * columnas + (c+1)
            if(cc < (columnas*2-2)):
                umat[ff, cc+1] = np.sqrt(np.sum((w[n1,:]-w[n2,:])**2))                
                suma = suma + umat[ff, cc+1]
                n=n+1
            n2 = (f+1) * columnas + c
            if(ff < (filas*2-2)):
                umat[ff+1, cc] = np.sqrt(np.sum((w[n1,:]-w[n2,:])**2))                
                suma = suma + umat[ff+1, cc]
                n=n+1            
            if(n==2):
                umat[ff+1, cc+1] = suma / 2
                suma = suma + umat[ff+1, cc+1]
                n=n+1
            if(n>0):
                umat[ff, cc] = suma / n
    umat[filas*2-2, columnas*2-2] = (umat[filas*2-3, columnas*2-2] + umat[filas*2-2, columnas*2-3]) / 2
            
    return umat

def PrepararSOM(X, Y, CantColumnas):
    
    for j in range(X.shape[1]):
        MinCol = X[:,j].min()
        MaxCol = X[:,j].max()
        for i in range(X.shape[0]):
            X[i,j] = (X[i,j] - MinCol) / (MaxCol - MinCol)

    T = Y
    P = X[:,:CantColumnas]
    
    return (P, T)
