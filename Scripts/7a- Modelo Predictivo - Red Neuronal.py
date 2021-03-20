import pandas as pd
import numpy as np
import warnings #Para suprimir los warnings de deprecated
from sklearn.metrics import explained_variance_score, mean_absolute_error    #Para la confusion 
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras import optimizers
#from sklearn.model_selection import KFold
from itertools import product
import os
import shutil
import logging
os.chdir('C:\\Users\\Juani\\OneDrive\\Juani\\Maestria\\Tesis Clima\\Scripts')
import Plotter 

Grupos                  = ['G0','G1','G2','G3']
Meses                   = ['01','02','03','04','05','06','07','08','09','10','11','12']

NombreModelo            = 'NeuralNetwork'

RootFolderPredictores = 'C:\\Users\\Juani\\OneDrive\\Juani\\Maestria\\Tesis Clima\\Predictores\\'
PathClases            = 'C:\\Users\\Juani\\OneDrive\\Juani\\Maestria\\Tesis Clima\\Datos Lluvias\\Gran Chaco\\GC_GROUPMEDIANS.xlsx'
PathModelOutput       = 'C:\\Users\\Juani\\OneDrive\\Juani\\Maestria\\Tesis Clima\\Modelos\\'

NN_CapasOcultas          = [2, 6]
NN_NeuronasPorCapaOculta = [64, 128]
NN_FuncionesCapasOcultas = ['linear', 'relu']
NN_Optimizer             = ['nadam','adadelta']
NN_BatchSize             = [5]
NN_Epochs                = [200]
NN_DropOutRate           = [0, 0.1, 0.25]

ValidationYears          = (2009, 2019)


 
EliminarCorridasAnteriores = False
for Grupo in Grupos:
    if EliminarCorridasAnteriores and os.path.exists(PathModelOutput + Grupo):
        shutil.rmtree(PathModelOutput + Grupo, ignore_errors=True)
    if not os.path.exists(PathModelOutput + Grupo):
        os.mkdir(PathModelOutput + Grupo)

       
for GrupoMes in product(Grupos, Meses):
    Grupo = GrupoMes[0]
    Mes = GrupoMes[1]

    #Set outut folder
    OutputFolder = PathModelOutput + Grupo + '\\Modelo_' + Grupo + '_M' + Mes + '_Y' + str(ValidationYears[0]) + '-' + str(ValidationYears[1])    
    if not os.path.exists(OutputFolder):
        os.mkdir(OutputFolder)
    if not os.path.exists(OutputFolder + '\\' + NombreModelo):
        os.mkdir(OutputFolder + '\\' + NombreModelo)
    
    #Set log file
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    if os.path.exists(OutputFolder + '\\' + NombreModelo + '\\' + NombreModelo + '.log'):
        logging.basicConfig(format = '%(asctime)s %(message)s', filename=OutputFolder + '\\' + NombreModelo + '\\' + NombreModelo + '.log',level=logging.INFO)
    else:
        logging.basicConfig(format = '%(asctime)s %(message)s', filename=OutputFolder + '\\' + NombreModelo + '\\' + NombreModelo + '.log',filemode='w',level=logging.INFO)

    
    print('Processing models for Group [' + Grupo + '] Month [' + Mes + ']')
    
    #Step 1 =======================================================================
    #Read the right predictors file
    Predictores = pd.read_excel(RootFolderPredictores + Grupo + '\\Predictors\\Predictors_' + Grupo + '_M' + Mes + '.xlsx')
    Clase = pd.read_excel(PathClases, sheet_name = Grupo )    
    ColumnaMes = 'M' + Mes
    Predictores = np.matrix(Predictores)
    Clase = np.matrix(Clase[[ColumnaMes]])    
    Percentiles = np.percentile(Clase,[33,66])    
    XMean = Predictores.mean(axis=0)
    XStdDev = Predictores.std(axis=0)
    YMean = Clase.mean(axis=0)
    YStdDev = Clase.std(axis=0)    
    Predictores = (Predictores - XMean) / XStdDev
    Clase = (Clase - YMean) / YStdDev
    CantRowsTraining = ValidationYears[0] - 1979
    X_val = Predictores[CantRowsTraining:,:]
    y_val = Clase[CantRowsTraining:]    
    Predictores = Predictores[:CantRowsTraining,:]
    Clase = Clase[:CantRowsTraining]    
    if Predictores.shape[0] == 0:
        #This Group and Month have no predictors, no model can be built
        continue
          
    #Step 2 =======================================================================
    #Build the network
    def BuildRegressor(Red):
        Capas = Red[0]
        Neuronas = Red[1]
        Funcion = Red[2]
        Optimizer = Red[3]
        DropOut = Red[4]
        
        regressor = Sequential()
        regressor.add(Dense(units = Neuronas, kernel_initializer = 'normal', activation = Funcion, input_dim = Predictores.shape[1]))
        for j in range(1, Capas):
            regressor.add(Dense(units = Neuronas, kernel_initializer = 'normal', activation = Funcion))
            if DropOut > 0:
                regressor.add(Dropout(rate = DropOut))        
        regressor.add(Dense(units = 1, kernel_initializer = 'normal', activation = 'relu')) #Last layer is ReLU to avooid negative values
        if Optimizer == 'sgd':
            Optimizer = optimizers.SGD(learning_rate = 0.1, momentum = 0.1, nesterov= True)
        elif Optimizer == 'nadam':
            Optimizer = optimizers.Nadam()
        elif Optimizer == 'adadelta':
            Optimizer = optimizers.Adadelta()
        regressor.compile(optimizer = Optimizer, loss = 'mae', metrics = ['mae'])
        return regressor
    
    
    #Step 3  =======================================================================
    #Train all the models
    print("Training models to find best hyperparameters")
    CantModelos = len(NN_CapasOcultas)
    MetricasModelo = []
    X_train, X_test = Predictores[:], X_val[:] 
    y_train, y_test = Clase[:], y_val[:]
    i=1
    for Red in list(product(NN_CapasOcultas, NN_NeuronasPorCapaOculta, NN_FuncionesCapasOcultas,NN_Optimizer, NN_DropOutRate, NN_Epochs, NN_BatchSize)):
        Modelo = BuildRegressor(Red)
        Capas = Red[0]
        Neuronas = Red[1]
        Funcion = Red[2]
        Optimizer = Red[3]
        DropOut = Red[4]
        Epochs = Red[5]
        BatchSize = Red[6]
        history = Modelo.fit(x = X_train, y = y_train, batch_size = BatchSize, epochs = Epochs, validation_data = (X_test, y_test), verbose = False)
        y_pred = Modelo.predict(x = X_test, batch_size = BatchSize)
        y_pred.transpose()
        y_pred = np.nan_to_num(y_pred, 0)
        MeanAbsoluteErr = round(mean_absolute_error(y_test,y_pred),3)
        ExplainedVar = round(explained_variance_score(y_test,y_pred),3)
        print('\t\tResults for NN ' + str(i) + ' [Capas: ' + str(Capas) + ' Neuronas: ' + str(Neuronas) + ' Funcion: ' + Funcion + ' Optimizer: ' + Optimizer + ', Dropout: ' + str(DropOut) + '] -> MAE: ' + str(round(MeanAbsoluteErr,2)) + ' | ExpVar: ' +  str(round(ExplainedVar,2)))
        logging.info('Results for NN ' + str(i) + ' [Capas: ' + str(Capas) + ' Neuronas: ' + str(Neuronas) + ' Funcion: ' + Funcion + ' Optimizer: ' + Optimizer + ', Dropout: ' + str(DropOut) + '] -> MAE: ' + str(round(MeanAbsoluteErr,2)) + ' | ExpVar: ' +  str(round(ExplainedVar,2)))
        MetricasModelo.append(MeanAbsoluteErr)
        i = i + 1
    
    
    #Step 4 =======================================================================
    #Get best model
    print("\t\tBest Model:")
    logging.info('Best Model:')        
    MejorRed = min(MetricasModelo)
    
    i = MetricasModelo.index(MejorRed)
    MejorRed = list(product(NN_CapasOcultas, NN_NeuronasPorCapaOculta, NN_FuncionesCapasOcultas,NN_Optimizer, NN_DropOutRate, NN_Epochs, NN_BatchSize))[i]
    print('\t\t\tLayers:', MejorRed[0])
    logging.info('Layers:' + str(MejorRed[0]))
    print('\t\t\tNeurons per Layer:', MejorRed[1])
    logging.info('Neurons per Layer:' + str(MejorRed[1]))
    print('\t\t\tActivation f(x):', MejorRed[2])
    logging.info('Activation f(x):' + str(MejorRed[2]))
    print('\t\t\tOptimizer:', MejorRed[3])
    logging.info('Optimizer:' + str(MejorRed[3]))
    print('\t\t\tDropout:', MejorRed[4])
    logging.info('Dropout:' + str(MejorRed[4]))
    
    print("\t\tRetraining Best Model:")
    Modelo = BuildRegressor(MejorRed)
    #Retrain Best Model
    history = Modelo.fit(x = X_train, y = y_train, batch_size = MejorRed[6], epochs = MejorRed[5], validation_data = (X_test, y_test), verbose = False)
    y_pred = Modelo.predict(x = X_test, batch_size = MejorRed[6])
    y_pred.transpose()
    #Reverse Variable Standardization    
    y_val = (y_val * YStdDev) + YMean
    y_pred = (y_pred * YStdDev) + YMean
    y_test = (y_test * YStdDev) + YMean
    #Metrics
    MeanAbsoluteErr = round(mean_absolute_error(y_test,y_pred),3)
    ExplainedVar = round(explained_variance_score(y_test,y_pred),3)
    
    
    #Step 5 =======================================================================
    #Plot best model     
    y_pred = np.array(y_pred).flatten()
    y_val = np.array(y_val).flatten()
    Plotter.PlotNetworkPerformance(history, y_val, y_pred, MeanAbsoluteErr, ExplainedVar, ValidationYears, Percentiles, NombreModelo, Grupo, Mes, PathModelOutput)         
    
    
    #Step 6 =======================================================================
    #Save best model
    history = Modelo.fit(x = Predictores[:], y = Clase[:], batch_size =  MejorRed[6], epochs =  MejorRed[5], verbose = False)
    
    Modelo.save(OutputFolder + '\\' + NombreModelo)
    
    np.savetxt(OutputFolder + '\\' + NombreModelo + '\\BestMetric.txt', [ExplainedVar], delimiter = ',')
    np.savetxt(OutputFolder + '\\' + NombreModelo + '\\XMean.txt', XMean, delimiter = ',')
    np.savetxt(OutputFolder + '\\' + NombreModelo + '\\XStdDev.txt', XStdDev, delimiter = ',')
    np.savetxt(OutputFolder + '\\' + NombreModelo + '\\YMean.txt', YMean, delimiter = ',')
    np.savetxt(OutputFolder + '\\' + NombreModelo + '\\YStdDev.txt', YStdDev, delimiter = ',')
     
    
    