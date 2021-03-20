import pandas as pd
import numpy as np
import warnings #Para suprimir los warnings de deprecated
from sklearn.metrics import explained_variance_score, mean_absolute_error    #Para la confusion 
from sklearn.svm import SVR
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
#from sklearn.model_selection import KFold
from itertools import product
import os
import shutil
from pickle import dump
import logging
os.chdir('C:\\Users\\Juani\\OneDrive\\Juani\\Maestria\\Tesis Clima\\Scripts')
import Plotter 

Grupos                  = ['G0','G1','G2','G3']
Meses                   = ['01','02','03','04','05','06','07','08','09','10','11','12']

NombreModelo            = 'SupportVectorRegression'

RootFolderPredictores = 'C:\\Users\\Juani\\OneDrive\\Juani\\Maestria\\Tesis Clima\\Predictores\\'
PathClases            = 'C:\\Users\\Juani\\OneDrive\\Juani\\Maestria\\Tesis Clima\\Datos Lluvias\\Gran Chaco\\GC_GROUPMEDIANS.xlsx'
PathModelOutput       = 'C:\\Users\\Juani\\OneDrive\\Juani\\Maestria\\Tesis Clima\\Modelos\\'

SVM_Kernel   = ['rbf','rbf','rbf','rbf','rbf', 'poly','poly','poly','poly','poly', 'sigmoid','sigmoid','sigmoid','sigmoid','sigmoid','poly']
SVM_Degree   = [0,0,0,0,0,1,2,3,4,5,0,0,0,0,0,12]
SVM_Gamma    = ['auto','auto','auto','auto','auto','auto','auto','auto','auto','auto','auto','auto','auto','auto','auto','auto']
SVM_Coef     = [0,5,10,15,20,0,5,10,15,20,0,5,10,15,20,0]
SVM_C        = [1,2,3,4,5,1,2,3,4,5,1,2,3,4,5,1]

ValidationYears          = (2009, 2019)



        
 
EliminarCorridasAnteriores = False        
for Grupo in Grupos:
    if EliminarCorridasAnteriores and os.path.exists(PathModelOutput + Grupo):
        shutil.rmtree(PathModelOutput + Grupo, ignore_errors=True)
    if not os.path.exists(PathModelOutput + Grupo):
        os.mkdir(PathModelOutput + Grupo)

  
#Grupo, Mes = list(product(Grupos, Meses))[0]      
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
    def BuildRegressor(i):
        regressor = SVR(kernel = SVM_Kernel[i], degree = SVM_Degree[i], gamma = SVM_Gamma[i], coef0 = SVM_Coef[i], C = SVM_C[i])
        return regressor    
    
    #Step 3  =======================================================================
    #Train all the models
    print("Training models to find best hyperparameters")
    CantModelos = len(SVM_Kernel)
    MetricasModelo = [] * CantModelos
    for i in range(0,CantModelos):
        Modelo = BuildRegressor(i)
          
        X_train, X_test = Predictores[:], X_val[:] 
        y_train, y_test = Clase[:], y_val[:]
        #Retrain Best Model
        Modelo.fit(X = X_train, y = y_train)
        y_pred = Modelo.predict(X = X_test)
        #Metrics
        MeanAbsoluteErr = round(mean_absolute_error(y_test,y_pred),3)
        ExpVar = round(explained_variance_score(y_test,y_pred),3)
        
        print('\t\tResults for SVR', i + 1, ' [Kernel: ' + str(SVM_Kernel[i]) + ', Degree: ' + str(SVM_Degree[i]) + ', Gamma: ' + str(SVM_Gamma[i]) + ', Coef: ' + str(SVM_Coef[i]) + ', C: ' + str(SVM_C[i]) + '] ->', 'MAE:', round(MeanAbsoluteErr,2),'| ExpVar:', round(ExpVar,2))
        logging.info('Results for SVR ' + str(i + 1) + ' [Kernel: ' + str(SVM_Kernel[i]) + ', Degree: ' + str(SVM_Degree[i]) + ', Gamma: ' + str(SVM_Gamma[i]) + ', Coef: ' + str(SVM_Coef[i]) + ', C: ' + str(SVM_C[i]) + '] -> MAE:' + str(round(MeanAbsoluteErr,2)) + ' | ExpVar:' +  str(round(ExpVar,2)))
        
        MetricasModelo.append(MeanAbsoluteErr)
    
    
    #Step 4 =======================================================================
    #Get best model
    print("\t\tBest Model:")
    logging.info('Best Model:')
    MejorMetrica = min(MetricasModelo)
    MejorMetrica = MetricasModelo.index(MejorMetrica)
    print('\t\t\tKernel:', SVM_Kernel[MejorMetrica])
    logging.info('Kernel:' + str(SVM_Kernel[MejorMetrica]))
    print('\t\t\tDegree:', SVM_Degree[MejorMetrica])
    logging.info('Degree:' + str(SVM_Degree[MejorMetrica]))
    print('\t\t\tGamma:', SVM_Gamma[MejorMetrica])
    logging.info('Gamma:' + str(SVM_Gamma[MejorMetrica]))
    print('\t\t\tCoef:', SVM_Coef[MejorMetrica])
    logging.info('Coef:' + str(SVM_Coef[MejorMetrica]))
    print('\t\t\tC:', SVM_C[MejorMetrica])
    logging.info('C:' + str(SVM_C[MejorMetrica]))
     
    print("\t\tRetraining Best Model:")
    Modelo = BuildRegressor(MejorMetrica)
    #Retrain Best Model
    Modelo.fit(X = X_train, y = y_train)
    y_pred = Modelo.predict(X = X_test)
    #Reverse Variable Standardization    
    y_val = (y_val * YStdDev) + YMean
    y_pred = (y_pred * YStdDev[0,0]) + YMean
    y_test = (y_test * YStdDev) + YMean
    
    y_pred = y_pred.transpose()
    y_pred[y_pred<0] = 0
    
    #Metrics
    MeanAbsoluteErr = round(mean_absolute_error(y_test,y_pred),3)
    ExplainedVar = round(explained_variance_score(y_test,y_pred),3)
    
    #Step 5 =======================================================================
    #Plot best model        
    y_pred = np.array(y_pred).flatten()
    y_val = np.array(y_val).flatten()
    Plotter.PlotModelPerformance(y_val, y_pred, MeanAbsoluteErr, ExplainedVar, ValidationYears, Percentiles, NombreModelo, Grupo, Mes, PathModelOutput)         
     
    
    #Step 6 =======================================================================
    #Save best model
    Modelo.fit(X = Predictores[:], y = Clase[:])
        
    dump(Modelo, open(OutputFolder + '\\' + NombreModelo + '\\SVR.mdl', 'wb'))
    
    np.savetxt(OutputFolder + '\\' + NombreModelo + '\\BestMetric.txt', [ExplainedVar], delimiter = ',')
    np.savetxt(OutputFolder + '\\' + NombreModelo + '\\XMean.txt', XMean, delimiter = ',')
    np.savetxt(OutputFolder + '\\' + NombreModelo + '\\XStdDev.txt', XStdDev, delimiter = ',')
    np.savetxt(OutputFolder + '\\' + NombreModelo + '\\YMean.txt', YMean, delimiter = ',')
    np.savetxt(OutputFolder + '\\' + NombreModelo + '\\YStdDev.txt', YStdDev, delimiter = ',')
     
    