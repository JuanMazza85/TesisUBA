import pandas as pd
import numpy as np
import warnings #Para suprimir los warnings de deprecated
from sklearn.metrics import explained_variance_score, mean_absolute_error
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
import xgboost as xgb
from itertools import product
import os
import shutil
import logging
import math
from pickle import dump
os.chdir('C:\\Users\\Juani\\OneDrive\\Juani\\Maestria\\Tesis Clima\\Scripts')
import Plotter 

Grupos                  = ['G0','G1','G2','G3']
Meses                   = ['01','02','03','04','05','06','07','08','09','10','11','12']

NombreModelo            = 'XGBoost'

RootFolderPredictores = 'C:\\Users\\Juani\\OneDrive\\Juani\\Maestria\\Tesis Clima\\Predictores\\'
PathClases            = 'C:\\Users\\Juani\\OneDrive\\Juani\\Maestria\\Tesis Clima\\Datos Lluvias\\Gran Chaco\\GC_GROUPMEDIANS.xlsx'
PathModelOutput       = 'C:\\Users\\Juani\\OneDrive\\Juani\\Maestria\\Tesis Clima\\Modelos\\'

XGB_Target          = ['reg:linear', 'reg:gamma']
XGB_MaxDepth        = [2, 3, 6]
XGB_NEstimators     = [100, 250, 500, 1000]
XGB_RegAlpa         = [0, 0.1]
XGB_RegLambda       = [0, 0.1]
ValidationYears     = (2009, 2019)


 
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
    #Clase = (Clase - YMean) / YStdDev
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
    def BuildRegressor(Tree):
        Target = Tree[0]
        MaxDepth = Tree[1]
        NEstimators = Tree[2]
        Alpha = Tree[3]
        Lambda = Tree[4]
        XGBModel = xgb.XGBRegressor(objective=Target, max_depth = MaxDepth, n_estimators = NEstimators, eval_metric = 'mae', reg_alpha = Alpha, reg_lambda = Lambda, n_jobs = 2)
        return XGBModel
    
    
    #Step 3  =======================================================================
    #Train all the models
    print("Training models to find best hyperparameters")
    CantModelos = len(XGB_Target)
    MetricasModelo = [] 
    X_train, X_test = Predictores[:], X_val[:] 
    y_train, y_test = Clase[:], y_val[:]
    i=1
    for Tree in list(product(XGB_Target, XGB_MaxDepth, XGB_NEstimators, XGB_RegAlpa, XGB_RegLambda)):
        Modelo = BuildRegressor(Tree)
        Target = Tree[0]
        MaxDepth = Tree[1]
        NEstimators = Tree[2]
        Alpha = Tree[3]
        Lambda = Tree[4]        
        Modelo = Modelo.fit(X_train, y_train)
        y_pred = Modelo.predict(X_test) 
        y_pred = [0 if math.isnan(x) else x for x in y_pred]        
        MeanAbsoluteErr = round(mean_absolute_error(y_test,y_pred),3)
        ExplainedVar = round(explained_variance_score(y_test,y_pred),3)        
        print('\t\tResults for XGB ' + str(i) + ' [Target: ' + str(Target) + ', MaxDepth: ' + str(MaxDepth) + ', NEstimators: ' + str(NEstimators) + ', Alpha: ' + str(Alpha) + ', Lambda: ' + str(Lambda) + '] -> MAE: ' + str(round(MeanAbsoluteErr,2)) + ' | ExpVar: ' +  str(round(ExplainedVar,2)))
        logging.info('Results for XGB ' + str(i) + ' [Target: ' + str(Target) + ', MaxDepth: ' + str(MaxDepth) + ', NEstimators: ' + str(NEstimators) + ', Alpha: ' + str(Alpha) + ', Lambda: ' + str(Lambda) + '] -> MAE: ' + str(round(MeanAbsoluteErr,2)) + ' | ExpVar: ' +  str(round(ExplainedVar,2)))
        MetricasModelo.append(MeanAbsoluteErr)
        i = i + 1
    
    
    #Step 4 =======================================================================
    #Get best model
    print("\t\tBest Model:")
    logging.info('Best Model:')        
    MejorXGB = min(MetricasModelo)    
    i = MetricasModelo.index(MejorXGB)
    MejorXGB = list(product(XGB_Target, XGB_MaxDepth, XGB_NEstimators, XGB_RegAlpa, XGB_RegLambda))[i]
    print('\t\t\tTarget:', MejorXGB[0])
    logging.info('Target:' + str(MejorXGB[0]))
    print('\t\t\tMaxDepth:', MejorXGB[1])
    logging.info('MaxDepth:' + str(MejorXGB[1]))
    print('\t\t\tNEstimators:', MejorXGB[2])
    logging.info('NEstimators:' + str(MejorXGB[2]))
    print('\t\t\tAlpha:', MejorXGB[3])
    logging.info('Alpha:' + str(MejorXGB[3]))
    print('\t\t\tGamma:', MejorXGB[4])
    logging.info('Gamma:' + str(MejorXGB[4]))    
    print("\t\tRetraining Best Model:")
    Modelo = BuildRegressor(MejorXGB)
    #Retrain Best Model
    Modelo = Modelo.fit(X_train, y_train)
    y_pred = Modelo.predict(X_test) 
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
    Modelo = Modelo.fit(Predictores[:], Clase[:])    
    dump(Modelo, open(OutputFolder + '\\' + NombreModelo + '\\' + NombreModelo + '.mdl', 'wb'))
    np.savetxt(OutputFolder + '\\' + NombreModelo + '\\BestMetric.txt', [ExplainedVar], delimiter = ',')
    np.savetxt(OutputFolder + '\\' + NombreModelo + '\\XMean.txt', XMean, delimiter = ',')
    np.savetxt(OutputFolder + '\\' + NombreModelo + '\\XStdDev.txt', XStdDev, delimiter = ',')
    np.savetxt(OutputFolder + '\\' + NombreModelo + '\\YMean.txt', YMean, delimiter = ',')
    np.savetxt(OutputFolder + '\\' + NombreModelo + '\\YStdDev.txt', YStdDev, delimiter = ',')
     
    
    