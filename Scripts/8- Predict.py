import pandas as pd
import numpy as np
import warnings #Para suprimir los warnings de deprecated
from sklearn.metrics import explained_variance_score, mean_absolute_error    #Para la confusion 
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
from tensorflow.keras.models import load_model
from itertools import product
import os
from pickle import load
os.chdir('C:\\Users\\Juani\\OneDrive\\Juani\\Maestria\\Tesis Clima\\Scripts')
import Plotter 
#VARS =========================================================================

Grupos                  = ['G0','G1','G2','G3']
Meses                   = ['01','02','03','04','05','06','07','08','09','10','11','12']

RootModelos             = 'C:\\Users\\Juani\\OneDrive\\Juani\\Maestria\\Tesis Clima\\Modelos\\'
RootFolderPredictores  = 'C:\\Users\\Juani\\OneDrive\\Juani\\Maestria\\Tesis Clima\\Predictores\\'
PathClases             = 'C:\\Users\\Juani\\OneDrive\\Juani\\Maestria\\Tesis Clima\\Datos Lluvias\\Gran Chaco\\GC_GROUPMEDIANS.xlsx'
ValidationYears          = (2009, 2019)

MinBestMetric = 0.15
#FUNCS ========================================================================

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0) # only difference

#CODE =========================================================================

os.chdir(RootModelos)
for GrupoMes in product(Grupos, Meses):
    Grupo =  GrupoMes[0]
    Mes = GrupoMes[1]

    #Validate we have the models for this Group-Month pair
    RutaModelos = '.\\' + Grupo + '\\Modelo_' + Grupo + '_M' + Mes + '_Y2009-2019'
    if not os.path.exists(RutaModelos):
        print('The models for group',Grupo,'month',Mes,'could not be found')
        continue
    
    #Create the output folder for the ensemble
    OutputFolder =  RutaModelos + '\\Ensemble' 
    if not os.path.exists(OutputFolder):
        os.mkdir(OutputFolder)
        
    Predictores = pd.read_excel(RootFolderPredictores + Grupo + '\\Predictors\\Predictors_' + Grupo + '_M' + Mes + '.xlsx')
    Clase = pd.read_excel(PathClases, sheet_name = Grupo )    
    ColumnaMes = 'M' + Mes
    Predictores = np.matrix(Predictores)
    Clase = np.matrix(Clase[[ColumnaMes]])  
    Quantiles = np.percentile(Clase,[33,66]) 
    
    #Discover available models
    NombreModelos = os.listdir(RutaModelos)
    NombreModelos.remove('Ensemble')
    Modelos = []
    Metrics = []
    for NombreModelo in NombreModelos:
        BestMetric = float(np.loadtxt(RutaModelos + '\\' + NombreModelo + '\\BestMetric.txt'))
        if BestMetric < MinBestMetric:
            print('The model',NombreModelo,'for group',Grupo,'month',Mes,'had low score of', round(BestMetric,2))
            continue
        Metrics.append(BestMetric)
        XMean = np.loadtxt(RutaModelos + '\\' + NombreModelo + '\\XMean.txt', delimiter=',')
        YMean = float(np.loadtxt(RutaModelos + '\\' + NombreModelo + '\\YMean.txt'))
        XStdDev = np.loadtxt(RutaModelos + '\\' + NombreModelo + '\\XStdDev.txt', delimiter = ',')
        YStdDev = float(np.loadtxt(RutaModelos + '\\' + NombreModelo + '\\YStdDev.txt'))
        if NombreModelo == 'NeuralNetwork':
            Modelo = load_model(RutaModelos + '\\' + NombreModelo)
            PredictoresModelo = (Predictores - XMean) / XStdDev
        elif NombreModelo == 'SupportVectorRegression':
            Modelo = load(open(RutaModelos + '\\' + NombreModelo + '\\SVR.mdl', 'rb'))
            PredictoresModelo = (Predictores - XMean) / XStdDev
        elif NombreModelo == 'XGBoost':
            Modelo = load(open(RutaModelos + '\\' + NombreModelo + '\\XGBoost.mdl', 'rb'))
            PredictoresModelo = (Predictores - XMean) / XStdDev
        
        CantRowsTraining = ValidationYears[0] - 1979
        X_val = PredictoresModelo[CantRowsTraining:,:]
        y_val = Clase[CantRowsTraining:]    
    
        
        Modelos.append({
                        'Name':NombreModelo,
                        'BestMetric':BestMetric,
                        'Model':Modelo,
                        'YMean':YMean,
                        'YStdDev': YStdDev,
                        'Pred':Modelo.predict(X_val),
                        'Weight': None,   
                        'WeightedPred': None
                       })
        
    if len(Modelos) == 0:
        print('The models for group',Grupo,'month',Mes,'have all an ExpVar metric below 0')
        continue
    
    #Predictions come in different format, convert everything to list
    y_val = list(np.array(y_val)[:,0])
    for Modelo in Modelos:
        if Modelo['Name'] == 'NeuralNetwork':
            Modelo['Pred'] = list(Modelo['Pred'][:,0])
        elif Modelo['Name'] == 'SupportVectorRegression':
            Modelo['Pred'] = list(Modelo['Pred'])
        elif Modelo['Name'] == 'XGBoost':
            Modelo['Pred'] = list(Modelo['Pred'])
    
    #For NN and SVR, the predictions are Normalized, we have to reconstruct them
    for Modelo in Modelos:
        if Modelo['Name'] in ['NeuralNetwork','SupportVectorRegression']:
            Modelo['Pred'] = [(i * Modelo['YStdDev']) + Modelo['YMean'] for i in Modelo['Pred']]
    
    
    #Calculate and apply weights
    Pesos = softmax(Metrics)
    for Modelo, Peso in zip(Modelos, Pesos):
        Modelo['Weight'] =  Peso
        Modelo['WeightedPred'] = [i * Modelo['Weight'] for i in Modelo['Pred']] 
    
    #Calculate Final Pred
    FinalPred = []
    if len(Modelos) == 1:
        FinalPred = Modelos[0]['WeightedPred']
    if len(Modelos) == 2:
        for p1, p2 in zip(*[Modelo['WeightedPred'] for Modelo in Modelos]):
            FinalPred.append(np.sum([p1, p2]))
    if len(Modelos) == 3:
        for p1, p2, p3 in zip(*[Modelo['WeightedPred'] for Modelo in Modelos]):
            FinalPred.append(np.sum([p1, p2, p3]))
            
        
    
    MeanAbsoluteErr = round(mean_absolute_error(y_val,FinalPred),3)
    ExplainedVar = round(explained_variance_score(y_val,FinalPred),3)
    Plotter.PlotModelPerformance(y_val, FinalPred, MeanAbsoluteErr, ExplainedVar, ValidationYears, Quantiles, 'Ensemble', Grupo, Mes, RootModelos)         
      
    