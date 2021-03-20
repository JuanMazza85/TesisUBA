# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 13:48:17 2019

@author: Juani
@Desc: Analisis de Datos
"""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import xlrd
from math import pi


RootFolder    = 'C:\\Users\\Juani\\OneDrive\\Juani\\Maestria\\Tesis Clima\\Datos Lluvias\\Gran Chaco\\'
RainFile      = 'GC_PRE.xlsx'
StationsFile  = 'GC_METADATA.xlsx'

MonthColumns  = ['M01', 'M02', 'M03', 'M04', 'M05', 'M06', 'M07', 'M08', 'M09', 'M10', 'M11', 'M12']
SeasonColumns = ['DEF','MAM', 'JJA', 'SON']
YearAccumColumn = ['A_YEAR']
ColumnsToUse  = YearAccumColumn #One of the previous vars

#Read File =================================================================================
xls = xlrd.open_workbook(RootFolder + RainFile, on_demand=True)
StationsIDs = xls.sheet_names()
File = []
for st in StationsIDs:
    File.append(pd.read_excel(RootFolder + RainFile, sheet_name = st))
StationsNames = pd.read_excel(RootFolder + StationsFile)

 
#Check for missing values
for i in range(0, len(StationsIDs)):
    CurrentStationID = int(StationsIDs[i])
    CurrentStationName = StationsNames[StationsNames.idOMM == CurrentStationID].NomEstacion.values[0]
    print('======',CurrentStationName,'======')
    print(File[i].isna().sum())

#fill the missing values with the median of the month
for i in range(0, len(StationsIDs)):
    for col in File[i].columns:
        MedianValue = np.nanmedian(File[i][[col]])
        File[i][col].fillna(value = MedianValue, inplace = True)
#==========================================================================================

#Plot Precipitacion series for each station
%matplotlib inline
for i in range(0, len(StationsIDs)):
    CurrentStationID = int(StationsIDs[i])
    CurrentStationName = StationsNames[StationsNames.idOMM == CurrentStationID].NomEstacion.values[0]
    PlotSize = (10, 10)
    fig = plt.figure(figsize = PlotSize)
    ax = fig.add_subplot(111)
    ax.set_title(CurrentStationName)
    for j in range(0, len(ColumnsToUse)):
        ax.plot(File[i].YEAR, File[i][ColumnsToUse[j]], label = ColumnsToUse[j])
    ax.legend()
    plt.show()
    



#PLot boxplots
%matplotlib inline
PlotSize = (15, 10)
for i in range(0, len(StationsIDs)): 
    PlotData = []
    CurrentStationID = int(StationsIDs[i])
    CurrentStationName = StationsNames[StationsNames.idOMM == CurrentStationID].NomEstacion.values[0]
    for j in range(0, len(ColumnsToUse)):
        PlotData.append(File[i][ColumnsToUse[j]].values.tolist())
    fig = plt.figure(figsize = PlotSize)
    ax = fig.add_subplot(111)
    ax.set_title(CurrentStationName)
    ax.yaxis.grid(True)
    boxplots = ax.boxplot(PlotData, patch_artist=True, showcaps= True, sym="bo", labels = ColumnsToUse)
    for patch in boxplots['boxes']:
        patch.set_facecolor('lightblue')
    plt.show()
    
    



    
#Histograms per month per station
PlotSize = (20, 10)
for i in range(0, len(StationsIDs)):  
    CurrentStationID = int(StationsIDs[i])
    CurrentStationName = StationsNames[StationsNames.idOMM == CurrentStationID].NomEstacion.values[0]
    fig = plt.figure(figsize = PlotSize)
    ax = fig.add_subplot(111)
    ax.set_title(CurrentStationName)
    File[i][ColumnsToUse].hist(bins = 10, ax = ax)
    
    




