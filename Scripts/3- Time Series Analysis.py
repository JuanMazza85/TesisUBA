# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 13:48:17 2019

@author: Juani
@Desc: Analisis de Series de Tiempo
"""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import xlrd
import statsmodels.api as sm


RootFolder    = 'C:\\Users\\Juani\\OneDrive\\Juani\\Maestria\\Tesis Clima\\Datos Lluvias\\Gran Chaco\\' #'C:\\Users\\Juan.Mazza\\Desktop\\Gran Chaco\\'
RainFile      = 'GC_PRE.xlsx'
StationsFile  = 'GC_METADATA.xlsx'

MonthColumns  = ['M01', 'M02', 'M03', 'M04', 'M05', 'M06', 'M07', 'M08', 'M09', 'M10', 'M11', 'M12']
SeasonColumns = ['DEF','MAM', 'JJA', 'SON']
YearAccumColumn = ['A_YEAR']
ColumnsToUse  = MonthColumns #One of the previous vars

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
    
#set indeces for TS analysis
for i in range(0, len(StationsIDs)):
    date_rng = pd.date_range(start='1/1/' + str(File[i].YEAR.min()), end='1/01/' + str(File[i].YEAR.max()+ 1) , freq='Y')
    File[i]['YEAR'] = date_rng
    File[i]['YEAR'] = pd.to_datetime(date_rng)
    File[i] = File[i].set_index('YEAR')
    File[i][ColumnsToUse] = File[i][ColumnsToUse].resample('Y').mean()
    
    
#==========================================================================================
    
#Plot Precipitacion series for each season
%matplotlib inline
for i in range(0, len(StationsIDs)):
    CurrentStationID = int(StationsIDs[i])
    CurrentStationName = StationsNames[StationsNames.idOMM == CurrentStationID].NomEstacion.values[0]
    PlotSize = (5, 5)
    fig = plt.figure(figsize = PlotSize)
    ax = fig.add_subplot(111)
    ax.set_title(CurrentStationName)
    for j in range(0, len(ColumnsToUse)):
        ax.plot(File[i].index, File[i][ColumnsToUse[j]], label = ColumnsToUse[j])
    ax.legend()
    plt.show()


#==========================================================================================
    
#Plot trends using rolling average or each season
%matplotlib inline
MA = 5
Colors = ['blue','blue','blue','red','red','red','green','green','green','orange','orange','orange']
for i in range(len(StationsIDs)):
    CurrentStationID = int(StationsIDs[i])
    CurrentStationName = StationsNames[StationsNames.idOMM == CurrentStationID].NomEstacion.values[0]
    PlotSize = (5, 5)
    fig = plt.figure(figsize = PlotSize)
    ax = fig.add_subplot(111)
    ax.set_title(CurrentStationName)
    for j in range(0, len(ColumnsToUse)):
        ax.plot(File[i].index[-40:], File[i][ColumnsToUse[j]][-40:].rolling(MA).mean(), label = ColumnsToUse[j], color = Colors[j])
    ax.legend()
    plt.show()
    
#==========================================================================================
    
#Seasonal patterns using first order differentiation for each station, there are peaks every couple of years
%matplotlib inline
for i in range(0, len(StationsIDs)):
    CurrentStationID = int(StationsIDs[i])
    CurrentStationName = StationsNames[StationsNames.idOMM == CurrentStationID].NomEstacion.values[0]
    PlotSize = (5, 5)
    fig = plt.figure(figsize = PlotSize)
    ax = fig.add_subplot(111)
    ax.set_title(CurrentStationName)
    for j in range(0, len(ColumnsToUse)):
        ax.plot(File[i].index, File[i][ColumnsToUse[j]].diff(), label = ColumnsToUse[j])
    ax.legend()
    plt.show()
    
    

#==========================================================================================
    
#Autocorrelation, the idea is that if there is periodicty, the series will correlate high with itself every X years
%matplotlib inline
for i in range(0, len(StationsIDs)):
    CurrentStationID = int(StationsIDs[i])
    CurrentStationName = StationsNames[StationsNames.idOMM == CurrentStationID].NomEstacion.values[0]
    PlotSize = (5, 5)
    fig = plt.figure(figsize = PlotSize)
    ax = fig.add_subplot(111)
    ax.set_title(CurrentStationName)
    for j in range(0, len(ColumnsToUse)):
        pd.plotting.autocorrelation_plot(File[i][ColumnsToUse[j]], ax=ax)
    plt.show()