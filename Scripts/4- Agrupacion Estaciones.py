# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 13:48:17 2019

@author: Juani
@Desc: Agrupacion de Estaciones
"""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import xlrd
from math import pi
import os

os.chdir('C:\\Users\\Juani\\OneDrive\\Juani\\Maestria\\Tesis Clima\\Scripts')

%matplotlib inline

RootFolder          = 'C:\\Users\\Juani\\OneDrive\\Juani\\Maestria\\Tesis Clima\\Datos Lluvias\\Gran Chaco\\'
RainFile            = 'GC_PRE.xlsx'
StationsFile        = 'GC_METADATA.xlsx'
GroupsFile          = 'GC_GROUPMEDIANS.xlsx'
GroupedStationsFile = 'GC_GROUPEDSTATIONS.xlsx'

MonthColumns  = ['M01', 'M02', 'M03', 'M04', 'M05', 'M06', 'M07', 'M08', 'M09', 'M10', 'M11', 'M12']
SeasonColumns = ['DEF','MAM', 'JJA', 'SON']
ColumnsToUse  = MonthColumns #One of the previous vars

#Read File
print('Reading Excel Files')
xls = xlrd.open_workbook(RootFolder + RainFile, on_demand=True)
StationsIDs = xls.sheet_names()
File = []
for st in StationsIDs:
    print('\tCurrent Station ID:', st)
    File.append(pd.read_excel(RootFolder + RainFile, sheet_name = st))
StationsNames = pd.read_excel(RootFolder + StationsFile)



#Keep only the last N years
KeepLastYears = 40
YearThreshold = File[0].YEAR.max() - KeepLastYears
for i in range(len(File)):
    File[i] = File[i][File[i].YEAR >= YearThreshold]



print('Showing how many NAs there are')
for i in range(0, len(StationsIDs)):
    count = 0 
    for col in File[i].columns:
        count = count + File[i][col].isna().sum()
    print('File for',StationsNames.loc[i,'NomEstacion'],':', count, '(', round(count/1062*100,3),'%)')



print('Filling NAs with the median for the Month-Station')
#fill the missing values with the last valid observation of the month
for i in range(0, len(StationsIDs)):
    for col in File[i].columns:
        MedianValue = np.nanmedian(File[i][[col]])
        File[i][col].fillna(value = MedianValue, inplace = True)

 


#==============================================================================
#For each station we have a 58x12 matrix, we will compress that to 1 row per station with the median of each month, then I will use  this [stations x 12] matrix to train a SOM
print('Preparing Data for the Kohonen Network')
MediansPerStation = []
for i in range(0, len(StationsIDs)):  
    Medians = File[i][ColumnsToUse].median()
    StationID = pd.Series([StationsIDs[i]], dtype='float')
    DataRow = StationID
    DataRow = DataRow.append(Medians)
    MediansPerStation.append(DataRow)
MediansPerStation = np.matrix(MediansPerStation)


#Train the SOM Model
print('Training the Kohonen Network')
import SOM
#Train SOM
NeuronRows = 2 #lattice rows
NeuronCols = 2 #lattice cols
alfa = 0.8 #Learning rage
initialNeigborhood = 10 #Initial Neighborhood
funNeighborhood = 1 #Neighborhood function (1= linea, 2 gaussian)
sigma = 0.3 #Bell size, for gaussian only
NeighborhoodRadioReduction = 100 #Radio is reduced after this many iterations
draw = True
(W) = SOM.trainSOM(MediansPerStation[:,1:], NeuronRows, NeuronCols, alfa, initialNeigborhood, funNeighborhood, sigma, NeighborhoodRadioReduction, draw)
Neurons = W.shape[0]



    
def SpiderPlot(NeuronRows,NeuronCols,row, title, color, df):
     
    # number of variable
    categories=list(df)
    N = len(categories)
     
    # What will be the angle of each axis in the plot? (we divide the plot / number of variable)
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]
     
    # Initialise the spider plot
    ax = plt.subplot(NeuronRows,NeuronCols,row+1, polar=True, )
     
    # If you want the first axis to be on top:
    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)
     
    # Draw one axe per variable + add labels labels yet
    plt.xticks(angles[:-1], categories, color='grey', size=8)
     
    # Draw ylabels
    ax.set_rlabel_position(0)
     
    # Ind1
    values=df.loc[row].values.flatten().tolist()
    values += values[:1]
    ax.plot(angles, values, color=color, linewidth=2, linestyle='solid')
    ax.fill(angles, values, color=color, alpha=0.4)
     
    # Add a title
    plt.title(title, size=11, color=color, y=1.1)
    


#Plot the vars distribution across each neuron
print('Plotting Neurons')
plt.figure(figsize=(15,15)) 
# Create a color palette:
my_palette = plt.cm.get_cmap("tab10", Neurons) 
# Loop to plot
for row in range(0, Neurons):
    SpiderPlot(NeuronRows,NeuronCols,row=row, title='Neuron '+ str(row), color=my_palette(row), df=pd.DataFrame(W, columns = ColumnsToUse))
plt.pause(0.000000000001) 

#show UMatrix
print('Plotting UMatrix')
umat = SOM.umatrix(W, NeuronRows, NeuronCols)
plt.figure(figsize=(5,5)) 
plt.imshow(umat, cmap=plt.cm.get_cmap("gray", Neurons))
plt.colorbar()
plt.show()
plt.pause(0.000000000001) 


#Map stations to neurons
print('Mapping Stations to Neurons')
ClusteredMediansPerStation = pd.DataFrame(MediansPerStation, columns=['idOMM'] + ColumnsToUse)
groups = []
for i in range(0, len(MediansPerStation)):    
    distancias = -np.sqrt(np.sum(np.power(W-MediansPerStation[i,1:14],2),1))
    ganadora = np.argmax(distancias)
    groups.append(ganadora)

ClusteredMediansPerStation['Group'] = groups
ClusteredMediansPerStation = ClusteredMediansPerStation.sort_values(by=['Group'])


#Check how many stations per group we have
ClusteredMediansPerStation.groupby(['Group']).size()


#Show which station belongs to each group
print('Saving Grouped Stations File ['+GroupedStationsFile+']')
ClusteredMediansPerStation = pd.merge(ClusteredMediansPerStation,  StationsNames[['idOMM','NomEstacion', 'Latitud', 'Longitud', 'Elevacion']], how='inner', on=['idOMM'])
print(ClusteredMediansPerStation[['idOMM', 'NomEstacion','Group']])
ClusteredMediansPerStation[['idOMM', 'NomEstacion','Group']].to_excel(RootFolder + GroupedStationsFile, sheet_name = "Groups", index=None )


#Plot Stations colored by the group they belong to
%matplotlib qt
print('Plotting Stations colored by the group they belong to')
plt.scatter(ClusteredMediansPerStation.Longitud, ClusteredMediansPerStation.Latitud,c=ClusteredMediansPerStation.Group, s=ClusteredMediansPerStation.Elevacion, cmap=plt.cm.get_cmap("tab10", Neurons))
for i in range(len(ClusteredMediansPerStation)):
    x = ClusteredMediansPerStation.Longitud[i]
    y = ClusteredMediansPerStation.Latitud[i]
    if ClusteredMediansPerStation.NomEstacion[i] in ['Chilecito Aero', 'Resistencia Aero','Ceres Aero', 'Cordoba Aero']:
        y -=  .3
    plt.text(x, y, ClusteredMediansPerStation.NomEstacion[i])
plt.ylabel('Latitud')
plt.xlabel('Longitud')
plt.pause(0.000000000001) 


print('Calculating Mean Series for each month and each group')
#Now that we have the groups. Let's calculate the mean series for each month for each group
ExcelWriter = pd.ExcelWriter(RootFolder + GroupsFile)
for group in set(ClusteredMediansPerStation.Group):
    M01, M02, M03, M04, M05, M06, M07, M08, M09, M10, M11, M12 = [], [], [], [], [], [], [], [], [], [], [], []
    print ('\tProcessing Group', group)
    for stationID in ClusteredMediansPerStation[ClusteredMediansPerStation.Group == group].idOMM.tolist():
        #with the staion ID (e.g 87022) I have to find the index in the array
        stationID = str(int(stationID))
        stationIndex = StationsIDs.index(stationID)
        print('\t\tStation ID', stationID, 'Station Index', stationIndex)        
        DF = File[stationIndex][File[stationIndex].YEAR >= 1979]
        M01.append(DF.M01.tolist())
        M02.append(DF.M02.tolist())
        M03.append(DF.M03.tolist())
        M04.append(DF.M04.tolist())
        M05.append(DF.M05.tolist())
        M06.append(DF.M06.tolist())
        M07.append(DF.M07.tolist())
        M08.append(DF.M08.tolist())
        M09.append(DF.M09.tolist())
        M10.append(DF.M10.tolist())
        M11.append(DF.M11.tolist())
        M12.append(DF.M12.tolist())
    
    #Once the monthly lists are ready, compute the median
    M01 = pd.DataFrame(np.median(np.matrix(M01).T, axis = 1))
    M02 = pd.DataFrame(np.median(np.matrix(M02).T, axis = 1))
    M03 = pd.DataFrame(np.median(np.matrix(M03).T, axis = 1))
    M04 = pd.DataFrame(np.median(np.matrix(M04).T, axis = 1))
    M05 = pd.DataFrame(np.median(np.matrix(M05).T, axis = 1))
    M06 = pd.DataFrame(np.median(np.matrix(M06).T, axis = 1))
    M07 = pd.DataFrame(np.median(np.matrix(M07).T, axis = 1))
    M08 = pd.DataFrame(np.median(np.matrix(M08).T, axis = 1))
    M09 = pd.DataFrame(np.median(np.matrix(M09).T, axis = 1))
    M10 = pd.DataFrame(np.median(np.matrix(M10).T, axis = 1))
    M11 = pd.DataFrame(np.median(np.matrix(M11).T, axis = 1))
    M12 = pd.DataFrame(np.median(np.matrix(M12).T, axis = 1))
    
    print('Saving group medians to [' + GroupsFile + '][G' + str(group) + ']')        
            
    GroupDF = pd.concat([M01, M02, M03, M04, M05, M06, M07, M08, M09, M10, M11, M12], axis = 1)
    GroupDF.columns = columns = ['M01', 'M02', 'M03', 'M04', 'M05', 'M06', 'M07', 'M08', 'M09', 'M10', 'M11', 'M12']
    GroupDF.index = range(1979,DF.YEAR.max() + 1)
    
    GroupDF.to_excel(ExcelWriter, sheet_name = "G"+str(group), index_label='YEAR' )

ExcelWriter.close()