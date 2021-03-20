import xlrd
from netCDF4 import Dataset
import pandas as pd
import os
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.gridspec as gridspec
import numpy as np
from scipy.stats.stats import pearsonr
import cv2
import scipy.misc as scp
import warnings
from datetime import date, timedelta
import seaborn as sns
warnings.simplefilter("ignore")

%matplotlib inline
RootFolderXLS           = 'C:\\Users\\Juani\\OneDrive\\Juani\\Maestria\\Tesis Clima\\Datos Lluvias\\Gran Chaco\\'
GroupsFile              = 'GC_GROUPMEDIANS.xlsx'
GroupedStationsFile     = 'GC_GROUPEDSTATIONS.xlsx'

RootFolderNC            = 'C:\\Users\\Juani\\OneDrive\\Juani\\Maestria\\Tesis Clima\\Archivos NC\\'
OutputFolderPredictors  = 'C:\\Users\\Juani\\OneDrive\\Juani\\Maestria\\Tesis Clima\\Predictores\\'

RootFolderVariableMasks = 'C:\\Users\\Juani\\OneDrive\\Juani\\Maestria\\Tesis Clima\\Scripts\\Mascaras\\'

MinAreaPoints           = 15   #How many points must an area have to consider it important
NCFilesOffset           = 372  #Position of the Time Dimension that belongs to Jan 1st 1979
NCFilesBaseDate         = date(1800,1,1)

GenerateMaps            = False
GenerateExcelFiles      = True

#Step 1 =======================================================================
#For each cluster I found, load the monthly median series
print('Reading Excel Files')
xls = xlrd.open_workbook(RootFolderXLS + GroupsFile, on_demand=True)
GroupNames = xls.sheet_names()
FileMedianSeries = []
for Group in GroupNames:
    print('\tCurrent Group:', Group)
    FileMedianSeries.append(pd.read_excel(RootFolderXLS + GroupsFile, sheet_name = Group))
FileStations = pd.read_excel(RootFolderXLS + GroupedStationsFile)
MinYear = FileMedianSeries[0].YEAR.min()
MaxYear = FileMedianSeries[0].YEAR.max()
YearsCount = MaxYear - MinYear + 1

NormalTest = 1.96 / np.sqrt(YearsCount - 2)
CorrelationThreshold    = [-NormalTest, NormalTest]

#Step 2 =======================================================================
#For each potential predictor (NC Files) calculate the correlation
def CreateDir(Path):
    if not os.path.exists(Path):
        print('Creating [' + Path + ']')
        os.mkdir(Path)

CreateDir(OutputFolderPredictors)
print('Reading NC Files')
NCFiles = os.listdir(RootFolderNC)
NCFiles.remove('lsm.nc')
print('Reading land surface mask')
LSMFile = Dataset(RootFolderNC + 'lsm.nc', "r", format="NETCDF4")
LSM = LSMFile.variables['msk'][:]
plt.imshow(LSM, cmap=plt.cm.Blues)
plt.pause(0.000000000001)
for Group in GroupNames:
    print('Generating Data for group:', Group)
    CreateDir(OutputFolderPredictors + Group)
    CreateDir(OutputFolderPredictors + Group + '\\Maps')
    CreateDir(OutputFolderPredictors + Group + '\\Predictors')
    CreateDir(OutputFolderPredictors + Group + '\\Masks')
    CreateDir(OutputFolderPredictors + Group + '\\CorrPlots')
    GroupIndex = GroupNames.index(Group)
    
    for NCFileName in NCFiles:
        print('\tCurrent File:', NCFileName)    
        NCFile = Dataset(RootFolderNC + NCFileName, "r", format="NETCDF4")
        print('\tReading Dimensions and Variable')
        Lat = NCFile.variables['lat'][:].tolist()
        Lat.reverse()
        Lon = NCFile.variables['lon'][:].tolist()
        Lon = [x - 180 for x in Lon]
        Time = NCFile.variables['time'][:].tolist()
        print('\tMIN TIME', NCFilesBaseDate + timedelta(int(np.ma.getdata(min(Time)/24))))
        print('\tMAX TIME', NCFilesBaseDate + timedelta(int(np.ma.getdata(max(Time)/24))))
        #Predict the variable name 
        NCFileDims = []
        NCFileKeys = []
        [NCFileDims.append(Key) for Key in NCFile.dimensions.keys()]
        [NCFileKeys.append(Key) for Key in NCFile.variables.keys() if Key not in NCFileDims]
        VarName = NCFileKeys[0]
        Var = NCFile.variables[VarName][:]
        
        for VariableMonth in range(1,13): 
            if VariableMonth <= 11:
                RainsMonth = VariableMonth + 1
            else:
                RainsMonth = 1
            print('\tCalculating Correlation Matrix for Month [' + str(VariableMonth) + '] of', VarName, 'and Month [' + str(RainsMonth) + '] of the group')
            RainsColumnName = 'M' + ('0' + str(RainsMonth))[-2:]
            PedictorMonthName = 'M' + ('0' + str(VariableMonth))[-2:]
            GroupSeries = FileMedianSeries[GroupIndex][RainsColumnName].tolist()[:YearsCount]
            CorrelationMatrix = np.zeros(LSM.shape, dtype=float)
            TimeIndices = [t for t in range(NCFilesOffset + VariableMonth - 1, NCFilesOffset + (YearsCount * 12), 12)]
            print('\tFROM TIME', NCFilesBaseDate + timedelta(int(np.ma.getdata(Time[TimeIndices[0]])/24)))
            print('\tTO TIME', NCFilesBaseDate + timedelta(int(np.ma.getdata(Time[TimeIndices[-1]])/24)))
            for lat in range(len(Lat)):
                for lon in range(len(Lon)):
                    if len(NCFileDims) == 3:                
                        VarSeries = Var[TimeIndices, lat, lon].data.tolist()
                    elif len(NCFileDims) == 4:
                        VarSeries = Var[TimeIndices, 0, lat, lon].tolist()
                    CorrelationMatrix[lat,lon] = pearsonr(VarSeries, GroupSeries)[0]
            #plt.imshow(CorrelationMatrix)
            #plt.pause(0.000000000001)
            
            print('\tCalculating high correlation areas')
            #Define a mask matrix with 1 where the corr is significant and 0 otherwise
            CorrelationMask = np.zeros(LSM.shape, dtype=np.uint8)
            for lat in range(len(Lat)):
                for lon in range(len(Lon)):
                    CorrelationMask[lat, lon] = (CorrelationMatrix[lat, lon] < CorrelationThreshold[0] or CorrelationMatrix[lat, lon] > CorrelationThreshold[1])
            #plt.imshow(CorrelationMask, cmap=plt.cm.Blues)
            #plt.pause(0.000000000001)
            
            
            print('\tLoading variable mask')
            
            VarMask = scp.imread(RootFolderVariableMasks + NCFileName.replace('nc','png'), mode='L')
            VarMask[VarMask == 255] = 100
            VarMask[VarMask == 0] = 255
            VarMask[VarMask == 100] = 0            
            
            fig = plt.figure()    
            fig.set_figheight(8)
            fig.set_figwidth(15)
            fig.suptitle('Pedictor [' + NCFileName.split('.')[0] + '] Month [' + PedictorMonthName + '] for Group [' + Group + '] Month [' + RainsColumnName + ']', fontsize=16)
                
            plt.subplot(221)
            EarthMask = np.ma.masked_where(LSM == 1, LSM)
            Ecuador = np.repeat(36,144)
            plt.imshow(EarthMask, cmap = plt.cm.Greys_r, interpolation='none', alpha=1)
            plt.plot(np.arange(144), Ecuador, 'r--') 
            plt.imshow(CorrelationMask, cmap = plt.cm.Blues, interpolation='nearest', alpha=0.8)
            plt.title('High correlation areas')
            plt.xticks([])
            plt.yticks([])
            
            plt.subplot(222)
            plt.imshow(EarthMask, cmap = plt.cm.Greys_r, interpolation='none', alpha=1)
            plt.plot(np.arange(144), Ecuador, 'r--') 
            plt.imshow(VarMask, cmap = plt.cm.Reds, interpolation='nearest', alpha=0.8)
            plt.title('Variable Mask')
            plt.xticks([])
            plt.yticks([])
            
            plt.subplot(223)
            plt.imshow(EarthMask, cmap = plt.cm.Greys_r, interpolation='none', alpha=1)
            plt.plot(np.arange(144), Ecuador, 'r--') 
            plt.imshow(CorrelationMask, cmap = plt.cm.Blues, interpolation='nearest', alpha=0.8)
            plt.imshow(VarMask, cmap = plt.cm.Reds, interpolation='nearest', alpha=0.3)
            plt.title('Layers overlapped')
            plt.xticks([])
            plt.yticks([])
            
            CroppedAreas = cv2.subtract(CorrelationMask,VarMask)
            
            plt.subplot(224)
            plt.imshow(EarthMask, cmap = plt.cm.Greys_r, interpolation='none', alpha=1)
            plt.plot(np.arange(144), Ecuador, 'r--') 
            plt.imshow(CroppedAreas, cmap = plt.cm.Blues, interpolation='nearest', alpha=0.8)
            plt.imshow(VarMask, cmap = plt.cm.Reds, interpolation='nearest', alpha=0.3)
            plt.title('Final areas')
            plt.xticks([])
            plt.yticks([])
            
            PlotFileName = NCFileName.replace('.nc','') + '_'+ PedictorMonthName + '_' + Group + '_' + RainsColumnName + '.png'
            plt.savefig(OutputFolderPredictors + Group + '\\Masks\\' + PlotFileName, format='png', bbox_inches='tight')
                

            plt.pause(0.000000000001)
            
            
            CorrelationMask = CroppedAreas
            
            #cv2 has a function to find the connectedComponents
            CorrelationAreasCount, CorrelationAreasLabels = cv2.connectedComponents(image = CorrelationMask, connectivity = 8)
            ImportantAreas = []
            for i in range(1, CorrelationAreasCount): #starts from 1 because 0 is the background
                CorrelationAreaIndices = np.argwhere(CorrelationAreasLabels==i)
                if len(CorrelationAreaIndices) < MinAreaPoints:
                    print('\t\tCorrelation Area [' + str(i) + '] has only', len(CorrelationAreaIndices), 'points. It will NOT be considered')
                    for Point in CorrelationAreaIndices.tolist():
                        CorrelationMask[Point[0], Point[1]] = 0
                else:
                    print('\t\tCorrelation Area [' + str(i) + '] has', len(CorrelationAreaIndices), 'points and It will used')
                    ImportantAreas.append(CorrelationAreaIndices)
            
            
            print('\tCalculating impotant areas\' contours')        
            (Contours, _) = cv2.findContours(image = CorrelationMask, mode = cv2.RETR_EXTERNAL, method = cv2.CHAIN_APPROX_SIMPLE)
            ContoursImage = np.zeros(CorrelationMask.shape, np.uint8)
            PlotAnnotations = []
            for Contour in Contours:
                AreaMeanCorrelation = 0
                for AreaPoint in Contour[:,0,:]:
                    AreaMeanCorrelation +=  CorrelationMatrix[AreaPoint[1], AreaPoint[0]]
                AreaMeanCorrelation = AreaMeanCorrelation / len(Contour[:,0,:])
                AreaX = Contour[:,0,:][0][0]
                AreaY = Contour[:,0,:][0][1]
                AreaCorrelationValue = round(AreaMeanCorrelation, 3) #X and Y are inverted because the matrix uses [Rows, Cols] instead of [X,Y]
                PlotAnnotations.append((AreaX, AreaY, AreaCorrelationValue))
                cv2.drawContours(ContoursImage, [Contour], -1, color=1, thickness=1)
                
            #plt.imshow(ContoursImage)
            #plt.pause(0.000000000001)
            if GenerateMaps:
                print('\tGenerating Correlation Map')
                EarthMask = np.ma.masked_where(LSM == 1, LSM)
                fig = plt.figure(figsize=(15,15))
                ax = plt.gca()
                im = ax.imshow(CorrelationMatrix, interpolation='none')
                ContoursImage = np.ma.masked_where(ContoursImage == 0, ContoursImage)
                ax.imshow(ContoursImage, cmap = plt.cm.autumn, interpolation='nearest')
                plt.title('Pedictor [' + NCFileName.split('.')[0] + '] Month [' + PedictorMonthName + '] for Group [' + Group + '] Month [' + RainsColumnName + ']')
                plt.ylabel('Latitud')
                plt.xlabel('Longitud')
                
                #fix ticks
                locs, labels = plt.xticks()
                locs = np.linspace(plt.xlim()[0], plt.xlim()[1], 37)
                labels = np.concatenate((np.arange(0,180,10), np.arange(-180,1,10)))
                dummy = plt.xticks(locs, labels, rotation = 45)
                    
                locs, labels = plt.yticks()
                locs = np.linspace(plt.ylim()[0], plt.ylim()[1], 19)
                labels = np.arange(-90,91,10)
                dummy = plt.yticks(locs, labels)
                
                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="5%", pad=0.05)
                plt.colorbar(im, cax)
                if VarName == 'skt':
                    alphaValue = 1
                else:
                    alphaValue = 0.5
                ax.imshow(EarthMask, cmap = plt.cm.gray, interpolation='none', alpha=alphaValue)
                
                for Annotation in PlotAnnotations:
                    bbox_props = dict(boxstyle="round,pad=0.5", fc="sandybrown", ec="k", lw=2)  
                    if Annotation[0] >= ContoursImage.shape[1] - 20:
                        AnnotationX = Annotation[0] - 20
                    elif Annotation[0] < 20:
                        AnnotationX = 2
                    else:
                        AnnotationX = Annotation[0] - 5
                    if Annotation[1] <= 5:
                        AnnotationY = Annotation[1] + 5
                    else:
                        AnnotationY = Annotation[1] - 5
                    ax.annotate('Correlation: ' + str(Annotation[2]),
                                xy=(Annotation[0], Annotation[1]),
                                xytext=(AnnotationX, AnnotationY),
                                bbox=bbox_props,
                                arrowprops=
                                    dict(facecolor='red', shrink=0.05),
                                    horizontalalignment='left',
                                    verticalalignment='center')
                PlotFileName = NCFileName.replace('.nc','') + '_'+ PedictorMonthName + '_' + Group + '_' + RainsColumnName + '.png'
                plt.savefig(OutputFolderPredictors + Group + '\\Maps\\' + PlotFileName, format='png', bbox_inches='tight')
                plt.pause(0.000000000001)
        
            if GenerateExcelFiles:
                print('\tCalculating Mean Series for each important area')    
                ExcelFileName = 'Predictors_' + Group + '_' + RainsColumnName + '.xlsx'
                ExcelPath = OutputFolderPredictors + Group + '\\Predictors\\' + ExcelFileName
                if os.path.exists(ExcelPath):
                    OutputPredictorsFile = pd.read_excel(ExcelPath)
                else:
                    OutputPredictorsFile = pd.DataFrame()
                AreaIndex = 0
                for ImportantArea in ImportantAreas:
                    AreaName = NCFileName.replace('.nc','') + '_' + PedictorMonthName + '_A' + ('0' + str(AreaIndex))[-2:] 
                    print('\t\t' + AreaName)    
                    AreaSeries = np.zeros((len(TimeIndices), len(ImportantArea)), dtype = float)
                    PointIndex = 0
                    for Point in ImportantArea:
                        ImportantAreaLat = Point[0]
                        ImportantAreaLon = Point[1]
                        if len(NCFileDims) == 3:                
                            VarSeries = Var[TimeIndices, ImportantAreaLat, ImportantAreaLon].data.tolist()
                        elif len(NCFileDims) == 4:
                            VarSeries = Var[TimeIndices, 0, ImportantAreaLat, ImportantAreaLon].tolist()
                        AreaSeries[:,PointIndex] = VarSeries
                        PointIndex = PointIndex + 1
                    AreaMeanSeries = np.mean(AreaSeries, axis = 1)
                    OutputPredictorsFile[AreaName] = AreaMeanSeries
                    AreaIndex = AreaIndex + 1
                print('\tWriting Excel File')
                OutputPredictorsFile.to_excel(ExcelPath, sheet_name = "Predictors", index=None)
        
        
                #Correlation plot of the predictors and the variables 
                CorrPlotFileName = 'CorrPlot_' + Group + '_' + RainsColumnName + '.png'
                CorrPlotPath = OutputFolderPredictors + Group + '\\CorrPlots\\' + CorrPlotFileName
                M = OutputPredictorsFile.copy()
                M['Rain'] = GroupSeries
                Corr = M.corr()
                plt.figure(figsize=(10,10), dpi= 80)
                CorrPlot = sns.heatmap(Corr, xticklabels=Corr.columns, yticklabels=Corr.columns, cmap='RdYlGn', center=0, annot=True)
                plt.title('Correlogram for ' + Group + '_' + RainsColumnName, fontsize=22)
                plt.xticks(fontsize=12)
                plt.yticks(fontsize=12)
                plt.tight_layout()
                fig = CorrPlot.get_figure()
                fig.savefig(CorrPlotPath)
                plt.pause(0.0000000001)    
                
        NCFile.close()