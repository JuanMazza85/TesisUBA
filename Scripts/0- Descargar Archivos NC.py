import shutil
import urllib.request as request
from contextlib import closing
import os
from netCDF4 import Dataset
import numpy as np
from datetime import date, timedelta



RootFolder  = 'C:\\Users\\Juani\\OneDrive\\Juani\\Maestria\\Tesis Clima\\'
TempFolder  = 'Temp\\'
NCFolder    = 'Archivos NC\\'
                   
PredictorsURLs = ['ftp://ftp.cdc.noaa.gov/Datasets/ncep.reanalysis.derived/pressure/hgt.mon.mean.nc',
                  'ftp://ftp.cdc.noaa.gov/Datasets/ncep.reanalysis.derived/pressure/uwnd.mon.mean.nc',
                  'ftp://ftp.cdc.noaa.gov/Datasets/ncep.reanalysis.derived/pressure/vwnd.mon.mean.nc',
                  'ftp://ftp.cdc.noaa.gov/Datasets/ncep.reanalysis.derived/surface/pr_wtr.mon.mean.nc',
                  'ftp://ftp.cdc.noaa.gov/Datasets/noaa.oisst.v2/sst.mnmean.nc']


DownloadFiles  = False



#Step 1 ======================================================================= 
#Prepare tmp folder
if DownloadFiles:
    print('Creating Temp Folder')
    os.chdir(RootFolder)
    if os.path.exists(RootFolder + TempFolder):
      os.rmdir(RootFolder + TempFolder)

    os.mkdir(RootFolder + TempFolder)

os.chdir(RootFolder + TempFolder)



#Step 2 ======================================================================= 
#Prepare tmp folder
if DownloadFiles:
    print('Downloading Files')
    for PredictorURL in PredictorsURLs:
        AuxFile = PredictorURL.split('/')[-1]
        print('\t' + AuxFile)
        with closing(request.urlopen(PredictorURL)) as r:
            with open(AuxFile, 'wb') as f:
                shutil.copyfileobj(r, f)
                



#Step 2 ======================================================================= 
def CreateNCFile(FilePath, FileName, NCFile, level):    
    
    NCFileDims = []
    NCFileKeys = []
    [NCFileDims.append(Key) for Key in NCFile.dimensions.keys()]
    [NCFileKeys.append(Key) for Key in NCFile.variables.keys() if Key not in NCFileDims]
    VarName = NCFileKeys[0]
    
    if os.path.exists(FilePath+ VarName + level + '.nc'):
        os.remove(FilePath+ VarName + level + '.nc')
    
    root_grp = Dataset(FilePath + FileName + '.nc', 'w', format='NETCDF4')
     
    # dimensions
    for key in NCFile.dimensions.keys():
        if key != 'level':
            root_grp.createDimension(key, NCFile.dimensions[key].size)
      
    # variables
    time = root_grp.createVariable('time', 'f8', ('time',))
    lat = root_grp.createVariable('lat', 'f4', ('lat',))
    lon = root_grp.createVariable('lon', 'f4', ('lon',))
    field = root_grp.createVariable(VarName, 'f8', ('time', 'lat', 'lon',))
    
    
    # data
    lat[:] = NCFile.variables['lat'][:]
    lon[:] = NCFile.variables['lon'][:]
    time[:] = NCFile.variables['time'][:]
    
    BaseDate = date(1800,1,1)
    print('===',FileName,'===')
    print('\tLATs',len(lat))
    print('\tLONs',len(lon))
    print('\tTIMEs',len(time))
    print('\tMIN TIME', BaseDate + timedelta(min(time)/24))
    print('\tMAX TIME', BaseDate + timedelta(max(time)/24))
        
    #Level Index
    if 'level' in NCFile.dimensions.keys() :
        Levels = np.array(NCFile.variables['level'][:])
        LevelIndex = np.where(Levels == float(level))[0][0]
    
        field[:,:,:] = NCFile.variables[VarName][:,LevelIndex,:,:]
    else:
        field[:,:,:] = NCFile.variables[VarName][:,:,:]
    
    root_grp.close()            
    
    
    
def CreateNCFileSST(FilePath, FileName, NCFile, level):    
    NCFileDims = []
    NCFileKeys = []
    [NCFileDims.append(Key) for Key in NCFile.dimensions.keys()]
    [NCFileKeys.append(Key) for Key in NCFile.variables.keys() if Key not in NCFileDims]
    VarName = NCFileKeys[0]
    
    if os.path.exists(FilePath+ VarName + level + '.nc'):
        os.remove(FilePath+ VarName + level + '.nc')
    
    root_grp = Dataset(FilePath + FileName + '.nc', 'w', format='NETCDF4')
     
    # dimensions
    for key in NCFile.dimensions.keys():
        if key != 'nbnds':
            root_grp.createDimension(key, NCFile.dimensions[key].size)
      
    # variables
    time = root_grp.createVariable('time', 'f8', ('time',))
    lat = root_grp.createVariable('lat', 'f4', ('lat',))
    lon = root_grp.createVariable('lon', 'f4', ('lon',))
    field = root_grp.createVariable(VarName, 'f8', ('time', 'lat', 'lon',))
    
    
    # data
    lat[:] = NCFile.variables['lat'][:]
    lon[:] = NCFile.variables['lon'][:]
    time[:] = NCFile.variables['time'][:]
    
    BaseDate = date(1800,1,1)
    print('===',FileName,'===')
    print('\tLATs',len(lat))
    print('\tLONs',len(lon))
    print('\tTIMEs',len(time))
    #Position 372 is Jan 1st 1979
    print('\tMIN TIME', BaseDate + timedelta(int(np.ma.getdata(min(time)))))
    print('\tMAX TIME', BaseDate + timedelta(int(np.ma.getdata(max(time)))))
        
    #Level Index
    if 'level' in NCFile.dimensions.keys() :
        Levels = np.array(NCFile.variables['level'][:])
        LevelIndex = np.where(Levels == float(level))[0][0]
    
        field[:,:,:] = NCFile.variables[VarName][:,LevelIndex,:,:]
    else:
        field[:,:,:] = NCFile.variables[VarName][:,:,:]
    
    root_grp.close()
            
    
    
    
    
    
#Prepare tmp folder
print('Preparing Files')
for PredictorURL in PredictorsURLs:
    AuxFile = PredictorURL.split('/')[-1]
    print('\t' + AuxFile)
    NCFile = Dataset(AuxFile)        
    if AuxFile.startswith("hgt"):
        CreateNCFile(RootFolder + NCFolder, "hgt200", NCFile, "200")
        CreateNCFile(RootFolder + NCFolder, "hgt500", NCFile, "500")
        CreateNCFile(RootFolder + NCFolder, "hgt1000", NCFile,  "1000")
    elif AuxFile.startswith("uwnd"):
        CreateNCFile(RootFolder + NCFolder, "u850", NCFile, "850")
    elif AuxFile.startswith("vwnd"):
        CreateNCFile(RootFolder + NCFolder, "v850", NCFile, "850")
    elif AuxFile.startswith("pr_wtr"):
        CreateNCFile(RootFolder + NCFolder, "tcw", NCFile, "")
    elif AuxFile.startswith("sst"):
        CreateNCFileSST(RootFolder + NCFolder, "sst", NCFile, "nbnds")