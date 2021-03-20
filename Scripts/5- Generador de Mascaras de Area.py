import matplotlib.pyplot as plt
import numpy as np
from netCDF4 import Dataset


RootFolder           = 'C:\\Users\\Juani\\OneDrive\\Juani\\Maestria\\Tesis Clima\\Scripts\\Mascaras\\'
MasksFile            = 'mascaras.csv'

RootFolderNC            = 'C:\\Users\\Juani\\OneDrive\\Juani\\Maestria\\Tesis Clima\\Archivos NC\\'


print('Reading land surface mask')
LSMFile = Dataset(RootFolderNC + 'lsm.nc', "r", format="NETCDF4")
LSM = LSMFile.variables['msk'][:]

print('Reading masks data')
with open(RootFolder + MasksFile) as file:
    FileData = file.readlines()
    
for Line in FileData:
    Line = Line.replace('\n','')
        
    Mask = np.zeros((73, 144), dtype=int)
    VarName = Line.split(':')[0]
    VarBoxes = Line.split(':')[1].split('|')
    
    for VarBox in VarBoxes:
        CurrentBox = VarBox.split(',')
        CurrentBox = [int(int(x) / 2.5) for x in CurrentBox]
        LatSup = 36 + CurrentBox[0] * -1 
        LatInf = 36 - CurrentBox[1]
        if CurrentBox[2] > -72 and CurrentBox[2] < 72 :
            LonIzq = CurrentBox[2] + 72 * 2
        else:
            LonIzq = CurrentBox[2] + 72
        if CurrentBox[3] > -72 and CurrentBox[3] < 72:
            LonDer = CurrentBox[3] + 72 * 2
        else:
            LonDer = CurrentBox[3] + 72
        if LonDer > 144: 
            LonDer = LonDer - 144
            
        if LatSup > LatInf:
            LatSup, LatInf = LatInf, LatSup
        if LonIzq > LonDer:
            LonIzq, LonDer = LonDer, LonIzq
        Mask[LatSup:LatInf + 1,LonIzq:LonDer + 1] = 1
    
    if VarName == 'sst': #For SST I also need to overlap the continents as part of the mask
      Mask = Mask - (1 - LSM)
      for i in range(Mask.shape[0]):
          for j in range(Mask.shape[1]):
              if Mask[i,j] < 0:
                  Mask[i,j] = 0
      
    plt.imsave(RootFolder + VarName + '.png', Mask, cmap=plt.cm.gray)
    
    EarthMask = np.ma.masked_where(LSM == 1, LSM)
    fig = plt.figure(figsize=(15,15))
    ax = plt.gca()
    ax.imshow(EarthMask, cmap = plt.cm.Greys_r, interpolation='none', alpha=1)
    Ecuador = np.repeat(36,144)
    plt.plot(np.arange(144), Ecuador, 'r--') 
    ax.imshow(Mask, cmap = plt.cm.Blues, interpolation='nearest', alpha=0.8)
    plt.title(VarName + ': ' + ' '.join(['[' + x + ']' for x in VarBoxes]))
    
    locs, labels = plt.xticks()
    locs = np.linspace(plt.xlim()[0], plt.xlim()[1], 37)
    labels = np.concatenate((np.arange(0,180,10), np.arange(-180,1,10)))
    dummy = plt.xticks(locs, labels, rotation = 45)
        
    locs, labels = plt.yticks()
    locs = np.linspace(plt.ylim()[0], plt.ylim()[1], 19)
    labels = np.arange(-90,91,10)
    dummy = plt.yticks(locs, labels)

    fig.savefig(RootFolder + '\\Mapas Mascaras\\' + VarName + '.jpg')
    plt.pause(0.000000000001)
    
    