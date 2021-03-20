import numpy as np
import pandas as pd
import os
import xlsxwriter

RootFolder = 'C:\\Users\\Juani\\OneDrive\\Juani\\Maestria\\Tesis Clima\\Datos Lluvias\\Gran Chaco\\'
Lluvias2019  = 'Pre_2019.xlsx'
LluviasGC  = 'GC_PRE.xlsx'


os.chdir (RootFolder)


DF = pd.read_excel(Lluvias2019)
DF['MES'] = [mes.month for mes in DF.Fecha]
DFG = DF.groupby(by=['idOMM','MES']).sum()


GC = pd.read_excel(LluviasGC)

EstacionConDatosFaltantes = []
for IDEstacion, Mes in DFG.index:
    if IDEstacion in EstacionConDatosFaltantes:
        continue
    
    if IDEstacion in [87691,87741]: #Estaciones con datos faltantes
        EstacionConDatosFaltantes.append(IDEstacion)
        Row = [2019,'','','','','','','','','','','','','','','','','']
        print (IDEstacion, '>>>',str(Row).replace(' ','').replace('[','').replace(']',''))
        continue
    
    if Mes == 1:
        Row = []
        Row.append(2019)
                
    Row.append(round(DFG.loc[IDEstacion, Mes].Prcp,2))
    if Mes == 12:
        DEF = Row[12] + Row[1] + Row[2]
        MAM = Row[3] + Row[4] + Row[5]
        JJA = Row[6] + Row[7] + Row[8]
        SON = Row[9] + Row[10] + Row[11]
        ANO = DEF + MAM + JJA + SON
        Row.append(round(DEF,2))
        Row.append(round(MAM,2))
        Row.append(round(JJA,2))
        Row.append(round(SON,2))
        Row.append(round(ANO,2))
        
           
                    


        print (IDEstacion, '>>>', str(Row).replace(' ','').replace('[','').replace(']',''))
            

    