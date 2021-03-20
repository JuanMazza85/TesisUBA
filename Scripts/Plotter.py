from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, roc_curve, auc    
import matplotlib.pyplot as plt   
import numpy as np 
from sklearn.preprocessing import label_binarize
import pandas as pd

IntervalThresholdmm = 5

def AccomodatePercentiles(Percentiles):
    #For months where data is heavily collapsed around a single value, intervals could have range 0.
    #Intervals with less than IntervalThreshold mm are merged
    Q = Percentiles.copy()
    K = Percentiles.copy()
    if Q[0] < IntervalThresholdmm:
        del(Q[0])
        del(K[0])
    j=0
    for i in range(len(Q)-1):
        if Q[i+1] - Q[i] < IntervalThresholdmm:
            del K[j]
        else:
            j=j+1            
    if len(K) == 2:
        if K[0] < IntervalThresholdmm and K[1] - K[0] > IntervalThresholdmm:
            Offset = K[1] - K[0] - 5
            K[0] = K[0] + Offset
    elif len(K) == 1 and K[0] < IntervalThresholdmm:
        K[0] = IntervalThresholdmm
    return(np.round(K,2))
    
def NameClass(NClasses, label):
    if NClasses == 2:
        if label == 0:
            return ('Normal')
        elif label == 1:
            return ('Supranormal')
    elif NClasses == 3:
        if label == 0:
            return ('Subnormal')
        elif label == 1:
            return ('Normal')
        elif label == 2:
            return ('Supranormal')
    elif NClasses == 4:
        if label == 0:
            return ('Subnormal')
        elif label == 1:
            return ('Normal')
        elif label == 2:
            return ('Supranormal')
        elif label == 3:
            return ('Muy supranormal')
    elif NClasses == 5:
        if label == 0:
            return ('Muy subnormal')
        elif label == 1:
            return ('Subnormal')
        elif label == 2:
            return ('Normal')
        elif label == 3:
            return ('Supranormal')
        elif label == 4:
            return ('Muy supranormal')
    
def DefineRange(Percentiles, NClasses, label):
    if label == 0:
        return("[0 - " + str(round(Percentiles[label],2)) + "]")
    elif label == NClasses - 1:
        return("[" + str(round(Percentiles[label - 1],2)) +  " +]")
    else:
        return("[" + str(round(Percentiles[label - 1],2)) +  " - " + str(round(Percentiles[label],2)) + "]")


def PlotModelPerformance(y_val, y_pred, MeanAbsoluteErr, ExplainedVar, ValidationYears, Percentiles, NombreModelo, Grupo, Mes, PathModelOutput):
        #Prepare Data
        
        Percentiles = AccomodatePercentiles(list(Percentiles))   
        NClasses = len(Percentiles) + 1             
                
        TargetYears = np.linspace(ValidationYears[0], ValidationYears[1], ValidationYears[1] - ValidationYears[0] + 1, dtype=int)
        ClaseValDiscretizada = np.digitize(y_val, Percentiles) 
        ClasePredDiscretizada = np.digitize(y_pred, Percentiles)
        classNames = []
        classRange = []
        classLabels = []
        if 0 in ClaseValDiscretizada or 0 in ClasePredDiscretizada: 
            classNames.append(NameClass(NClasses,0))
            classLabels.append(0)
            classRange.append(DefineRange(Percentiles, NClasses, 0))
        if 1 in ClaseValDiscretizada or 1 in ClasePredDiscretizada: 
            classNames.append(NameClass(NClasses,1))
            classLabels.append(1)
            classRange.append(DefineRange(Percentiles, NClasses, 1))
        if 2 in ClaseValDiscretizada or 2 in ClasePredDiscretizada: 
            classNames.append(NameClass(NClasses,2))
            classLabels.append(2)
            classRange.append(DefineRange(Percentiles, NClasses, 2))
        if 3 in ClaseValDiscretizada or 3 in ClasePredDiscretizada: 
            classNames.append(NameClass(NClasses,3))
            classLabels.append(3)
            classRange.append(DefineRange(Percentiles, NClasses, 3))
        if 4 in ClaseValDiscretizada or 4 in ClasePredDiscretizada: 
            classNames.append(NameClass(NClasses,4))
            classLabels.append(4)
            classRange.append(DefineRange(Percentiles, NClasses, 4))
        cm  = confusion_matrix(ClaseValDiscretizada, ClasePredDiscretizada)
        PrecRecallF1 = precision_recall_fscore_support(ClaseValDiscretizada, ClasePredDiscretizada, average=None, labels=classLabels, warn_for=())
        
        
       
        
        #Plot #1 - Predicted vs Actual
        plt.figure(figsize=(30,5))
        plt.subplot(151)    
        plt.plot(y_pred, y_val, 'bo')
        [plt.text(y_pred[i], y_val[i], TargetYears[i]) for i in range(len(TargetYears))]
        plt.plot(y_val, y_val, 'go')
        plt.plot([np.min(y_val), np.max(y_val)], [np.min(y_val), np.max(y_val)], 'r--')
        plt.xlabel('Predicted value')
        plt.ylabel('Actual value')
        plt.suptitle(NombreModelo + '_' + Grupo + '_M' + Mes + '_Y' + str(ValidationYears[0]) + '-' + str(ValidationYears[1]), fontsize = 'xx-large', weight = 'heavy')
        
        
        
        #Plot #2 - Confusion Matrix
        plt.subplot(152)    
        CMPlot = np.zeros(cm.shape, dtype = int)
        for i in range(CMPlot.shape[0]):
            for j in range(CMPlot.shape[1]):
                if np.mod(i+j, 2) == 0: 
                    CMPlot[i,j] = 2
                else:
                    CMPlot[i,j] = 0
        plt.imshow(np.rot90(CMPlot), interpolation='nearest', cmap=plt.cm.Pastel2)
        plt.ylabel('Actual Class')
        plt.xlabel('Predicted Class')
        tick_marks = np.arange(len(classNames))
        plt.xticks(tick_marks, classNames, rotation=25)
        YLabels = []
        for i in range(len(classNames)):
            YLabels.append(classNames[i] + '\n' + classRange[i])
        plt.yticks(tick_marks, YLabels)
        Usados = [] #para evitar que los numeros se pisen
        OffsetsY = [-0.4, -0.2, 0, 0.2, -0.4, -0.2, 0, 0.2, -0.4, -0.2, 0, 0.2, 0.6, -0.6, 0.8, -0.8]
        OffsetsX = [-0.45, -0.05, 0.35]
        OffsetX = 0
        if NClasses > 2:
            GeneralYOffset = 0.1
        else:
            GeneralYOffset = 0
        for i in range(len(ClaseValDiscretizada)):
            Pred = ClasePredDiscretizada[i]
            Real = ClaseValDiscretizada[i]
            Year = TargetYears[i]
            OffsetY = int(np.sum([ 1 for x, y in Usados if x  == Real and y == Pred ]))
            if np.mod(OffsetY,4) == 0 and OffsetY != 0 and OffsetX < 2:
                OffsetX = OffsetX + 1
            Usados.append((Real, Pred))
            plt.text(classLabels.index(Pred) + OffsetsX[OffsetX] ,classLabels.index(Real) + OffsetsY[OffsetY] + GeneralYOffset, str(Year))    
            
         
            
        #Plot #3 - Model metrics    
        plt.subplot(153)
        plt.axis('off')
        bbox_props = dict(boxstyle="Round,pad=0.3", fc="lightcoral", ec="r", lw=2)
        plt.text(.25, .33, "Expl. Var.: " + str(ExplainedVar), ha="center", va="center", rotation=0, size=25, bbox=bbox_props)
        plt.text(.25, .66, "MAE: " + str(MeanAbsoluteErr), ha="center", va="center", rotation=0, size=25, bbox=bbox_props)


        #Plot #4 - Precision, Recall, F1-Score, Support    
        ax = plt.subplot(154)
        plt.axis('off')
        Cols = ['Precision', 'Recall', 'F1-Score']
        Rows = classNames 
        PrecRecallF1 = PrecRecallF1[0:3] #Remove support metric
        Cells = np.matrix(PrecRecallF1)
        Cells = np.rot90(Cells, k=3)
        Cells = Cells.round(2)
        
        Table = plt.table(cellText=np.array(Cells),
                      rowLabels=Rows,
                      rowColours=["#a9a9a9"] * 5,
                      colColours = ["#a9a9a9"] * 3,
                      colLabels=Cols,
                      loc='center')
        Table.auto_set_font_size = False
        Table.set_fontsize(16)
        Table.scale(0.75,4)
        
        
        #Plot #5 - ROC curves
        
        ClaseBinarizada = label_binarize(ClaseValDiscretizada, classes=classLabels + [-1])
        PredBinarizada = label_binarize(ClasePredDiscretizada, classes=classLabels + [-1])
        n_classes = ClaseBinarizada.shape[1] - 1
        
        # Compute ROC curve and ROC area for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(ClaseBinarizada[:,i], PredBinarizada[:,i])
            if str(tpr[i]) == '[nan nan]':
                tpr[i] = [1, 1]
            if str(fpr[i]) == '[nan nan]':
                fpr[i] = [0,1]
            roc_auc[i] = auc(fpr[i], tpr[i])
        
        plt.subplot(155)
        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        colors = ['royalblue', 'tomato', 'forestgreen', 'darkviolet', 'darkorange']
        for i, color in zip(range(n_classes), colors):
            plt.plot(fpr[i], tpr[i], color=color, lw=2, label= classNames[i] + '. Area = ' + str(round(roc_auc[i],2)))
        plt.legend(loc="lower right")
        
        #Save and Show
        plt.tight_layout()
        plt.savefig(PathModelOutput + Grupo + '\\Modelo_' + Grupo + '_M' + Mes + '_Y' + str(ValidationYears[0]) + '-' + str(ValidationYears[1]) + '\\' + NombreModelo + '\\' + NombreModelo + '_' + Grupo + '_M' + Mes + '_Y' + str(ValidationYears[0]) + '-' + str(ValidationYears[1]) + '.png')
        plt.show()  
        
        



def PlotNetworkPerformance(history, y_val, y_pred, MeanAbsoluteErr, ExplainedVar, ValidationYears, Percentiles, NombreModelo, Grupo, Mes, PathModelOutput):
        #Prepare Data
        Percentiles = AccomodatePercentiles(list(Percentiles))   
        NClasses = len(Percentiles) + 1             
                
        TargetYears = np.linspace(ValidationYears[0], ValidationYears[1], ValidationYears[1] - ValidationYears[0] + 1, dtype=int)
        ClaseValDiscretizada = np.digitize(y_val, Percentiles) 
        ClasePredDiscretizada = np.digitize(y_pred, Percentiles)
        classNames = []
        classRange = []
        classLabels = []
        if 0 in ClaseValDiscretizada or 0 in ClasePredDiscretizada: 
            classNames.append(NameClass(NClasses,0))
            classLabels.append(0)
            classRange.append(DefineRange(Percentiles, NClasses, 0))
        if 1 in ClaseValDiscretizada or 1 in ClasePredDiscretizada: 
            classNames.append(NameClass(NClasses,1))
            classLabels.append(1)
            classRange.append(DefineRange(Percentiles, NClasses, 1))
        if 2 in ClaseValDiscretizada or 2 in ClasePredDiscretizada: 
            classNames.append(NameClass(NClasses,2))
            classLabels.append(2)
            classRange.append(DefineRange(Percentiles, NClasses, 2))
        if 3 in ClaseValDiscretizada or 3 in ClasePredDiscretizada: 
            classNames.append(NameClass(NClasses,3))
            classLabels.append(3)
            classRange.append(DefineRange(Percentiles, NClasses, 3))
        if 4 in ClaseValDiscretizada or 4 in ClasePredDiscretizada: 
            classNames.append(NameClass(NClasses,4))
            classLabels.append(4)
            classRange.append(DefineRange(Percentiles, NClasses, 4))
        cm  = confusion_matrix(ClaseValDiscretizada, ClasePredDiscretizada)
        PrecRecallF1 = precision_recall_fscore_support(ClaseValDiscretizada, ClasePredDiscretizada, average=None, labels=classLabels, warn_for=())
        
        
        #Plot 0 - Loss function
        training_loss = pd.DataFrame(history.history['loss'], columns = ['val'])
        training_loss = np.array(training_loss.val.rolling(window=10).mean())
        if 'val_loss' in history.history.keys():
            #test_loss = history.history['val_loss']    
            test_loss = pd.DataFrame(history.history['val_loss'], columns = ['val'])
            test_loss = np.array(test_loss.val.rolling(window=10).mean())
                
        epoch_count = range(1, len(training_loss) + 1)
        plt.figure(figsize=(20,9))
        plt.subplot(231)    
        plt.title("Loss function")
        plt.plot(epoch_count, training_loss, 'r--')
        if 'val_loss' in history.history.keys():
            plt.plot(epoch_count, test_loss, 'b-')
            plt.legend(['Training Loss', 'Test Loss'])
            plt.ylabel('Loss')   
        plt.xlabel('Epoch')
        
        
        #Plot #1 - Predicted vs Actual
        plt.subplot(232)    
        plt.plot(y_pred, y_val, 'bo')
        [plt.text(y_pred[i], y_val[i], TargetYears[i]) for i in range(len(TargetYears))]
        plt.plot(y_val, y_val, 'go')
        plt.plot([np.min(y_val), np.max(y_val)], [np.min(y_val), np.max(y_val)], 'r--')
        plt.xlabel('Predicted value')
        plt.ylabel('Actual value')
        plt.suptitle(NombreModelo + '_' + Grupo + '_M' + Mes + '_Y' + str(ValidationYears[0]) + '-' + str(ValidationYears[1]), fontsize = 'xx-large', weight = 'heavy')
        
        
        
        #Plot #2 - Confusion Matrix
        plt.subplot(233)    
        CMPlot = np.zeros(cm.shape, dtype = int)
        for i in range(CMPlot.shape[0]):
            for j in range(CMPlot.shape[1]):
                if np.mod(i+j, 2) == 0: 
                    CMPlot[i,j] = 2
                else:
                    CMPlot[i,j] = 0
        plt.imshow(np.rot90(CMPlot), interpolation='nearest', cmap=plt.cm.Pastel2)
        plt.ylabel('Actual Class')
        plt.xlabel('Predicted Class')
        tick_marks = np.arange(len(classNames))
        plt.xticks(tick_marks, classNames, rotation=25)
        YLabels = []
        for i in range(len(classNames)):
            YLabels.append(classNames[i] + '\n' + classRange[i])
        plt.yticks(tick_marks, YLabels)
        Usados = [] #para evitar que los numeros se pisen
        OffsetsY = [-0.4, -0.2, 0, 0.2, -0.4, -0.2, 0, 0.2, -0.4, -0.2, 0, 0.2, 0.6, -0.6, 0.8, -0.8]
        OffsetsX = [-0.45, -0.05, 0.35]
        if NClasses > 2:
            GeneralYOffset = 0.1
        else:
            GeneralYOffset = 0
        for i in range(len(ClaseValDiscretizada)):
            Pred = ClasePredDiscretizada[i]
            Real = ClaseValDiscretizada[i]
            Year = TargetYears[i]
            OffsetY = int(np.sum([ 1 for x, y in Usados if x  == Real and y == Pred ]))
            OffsetX = 0
            if np.mod(OffsetY,4) == 0 and OffsetY != 0 and OffsetX < 2:
                OffsetX = OffsetX + 1
            Usados.append((Real, Pred))
            plt.text(classLabels.index(Pred) + OffsetsX[OffsetX] ,classLabels.index(Real) + OffsetsY[OffsetY] + GeneralYOffset, str(Year))    
            
         
            
        #Plot #3 - Model metrics    
        plt.subplot(234)
        plt.axis('off')
        bbox_props = dict(boxstyle="Round,pad=0.3", fc="lightcoral", ec="r", lw=2)
        plt.text(.25, .33, "Expl. Var.: " + str(ExplainedVar), ha="center", va="center", rotation=0, size=25, bbox=bbox_props)
        plt.text(.25, .66, "MAE: " + str(MeanAbsoluteErr), ha="center", va="center", rotation=0, size=25, bbox=bbox_props)


        #Plot #4 - Precision, Recall, F1-Score, Support    
        ax = plt.subplot(235)
        plt.axis('off')
        Cols = ['Precision', 'Recall', 'F1-Score']
        Rows = classNames 
        PrecRecallF1 = PrecRecallF1[0:3] #Remove support metric
        Cells = np.matrix(PrecRecallF1)
        Cells = np.rot90(Cells, k=3)
        Cells = Cells.round(2)
        
        Table = plt.table(cellText=np.array(Cells),
                      rowLabels=Rows,
                      rowColours=["#a9a9a9"] * 5,
                      colColours = ["#a9a9a9"] * 3,
                      colLabels=Cols,
                      loc='center')
        Table.auto_set_font_size = False
        Table.set_fontsize(16)
        Table.scale(0.75,4)
        
        
        #Plot #5 - ROC curves
        ClaseBinarizada = label_binarize(ClaseValDiscretizada, classes=classLabels + [-1])
        PredBinarizada = label_binarize(ClasePredDiscretizada, classes=classLabels + [-1])
        n_classes = ClaseBinarizada.shape[1] - 1
        
        # Compute ROC curve and ROC area for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(ClaseBinarizada[:,i], PredBinarizada[:,i])
            if str(tpr[i]) == '[nan nan]':
                tpr[i] = [1, 1]
            if str(fpr[i]) == '[nan nan]':
                fpr[i] = [0,1]
            roc_auc[i] = auc(fpr[i], tpr[i])
        
        plt.subplot(236)
        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        colors = ['royalblue', 'tomato', 'forestgreen', 'darkviolet', 'darkorange']
        for i, color in zip(range(n_classes), colors):
            plt.plot(fpr[i], tpr[i], color=color, lw=2, label= classNames[i] + '. Area = ' + str(round(roc_auc[i],2)))
        plt.legend(loc="lower right")
        
        #Save and Show
        plt.savefig(PathModelOutput + Grupo + '\\Modelo_' + Grupo + '_M' + Mes + '_Y' + str(ValidationYears[0]) + '-' + str(ValidationYears[1]) + '\\' + NombreModelo + '\\' + NombreModelo + '_' + Grupo + '_M' + Mes + '_Y' + str(ValidationYears[0]) + '-' + str(ValidationYears[1]) + '.png')
        plt.show()  