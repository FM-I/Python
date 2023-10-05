import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
import functions as filimonhuk_func
from sklearn.metrics import f1_score
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

df = pd.read_csv('data_metrics.csv')
df.head()

thresh = 0.5
df['predicted_RF'] = (df.model_RF >= 0.5).astype('int')
df['predicted_LR'] = (df.model_LR >= 0.5).astype('int')
df.head()

print(confusion_matrix(df.actual_label.values, df.predicted_RF.values))


print('TP:', filimonhuk_func.find_TP(df.actual_label.values, df.predicted_RF.values))
print('FN:', filimonhuk_func.find_FN(df.actual_label.values, df.predicted_RF.values))
print('FP:', filimonhuk_func.find_FP(df.actual_label.values, df.predicted_RF.values))
print('TN:', filimonhuk_func.find_TN(df.actual_label.values, df.predicted_RF.values))

assert np.array_equal(filimonhuk_func.filimonchuk_confusion_matrix(
    df.actual_label.values,
    df.predicted_RF.values),
    confusion_matrix(
        df.actual_label.values,
        df.predicted_RF.values)),\
    'filimonhuk_func.filimonchuk_confusion_matrix() is not correct for RF'

assert np.array_equal(filimonhuk_func.filimonchuk_confusion_matrix(
    df.actual_label.values,
    df.predicted_LR.values),
    confusion_matrix(
        df.actual_label.values,
        df.predicted_LR.values)),\
    'filimonhuk_func.filimonchuk_confusion_matrix() is not correct for LR'

print(accuracy_score(df.actual_label.values, df.predicted_RF.values))


assert (filimonhuk_func.filimonchuk_accuracy_score(df.actual_label.values, df.predicted_RF.values)
        == accuracy_score(df.actual_label.values, df.predicted_RF.values)),\
    'filimonhuk_func.filimonchuk_accuracy_score failed on RF'
assert (filimonhuk_func.filimonchuk_accuracy_score(df.actual_label.values, df.predicted_LR.values)
        == accuracy_score(df.actual_label.values, df.predicted_LR.values)),\
    'filimonhuk_func.filimonchuk_accuracy_score failed on LR'

print('Accuracy RF:%.3f'%(filimonhuk_func.filimonchuk_accuracy_score(df.actual_label.values, df.predicted_RF.values)))
print('Accuracy LR:%.3f'%(filimonhuk_func.filimonchuk_accuracy_score(df.actual_label.values, df.predicted_LR.values)))

print(recall_score(df.actual_label.values, df.predicted_RF.values))

assert (filimonhuk_func.filimonchuk_recall_score(df.actual_label.values, df.predicted_RF.values)
        == recall_score(df.actual_label.values, df.predicted_RF.values)),\
    'filimonhuk_func.filimonchuk_accuracy_score failed on RF'
assert (filimonhuk_func.filimonchuk_recall_score(df.actual_label.values, df.predicted_LR.values)
        == recall_score(df.actual_label.values, df.predicted_LR.values)),\
    'filimonhuk_func.filimonchuk_accuracy_score failed on LR'

print('Recall RF: %.3f'%(filimonhuk_func.filimonchuk_recall_score(df.actual_label.values, df.predicted_RF.values)))
print('Recall LR: %.3f'%(filimonhuk_func.filimonchuk_recall_score(df.actual_label.values, df.predicted_LR.values)))

print(precision_score(df.actual_label.values, df.predicted_RF.values))

assert (filimonhuk_func.filimonchuk_precision_score(
    df.actual_label.values,
    df.predicted_RF.values)
        == precision_score(
            df.actual_label.values,
            df.predicted_RF.values)),\
    'filimonhuk_func.filimonchuk_accuracy_score failed on RF'

assert (filimonhuk_func.filimonchuk_precision_score(
    df.actual_label.values,
    df.predicted_LR.values)
        == precision_score(
            df.actual_label.values,
            df.predicted_LR.values)),\
    'filimonhuk_func.filimonchuk_accuracy_score failed on LR'

print('Precision RF: %.3f'%(filimonhuk_func.filimonchuk_precision_score(
    df.actual_label.values, df.predicted_RF.values)))

print('Precision LR: %.3f'%(filimonhuk_func.filimonchuk_precision_score(
    df.actual_label.values, df.predicted_LR.values)))

print(f1_score(df.actual_label.values, df.predicted_RF.values))


assert (filimonhuk_func.filimonchuk_f1_score(
    df.actual_label.values,
    df.predicted_RF.values)
        ==  f1_score(
            df.actual_label.values,
            df.predicted_RF.values)),\
    'filimonhuk_func.filimonchuk_accuracy_score failed on RF'

assert (filimonhuk_func.filimonchuk_f1_score(
    df.actual_label.values,
    df.predicted_LR.values)
        == f1_score(
            df.actual_label.values,
            df.predicted_LR.values)),\
    'filimonhuk_func.filimonchuk_accuracy_score failed on LR'

print('F1 RF: %.3f'%(filimonhuk_func.filimonchuk_f1_score(
    df.actual_label.values, df.predicted_RF.values)))

print('F1 LR: %.3f'%(filimonhuk_func.filimonchuk_f1_score(
    df.actual_label.values, df.predicted_LR.values)))

print('scores with threshold = 0.5')

print('Accuracy RF: %.3f'%(filimonhuk_func.filimonchuk_accuracy_score(df.actual_label.values, df.predicted_RF.values)))
print('Recall RF: %.3f'%(filimonhuk_func.filimonchuk_recall_score(df.actual_label.values, df.predicted_RF.values)))
print('Precision RF: %.3f'%(filimonhuk_func.filimonchuk_precision_score(df.actual_label.values, df.predicted_RF.values)))
print('F1 RF: %.3f'%(filimonhuk_func.filimonchuk_f1_score(df.actual_label.values, df.predicted_RF.values)))
print('')
print('scores with threshold = 0.25')
print('Accuracy RF: %.3f'%(filimonhuk_func.filimonchuk_accuracy_score(df.actual_label.values, (df.model_RF >= 0.25).astype('int').values)))
print('Recall RF: %.3f'%(filimonhuk_func.filimonchuk_recall_score(df.actual_label.values, (df.model_RF >= 0.25).astype('int').values)))
print('Precision RF: %.3f'%(filimonhuk_func.filimonchuk_precision_score(df.actual_label.values, (df.model_RF >= 0.25).astype('int').values)))
print('F1 RF: %.3f'%(filimonhuk_func.filimonchuk_f1_score(df.actual_label.values, (df.model_RF >= 0.25).astype('int').values)))


fpr_RF, tpr_RF, thresholds_RF = roc_curve(df.actual_label.values, df.model_RF.values)
fpr_LR, tpr_LR, thresholds_LR =  roc_curve(df.actual_label.values, df.model_LR.values)

plt.plot(fpr_RF, tpr_RF,'r-',label = 'RF')
plt.plot(fpr_LR,tpr_LR,'b-', label= 'LR')
plt.plot([0,1],[0,1],'k-',label='random')
plt.plot([0,0,1,1],[0,1,1,1],'g-',label='perfect')
plt.legend()
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.show()

auc_RF = roc_auc_score(df.actual_label.values, df.model_RF.values)
auc_LR = roc_auc_score(df.actual_label.values, df.model_LR.values)
print('AUC RF:%.3f'% auc_RF)
print('AUC LR:%.3f'% auc_LR)

plt.plot(fpr_RF, tpr_RF,'r-',label = 'RF AUC: %.3f'%auc_RF)
plt.plot(fpr_LR,tpr_LR,'b-', label= 'LR AUC: %.3f'%auc_LR)
plt.plot([0,1],[0,1],'k-',label='random')
plt.plot([0,0,1,1],[0,1,1,1],'g-',label='perfect')
plt.legend()
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.show()
