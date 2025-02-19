from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix, balanced_accuracy_score
from sklearn.metrics import matthews_corrcoef

import re
#regex='\s[01]\s[01]\s'
regex='\s[01]\s[01]\s'

model_name='bert'
run='run3'
dataset='eicu'
variable='gender_race'
#model_dataset='bert_mimic3_race'

preds=[]
test_y=[]
f=open('/home/DifferentialImpact/Prediction/'+model_name+'_'+run+'/'+dataset+'/'+variable+'/valid_prediction_probs.txt', 'r')
for line in f:
    line=line.strip()
    tokens=line.split()
    prob=float(tokens[0])
    true=int(tokens[1])

    preds.append(prob)
    test_y.append(true)

f.close()

tr_list=[]

#best_recall=0.0
#best_f1=0.0
best_f1_minor=0.0
#recall_with_best_f1=0.0
#f1_with_best_recall=0.0
#tr_f1=0.0
#tr_recall=0.0
tr_f1_minor=0.0

tr=0.0
while(tr<=1.0):
    predictions=[]
    for pred in preds:
        if pred<tr:
            predictions.append(0)
        else:
            predictions.append(1)

    #accuracy=accuracy_score(test_y, predictions)
    #precision=precision_score(test_y, predictions)
    #recall=recall_score(test_y, predictions)
    f1=f1_score(test_y, predictions)

    tn, fp, fn, tp = confusion_matrix(test_y, predictions).ravel()
    recall0 = tn / (tn + fp)
    precision0 = tn / (tn + fn)
    recall1 = tp / (tp + fn)
    precision1 = tp / (tp + fp)

    #f1_minor = 2 * precision0 * recall0 / (precision0 + recall0)
    f1_minor = 2 * precision1 * recall1 / (precision1 + recall1)

    '''if f1>best_f1:
        best_f1=f1
        tr_f1=tr
        tr_list.insert(0, tr_f1)'''
    if f1_minor>best_f1_minor:
        best_f1_minor=f1_minor
        tr_f1_minor=tr
        tr_list.insert(0, tr_f1_minor)
    '''if recall>best_recall:
        best_recall=recall
        tr_recall=tr'''

    tr=tr+0.01
    tr=round(tr, 2)

#print('Optimal threshold: ', str(tr_f1))
#print('Optimal threshold: ', str(tr_f1_minor))
#print('Optimal threshold: ', str(tr_recall))
print('Ranked thresholds: ', str(tr_list))


preds=[]
test_y=[]
text_list=[]
f=open('/home/DifferentialImpact/Prediction/'+model_name+'_'+run+'/'+dataset+'/'+variable+'/valid_prediction_probs.txt', 'r')
#f=open('/home/Foundation/Prediction/'+model_dataset+'/prediction_probs_label_text.txt', 'r')
for line in f:
    line=line.strip()
    tokens=line.split()
    prob=float(tokens[0])
    true=int(tokens[1])

    preds.append(prob)
    test_y.append(true)

    '''match_list = re.split(regex, line)
    text=match_list[1]
    text_list.append(text)'''

f.close()


best_balanced=0.0
best_threshold=0.0
for i in range(3):
    tr=tr_list[i]

    predictions=[]
    for pred in preds:
        if pred<tr:
            predictions.append(0)
        else:
            predictions.append(1)

    accuracy=accuracy_score(test_y, predictions)
    #precision=precision_score(test_y, predictions)
    #recall=recall_score(test_y, predictions)
    #f1=f1_score(test_y, predictions)
    balanced_accuracy=balanced_accuracy_score(test_y, predictions)

    tn, fp, fn, tp = confusion_matrix(test_y, predictions).ravel()
    recall0 = tn / (tn + fp)
    precision0 = tn / (tn + fn)
    recall1 = tp / (tp + fn)
    precision1 = tp / (tp + fp)

    f1_0 = 2 * precision0 * recall0 / (precision0 + recall0)
    f1_1 = 2 * precision1 * recall1 / (precision1 + recall1)

    auc=roc_auc_score(test_y, preds)
    mcc=matthews_corrcoef(test_y, predictions)

    print(f'##### Score for threshold {tr:.2f} #####')
    print(f'Accuracy: {accuracy:.3f}')
    print(f'Precision (Class 0): {precision0:.3f}')
    print(f'Recall (Class 0): {recall0:.3f}')
    print(f'F1 (Class 0): {f1_0:.3f}')
    print(f'Precision (Class 1): {precision1:.3f}')
    print(f'Recall (Class 1): {recall1:.3f}')
    print(f'F1 (Class 1): {f1_1:.3f}')
    #print(f'F1 minor: {f1_0:.3f}')
    print(f'F1 minor: {f1_1:.3f}')
    print(f'Balanced accuracy: {balanced_accuracy:.3f}')
    print(f'ROC AUC: {auc:.3f}')
    print(f'Matthews corr coef: {mcc:.3f}')
    print('\n')

    if balanced_accuracy>best_balanced:
        best_balanced=balanced_accuracy
        best_threshold=tr


'''f=open('/home/Foundation/Results/'+model_dataset+'_wrong_preds.txt', 'w')
prediction=0
for i in range(len(preds)):
    if preds[i]<best_threshold:
        prediction=0
    else:
        prediction=1

    if prediction!=test_y[i]:
        f.write(str(preds[i])+' '+str(best_threshold)+' '+str(prediction)+' '+str(test_y[i])+' '+text_list[i]+'\n')

f.close()

f=open('/home/Foundation/Results/'+model_dataset+'_text.txt', 'w')
prediction=0
for i in range(len(preds)):
    if preds[i]<best_threshold:
        prediction=0
    else:
        prediction=1

    f.write(str(prediction)+' '+str(test_y[i])+' '+text_list[i]+'\n')

f.close()'''
