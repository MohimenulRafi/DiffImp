import pandas as pd
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix, balanced_accuracy_score
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import precision_recall_fscore_support
import json

score_dict={'acc':[], 'prec_c0':[], 'rec_c0':[], 'f1_c0':[], 'prec_c1':[], 'rec_c1':[], 'f1_c1':[], 'micro_prec':[], 'micro_rec':[], 'micro_f1':[], 'macro_prec':[], 'macro_rec':[], 'macro_f1':[], 'weighted_prec':[], 'weighted_rec':[], 'weighted_f1':[], 'bal_acc':[], 'auc':[], 'mcc':[]}

def score(y_true, y_pred, y_prob):
    accuracy=accuracy_score(y_true, y_pred)
    balanced_accuracy=balanced_accuracy_score(y_true, y_pred)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    recall0 = tn / (tn + fp)
    precision0 = tn / (tn + fn)
    recall1 = tp / (tp + fn)
    precision1 = tp / (tp + fp)

    f1_0 = 2 * precision0 * recall0 / (precision0 + recall0)
    f1_1 = 2 * precision1 * recall1 / (precision1 + recall1)

    precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(y_true, y_pred, average='micro')
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(y_true, y_pred, average='macro')
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')

    auc=roc_auc_score(y_true, y_prob) #need probabilities
    mcc=matthews_corrcoef(y_true, y_pred)

    #print('##### Overall scores without demographic information #####')
    print(f'Accuracy: {accuracy:.3f}')
    print(f'Precision (Class 0): {precision0:.3f}')
    print(f'Recall (Class 0): {recall0:.3f}')
    print(f'F1 (Class 0): {f1_0:.3f}')
    print(f'Precision (Class 1): {precision1:.3f}')
    print(f'Recall (Class 1): {recall1:.3f}')
    print(f'F1 (Class 1): {f1_1:.3f}')
    print(f'Micro Precision: {precision_micro:.3f}')
    print(f'Micro Recall: {recall_micro:.3f}')
    print(f'Micro F1: {f1_micro:.3f}')
    print(f'Macro Precision: {precision_macro:.3f}')
    print(f'Macro Recall: {recall_macro:.3f}')
    print(f'Macro F1: {f1_macro:.3f}')
    print(f'Weighted Precision: {precision_weighted:.3f}')
    print(f'Weighted Recall: {recall_weighted:.3f}')
    print(f'Weighted F1: {f1_weighted:.3f}')
    #print(f'F1 minor: {f1_0:.3f}')
    #print(f'F1 minor: {f1_1:.3f}')
    print(f'Balanced accuracy: {balanced_accuracy:.3f}')
    print(f'ROC AUC: {auc:.3f}') #need probabilities
    print(f'Matthews corr coef: {mcc:.3f}')

    score_dict['acc'].append(round(accuracy, 3))
    score_dict['prec_c0'].append(round(precision0, 3))
    score_dict['rec_c0'].append(round(recall0, 3))
    score_dict['f1_c0'].append(round(f1_0, 3))
    score_dict['prec_c1'].append(round(precision1, 3))
    score_dict['rec_c1'].append(round(recall1, 3))
    score_dict['f1_c1'].append(round(f1_1, 3))
    score_dict['micro_prec'].append(round(precision_micro, 3))
    score_dict['micro_rec'].append(round(recall_micro, 3))
    score_dict['micro_f1'].append(round(f1_micro, 3))
    score_dict['macro_prec'].append(round(precision_macro, 3))
    score_dict['macro_rec'].append(round(recall_macro, 3))
    score_dict['macro_f1'].append(round(f1_macro, 3))
    score_dict['weighted_prec'].append(round(precision_weighted, 3))
    score_dict['weighted_rec'].append(round(recall_weighted, 3))
    score_dict['weighted_f1'].append(round(f1_weighted, 3))
    score_dict['bal_acc'].append(round(balanced_accuracy, 3))
    score_dict['auc'].append(round(auc, 3))
    score_dict['mcc'].append(round(mcc, 3))

    print('\n')

model_name='bert'
run='run3'
dataset='eicu'
variable='gender_race'
out_variable='race_gender'

################### Without demographic info #####################
prediction_file='/home/danfeng/DifferentialBias/ihm_prediction/'+model_name+'_'+run+'/'+dataset+'/ihm_nodemography.csv'

df_nd=pd.read_csv(prediction_file) #nd -> no demographic info

total_instances=len(df_nd)
true_0=df_nd[df_nd['TrueLabel']==0]
true_1=df_nd[df_nd['TrueLabel']==1]
y_pred=df_nd['Prediction']
y_true=df_nd['TrueLabel']
y_prob=df_nd['Prob']

correct_nd=df_nd['CorrectFlag']

print('##### Overall scores without demographic information #####')
score(y_true, y_pred, y_prob)

################### With demographic info #####################

prediction_file='/home/danfeng/DifferentialBias/ihm_prediction/'+model_name+'_'+run+'/'+dataset+'/ihm_'+variable+'.csv'

df=pd.read_csv(prediction_file)

total_instances=len(df)
true_0=df[df['TrueLabel']==0]
true_1=df[df['TrueLabel']==1]
y_pred=df['Prediction']
y_true=df['TrueLabel']
y_prob=df['Prob']

correct=df['CorrectFlag']

print('##### Overall scores with demographic information #####')
score(y_true, y_pred, y_prob)


demographic_file='/home/danfeng/Data/eICU/ihm_demography/demography_test.csv'
df_demography=pd.read_csv(demographic_file)

gender=df_demography['Gender']
race=df_demography['Ethnicity']

case_dict={'Change to death':[], 'Change to survive':[], 'Wrong change to death':[], 'Wrong change to survive':[], 'Death prediction':[]}

actual_death_asian=0
actual_death_white=0
actual_death_black=0
actual_death_hispanic=0
actual_survive_asian=0
actual_survive_white=0
actual_survive_black=0
actual_survive_hispanic=0

for index, row in df_nd.iterrows():
    row_demographic=df.iloc[[index]]
    patient_race=race.iloc[[index]].item()
    if row['CorrectFlag']!=row_demographic['CorrectFlag'].item():
        patient_gender=gender.iloc[[index]].item()
        patient_race=race.iloc[[index]].item()

    if row['CorrectFlag']==0 and row_demographic['CorrectFlag'].item()==1 and row_demographic['TrueLabel'].item()==1:
        #print(index)
        #case_dict['Change to death'].append(gender.iloc[[index]].item())
        case_dict['Change to death'].append(race.iloc[[index]].item())
    if row['CorrectFlag']==0 and row_demographic['CorrectFlag'].item()==1 and row_demographic['TrueLabel'].item()==0:
        #print(index)
        #case_dict['Change to survive'].append(gender.iloc[[index]].item())
        case_dict['Change to survive'].append(race.iloc[[index]].item())

    if row['CorrectFlag']==1 and row_demographic['CorrectFlag'].item()==0 and row_demographic['TrueLabel'].item()==0:
        #print(index)
        #case_dict['Wrong change to death'].append(gender.iloc[[index]].item())
        case_dict['Wrong change to death'].append(race.iloc[[index]].item())
    if row['CorrectFlag']==1 and row_demographic['CorrectFlag'].item()==0 and row_demographic['TrueLabel'].item()==1:
        #print(index)
        #case_dict['Wrong change to survive'].append(gender.iloc[[index]].item())
        case_dict['Wrong change to survive'].append(race.iloc[[index]].item())

    
    if row['TrueLabel']==1 and race.iloc[[index]].item()=='Asian':
        actual_death_asian+=1
    if row['TrueLabel']==1 and race.iloc[[index]].item()=='White':
        actual_death_white+=1
    if row['TrueLabel']==1 and race.iloc[[index]].item()=='Black':
        actual_death_black+=1
    if row['TrueLabel']==1 and race.iloc[[index]].item()=='Hispanic':
        actual_death_hispanic+=1
    if row['TrueLabel']==0 and race.iloc[[index]].item()=='Asian':
        actual_survive_asian+=1
    if row['TrueLabel']==0 and race.iloc[[index]].item()=='White':
        actual_survive_white+=1
    if row['TrueLabel']==0 and race.iloc[[index]].item()=='Black':
        actual_survive_black+=1
    if row['TrueLabel']==0 and race.iloc[[index]].item()=='Hispanic':
        actual_survive_hispanic+=1


print('Actual number of death (Asian): '+str(actual_death_asian))
print('Actual number of death (White): '+str(actual_death_white))
print('Actual number of death (Black): '+str(actual_death_black))
print('Actual number of death (Hispanic): '+str(actual_death_hispanic))
print('Actual number of survival (Asian): '+str(actual_survive_asian))
print('Actual number of survival (White): '+str(actual_survive_white))
print('Actual number of survival (Black): '+str(actual_survive_black))
print('Actual number of survival (Hispanic): '+str(actual_survive_hispanic))
print('\n')

print('Change to death (correct)')
#print(case_dict['Channge to death'])

change_to_death_asian=len([x for x in case_dict['Change to death'] if x=='Asian'])
change_to_death_white=len([x for x in case_dict['Change to death'] if x=='White'])
change_to_death_black=len([x for x in case_dict['Change to death'] if x=='Black'])
change_to_death_hispanic=len([x for x in case_dict['Change to death'] if x=='Hispanic'])
print(f'Asian: {change_to_death_asian}')
print(f'White: {change_to_death_white}')
print(f'Black: {change_to_death_black}')
print(f'Hispanic: {change_to_death_hispanic}')

print('Change to survive (correct)')
#print(case_dict['Change to survive'])
change_to_survive_asian=len([x for x in case_dict['Change to survive'] if x=='Asian'])
change_to_survive_white=len([x for x in case_dict['Change to survive'] if x=='White'])
change_to_survive_black=len([x for x in case_dict['Change to survive'] if x=='Black'])
change_to_survive_hispanic=len([x for x in case_dict['Change to survive'] if x=='Hispanic'])
print(f'Asian: {change_to_survive_asian}')
print(f'White: {change_to_survive_white}')
print(f'Black: {change_to_survive_black}')
print(f'Hispanic: {change_to_survive_hispanic}')

print('Change to death (wrong)')
#print(case_dict['Channge to death'])
wrong_change_to_death_asian=len([x for x in case_dict['Wrong change to death'] if x=='Asian'])
wrong_change_to_death_white=len([x for x in case_dict['Wrong change to death'] if x=='White'])
wrong_change_to_death_black=len([x for x in case_dict['Wrong change to death'] if x=='Black'])
wrong_change_to_death_hispanic=len([x for x in case_dict['Wrong change to death'] if x=='Hispanic'])
print(f'Asian: {wrong_change_to_death_asian}')
print(f'White: {wrong_change_to_death_white}')
print(f'Black: {wrong_change_to_death_black}')
print(f'Hispanic: {wrong_change_to_death_hispanic}')

print('Change to survive (wrong)')
#print(case_dict['Change to survive'])
wrong_change_to_survive_asian=len([x for x in case_dict['Wrong change to survive'] if x=='Asian'])
wrong_change_to_survive_white=len([x for x in case_dict['Wrong change to survive'] if x=='White'])
wrong_change_to_survive_black=len([x for x in case_dict['Wrong change to survive'] if x=='Black'])
wrong_change_to_survive_hispanic=len([x for x in case_dict['Wrong change to survive'] if x=='Hispanic'])
print(f'Asian: {wrong_change_to_survive_asian}')
print(f'White: {wrong_change_to_survive_white}')
print(f'Black: {wrong_change_to_survive_black}')
print(f'Hispanic: {wrong_change_to_survive_hispanic}')

print('\n')

######### Race wise performance metrics #########
y_true_asian=[]
y_pred_asian=[]
y_prob_asian=[]
y_true_white=[]
y_pred_white=[]
y_prob_white=[]
y_true_black=[]
y_pred_black=[]
y_prob_black=[]
y_true_hispanic=[]
y_pred_hispanic=[]
y_prob_hispanic=[]

for index, row in df_nd.iterrows():
    row_nd=df_nd.iloc[[index]]
    patient_race=race.iloc[[index]].item()
    if patient_race=='Asian':
        y_true_asian.append(row_nd['TrueLabel'].item())
        y_pred_asian.append(row_nd['Prediction'].item())
        y_prob_asian.append(row_nd['Prob'].item())
    elif patient_race=='White':
        y_true_white.append(row_nd['TrueLabel'].item())
        y_pred_white.append(row_nd['Prediction'].item())
        y_prob_white.append(row_nd['Prob'].item())
    elif patient_race=='Black':
        y_true_black.append(row_nd['TrueLabel'].item())
        y_pred_black.append(row_nd['Prediction'].item())
        y_prob_black.append(row_nd['Prob'].item())
    elif patient_race=='Hispanic':
        y_true_hispanic.append(row_nd['TrueLabel'].item())
        y_pred_hispanic.append(row_nd['Prediction'].item())
        y_prob_hispanic.append(row_nd['Prob'].item())

print('##### Scores for asian without demographic information #####')
score(y_true_asian, y_pred_asian, y_prob_asian)

print('##### Scores for white without demographic information #####')
score(y_true_white, y_pred_white, y_prob_white)

print('##### Scores for black without demographic information #####')
score(y_true_black, y_pred_black, y_prob_black)

print('##### Scores for hispanic without demographic information #####')
score(y_true_hispanic, y_pred_hispanic, y_prob_hispanic)

##### with race information #####
y_true_asian=[]
y_pred_asian=[]
y_prob_asian=[]
y_true_white=[]
y_pred_white=[]
y_prob_white=[]
y_true_black=[]
y_pred_black=[]
y_prob_black=[]
y_true_hispanic=[]
y_pred_hispanic=[]
y_prob_hispanic=[]

for index, row in df_nd.iterrows():
    row_demographic=df.iloc[[index]]
    #row_nd=df_nd.iloc[[index]]
    patient_race=race.iloc[[index]].item()
    if patient_race=='Asian':
        y_true_asian.append(row_demographic['TrueLabel'].item())
        y_pred_asian.append(row_demographic['Prediction'].item())
        y_prob_asian.append(row_demographic['Prob'].item())
    elif patient_race=='White':
        y_true_white.append(row_demographic['TrueLabel'].item())
        y_pred_white.append(row_demographic['Prediction'].item())
        y_prob_white.append(row_demographic['Prob'].item())
    elif patient_race=='Black':
        y_true_black.append(row_demographic['TrueLabel'].item())
        y_pred_black.append(row_demographic['Prediction'].item())
        y_prob_black.append(row_demographic['Prob'].item())
    elif patient_race=='Hispanic':
        y_true_hispanic.append(row_demographic['TrueLabel'].item())
        y_pred_hispanic.append(row_demographic['Prediction'].item())
        y_prob_hispanic.append(row_demographic['Prob'].item())

print('##### Scores for asian with demographic information #####')
score(y_true_asian, y_pred_asian, y_prob_asian)

print('##### Scores for white with demographic information #####')
score(y_true_white, y_pred_white, y_prob_white)

print('##### Scores for black with demographic information #####')
score(y_true_black, y_pred_black, y_prob_black)

print('##### Scores for hispanic with demographic information #####')
score(y_true_hispanic, y_pred_hispanic, y_prob_hispanic)

with open('/home/danfeng/DifferentialBias/ihm_result/'+model_name+'_'+run+'/'+dataset+'/scores_'+out_variable+'.json', 'w') as fp:
    json.dump(score_dict, fp)