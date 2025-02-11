import pandas as pd

# best epochs: 28 for no demography, 71 for gender, 50 for race, 65 for age

def writeLabel(f, best_threshold, pred_dict, pred_list):
    #first_line=True
    for line in f:
        line=line.strip()
        #if first_line:
        #    first_line=False
        #    continue
        
        #tokens=line.split(',')
        tokens=line.split(' ')

        #file=tokens[0]
        #patient_id=file.split('_')[0]

        #prediction=float(tokens[1])
        #y_true=int(tokens[2])
        prediction=float(tokens[0])
        y_true=int(tokens[1])

        y_pred=0
        if prediction<best_threshold:
            y_pred=0
        else:
            y_pred=1

        #pred_dict[patient_id]=[y_pred, y_true, prediction]
        pred_list.append([y_pred, y_true, prediction])

#threshold_1= 0.19 # 0.19 for no demography
#run1
# mimic3: 0.15 for no demography, 0.18 for gender, 0.19 for race, 0.13 for gender race
# eicu: 0.13 for no demography, 0.13 for gender, 0.13 for race, 0.12 for gender race
#run2
# mimic3: 0.19 for no demography, 0.15 for gender, 0.15 for race, 0.16 for gender race
# eicu: 0.14 for no demography, 0.19 for gender, 0.17 for race, 0.21 for gender race
#run3
# mimic3: 0.13 for no demography, 0.14 for gender, 0.13 for race, 0.10 for gender race
# eicu: 0.19 for no demography, 0.18 for gender, 0.13 for race, 0.12 for gender race
threshold=0.12

'''pred_dict_1={}
f1=open('/home/mohimenul/Differential_Bias/no_demography/test_predictions/k_lstm.n16.d0.3.dep2.bs8.ts1.0.epoch28.test0.2843273498238169.state.csv', 'r')
writeLabel(f1, threshold_1, pred_dict_1)
f1.close()'''

model_name='bert'
run='run3'
dataset='eicu'
variable='gender_race'

pred_dict={}
pred_list=[]
f=open('/home/DifferentialImpact/Prediction/'+model_name+'_'+run+'/'+dataset+'/'+variable+'/prediction_probs.txt', 'r')
writeLabel(f, threshold, pred_dict, pred_list)
f.close()


fw=open('/home/DifferentialImpact/ihm_prediction/'+model_name+'_'+run+'/'+dataset+'/ihm_'+variable+'.csv', 'w')
fw.write('Prediction,TrueLabel,CorrectFlag,Prob\n')

'''demography_dict={}
f=open('/home/mohimenul/Data/mimic 3/ihm_demography/demography.csv', 'r')

first_line=True
for line in f:
    line=line.strip()
    if first_line:
        first_line=False
        continue

    tokens=line.split(',')
    file=tokens[0]
    patient_id=file.split('_')[0]

    label_pairs=pred_dict_2[patient_id]
    pred=label_pairs[0]
    true=label_pairs[1]
    prob=label_pairs[2]

    correct_flag=0
    if pred==true:
        correct_flag=1
    else:
        correct_flag=0

    fw2.write(str(pred)+','+str(true)+','+str(correct_flag)+','+str(prob)+'\n')

f.close()'''

for pairs in pred_list:
    label_pairs=pairs
    pred=label_pairs[0]
    true=label_pairs[1]
    prob=label_pairs[2]

    correct_flag=0
    if pred==true:
        correct_flag=1
    else:
        correct_flag=0

    fw.write(str(pred)+','+str(true)+','+str(correct_flag)+','+str(prob)+'\n')

fw.close()
