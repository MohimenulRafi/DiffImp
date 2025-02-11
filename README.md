# DiffImp

## BERT Instruction ##

**Run the bert.py file** <br />
python3 bert.py --data_file_train='/home/Data/mimic 3/ihm_train/train_gender.csv' --data_file_val='/home/Data/mimic 3/ihm_val/val_gender.csv' --max_token_length=300 --batch_size=32 --epochs=10 --seed=42 --model_dir='/home/DifferentialImpact/SavedModel' --model_name='bert' --run='run1' --dataset_name='mimic3' --variable='gender' <br />

**Run the predict.py file** <br />
python3 predict.py --data_file_test='/home/Data/mimic 3/ihm_test/test_gender.csv' --data_file_val='/home/Data/mimic 3/ihm_val/val_gender.csv' --max_token_length=300 --batch_size=1 --model_dir='/home/DifferentialImpact/SavedModel' --model_name='bert' --run='run1' --pred_dir='/home/DifferentialImpact/Prediction' --dataset_name='mimic3' --variable='gender' </br>

**Run the threshold.py file** <br />
Set the model_name, run, dataset, variable in the file <br />
Variable can be nodemography, gender, race, gender_race <br />
For example, <br />
model_name='bert' <br />
run='run1' <br />
dataset='mimic3' <br />
variable='gender' <br />

python3 threshold.py > /home/DifferentialImpact/ihm_result/bert_run1/mimic3/threshold_gender.txt <br />

**Run the write_label.py file** <br />
Set the model_name, run, dataset, variable in the file <br />
Set the threshold in the file; Threshold will be selected from the related threshold.txt file based on the highest balanced accuracy; For example, for gender variable, threshold_gender.txt file from the corresponding result folder will be used. <br />
model_name='bert' <br />
run='run1' <br />
dataset='mimic3' <br />
variable='gender' <br />

python3 write_label.py
