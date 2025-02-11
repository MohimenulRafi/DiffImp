# DiffImp

## BERT Instruction ##

Run the bert.py file <br />
python3 bert.py --data_file_train='/home/Data/mimic 3/ihm_train/train_gender.csv' --data_file_val='/home/Data/mimic 3/ihm_val/val_gender.csv' --max_token_length=300 --batch_size=32 --epochs=10 --seed=42 --model_dir='/home/DifferentialImpact/SavedModel' --model_name='bert' --run='run1' --dataset_name='mimic3' --variable='gender' <br />

Run the predict.py file <br />
python3 predict.py --data_file_test='/home/Data/mimic 3/ihm_test/test_gender.csv' --data_file_val='/home/Data/mimic 3/ihm_val/val_gender.csv' --max_token_length=300 --batch_size=1 --model_dir='/home/DifferentialImpact/SavedModel' --model_name='bert' --run='run1' --pred_dir='/home/DifferentialImpact/Prediction' --dataset_name='mimic3' --variable='gender' </br>

