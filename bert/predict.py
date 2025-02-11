import argparse
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import transformers
from transformers import AutoModel, BertTokenizerFast, AutoTokenizer, AutoConfig

from model import LM_Arch, LM_Arch_Lora, LM_Gator_Arch, LM_Gator_Arch_Lora

from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

# Defining arguments
parser=argparse.ArgumentParser()

parser.add_argument('--data_file_test', required=True, help='data file path for test')
parser.add_argument('--data_file_val', required=True, help='data file path for validation')
parser.add_argument('--max_token_length', type=int, required=True, help='maximum length for tokens')
parser.add_argument('--batch_size', type=int, default=1, help='batch size for test')
parser.add_argument('--model_dir', required=True, help='path for saved model')
parser.add_argument('--model_name', required=True, help='name for the saved model')
parser.add_argument('--run', required=True, help='trial number; trial 1 can be run1')
parser.add_argument('--pred_dir', required=True, help='path for saving the predictions')
parser.add_argument('--dataset_name', default='foundation', help='name of the dataset')
parser.add_argument('--variable', required=True, help='variable name; no_demography, gender, race etc.')

args=parser.parse_args()

data_file_val=args.data_file_val
data_file_test=args.data_file_test

max_token_length=args.max_token_length

if not os.path.exists(args.model_dir):
    print('Provide the right path for the model')
#model_path=args.model_dir+'/'+args.model_name+'_'+args.dataset_name+'.pt'
model_path=args.model_dir+'/'+args.run+'/'+args.model_name+'_'+args.dataset_name+'_'+args.variable+'.pt'

#pred_path=args.pred_dir+'/'+args.model_name+'_'+args.dataset_name
pred_path=args.pred_dir+'/'+args.model_name+'_'+args.run+'/'+args.dataset_name+'/'+args.variable
if not os.path.exists(pred_path):
    os.makedirs(pred_path)

# specify GPU
device = torch.device('cuda')

# Read dataset
df_val=pd.read_csv(data_file_val)
df_test=pd.read_csv(data_file_test)
print(df_test.head())

# Train test split
val_text, val_labels = df_val['Text'], df_val['Class']
test_text, test_labels = df_test['Text'], df_test['Class']
#val_text, val_labels = df_val['Text'], df_val['y_true']
#test_text, test_labels = df_test['Text'], df_test['y_true']

# Defining parameters
token_max_length=max_token_length # 55 for wisconsin and pima

# import BERT-base pretrained model
bert = AutoModel.from_pretrained('bert-base-uncased', return_dict=False) # Added return_dict=False
#bert = AutoModel.from_pretrained('dmis-lab/biobert-base-cased-v1.2', return_dict=False) # Added return_dict=False
#bert = AutoModel.from_pretrained('UFNLP/gatortron-base', return_dict=False) # Added return_dict=False

# Load the BERT tokenizer
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
#tokenizer = AutoTokenizer.from_pretrained('dmis-lab/biobert-base-cased-v1.2')
#tokenizer = BertTokenizerFast.from_pretrained('UFNLP/gatortron-base')

# sample data
text = ["this is a bert model tutorial", "we will fine-tune a bert model"]

# encode text
sent_id = tokenizer.batch_encode_plus(text, padding=True)

# output
print(sent_id)

# tokenize and encode sequences in the validation set
tokens_val = tokenizer.batch_encode_plus(
    val_text.tolist(),
    max_length = token_max_length,
    pad_to_max_length=True,
    truncation=True
)

# tokenize and encode sequences in the test set
tokens_test = tokenizer.batch_encode_plus(
    test_text.tolist(),
    max_length = token_max_length,
    pad_to_max_length=True,
    truncation=True
)

## convert lists to tensors
val_seq = torch.tensor(tokens_val['input_ids'])
val_mask = torch.tensor(tokens_val['attention_mask'])
val_y = torch.tensor(val_labels.tolist())

test_seq = torch.tensor(tokens_test['input_ids'])
test_mask = torch.tensor(tokens_test['attention_mask'])
test_y = torch.tensor(test_labels.tolist())

from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

#define a batch size
batch_size = args.batch_size

# wrap tensors
val_data = TensorDataset(val_seq, val_mask, val_y)

# sampler for sampling the data during training
val_sampler = SequentialSampler(val_data)

# dataloader for tuning the threshold
val_dataloader_tr = DataLoader(val_data, sampler = val_sampler, batch_size=1)

# For test data - Mohimenul
# wrap tensors
test_data = TensorDataset(test_seq, test_mask, test_y)

# sampler for sampling the data during training
test_sampler = SequentialSampler(test_data)

# dataLoader for validation set
test_dataloader = DataLoader(test_data, sampler = test_sampler, batch_size=1)


# pass the pre-trained BERT to our define architecture
model = LM_Arch(bert)
#model = LM_Arch_Lora(bert)
#model = LM_Gator_Arch(bert)
#model = LM_Gator_Arch_Lora(bert)

# push the model to GPU
model = model.to(device)

# Define test function - Mohimenul
def test(input_dataloader):
    x_dataloader=input_dataloader
    print("\nTesting...")
  
    # deactivate dropout layers
    #model.eval()
  
    # empty list to save the model predictions
    all_preds = []

    # iterate over batches
    for step,batch in enumerate(x_dataloader):
        
        # Progress update every 50 batches.
        if step % 50 == 0 and not step == 0:  
            # Report progress.
            print('  Batch {:>5,}  of  {:>5,}.'.format(step, len(test_dataloader)))

        # push the batch to gpu
        batch = [t.to(device) for t in batch]

        sent_id, mask, labels = batch

        # deactivate autograd
        with torch.no_grad():
            # model predictions
            preds = model(sent_id, mask)
            preds = preds.detach().cpu().numpy()

            all_preds.append(preds)

    # reshape the predictions in form of (number of samples, no. of classes)
    all_preds  = np.concatenate(all_preds, axis=0)

    return all_preds

#load weights of best model
path = model_path
model.load_state_dict(torch.load(path))

# get predictions for validation data
validation_preds=test(val_dataloader_tr)

fw=open(pred_path+'/valid_prediction_probs.txt', 'w')
'''for pred in preds:
    fw.write(str(pred[0])+'\n')'''
for i in range(len(validation_preds)):
    fw.write(str(validation_preds[i][0])+' '+str(val_y[i].item())+'\n')
fw.close()

# get predictions for test data
preds=test(test_dataloader)

fw=open(pred_path+'/prediction_probs.txt', 'w')
'''for pred in preds:
    fw.write(str(pred[0])+'\n')'''
for i in range(len(preds)):
    fw.write(str(preds[i][0])+' '+str(test_y[i].item())+'\n')
fw.close()

'''fw=open(pred_path+'/prediction_probs_label_text.txt', 'w')
#for pred in preds:
#    fw.write(str(pred[0])+'\n')
for i in range(len(preds)):
    fw.write(str(preds[i][0])+' '+str(test_y[i].item())+' '+str(test_labels[i])+' '+test_text[i]+'\n')
fw.close()'''

preds = preds.round()
print(classification_report(test_y, preds))

test_fpr, test_tpr, te_thresholds = roc_curve(test_y, preds)
test_auc = auc(test_fpr, test_tpr)

accuracy=accuracy_score(test_y, preds.round())
precision=precision_score(test_y, preds.round())
recall=recall_score(test_y, preds.round())
f1=f1_score(test_y, preds.round())

print(f'Accuracy: {accuracy:.3f}')
print(f'Precision: {precision:.3f}')
print(f'Recall: {recall:.3f}')
print(f'F1: {f1:.3f}')
print(f'AUC: {test_auc:.3f}')

