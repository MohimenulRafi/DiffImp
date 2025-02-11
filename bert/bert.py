import argparse
import os

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import transformers
from transformers import AutoModel, BertTokenizerFast

from model import LM_Arch

# Defining arguments
parser=argparse.ArgumentParser()

parser.add_argument('--data_file_train', required=True, help='data file path for training')
parser.add_argument('--data_file_val', required=True, help='data file path for validation')
parser.add_argument('--max_token_length', type=int, required=True, help='maximum length for tokens')
parser.add_argument('--batch_size', type=int, default=32, help='batch size for training and validation')
parser.add_argument('--epochs', type=int, default=10, help='number of training epochs')
parser.add_argument('--seed', type=int, default=42, help='set the seed')
parser.add_argument('--model_dir', required=True, help='path for saving the trained model')
parser.add_argument('--model_name', required=True, help='name for the saved model')
parser.add_argument('--run', required=True, help='trial number; trial 1 can be run1')
#parser.add_argument('--freeze', action='store_true', help='freeze the lm parameters')
#parser.add_argument('--pred_dir', required=True, help='path for saving the predictions')
parser.add_argument('--dataset_name', default='foundation', help='name of the dataset')
parser.add_argument('--variable', required=True, help='variable name; no_demography, gender, race etc.')

args=parser.parse_args()

data_file_train=args.data_file_train
data_file_val=args.data_file_val

max_token_length=args.max_token_length
seed=args.seed # used 42

if not os.path.exists(args.model_dir):
    os.makedirs(args.model_dir)
#model_path=args.model_dir+'/'+args.model_name+'_'+args.dataset_name+'.pt'
model_path=args.model_dir+'/'+args.run+'/'+args.model_name+'_'+args.dataset_name+'_'+args.variable+'.pt'

#freeze=args.freeze

'''pred_path=args.pred_dir+'/'+args.model_name+'_'+args.dataset_name
if not os.path.exists(pred_path):
    os.makedirs(pred_path)'''

torch.manual_seed(seed)

# specify GPU
device = torch.device('cuda')

# Read dataset
df_train=pd.read_csv(data_file_train)
df_val=pd.read_csv(data_file_val)
print(df_train.head())

# Train test split
train_text, train_labels = df_train['Text'], df_train['Class']
val_text, val_labels = df_val['Text'], df_val['Class']
#train_text, train_labels = df_train['Text'], df_train['y_true']
#val_text, val_labels = df_val['Text'], df_val['y_true']

# Defining parameters
token_max_length=max_token_length # 55 for wisconsin and pima

# import BERT-base pretrained model
bert = AutoModel.from_pretrained('bert-base-uncased', return_dict=False) # Added return_dict=False

# Load the BERT tokenizer
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

# sample data
text = ["this is a bert model tutorial", "we will fine-tune a bert model"]

# encode text
sent_id = tokenizer.batch_encode_plus(text, padding=True)

# output
print(sent_id)

# tokenize and encode sequences in the training set
tokens_train = tokenizer.batch_encode_plus(
    train_text.tolist(),
    max_length = token_max_length,
    pad_to_max_length=True,
    truncation=True
)

# tokenize and encode sequences in the validation set
tokens_val = tokenizer.batch_encode_plus(
    val_text.tolist(),
    max_length = token_max_length,
    pad_to_max_length=True,
    truncation=True
)


## convert lists to tensors

train_seq = torch.tensor(tokens_train['input_ids'])
train_mask = torch.tensor(tokens_train['attention_mask'])
train_y = torch.tensor(train_labels.tolist())

val_seq = torch.tensor(tokens_val['input_ids'])
val_mask = torch.tensor(tokens_val['attention_mask'])
val_y = torch.tensor(val_labels.tolist())

from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

#define a batch size
batch_size = args.batch_size

# wrap tensors
train_data = TensorDataset(train_seq, train_mask, train_y)

# sampler for sampling the data during training
train_sampler = RandomSampler(train_data)

# dataLoader for train set
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

# wrap tensors
val_data = TensorDataset(val_seq, val_mask, val_y)

# sampler for sampling the data during training
val_sampler = SequentialSampler(val_data)

# dataLoader for validation set
val_dataloader = DataLoader(val_data, sampler = val_sampler, batch_size=batch_size)

# dataloader for tuning the threshold
val_dataloader_tr = DataLoader(val_data, sampler = val_sampler, batch_size=1)

# freeze all the parameters
'''for param in bert.parameters():
    param.requires_grad = False'''

model = LM_Arch(bert)

# push the model to GPU
model = model.to(device)

# optimizer from hugging face transformers
from transformers import AdamW

# define the optimizer
optimizer = AdamW(model.parameters(), lr = 1e-5) # learning rate

from sklearn.utils.class_weight import compute_class_weight

#compute the class weights
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(train_labels), y=train_labels)

print("Class Weights:",class_weights)

# converting list of class weights to a tensor
weights= torch.tensor(class_weights,dtype=torch.float)

# push to GPU
weights = weights.to(device)

# define the loss function
cross_entropy  = nn.BCELoss()

# number of training epochs
epochs = args.epochs

# function to train the model
def train():
  
    model.train()

    total_loss, total_accuracy = 0, 0

    # empty list to save model predictions
    total_preds=[]
  
    # iterate over batches
    for step,batch in enumerate(train_dataloader):
    
        # progress update after every 50 batches.
        if step % 50 == 0 and not step == 0:
            print('  Batch {:>5,}  of  {:>5,}.'.format(step, len(train_dataloader)))

        # push the batch to gpu
        batch = [r.to(device) for r in batch]

        sent_id, mask, labels = batch
        #print(sent_id, mask)

        # clear previously calculated gradients 
        model.zero_grad()        

        # get model predictions for the current batch
        preds = model(sent_id, mask)
        labels = labels.unsqueeze(1)
        labels = labels.float()
        #print(labels)
        #print(preds)

        # compute the loss between actual and predicted values
        loss = cross_entropy(preds, labels)

        # add on to the total loss
        total_loss = total_loss + loss.item()

        # backward pass to calculate the gradients
        loss.backward()

        # clip the the gradients to 1.0. It helps in preventing the exploding gradient problem
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # update parameters
        optimizer.step()

        # model predictions are stored on GPU. So, push it to CPU
        preds=preds.detach().cpu().numpy()

        # append the model predictions
        total_preds.append(preds)

    # compute the training loss of the epoch
    avg_loss = total_loss / len(train_dataloader)

    # predictions are in the form of (no. of batches, size of batch, no. of classes).
    # reshape the predictions in form of (number of samples, no. of classes)
    total_preds  = np.concatenate(total_preds, axis=0)

    #returns the loss and predictions
    return avg_loss, total_preds

# function for evaluating the model
def evaluate():
  
    print("\nEvaluating...")

    # deactivate dropout layers
    model.eval()

    total_loss, total_accuracy = 0, 0

    # empty list to save the model predictions
    total_preds = []

    # iterate over batches
    for step,batch in enumerate(val_dataloader):
        
        # Progress update every 50 batches.
        if step % 50 == 0 and not step == 0:
          
            # Calculate elapsed time in minutes.
            #elapsed = format_time(time.time() - t0)
                
            # Report progress.
            print('  Batch {:>5,}  of  {:>5,}.'.format(step, len(val_dataloader)))

        # push the batch to gpu
        batch = [t.to(device) for t in batch]

        sent_id, mask, labels = batch

        # deactivate autograd
        with torch.no_grad():
          
            # model predictions
            preds = model(sent_id, mask)
            labels = labels.unsqueeze(1)
            labels = labels.float()

            # compute the validation loss between actual and predicted values
            loss = cross_entropy(preds,labels)

            total_loss = total_loss + loss.item()

            preds = preds.detach().cpu().numpy()

            total_preds.append(preds)

    # compute the validation loss of the epoch
    avg_loss = total_loss / len(val_dataloader) 

    # reshape the predictions in form of (number of samples, no. of classes)
    total_preds  = np.concatenate(total_preds, axis=0)

    return avg_loss, total_preds


# set initial loss to infinite
best_valid_loss = float('inf')

# empty lists to store training and validation loss of each epoch
train_losses=[]
valid_losses=[]

#for each epoch
for epoch in range(epochs):
     
    print('\n Epoch {:} / {:}'.format(epoch + 1, epochs))

    #train model
    train_loss, _ = train()

    #evaluate model
    valid_loss, _ = evaluate()

    #save the best model
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), model_path)

    # append training and validation loss
    train_losses.append(train_loss)
    valid_losses.append(valid_loss)

    print(f'\nTraining Loss: {train_loss:.3f}')
    print(f'Validation Loss: {valid_loss:.3f}')

