import torch
import torch.nn as nn
import transformers
from peft import get_peft_model, LoraConfig

class LM_Arch(nn.Module):

    def __init__(self, lm_model):
        
        super(LM_Arch, self).__init__()
        
        self.bert = lm_model
      
        # dropout layer
        self.dropout = nn.Dropout(0.1)
          
        # relu activation function
        self.relu =  nn.ReLU()

        # dense layer 1
        self.fc1 = nn.Linear(768,128)
          
        # dense layer 2 (Output layer)
        self.fc2 = nn.Linear(128,1)

        #sigmoid activation function
        self.sigmoid = nn.Sigmoid()

    #define the forward pass
    def forward(self, sent_id, mask):
        
        #pass the inputs to the model
        _, cls_hs = self.bert(sent_id, attention_mask=mask)
        
        x = self.fc1(cls_hs)
        
        x = self.relu(x)
        
        x = self.dropout(x)
        
        # output layer
        x = self.fc2(x)
        
        # apply sigmoid activation
        x = self.sigmoid(x)
        
        return x


class LM_Arch_Lora(nn.Module):

    def __init__(self, lm_model):
        
        super(LM_Arch_Lora, self).__init__()

        # Define LoRA configuration
        lora_config = LoraConfig(
            r=32,  # rank of the low-rank matrices
            lora_alpha=32,  # scaling factor
            lora_dropout=0.1,  # dropout rate
            bias="none"  # bias type
        )
        
        self.bert = get_peft_model(lm_model, lora_config)
      
        # dropout layer
        self.dropout = nn.Dropout(0.1)
          
        # relu activation function
        self.relu =  nn.ReLU()

        # dense layer 1
        self.fc1 = nn.Linear(768,128)
          
        # dense layer 2 (Output layer)
        self.fc2 = nn.Linear(128,1)

        #sigmoid activation function
        self.sigmoid = nn.Sigmoid()

    #define the forward pass
    def forward(self, sent_id, mask):
        
        #pass the inputs to the model
        _, cls_hs = self.bert(sent_id, attention_mask=mask)
        
        x = self.fc1(cls_hs)
        
        x = self.relu(x)
        
        x = self.dropout(x)
        
        # output layer
        x = self.fc2(x)
        
        # apply sigmoid activation
        x = self.sigmoid(x)
        
        return x


class LM_Gator_Arch(nn.Module):

    def __init__(self, lm_model):
        
        super(LM_Gator_Arch, self).__init__()
        
        self.bert = lm_model
      
        # dropout layer
        self.dropout = nn.Dropout(0.1)
          
        # relu activation function
        self.relu =  nn.ReLU()

        # dense layer 1
        self.fc1 = nn.Linear(1024,128)
          
        # dense layer 2 (Output layer)
        self.fc2 = nn.Linear(128,1)

        #sigmoid activation function
        self.sigmoid = nn.Sigmoid()

    #define the forward pass
    def forward(self, sent_id, mask):
        
        #pass the inputs to the model
        _, cls_hs = self.bert(sent_id, attention_mask=mask)
        
        x = self.fc1(cls_hs)
        
        x = self.relu(x)
        
        x = self.dropout(x)
        
        # output layer
        x = self.fc2(x)
        
        # apply sigmoid activation
        x = self.sigmoid(x)
        
        return x


class LM_Gator_Arch_Lora(nn.Module):

    def __init__(self, lm_model):
        
        super(LM_Gator_Arch_Lora, self).__init__()

        # Define LoRA configuration
        lora_config = LoraConfig(
            r=32,  # rank of the low-rank matrices
            lora_alpha=32,  # scaling factor
            lora_dropout=0.1,  # dropout rate
            bias="none",  # bias type
            target_modules=[
                "query",
                "key",
                "value",
                ]
        )
        
        self.bert = get_peft_model(lm_model, lora_config)
      
        # dropout layer
        self.dropout = nn.Dropout(0.1)
          
        # relu activation function
        self.relu =  nn.ReLU()

        # dense layer 1
        self.fc1 = nn.Linear(1024,128)
          
        # dense layer 2 (Output layer)
        self.fc2 = nn.Linear(128,1)

        #sigmoid activation function
        self.sigmoid = nn.Sigmoid()

    #define the forward pass
    def forward(self, sent_id, mask):
        
        #pass the inputs to the model
        _, cls_hs = self.bert(sent_id, attention_mask=mask)
        
        x = self.fc1(cls_hs)
        
        x = self.relu(x)
        
        x = self.dropout(x)
        
        # output layer
        x = self.fc2(x)
        
        # apply sigmoid activation
        x = self.sigmoid(x)
        
        return x

class LM_Attn_Arch(nn.Module):

    def __init__(self, lm_model):
        
        super(LM_Attn_Arch, self).__init__()
        
        self.bert = lm_model
      
        # dropout layer
        self.dropout = nn.Dropout(0.1)
          
        # relu activation function
        self.relu =  nn.ReLU()

        # dense layer 1
        self.fc1 = nn.Linear(768,128)
          
        # dense layer 2 (Output layer)
        self.fc2 = nn.Linear(128,1)

        #sigmoid activation function
        self.sigmoid = nn.Sigmoid()

    #define the forward pass
    def forward(self, sent_id, mask):
        
        #pass the inputs to the model
        outputs = self.bert(sent_id, attention_mask=mask, output_attentions=True)
        last_hidden_state = outputs[0]  # Extract last_hidden_state
        cls_hs = last_hidden_state[:, 0, :]  # Get CLS token's hidden state
        attention_scores = outputs[2]  # Extract attention scores
        #_, cls_hs = self.bert(sent_id, attention_mask=mask)
        
        x = self.fc1(cls_hs)
        
        x = self.relu(x)
        
        x = self.dropout(x)
        
        # output layer
        x = self.fc2(x)
        
        # apply sigmoid activation
        x = self.sigmoid(x)
        
        return x, attention_scores