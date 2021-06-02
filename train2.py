
  



import os
import random
import time

import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import torch
from torch import nn
import torch.optim as optim
from torchtext import data, datasets

from transformers import BertTokenizer, BertModel
from transformers import DistilBertTokenizer, DistilBertModel
# from transformers import AutoTokenizer, AutoModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from model import GRUBERT, GRUBERT2, LSTMBERT

ID_COL = 'Id'
TARGET_COL = 'Category'
SOURCE_COL = 'Sentence'

embedding_dim = {
    'bert-base-cased': 'hidden_size',
    'bert-base-uncased': 'hidden_size',
    'distilbert-base-cased': 'dim',
    'distilbert-base-uncased': 'dim',
}
def predict(model, iterator):
    
    model.eval()
    
    predictions = []
    
    with torch.no_grad():
    
        for batch in iterator:
            
            data = batch.Sentence
            
            prediction = model(data)
            print(prediction.logits[1])
            prediction = torch.argmax(torch.Tensor(prediction.logits.cpu()), dim=1)
            
            predictions.extend(prediction.tolist())
        
    return predictions
  
tokenizer = AutoTokenizer.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")
model = AutoModelForSequenceClassification.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")

# tokenizer.to('cuda')
model.to('cuda')

# for name, param in model.named_parameters(): 
#     print(name)

def tokenize_and_cut(sentence):
        tokens = tokenizer.tokenize(sentence)
        max_input_length = 512
        tokens = tokens[:max_input_length-2]
        return tokens

TEXT = data.Field(batch_first = True,
                    use_vocab = False,
                    tokenize = tokenize_and_cut,
                    preprocessing = tokenizer.convert_tokens_to_ids,
                    init_token = tokenizer.cls_token_id,
                    eos_token = tokenizer.sep_token_id,
                    pad_token = tokenizer.pad_token_id,
                    unk_token = tokenizer.unk_token_id)

test = data.TabularDataset('dataset/eval_final_open.csv', format='csv', skip_header=True,
                               fields=[(ID_COL, None), (SOURCE_COL, TEXT)])
test_iter = data.Iterator(test, batch_size=32, train=False, sort=False, device='cuda')

model.eval()
predictions = predict(model, test_iter)
test = pd.read_csv('dataset/eval_final_open.csv')
test[TARGET_COL] = predictions
del test[SOURCE_COL]
test.to_csv(os.path.join('submission.csv'), index=False)