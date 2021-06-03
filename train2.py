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
from transformers import AutoTokenizer, AutoModel

from model import GRUBERT, GRUBERT2, LSTMBERT

ID_COL = 'Id'
TARGET_COL = 'Category'
SOURCE_COL = 'Sentence'




def prepare_parser():
    parser = argparse.ArgumentParser(description='DistilBERT with Bi-GRU')
    # Hyperparam
    parser.add_argument('--bert_ep', default=20, type=int, help='Number of epochs for BERT training')
    parser.add_argument('--gru_ep', default=20, type=int, help='Number of epochs for GRU training')
    parser.add_argument('--bs', default=256, type=int, help='Batch size')
    parser.add_argument('--bert_lr', default=1e-5, type=float, help='Bert learning rate')
    parser.add_argument('--gru_lr', default=1e-3, type=float, help='GRU learning rate')

    # Model Selection
    parser.add_argument('--out_dim', default=5, type=int, help='Dimension of output label')
    parser.add_argument('--pretrained_model', default='distilbert-base-cased', type=str, help='Pretrained model for sentence classification')
    parser.add_argument('--gru_only', action='store_true', help='Train only GRU')

    # Randomness
    parser.add_argument('--seed', default=2021, type=int, help='Random seed to reproduce prediction result')

    # Directories
    parser.add_argument('--train_data', default='train_final_clean.csv', type=str, help='XXX')
    parser.add_argument('--test_data', default='eval_final_clean.csv', type=str, help='XXX')
    parser.add_argument('--datadir', default='./dataset', type=str, help='XXX')
    parser.add_argument('--outdir', default='./models', type=str, help='XXX')
    parser.add_argument('--submitdir', default='./submissions', type=str, help='XXX')

    parser.add_argument('--debug', action='store_true', help='Debug mode')


    return parser

def seed_all(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    
    if torch.cuda.is_available(): 
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False



def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def accuracy(prediction, label):
    """
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    """
    prediction = torch.argmax(nn.functional.softmax(prediction, dim=1), dim=1)
    acc = torch.sum(prediction == label).float() / len(prediction == label)
    return acc

def train_fn(model, iterator, optimizer, criterion):
    
    epoch_loss = 0
    epoch_acc = 0
    
    model.train()
    # print(f'TRAIN: The model has {count_parameters(model):,} trainable parameters')

    for batch in iterator:
        
        optimizer.zero_grad()
        
        data = batch.Sentence
        label = batch.Category
        
        prediction = model(data)
        
        loss = criterion(prediction, label)
        
        acc = accuracy(prediction, label)
        
        loss.backward()
        
        optimizer.step()
        
        epoch_loss += loss.item()
        epoch_acc += acc.item()
        
    return epoch_loss / len(iterator), epoch_acc / len(iterator)

def eval_fn(model, iterator, criterion):
    
    epoch_loss = 0
    epoch_acc = 0
    
    model.eval()
    # print(f'EVAL: The model has {count_parameters(model):,} trainable parameters')
    with torch.no_grad():
    
        for batch in iterator:
            
            data = batch.Sentence
            label = batch.Category
            
            prediction = model(data)
            
            loss = criterion(prediction, label)
            
            acc = accuracy(prediction, label)

            epoch_loss += loss.item()
            epoch_acc += acc.item()
        
    return epoch_loss / len(iterator), epoch_acc / len(iterator)

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

def predict(model, iterator):
    
    model.eval()
    
    predictions = []
    
    with torch.no_grad():
    
        for batch in iterator:
            
            data = batch.Sentence
            
            prediction = model(data)
            
            prediction = torch.argmax(nn.functional.softmax(prediction, dim=1), dim=1)
            
            predictions.extend(prediction.tolist())
        
    return predictions

def run(config):
    print(config)

    seed_all(config['seed'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Currently running on:", device)

    tokenizer = AutoTokenizer.from_pretrained(config['pretrained_model'])

    def tokenize_and_cut(sentence):
        tokens = tokenizer.tokenize(sentence)
        max_input_length = tokenizer.max_model_input_sizes[config['pretrained_model']]
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

    LABEL = data.Field(sequential=False, use_vocab=False)

    
    train = pd.read_csv(os.path.join(config['datadir'], config['train_data']))
    test = pd.read_csv(os.path.join(config['datadir'], config['test_data']))
    train, valid = train_test_split(train, test_size=0.2)
    train.to_csv(os.path.join(config['datadir'],'temp_train.csv'), index=False)
    valid.to_csv(os.path.join(config['datadir'],'temp_val.csv'), index=False)

    train, valid = data.TabularDataset.splits(path=config['datadir'], train='temp_train.csv', validation='temp_val.csv', 
                                              format='csv', skip_header=True,
                                              fields=[(ID_COL, None), (TARGET_COL, LABEL), (SOURCE_COL, TEXT), ])
    test = data.TabularDataset(os.path.join(config['datadir'], config['test_data']), format='csv', skip_header=True,
                               fields=[(ID_COL, None), (SOURCE_COL, TEXT)])
    if config['debug']:
        print(f"Number of training examples: {len(train)}")
        print(f"Number of validation examples: {len(valid)}")
        print(f"Number of testing examples: {len(test)}")
        print(vars(train[0]))
        print(tokenizer.convert_ids_to_tokens(vars(train[0])[SOURCE_COL]))

    train_iter = data.BucketIterator(train, batch_size=config['bs'], shuffle=True, device=device)
    valid_iter = data.BucketIterator(valid, batch_size=config['bs'], shuffle=True, device=device)
    test_iter = data.Iterator(test, batch_size=config['bs'], train=False, sort=False, device=device)

    batch = next(iter(train_iter))
    sentence = batch.Sentence
    category = batch.Category
    if config['debug']:
        print(sentence.shape)
        print(sentence)
        print(category.shape)
        print(category)


    if config['gru_only']:
        model_name = f"model_{config['pretrained_model']}_gru_lr{config['gru_lr']}_bs{config['bs']}_ep{config['gru_ep']}.pt"
        model_fullname = f"model_full_{config['pretrained_model']}_gru_lr{config['gru_lr']}_bs{config['bs']}_ep{config['gru_ep']}.pt"
        submission_name = f"submission_{config['pretrained_model']}_gru_lr{config['gru_lr']}_bs{config['bs']}_ep{config['gru_ep']}.csv"
    else:
        model_name = f"model_{config['pretrained_model']}_bert_lr{config['bert_lr']}_gru_lr{config['gru_lr']}_bs{config['bs']}_ep{config['gru_ep']}.pt"
        model_fullname = f"model_full_{config['pretrained_model']}_bert_lr{config['bert_lr']}_gru_lr{config['gru_lr']}_bs{config['bs']}_ep{config['gru_ep']}.pt"
        submission_name = f"submission_{config['pretrained_model']}_bert_lr{config['bert_lr']}_gru_lr{config['gru_lr']}_bs{config['bs']}_ep{config['gru_ep']}.csv"


    bert = AutoModel.from_pretrained(config['pretrained_model'])
    model = GRUBERT(bert=bert, config=config).to(device)



    for name, param in model.named_parameters(): 
        if name.startswith('bert'):
            param.requires_grad = False # Lock BERT

        else:
            param.requires_grad = True
    
    optimizer = optim.AdamW(model.parameters(), lr=config['gru_lr'])
    criterion = nn.CrossEntropyLoss().to(device)

    if config['debug']:
        print(f'The model has {count_parameters(model):,} trainable parameters')
        for name, param in model.named_parameters():                
            if param.requires_grad:
                print(name)
    

    best_epoch = 0
    best_valid_loss = float('inf')
    best_valid_acc = 0.0

    print('training GRU')
    for epoch in range(config['gru_ep'] + config['bert_ep']):
        
        if epoch == config['gru_ep']:
            param_bert = []
            param_gru = []
            
            for name, param in model.named_parameters(): 
                if name.startswith('bert'):
                    param.requires_grad = True # Open BERT
                    param_bert.append(param)
                    
                else:
                    param.requires_grad = True
                    param_gru.append(param)

            optimizer = optim.AdamW([{'params': param_bert, 'lr': config['bert_lr']}, {'params': param_gru, 'lr': config['gru_lr']}])
            criterion = nn.CrossEntropyLoss().to(device)

            print('training GRU+BERT')
        

        start_time = time.time()
        
        train_loss, train_acc = train_fn(model, train_iter, optimizer, criterion)
        valid_loss, valid_acc = eval_fn(model, valid_iter, criterion)
            
        end_time = time.time()
            
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
            
        # if valid_loss < best_valid_loss:
        #     best_epoch = epoch + 1
        #     best_valid_loss = valid_loss
        #     torch.save(model.state_dict(), os.path.join(config['outdir'],model_name))
        
        if valid_acc > best_valid_acc:
            best_epoch = epoch + 1
            best_valid_acc = valid_acc
            torch.save(model.state_dict(), os.path.join(config['outdir'],model_name))

        print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%')


    train_full = data.TabularDataset(os.path.join(config['datadir'], config['train_data']), format='csv', skip_header=True,
                                     fields=[(ID_COL, None), (TARGET_COL, LABEL), (SOURCE_COL, TEXT)])
    train_full_iter = data.BucketIterator(train_full, batch_size=config['bs'], shuffle=True, device=device)

    
    bert = AutoModel.from_pretrained(config['pretrained_model'])
    model = GRUBERT(bert=bert, config=config).to(device)


    for name, param in model.named_parameters(): 
        if name.startswith('bert'):
            param.requires_grad = False # Lock BERT
        else:
            param.requires_grad = True
    
    optimizer = optim.AdamW(model.parameters(), lr=config['gru_lr'])
    criterion = nn.CrossEntropyLoss().to(device)
    
    print('training GRU')
    for epoch in range(config['gru_ep'] + config['bert_ep']):

        if epoch == config['gru_ep']:
            param_bert = []
            param_gru = []
            
            for name, param in model.named_parameters(): 
                if name.startswith('bert'):
                    param.requires_grad = True # Open BERT
                    param_bert.append(param)
                else:
                    param.requires_grad = True
                    param_gru.append(param)

            optimizer = optim.AdamW([{'params': param_bert, 'lr': config['bert_lr']}, {'params': param_gru, 'lr': config['gru_lr']}])
            criterion = nn.CrossEntropyLoss().to(device)

            print('training GRU+BERT')
        
        start_time = time.time()
        
        train_loss, train_acc = train_fn(model, train_full_iter, optimizer, criterion)
            
        end_time = time.time()
            
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
            
        if epoch == best_epoch:
            torch.save(model.state_dict(), os.path.join(config['outdir'], model_fullname))
        
        print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')


    model.eval()
    predictions = predict(model, test_iter)
    test = pd.read_csv(os.path.join(config['datadir'], config['test_data']))
    test[TARGET_COL] = predictions
    del test[SOURCE_COL]
    test.to_csv(os.path.join(config['submitdir'], submission_name), index=False)

if __name__ == "__main__":
    parser = prepare_parser()
    config = vars(parser.parse_args())

    run(config)

