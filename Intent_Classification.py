import torch
import torch.nn as nn

import os
import pandas as pd
import numpy as np
import random
import time
import datetime
import urllib.request
import matplotlib.pyplot as plt

from tqdm import tqdm, trange
from transformers import BertModel
from transformers import BertTokenizer
from transformers import BertForSequenceClassification, AdamW, BertConfig
from transformers import get_linear_schedule_with_warmup
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler, random_split
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report
from tokenization_kobert import KoBertTokenizer

test_total = pd.read_csv('Test_total_label.txt', delimiter = ',', error_bad_lines=False)
EDA_10 = pd.read_csv('10EDA.txt', delimiter = ',', error_bad_lines=False)
EDA_20 = pd.read_csv('20EDA.txt', delimiter = ',', error_bad_lines=False)
EDA_30 = pd.read_csv('30EDA.txt', delimiter = ',', error_bad_lines=False)
EMB_10 = pd.read_csv('10EMB.txt', delimiter = ',', error_bad_lines=False)
EMB_20 = pd.read_csv('20EMB.txt', delimiter = ',', error_bad_lines=False)
EMB_30 = pd.read_csv('30EMB.txt', delimiter = ',', error_bad_lines=False)

test_total.columns=['question','label']
EDA_10.columns=['question','label']
EDA_20.columns=['question','label']
EDA_30.columns=['question','label']
EMB_10.columns=['question','label']
EMB_20.columns=['question','label']
EMB_30.columns=['question','label']

f_sang_read = open("kss_sang.txt", "r")
f_no_read = open("kss_no.txt", "r")
f_ju_read = open("kss_ju.txt", "r")

sang_line = f_sang_read.read().splitlines()
no_line = f_no_read.read().splitlines()
ju_line = f_ju_read.read().splitlines()

sang_train, sang_test = train_test_split(sang_line, test_size=0.2, shuffle=True, random_state=34)
no_train, no_test = train_test_split(no_line, test_size=0.2, shuffle=True, random_state=34)
ju_train, ju_test = train_test_split(ju_line, test_size=0.2, shuffle=True, random_state=34)

sang_train, sang_val = train_test_split(sang_train, test_size=0.2, shuffle=True, random_state=34)
no_train, no_val = train_test_split(no_train, test_size=0.2, shuffle=True, random_state=34)
ju_train, ju_val = train_test_split(ju_train, test_size=0.2, shuffle=True, random_state=34)

data = {
    'question' : [],
    'label' : []
}

ori = pd.DataFrame(data)

for i in sang_train:
    ori.loc[len(ori)] = [i,'ID']
    
for i in no_train:
    ori.loc[len(ori)] = [i,'ALM']
    
for i in ju_train:
    ori.loc[len(ori)] = [i,'SPD']

ori_question = ori.question
ori_label = ori.label

val = pd.DataFrame(data)

for i in sang_val:
    val.loc[len(val)] = [i,'ID']
    
for i in no_val:
    val.loc[len(val)] = [i,'ALM']
    
for i in ju_val:
    val.loc[len(val)] = [i,'SPD']

ori_question = ori.question
ori_label = ori.label
val_question = val.question
val_label = val.label

def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

def format_time(elapsed):
    elapsed_rounded = int(round((elapsed)))
    return str(datetime.timedelta(seconds=elapsed_rounded))

def decode_sentiment(label):
    decode_map = {"ALM" : 0, "ID": 1, "SPD": 2}
    return decode_map[str(label)]

def make_TensorDataset(train_df, val_df, test_df, tokenizer):
    MAX_LEN = 512
    BATCH_SIZE = 32
    train_input_ids, train_token_type_ids, train_attention_masks = [], [], []
    val_input_ids, val_token_type_ids, val_attention_masks = [], [], []
    test_input_ids, test_token_type_ids, test_attention_masks = [], [], []

    train_label = train_df["label"].apply(lambda x: decode_sentiment(x))
    val_label = val_df["label"].apply(lambda x: decode_sentiment(x))
    test_label = test_df["label"].apply(lambda x: decode_sentiment(x))
    
    for i in train_df.question:
        pair = tokenizer(i)
        train_input_ids.append(pair['input_ids'])
        train_token_type_ids.append(pair['token_type_ids'])
        train_attention_masks.append(pair['attention_mask'])
        
    for i in val_df.question:
        pair = tokenizer(i)
        val_input_ids.append(pair['input_ids'])
        val_token_type_ids.append(pair['token_type_ids'])
        val_attention_masks.append(pair['attention_mask'])

    for i in test_df.question:
        pair = tokenizer(i)
        test_input_ids.append(pair['input_ids'])
        test_token_type_ids.append(pair['token_type_ids'])
        test_attention_masks.append(pair['attention_mask'])

    train_input_ids = pad_sequences(train_input_ids, maxlen=MAX_LEN, dtype='long', truncating='post', padding='post')
    train_token_type_ids = pad_sequences(train_token_type_ids, maxlen=MAX_LEN, dtype='long', truncating='post', padding='post')
    train_attention_masks = pad_sequences(train_attention_masks, maxlen=MAX_LEN, dtype='long', truncating='post', padding='post')
    
    val_input_ids = pad_sequences(val_input_ids, maxlen=MAX_LEN, dtype='long', truncating='post', padding='post')
    val_token_type_ids = pad_sequences(val_token_type_ids, maxlen=MAX_LEN, dtype='long', truncating='post', padding='post')
    val_attention_masks = pad_sequences(val_attention_masks, maxlen=MAX_LEN, dtype='long', truncating='post', padding='post')

    test_input_ids = pad_sequences(test_input_ids, maxlen=MAX_LEN, dtype='long', truncating='post', padding='post')
    test_token_type_ids = pad_sequences(test_token_type_ids, maxlen=MAX_LEN, dtype='long', truncating='post', padding='post')
    test_attention_masks = pad_sequences(test_attention_masks, maxlen=MAX_LEN, dtype='long', truncating='post', padding='post')
    
    train_input_ids = torch.tensor(train_input_ids)
    train_token_type_ids = torch.tensor(train_token_type_ids)
    train_attention_masks = torch.tensor(train_attention_masks)
    train_label = train_label.reset_index(drop=True)
    train_label = torch.tensor(train_label)

    validation_input_ids = torch.tensor(val_input_ids)
    validation_token_type_ids = torch.tensor(val_token_type_ids)
    validation_attention_masks = torch.tensor(val_attention_masks)
    validation_label = val_label.reset_index(drop=True)
    validation_label = torch.tensor(validation_label)

    test_input_ids = torch.tensor(test_input_ids)
    test_token_type_ids = torch.tensor(test_token_type_ids)
    test_attention_masks = torch.tensor(test_attention_masks)
    test_label = test_label.reset_index(drop=True)
    test_label = torch.tensor(test_label)

    train_dataset = TensorDataset(train_input_ids, train_token_type_ids, train_attention_masks, train_label)
    train_dataloader = DataLoader(
            train_dataset,
            sampler = RandomSampler(train_dataset),
            batch_size = BATCH_SIZE
            )

    val_dataset = TensorDataset(validation_input_ids, validation_token_type_ids, validation_attention_masks, validation_label)
    validation_dataloader = DataLoader(
            val_dataset,
            sampler = SequentialSampler(val_dataset),
            batch_size = BATCH_SIZE
            )

    test_dataset = TensorDataset(test_input_ids, test_token_type_ids, test_attention_masks, test_label)
    test_dataloader = DataLoader(
            test_dataset,
            sampler = RandomSampler(test_dataset),
            batch_size = BATCH_SIZE
            )
    return train_dataloader, validation_dataloader, test_dataloader

def train(device, model, optimizer, scheduler, train_dataloader, validation_dataloader):
    step_cnt = 0
    total_train_loss = 0
    total_train_acc = 0
    nb_train_steps = 0
    best_valid_loss = 1

    epoch_iterator = tqdm(train_dataloader, desc="training")

    for step, batch in enumerate(epoch_iterator):
        model.train()
        model.zero_grad()

        step_cnt+=1

        batch = tuple(t.to(device) for t in batch)
        b_input_ids = batch[0].to(device)
        b_token_type_ids = batch[1].to(device)
        b_attention_mask = batch[2].to(device)
        b_label = batch[3].to(device)

        outputs = model(b_input_ids, 
                        token_type_ids=b_token_type_ids, 
                        attention_mask=b_attention_mask, 
                        labels=b_label)

        loss = outputs["loss"]
        logits = outputs["logits"]

        loss = loss.mean()
        total_train_loss += loss.item()
        loss.backward()

        logits = logits.detach().cpu().numpy()
        label_ids = b_label.to('cpu').numpy()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        tmp_train_accuracy = flat_accuracy(logits, label_ids)
        total_train_acc += tmp_train_accuracy
        nb_train_steps += 1

    avg_train_loss = total_train_loss / len(train_dataloader)
    avg_train_acc = total_train_acc / nb_train_steps
    print("==> Step {0}, Train acc : {1:.3f}, loss : {2:.3f}".format(step_cnt, avg_train_acc, avg_train_loss))
    avg_val_loss, avg_val_acc = evaluate(step_cnt, device, model, validation_dataloader)

    return avg_val_loss

def evaluate(step_cnt, device, model, validation_dataloader):
    model.eval()
    
    total_eval_loss, total_eval_accuracy = 0, 0
    nb_eval_steps = 0

    for batch in validation_dataloader:
        batch = tuple(t.to(device) for t in batch)

        b_input_ids = batch[0].to(device)
        b_token_type_ids = batch[1].to(device)
        b_attention_mask = batch[2].to(device)
        b_label = batch[3].to(device)

        with torch.no_grad():   
            outputs = model(b_input_ids, 
                        token_type_ids=b_token_type_ids, 
                        attention_mask=b_attention_mask, 
                        labels=b_label)

        loss = outputs["loss"]
        logits = outputs["logits"]

        loss = loss.mean()
        total_eval_loss += loss.item()

        logits = logits.detach().cpu().numpy()

        label_ids = b_label.to('cpu').numpy()

        tmp_eval_accuracy = flat_accuracy(logits, label_ids)
        total_eval_accuracy += tmp_eval_accuracy
        nb_eval_steps += 1

    avg_val_loss = total_eval_loss / len(validation_dataloader)
    avg_val_acc = total_eval_accuracy/nb_eval_steps

    print("==> Step {0}, Eval acc : {1:.3f}, loss : {2:.3f}".format(step_cnt, avg_val_acc, avg_val_loss))

    return avg_val_loss, avg_val_acc

def test(device, test_dataloader, file_name):
    model = torch.load(file_name)
    model.eval()
    
    all_predicts = []
    all_labels = []

    for batch in test_dataloader:
        batch = tuple(t.to(device) for t in batch)

        b_input_ids = batch[0].to(device)
        b_token_type_ids = batch[1].to(device)
        b_attention_mask = batch[2].to(device)
        b_label = batch[3].to(device)

        with torch.no_grad():   
            outputs = model(b_input_ids, 
                        token_type_ids=b_token_type_ids, 
                        attention_mask=b_attention_mask, 
                        labels=b_label)


        logits = outputs["logits"]
        logits = logits.detach().cpu().numpy()
        label_ids = b_label.to('cpu').numpy()
        
        all_predicts.extend([np.argmax(pp) for pp in logits])
        all_labels.extend(label_ids)
        
    print("Precision, Recall and F1-Score:\n\n", classification_report(all_labels, all_predicts, digits=3))
    
def Classification(model_file_name, model_name, train_df, val_df, test_df, epoch):
    best_valid_loss= 10
    epochs = epoch

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print('There are %d GPU(s) available.' % torch.cuda.device_count())
        print('We will use the GPU:', torch.cuda.get_device_name(0))
    else:
        device = torch.device("cpu")
        print('No GPU available, using the CPU instead.')
    
    if model_name == 'BERT':
        model = model = BertForSequenceClassification.from_pretrained("bert-base-multilingual-cased", num_labels=3)
        tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased', do_lower_case=False)

    elif model_name == 'KoBERT':
        model = BertForSequenceClassification.from_pretrained("monologg/kobert", num_labels=3)
        tokenizer = KoBertTokenizer.from_pretrained('monologg/kobert')

    model = nn.DataParallel(model)
    model.cuda()

    train_dataloader, validation_dataloader, test_dataloader = make_TensorDataset(train_df, val_df, test_df, tokenizer)
    
    optimizer = AdamW(model.parameters(),
                      lr = 2e-5, 
                      eps = 1e-6)

    total_steps = len(train_dataloader) * epochs

    scheduler = get_linear_schedule_with_warmup(optimizer, 
                                            num_warmup_steps = 0,
                                            num_training_steps = total_steps)

    break_count = 3
    for epoch in range(epochs):
        print('\n Epoch {:} / {:}'.format(epoch + 1, epochs))
        total_loss = train(device, model, optimizer, scheduler, train_dataloader, validation_dataloader)
        
        if best_valid_loss > total_loss:
            best_valid_loss = total_loss
            torch.save(model, model_file_name)
        else:
            break_count -= 1
            if break_count == 0:
                break

    test(device, test_dataloader, model_file_name)
    

Classification("BERT_Classification_original_epoch10_v1", 'BERT', ori, val, test_total, 10)
Classification("BERT_Classification_EDA10_epoch10_v1", 'BERT', EDA_10, val, test_total, 10)
Classification("BERT_Classification_EDA20_epoch10_v1", 'BERT', EDA_20, val, test_total, 10)
Classification("BERT_Classification_EDA30_epoch10_v1", 'BERT', EDA_30, val, test_total, 10)
Classification("BERT_Classification_EMB10_epoch10_v1", 'BERT', EMB_10, val, test_total, 10)
Classification("BERT_Classification_EMB20_epoch10_v1", 'BERT', EMB_20, val, test_total, 10)
Classification("BERT_Classification_EMB30_epoch10_v1", 'BERT', EMB_30, val, test_total, 10)
Classification("KoBERT_Classification_original_epoch10_v1", 'KoBERT', ori, val, test_total, 10)
Classification("KoBERT_Classification_EDA10_epoch10_v1", 'KoBERT', EDA_10, val, test_total, 10)
Classification("KoBERT_Classification_EDA20_epoch10_v1", 'KoBERT', EDA_20, val, test_total, 10)
Classification("KoBERT_Classification_EDA30_epoch10_v1", 'KoBERT', EDA_30, val, test_total, 10)
Classification("KoBERT_Classification_EMB10_epoch10_v1", 'KoBERT', EMB_10, val, test_total, 10)
Classification("KoBERT_Classification_EMB20_epoch10_v1", 'KoBERT', EMB_20, val, test_total, 10)
Classification("KoBERT_Classification_EMB30_epoch10_v1", 'KoBERT', EMB_30, val, test_total, 10)