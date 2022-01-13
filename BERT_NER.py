import torch
import torch.nn as nn
import tensorflow as tf
import torch.nn.functional as F
import os
import pandas as pd
import numpy as np
import random
import time
import datetime
import matplotlib.pyplot as plt
import re

from tqdm import tqdm, trange

from transformers import BertModel
from transformers import BertTokenizer
from transformers import BertForSequenceClassification, BertForTokenClassification, AdamW, BertConfig
from transformers import get_linear_schedule_with_warmup
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report
import urllib.request
from tokenization_kobert import KoBertTokenizer
from sklearn import preprocessing
from sklearn.preprocessing import MultiLabelBinarizer

# It's omitted due to security issues.
da_who_list = []
da_when_list = []
da_where_list = []
da_no_list = []
da_action_list = []

test = pd.read_csv('Test_total_label.txt', delimiter = ',', error_bad_lines=False)
EDA_10 = pd.read_csv('10EDA.txt', delimiter = ',', error_bad_lines=False)
EDA_20 = pd.read_csv('20EDA.txt', delimiter = ',', error_bad_lines=False)
EDA_30 = pd.read_csv('30EDA.txt', delimiter = ',', error_bad_lines=False)
EMB_10 = pd.read_csv('10EMB.txt', delimiter = ',', error_bad_lines=False)
EMB_20 = pd.read_csv('20EMB.txt', delimiter = ',', error_bad_lines=False)
EMB_30 = pd.read_csv('30EMB.txt', delimiter = ',', error_bad_lines=False)

test.columns=['question','label']
EDA_10.columns=['question','label']
EDA_20.columns=['question','label']
EDA_30.columns=['question','label']
EMB_10.columns=['question','label']
EMB_20.columns=['question','label']
EMB_30.columns=['question','label']

test_question = test.question
test_label = test.label

EDA_10_question = EDA_10.question
EDA_10_label = EDA_10.label
EDA_20_question = EDA_20.question
EDA_20_label = EDA_20.label
EDA_30_question = EDA_30.question
EDA_30_label = EDA_30.label

EMB_10_question = EMB_10.question
EMB_10_label = EMB_10.label
EMB_20_question = EMB_20.question
EMB_20_label = EMB_20.label
EMB_30_question = EMB_30.question
EMB_30_label = EMB_30.label

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

val_question = val.question
val_label = val.label

def tagging_NER(question):
    qquestion = question.copy()
    for i in range(len(question)):
        bb = []
        for j in question[i].split():
            if j in da_who_list:
                bb.append("<" + j + ":WHO>")
            elif j in da_when_list:
                bb.append("<" + j + ":WEN>")
            elif j in da_where_list:
                bb.append("<" + j + ":WEE>")
            elif j in da_no_list:
                bb.append("<" + j + ":NOO>")
            elif j in da_action_list:
                bb.append("<" + j + ":ACT>")
            else :
                bb.append(j)
        cc = " ".join(bb)
        qquestion[i] = cc
        
    return qquestion

EDA_10_question_target = tagging_NER(EDA_10_question)
EDA_20_question_target = tagging_NER(EDA_20_question)
EDA_30_question_target = tagging_NER(EDA_30_question)
EMB_10_question_target = tagging_NER(EMB_10_question)
EMB_20_question_target = tagging_NER(EMB_20_question)
EMB_30_question_target = tagging_NER(EMB_30_question)
test_question_target = tagging_NER(test_question)
ori_question_target = tagging_NER(ori_question)
val_question_target = tagging_NER(val_question)

tag_dict = {'[PAD]' : 0, '[CLS]': 1, '[SEP]': 2, 'O' : 3,
            'B-ACT': 4, 'I-ACT': 5, 'B-WHO': 6, 'I-WHO': 7,
            'B-WEN': 8, 'I-WEN': 9, 'B-WEE': 10, 'I-WEE': 11,
            'B-NOO': 12, 'I-NOO': 13}

tag_dict_decode = {v: k for k, v in tag_dict.items()}

def make_TensorDataset(intent_train, intent_train_target, intent_val, intent_val_target, tokenizer):
    MAX_LEN = 512
    BATCH_SIZE = 16
    
    train_input_ids, train_attention_masks = [], []
    val_input_ids, val_attention_masks = [], []
    train_labels_ids, val_abels_ids = [], []
    
    for i in intent_train:
        pair = tokenizer(i)
        train_input_ids.append(pair['input_ids'])
        train_attention_masks.append(pair['attention_mask'])
        
    for i in intent_val:
        pair = tokenizer(i)
        val_input_ids.append(pair['input_ids'])
        val_attention_masks.append(pair['attention_mask'])

    train_ner_label_list_result = preprocessing_target(intent_train, intent_train_target, tokenizer)
    for label_list in train_ner_label_list_result:
        labels_ids_encoding = [tag_dict.get(x) for x in label_list]
        train_labels_ids.append(labels_ids_encoding)
        
    val_ner_label_list_result = preprocessing_target(intent_val, intent_val_target, tokenizer)
    for label_list in val_ner_label_list_result:
        labels_ids_encoding = [tag_dict.get(x) for x in label_list]
        val_abels_ids.append(labels_ids_encoding)
    
    train_input_ids = pad_sequences(train_input_ids, maxlen=MAX_LEN, dtype='long', truncating='post', padding='post')
    train_attention_masks = pad_sequences(train_attention_masks, maxlen=MAX_LEN, dtype='long', truncating='post', padding='post')
    train_labels_ids = pad_sequences(train_labels_ids, maxlen=MAX_LEN, dtype='long', truncating='post', padding='post')
    
    val_input_ids = pad_sequences(val_input_ids, maxlen=MAX_LEN, dtype='long', truncating='post', padding='post')
    val_attention_masks = pad_sequences(val_attention_masks, maxlen=MAX_LEN, dtype='long', truncating='post', padding='post')
    val_abels_ids = pad_sequences(val_abels_ids, maxlen=MAX_LEN, dtype='long', truncating='post', padding='post')
    
    train_input_ids = torch.tensor(train_input_ids)
    train_attention_masks = torch.tensor(train_attention_masks)
    train_labels_ids = torch.tensor(train_labels_ids)

    validation_input_ids = torch.tensor(val_input_ids)
    validation_attention_masks = torch.tensor(val_attention_masks)
    val_labels_ids = torch.tensor(val_abels_ids)

    train_dataset = TensorDataset(train_input_ids, train_attention_masks, train_labels_ids)
    train_dataloader = DataLoader(
            train_dataset,
            sampler = RandomSampler(train_dataset),
            batch_size = BATCH_SIZE
            )

    val_dataset = TensorDataset(validation_input_ids, validation_attention_masks, val_labels_ids)
    validation_dataloader = DataLoader(
            val_dataset,
            sampler = SequentialSampler(val_dataset),
            batch_size = BATCH_SIZE
            )

    return train_dataloader, validation_dataloader

def preprocessing_target(input_list, target_list, tokenizer):
    tokenized_target_text = [tokenizer.tokenize(text_temp) for text_temp in target_list]
    tokenized_texts_withoutCLS = [tokenizer.tokenize(text) for text in input_list]

    new_result_target_list = matching_input_target(tokenized_texts_withoutCLS, tokenized_target_text)
    prefix_sum_of_token_start_index_list, list_of_ner_tag_list, list_of_ner_text_list, list_of_tuple_ner_start_end_list = searching_boundary(tokenized_texts_withoutCLS, new_result_target_list)
    list_of_ner_label_list = tagging_label(tokenized_texts_withoutCLS, prefix_sum_of_token_start_index_list, list_of_ner_tag_list, list_of_ner_text_list, list_of_tuple_ner_start_end_list)
    ner_label_list_result = append_CLS_SEP(list_of_ner_label_list)
    return ner_label_list_result

def matching_input_target(tokenized_texts_withoutCLS, tokenized_target_text):
    result_string_list_input = []
    for target_token in tokenized_texts_withoutCLS:
        result_string = ""
        for tkd in target_token:
            result_string += tkd
        result_string_list_input.append(result_string)
    result_string_list = []
    
    for target_token in tokenized_target_text:
        result_string = ""
        for tkd in target_token:
            result_string += tkd
        result_string_list.append(result_string)

    regex_ner = re.compile('<(.+?):[A-Z#]{3,5}>')

    list_of_ner_all_list = []
    list_of_ner_text_list = []

    for result_string in result_string_list:
        filterd_by_regex = regex_ner.finditer(result_string)
        list_of_ner_all = []
        list_of_ner_text = []

        for match_item in filterd_by_regex:
            ner_all = match_item[0]
            ner_text = match_item[1]

            list_of_ner_all.append(ner_all)
            list_of_ner_text.append(ner_text)

        list_of_ner_all_list.append(list_of_ner_all)
        list_of_ner_text_list.append(list_of_ner_text)

    result_target_list = result_string_list_input

    new_result_target_list = []
    for target, all, text in zip(result_target_list, list_of_ner_all_list, list_of_ner_text_list):
        is_same = ""
        for a, t in zip(all, text):
            if t == is_same:
                continue
            else:
                target = target.replace(t, a)
                is_same = t
                
        total_target=target.replace('##', '##')
        new_result_target_list.append(total_target)
    return new_result_target_list

def searching_boundary(tokenized_texts_withoutCLS, new_result_target_list):
    prefix_sum_of_token_start_index_list = []
    for tokenized_text_w in tokenized_texts_withoutCLS:
        prefix_sum_of_token_start_index = []
        sum = 0
        for i, token in enumerate(tokenized_text_w):
            if i == 0:
                prefix_sum_of_token_start_index.append(0)
                sum += len(token)
            else:
                prefix_sum_of_token_start_index.append(sum)
                sum += len(token)

        prefix_sum_of_token_start_index_list.append(prefix_sum_of_token_start_index)

    regex_ner = re.compile('<(.+?):[A-Z#]{3,5}>')

    list_of_ner_tag_list = []
    list_of_ner_text_list = []
    list_of_tuple_ner_start_end_list = []

    for result_string in new_result_target_list:
        filterd_by_regex = regex_ner.finditer(result_string)

        list_of_ner_tag = []
        list_of_ner_text = []
        list_of_tuple_ner_start_end = []

        count_of_match = 0

        for match_item in filterd_by_regex:
            
            ner_tag_before = match_item[0].replace("#","")
            ner_tag = ner_tag_before[-4:-1]
            
            ner_text = match_item[1]  
            start_index = match_item.start() - 8 * count_of_match 
            end_index = match_item.end() - 8 - 8 * count_of_match

            list_of_ner_tag.append(ner_tag)
            list_of_ner_text.append(ner_text)
            list_of_tuple_ner_start_end.append((start_index, end_index))
            count_of_match += 1

        list_of_ner_tag_list.append(list_of_ner_tag)
        list_of_ner_text_list.append(list_of_ner_text)
        list_of_tuple_ner_start_end_list.append(list_of_tuple_ner_start_end)

    return prefix_sum_of_token_start_index_list, list_of_ner_tag_list, list_of_ner_text_list, list_of_tuple_ner_start_end_list

def tagging_label(tokenized_texts_withoutCLS, prefix_sum_of_token_start_index_list, list_of_ner_tag_list, list_of_ner_text_list, list_of_tuple_ner_start_end_list):
    list_of_ner_label_list = []
    for i in range(0, len(tokenized_texts_withoutCLS)):
        list_of_ner_label = []
        entity_index = 0
        is_entity_still_B = True
        for tup in zip(tokenized_texts_withoutCLS[i], prefix_sum_of_token_start_index_list[i]): 
            token, index = tup
            if entity_index < len(list_of_tuple_ner_start_end_list[i]): 
                start, end = list_of_tuple_ner_start_end_list[i][entity_index] 
                if end <= index:  
                    is_entity_still_B = True
                    entity_index = entity_index + 1 if entity_index + 1 < len(list_of_tuple_ner_start_end_list[i]) else entity_index 
                    start, end = list_of_tuple_ner_start_end_list[i][entity_index]

                if start <= index and index < end: 
                    entity_tag = list_of_ner_tag_list[i][entity_index] 
                    if is_entity_still_B is True: 
                        entity_tag = 'B-' + entity_tag 
                        list_of_ner_label.append(entity_tag) 
                        is_entity_still_B = False 
                    else: 
                        entity_tag = 'I-' + entity_tag 
                        list_of_ner_label.append(entity_tag) 
                else:
                    is_entity_still_B = True 
                    entity_tag = 'O' 
                    list_of_ner_label.append(entity_tag) 
            else:
                entity_tag = 'O'
                list_of_ner_label.append(entity_tag)
        list_of_ner_label_list.append(list_of_ner_label)
    return list_of_ner_label_list

def append_CLS_SEP(list_of_ner_label_list):
    for ner_label in list_of_ner_label_list:
        ner_label.append("[SEP]")
        ner_label.insert(0, "[CLS]")
    return list_of_ner_label_list

def train(device, model, optimizer, scheduler, train_dataloader, validation_dataloader):
    step_cnt = 0
    total_train_loss = 0
    nb_train_steps = 0
    best_valid_loss = 1

    epoch_iterator = tqdm(train_dataloader, desc="training")

    for step, batch in enumerate(epoch_iterator):
        model.train()
        model.zero_grad()

        step_cnt+=1

        batch = tuple(t.to(device) for t in batch)
        b_input_ids = batch[0].to(device)
        b_attention_mask = batch[1].to(device)
        b_label = batch[2].to(device)

        outputs = model(b_input_ids, 
                        token_type_ids=None, 
                        attention_mask=b_attention_mask, 
                        labels=b_label)

        #loss = outputs["loss"]
        #logits = outputs["logits"]
        loss = outputs[0]
        logits = outputs[1]

        loss = loss.mean()
        total_train_loss += loss.item()
        loss.backward()

        logits = logits.detach().cpu().numpy()
        label_ids = b_label.to('cpu').numpy()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        nb_train_steps += 1

    avg_train_loss = total_train_loss / len(train_dataloader)
    print("==> Step {0}, Train_loss : {1:.3f}".format(step_cnt, avg_train_loss))
    avg_val_loss = evaluate(step_cnt, device, model, validation_dataloader)

    return avg_val_loss

def evaluate(step_cnt, device, model, validation_dataloader):
    model.eval()
    
    total_eval_loss = 0
    nb_eval_steps = 0

    for batch in validation_dataloader:
        batch = tuple(t.to(device) for t in batch)

        b_input_ids = batch[0].to(device)
        b_attention_mask = batch[1].to(device)
        b_label = batch[2].to(device)

        with torch.no_grad():   
            outputs = model(b_input_ids, 
                        token_type_ids=None, 
                        attention_mask=b_attention_mask, 
                        labels=b_label)

        #loss = outputs["loss"]
        #logits = outputs["logits"]
        loss = outputs[0]
        logits = outputs[1]

        loss = loss.mean()
        total_eval_loss += loss.item()

        logits = logits.detach().cpu().numpy()

        label_ids = b_label.to('cpu').numpy()

        nb_eval_steps += 1

    avg_val_loss = total_eval_loss / len(validation_dataloader)

    print("==> Step {0}, Eval loss : {1:.3f}".format(step_cnt, avg_val_loss))

    return avg_val_loss

def test(device, file_name, tokenizer, intent_test, intent_test_target):
    model = torch.load(file_name)
    model.eval()
    
    test_input_ids, test_attention_masks = [], []
    test_labels_ids = []
    pred_labels = []
    
    for i in intent_test:
        pair = tokenizer(i)
        test_input_ids.append(pair['input_ids'])
        test_attention_masks.append(pair['attention_mask'])

    test_target = preprocessing_target(intent_test, intent_test_target, tokenizer)
    
    for i in range(0, len(test_question)):
        stopFlag = False
        test_inputs = torch.tensor([test_input_ids[i]])
        test_masks = torch.tensor([test_attention_masks[i]])

        b_input_ids = test_inputs.to(device)
        b_input_mask = test_masks.to(device)
        
        with torch.no_grad():  
            outputs = model(b_input_ids, 
                        token_type_ids=None, 
                        attention_mask=b_input_mask)
            
        logits = np.argmax(outputs[0].to('cpu').numpy(), axis=2)
        tokens = tokenizer.convert_ids_to_tokens(b_input_ids.to('cpu').numpy()[0])
        new_labels = []
        
        for token, label_idx in zip(tokens, logits[0]):
            if stopFlag == True:
                break
            if token == '[SEP]':
                stopFlag = True
            new_labels.append(tag_dict_decode[label_idx])
        pred_labels.append(new_labels)
        
    num, total = 0, 0
    for i in range(0, len(pred_labels)):
        start_index = -1
        end_index = -1
        for t in range(1, len(pred_labels[i])-1):
            if pred_labels[i][t][0] == "B" and start_index == -1:
                start_index = t
            elif start_index != -1 and pred_labels[i][t][0] == "O":
                end_index = t
                if test_target[i][start_index:end_index] == pred_labels[i][start_index:end_index]:
                    num += 1
                total += 1
                start_index = -1
            elif start_index != -1 and pred_labels[i][t][0] == "B":
                end_index = t
                if test_target[i][start_index:end_index] == pred_labels[i][start_index:end_index]:
                    num += 1
                total += 1
                start_index = t
    
    lb = MultiLabelBinarizer()
    aa = lb.fit_transform(test_target)
    bb = lb.transform(pred_labels)

    tagset = set(lb.classes_) - {'O'}
    tagset = sorted(tagset, key=lambda tag: tag.split('-', 1)[::-1])
    class_indices = {cls: idx for idx, cls in enumerate(lb.classes_)}
    
    print("Precision, Recall and F1-Score:\n\n", classification_report(aa, bb, labels = [class_indices[cls] for cls in tagset], target_names = tagset, digits=3))
    print("Accuracy : ", num / total * 100, "%")
    print()
    
    
def BERT_NER(model_file_name, model_name, intent_train, intent_train_target, intent_val, intent_val_target, intent_test, intent_test_target, epoch):
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
        model = model = BertForTokenClassification.from_pretrained("bert-base-multilingual-cased", num_labels=14)
        tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased', do_lower_case=False)
    elif model_name == 'KoBERT':
        model = BertForTokenClassification.from_pretrained('monologg/kobert', num_labels = 14)
        tokenizer = KoBertTokenizer.from_pretrained('monologg/kobert')

    model = nn.DataParallel(model)
    model.cuda()

    train_dataloader, validation_dataloader = make_TensorDataset(intent_train, intent_train_target, intent_val, intent_val_target, tokenizer)
    
    optimizer = AdamW(model.parameters(),
                  lr = 5e-5,
                  eps = 1e-6 
                )
    
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

    test(device, model_file_name, tokenizer, intent_test, intent_test_target)
    
BERT_NER("BERT_NER_Original_epoch10_v1", "BERT", ori_question, ori_question_target, val_question, val_question_target, test_question, test_question_target, 10)
BERT_NER("BERT_NER_EDA10_epoch10_v1", 'BERT', EDA_10_question, EDA_10_question_target, val_question, val_question_target, test_question, test_question_target, 10)
BERT_NER("BERT_NER_EDA20_epoch10_v1", 'BERT', EDA_20_question, EDA_20_question_target, val_question, val_question_target, test_question, test_question_target, 10)
BERT_NER("BERT_NER_EDA30_epoch10_v1", 'BERT', EDA_30_question, EDA_30_question_target, val_question, val_question_target, test_question, test_question_target, 10)
BERT_NER("BERT_NER_EMB10_epoch10_v1", 'BERT', EMB_10_question, EMB_10_question_target, val_question, val_question_target, test_question, test_question_target, 10)
BERT_NER("BERT_NER_EMB20_epoch10_v1", 'BERT', EMB_20_question, EMB_20_question_target, val_question, val_question_target, test_question, test_question_target, 10)
BERT_NER("BERT_NER_EMB30_epoch10_v1", 'BERT', EMB_30_question, EMB_30_question_target, val_question, val_question_target, test_question, test_question_target, 10)