from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
import pytorch_lightning as pl
import torch.nn.functional as F
from transformers import BertPreTrainedModel, BertModel, BertTokenizer, AdamW, BertConfig
import pandas as pd
import torch.nn as nn
import numpy as np
import torch
import argparse
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import os
import random
from transformers import AdamW, get_linear_schedule_with_warmup
from pytorch_lightning.trainer.supporters import CombinedLoader

from model import *

class MixDataset(Dataset):
    def __init__(self, process_data, process_label, max_seq_len, label_list):
        self.data = process_data
        self.label = process_label

        self.max_seq_len = max_seq_len
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
        self.label_list = label_list
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        data = self.data[index]
        # print('data', data)
        features = self.tokenizer(str(data), padding='max_length', max_length= self.max_seq_len, truncation=True, return_tensors='pt') 

        input_ids = features['input_ids'].squeeze(0)
        attention_mask = features['attention_mask'].squeeze(0)
        token_type_ids = features['token_type_ids'].squeeze(0)

        ori_label = self.label[index]

        label_id = torch.zeros(len(self.label_list))

        for d in ori_label:
            if d in self.label_list:
                idx = self.label_list.index(d)
                label_id[idx] = 1
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'token_type_ids': token_type_ids,
            'label_id': label_id
        }



class MixDataModule(pl.LightningDataModule):
    def __init__(self, args):
        self.dataset = args.dataset
        self.data_path = os.path.join(args.data_path, self.dataset)
        self.train_data_path = f'{self.data_path}/train.txt'
        self.val_data_path = f'{self.data_path}/dev.txt'
        self.test_data_path = f'{self.data_path}/test.txt'

        self.max_seq_len = args.max_seq_len
        self.batch_size = args.batch_size
        self.known_cls_ratio = args.known_cls_ratio
        self.worker = args.num_workers
        self.labeled_ratio = args.labeled_ratio
        
        # data 불러오기
        print("self.train_data_path", self.train_data_path)
        self.train_texts, _, self.train_intents = self.__read_file(self.train_data_path)

        # knwon_label list 만들기
        self.all_label_list = self.make_label_list(self.train_intents)
        self.n_known_cls = round(len(self.all_label_list)*self.known_cls_ratio)
        self.known_label_list = np.random.choice(np.array(self.all_label_list, dtype=str), self.n_known_cls, replace=False)
        self.known_label_list = self.known_label_list.tolist()
        print("known_label_list", self.known_label_list)

        
        args.num_labels = self.num_labels = len(self.known_label_list)
        print("num_labels", self.num_labels)

        if self.dataset == 'oos':
            self.unseen_label = 'oos'
        else:
            self.unseen_label = '<UNK>'
        
        self.unseen_label_id = self.num_labels
        self.label_list = self.known_label_list + [self.unseen_label]
        
    def setup(self, stage):

        if stage in (None, 'fit'):
            # data 불러오기
            self.valid_texts, _, self.valid_intents = self.__read_file(self.val_data_path)

            # token 문장으로 이어붙이기
            self.train_sentences = [' '.join(text) for text in self.train_texts]
            self.val_sentences = [' '.join(text) for text in self.valid_texts]

            # '#' 나누기
            self.train_label = self.divide_label(self.train_intents)
            self.valid_label = self.divide_label(self.valid_intents)
            

            # IND만을 가져오기
            self.train_examples = []
            self.train_true_label = []
            self.val_examples = []
            self.val_true_label = []
            train_now = []
            valid_now = []

            for i, cur_label in enumerate(self.train_label):
                for k, label in enumerate(cur_label):
                    if (label in self.known_label_list) and (np.random.uniform(0,1)<=self.labeled_ratio):
                        train_now.append(label)
                    if k == (len(cur_label)-1):  # 하나의 데이터 속 전체 intent 가 knwon 일 경우만 저장
                        if len(train_now) == len(cur_label):
                            self.train_examples.append([self.train_sentences[i]])
                            self.train_true_label.append(cur_label)
                train_now = []

            for i, cur_label in enumerate(self.valid_label):
                for k, label in enumerate(cur_label):
                    if (label in self.known_label_list) and (np.random.uniform(0,1)<=self.labeled_ratio):
                        valid_now.append(label)
                    if k == (len(cur_label)-1):  # 하나의 데이터 속 전체 intent 가 knwon 일 경우만 저장
                        if len(valid_now) == len(cur_label):
                            self.val_examples.append([self.val_sentences[i]])
                            self.val_true_label.append(cur_label)
                valid_now = []

            # print("train_examples", len(self.train_examples))  # 12396
            # print("val_examples", len(self.val_examples))  #643
            # breakpoint()
            self.train = MixDataset(self.train_examples, self.train_true_label, self.max_seq_len, self.known_label_list)
            self.valid = MixDataset(self.val_examples, self.val_true_label, self.max_seq_len, self.known_label_list)

        elif stage in (None, 'test'):
            # data 불러오기
            self.test_texts, _, self.test_intents = self.__read_file(self.test_data_path)

            # token 문장으로 이어붙이기
            self.test_sentences = [' '.join(text) for text in self.test_texts]
            
            # '#' 나누기
            self.test_label = self.divide_label(self.test_intents)

            self.ind_examples = []
            self.ind_true_label = []
            self.mix_examples = []
            self.mix_true_label = []
            self.ood_examples = []
            self.ood_true_label = []
            test_now = []
            num_oos = 0

            # label 바꿔서 다시 저장하기
            for i, cur_label in enumerate(self.test_label):
                for k, label in enumerate(cur_label):
                    if (label in self.known_label_list) and (np.random.uniform(0,1)<=self.labeled_ratio):  # ind ood 구분
                        test_now.append(label)
                    else:
                        label = self.unseen_label
                        test_now.append(label)
                        num_oos = num_oos + 1
                    # pure, mix, ood 중 구분하기
                    if k == (len(cur_label)-1):  # 하나의 데이터에서 마지막 intent까지 확인한 경우
                        if num_oos == len(cur_label):
                            self.ood_examples.append(self.test_sentences[i])
                            self.ood_true_label.append(test_now)
                        elif num_oos == 0:
                            self.ind_examples.append(self.test_sentences[i])
                            self.ind_true_label.append(test_now)
                        else:
                            self.mix_examples.append(self.test_sentences[i])
                            self.mix_true_label.append(test_now)
                test_now = []
                num_oos = 0


            self.test_ind = MixDataset(self.ind_examples, self.ind_true_label, self.max_seq_len, self.known_label_list)
            self.test_mix = MixDataset(self.mix_examples, self.mix_true_label, self.max_seq_len, self.known_label_list)
            self.test_ood = MixDataset(self.ood_examples, self.ood_true_label, self.max_seq_len, self.known_label_list)
        
    def train_dataloader(self):
        sampler = RandomSampler(self.train)
        return DataLoader(self.train, batch_size=self.batch_size, num_workers= self.worker, sampler = sampler)
    
    def val_dataloader(self):
        sampler = SequentialSampler(self.valid)
        return DataLoader(self.valid, batch_size=self.batch_size, num_workers= self.worker, sampler = sampler)
    
    def test_dataloader(self):
        ind_loader = DataLoader(self.test_ind, batch_size=self.batch_size, num_workers= self.worker)
        mix_loader = DataLoader(self.test_mix, batch_size=self.batch_size, num_workers= self.worker)
        ood_loader = DataLoader(self.test_ood, batch_size=self.batch_size, num_workers= self.worker)

        loaders = {'ind': ind_loader, "mix": mix_loader, "ood": ood_loader}
    
        combined_loaders = CombinedLoader(loaders, mode="max_size_cycle")

        return combined_loaders


    
    def predict_dataloader(self):
        sampler = RandomSampler(self.train)
        return DataLoader(self.train, batch_size=self.batch_size, num_workers= self.worker, sampler = sampler)

    def make_label_list(self, intents):
        all_label_list = []
        for intent in intents:
            all_label_list.extend(intent[0].split("#"))
        all_label_list = list(set(all_label_list))
        
        return all_label_list
    
    def __read_file(self, file_path):
        """ Read data file of given path.
        :param file_path: path of data file.
        :return: list of sentence, list of slot and list of intent.
        """

        texts, slots, intents = [], [], []
        text, slot = [], []

        with open(file_path, 'r', encoding="utf8") as fr:
            for line in fr.readlines():
                items = line.strip().split()

                if len(items) == 1:
                    texts.append(text)
                    slots.append(slot)
                    if "/" not in items[0]:
                        intents.append(items)
                    else:
                        new = items[0].split("/")
                        intents.append([new[1]])

                    # clear buffer lists.
                    text, slot = [], []

                elif len(items) == 2:
                    text.append(items[0].strip())
                    slot.append(items[1].strip())

        return texts, slots, intents
    
    def divide_label(self, intents):
        div_label = []
        for items in intents:
            div_label.append(items[0].split("#"))
        return div_label