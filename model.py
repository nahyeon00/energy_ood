import pytorch_lightning as pl
import torch.nn.functional as F
from transformers import BertPreTrainedModel, BertModel, BertTokenizer, AdamW, BertConfig
import pandas as pd
import torch.nn as nn
import numpy as np
import torch
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import os
import random
import torch.nn as nn
from transformers import BertPreTrainedModel, BertModel, BertTokenizer, AdamW, BertConfig
from torch.optim.lr_scheduler import StepLR
from sklearn import metrics
from torchmetrics.classification import MultilabelPrecision
from torchmetrics.classification import MultilabelRecall
from sklearn.metrics import auc

from mix_data import *
from model import *
from utils import *

class en_model(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.save_hyperparameters(args)

        # data
        self.dataset = args.dataset  # 저장 파일명 위해 필요
        self.known_cls_ratio = args.known_cls_ratio  # 저장 파일명 위해 필요

        self.num_labels = args.num_labels
        
        # use pretrained BERT
        model_config = BertConfig.from_pretrained('bert-base-uncased', output_hidden_states=True)
        self.bert = BertModel(model_config)
        self.dense = nn.Linear(self.bert.config.hidden_size, self.bert.config.hidden_size)
        self.classifier = nn.Linear(self.bert.config.hidden_size, self.num_labels)

        self.activation = nn.ReLU()
        self.dropout =  nn.Dropout(self.bert.config.hidden_dropout_prob)
        self.ind_stack = np.empty([0, self.num_labels])
        self.ood_stack = np.empty([0, self.num_labels])

        self.__build_loss()

        
    def forward(self, input_ids, attention_mask, token_type_ids):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        last_hidden_layer = outputs[0]
        mean_pooling = last_hidden_layer.mean(dim=1)
        pooled_output = self.dense(mean_pooling)
        pooled_output = self.activation(pooled_output)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        return pooled_output, logits

    def training_step(self, batch, batch_idx):
        print("start training")
        # batch
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        token_type_ids = batch['token_type_ids']
        label_id = batch['label_id']
        
        # fwd
        _, logits = self.forward(input_ids, attention_mask, token_type_ids)

        # loss
        loss = self._loss(logits, label_id.squeeze(-1))
        
        # logs
        tensorboard_logs = {'train_loss': loss}

        self.log("train_loss", loss, on_epoch=True, prog_bar=True, logger=True)

        return {'loss': loss, 'log': tensorboard_logs}
    
    def validation_step(self, batch, batch_idx):
        print("start validation")
        # batch
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        token_type_ids = batch['token_type_ids']
        label_id = batch['label_id']
        # label_id = label_id.type(torch.int32)
        print('label_id', label_id.size())

        # fwd
        _, logits = self.forward(input_ids, attention_mask, token_type_ids)  # 128,4
        outputs = torch.sigmoid(logits.detach())

        y_pred = outputs.detach().cpu()
        y_true = label_id.detach().cpu()
        # breakpoint()
        premetric = MultilabelPrecision(num_labels=self.num_labels)
        precisions = premetric(y_pred, y_true)

        remetric = MultilabelRecall(num_labels=self.num_labels)
        recalls = remetric(y_pred, y_true)


        
        self.log('precisions', precisions)

        return {'precisions': precisions, 'recalls': recalls}
    
    # def validation_epoch_end(self, outputs):
    #     print("outputs", outputs)
    #     precisions = []
    #     recalls = []
    #     for output in outputs:
    #         precisions.append(output['precisions'])
    #         recalls.append(output['recalls'])
    #     print("pre", precisions)
    #     print("rec", recalls)
    #     precisions = torch.cat(precisions, dim=1)
    #     recalls = torch.cat(recalls, dim=1)

    #     FinalMAPs = []
    #     for i in range(self.n_classes):
    #         FinalMAPs.append(auc(recalls[:, i], precisions[:, i]))

    #     return {'FinalMAP': torch.mean(torch.tensor(FinalMAPs))}

    def test_step(self, batch, batch_idx):
        print("start test")
        # batch - ind, mix, oos
        ind_batch = batch['ind']
        mix_batch = batch['mix']
        ood_batch = batch['ood']

        # ind batch
        ind_input_ids = ind_batch['input_ids']
        ind_attention_mask = ind_batch['attention_mask']
        ind_token_type_ids = ind_batch['token_type_ids']
        ind_label_id = ind_batch['label_id']
        
        # mix batch
        mix_input_ids = mix_batch['input_ids']
        mix_attention_mask = mix_batch['attention_mask']
        mix_token_type_ids = mix_batch['token_type_ids']
        mix_label_id = mix_batch['label_id']

        # ood batch
        ood_input_ids = ood_batch['input_ids']
        ood_attention_mask = ood_batch['attention_mask']
        ood_token_type_ids = ood_batch['token_type_ids']
        ood_label_id = ood_batch['label_id']

        # fwd
        # ind fwd
        _, ind_logits = self.forward(ind_input_ids, ind_attention_mask, ind_token_type_ids)
        ind_logits_np = ind_logits.cpu().squeeze().numpy()
        self.ind_stack = np.vstack((self.ind_stack, ind_logits_np))

        ind_E_f = torch.log(1+torch.exp(ind_logits))
        ind_scores = torch.sum(ind_E_f, dim=1)

        
        # mix fwd

        
        # ood fwd
        _, ood_logits = self.forward(ood_input_ids, ood_attention_mask, ood_token_type_ids)
        ood_logits_np = ood_logits.cpu().squeeze().numpy()
        self.ood_stack = np.vstack((self.ood_stack, ood_logits_np))

        ood_E_f = torch.log(1+torch.exp(ood_logits))
        ood_scores = torch.sum(ood_E_f, dim=1)
        self.log_dict({'step ind_scores': ind_scores, 'step ood_scores': ood_scores})

        return {'ind_scores': ind_scores, 'ood_scores': ood_scores}

    def test_epoch_end(self, outputs):
        ind_logits = torch.from_numpy(self.ind_stack).cuda()
        ind_outputs = torch.sigmoid(ind_logits)
        ind_E_f = torch.log(1+torch.exp(ind_logits))
        ind_scores = torch.sum(ind_E_f, dim=1).cpu().numpy()
        ind_scores = np.mean(ind_scores)

        ood_logits = torch.from_numpy(self.ood_stack).cuda()
        ood_outputs = torch.sigmoid(ood_logits)
        ood_E_f = torch.log(1+torch.exp(ood_logits))
        ood_scores = torch.sum(ood_E_f, dim=1).cpu().numpy()
        ood_scores = np.mean(ood_scores)
        self.log_dict({'ind_scores': np.mean(ind_scores), 'ood_scores': np.mean(ood_scores)})

        return {'ind_scores': np.mean(ind_scores), 'ood_scores': np.mean(ood_scores)}







    def configure_optimizers(self):
        """Prepare optimizer and schedule (linear warmup and decay)"""
        parameters = []
        for p in self.parameters():
            if p.requires_grad:
                parameters.append(p)
            else:
                print(p)

        optimizer = torch.optim.Adam(parameters, lr=2e-05 , eps=1e-08)
        scheduler = StepLR(optimizer, step_size=5, gamma=0.5)

        return {'optimizer': optimizer, 'lr_scheduler': scheduler}
    
    def __build_loss(self):
        self._loss = nn.BCEWithLogitsLoss()
