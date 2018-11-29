import argparse
import csv
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

import torch.nn.utils.rnn as rnn

from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
import pdb
from torch.distributions.gumbel import Gumbel

class SpellerModel(nn.Module):

    def __init__(self,vocab_size,embed_size,hidden_size,listener_key_size):
        super(SpellerModel, self).__init__()
        self.vocab_size = vocab_size #attention projection size
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.nlayers = 3
        self.context_size = listener_key_size
        # self.batch_size = batch_size

        self.embedding = nn.Embedding(vocab_size, self.embed_size)  # Embedding layer
        self.lstm_cell1 = nn.LSTMCell(self.embed_size + self.context_size, self.hidden_size)
        self.lstm_cell2 = nn.LSTMCell(self.hidden_size, self.hidden_size)
        self.lstm_cell3 = nn.LSTMCell(self.hidden_size, self.hidden_size)

        self.projection_query = nn.Linear(self.hidden_size,self.context_size)
        self.projection_vocab = nn.Linear(self.hidden_size,self.vocab_size)

        self.softmax = nn.Softmax(dim=2)
        self.criterion = nn.CrossEntropyLoss(reduction='none')

        self.hx1 = nn.Parameter(torch.FloatTensor(1, self.hidden_size).zero_())
        self.cx1 = nn.Parameter(torch.FloatTensor(1, self.hidden_size).zero_())

        self.hx2 = nn.Parameter(torch.FloatTensor(1, self.hidden_size).zero_())
        self.cx2 = nn.Parameter(torch.FloatTensor(1, self.hidden_size).zero_())

        self.hx3 = nn.Parameter(torch.FloatTensor(1, self.hidden_size).zero_())
        self.cx3 = nn.Parameter(torch.FloatTensor(1, self.hidden_size).zero_())

        # self.init_weights()

    def forward(self, target, target_mask, attention_key, attention_val, attention_mask, flag, target_dict):

        # target is seq * batch size
        batch_size = attention_key.shape[0]

        hx1 = self.hx1.expand(batch_size,-1)
        cx1 = self.cx1.expand(batch_size,-1)
        
        hx2 = self.hx2.expand(batch_size,-1)
        cx2 = self.cx2.expand(batch_size,-1)
        
        hx3 = self.hx3.expand(batch_size,-1)
        cx3 = self.cx3.expand(batch_size,-1)

        prev_context = self.projection_query(hx1)

        # inner product requires this change to satisfy dim
        attention_key = attention_key.permute(0,2,1)

        output_list = []
        batch_loss = []

        max_allowed_seq_length = 300
        prev_output = None
        attention_map = []
        pred_list = []

        for i in range(max_allowed_seq_length):  # 1 word
            if i is 0:
                y = torch.zeros(target.shape[1]).long().cuda()
            else:
                if flag is 'train':
                    ##### TEACHER FORCING #########3
                    prob = torch.randint(0, 100, (1,))
                    if prob>=90:
                        y = prev_output
                    else:
                        y = target[i-1] # first input is sos
                elif flag is 'eval':
                    ########## ALWAYS SEND GROUND TRUTH #############
                    y = target[i-1]
                else:
                    ######### ALWAYS SEND PREVIOUS OUTPUT ###########
                    y = prev_output.long()  # first input is sos


            ############ EMBEDDING PART ################
            embed = self.embedding(y) # batch * embed size
            concat_input = torch.cat((embed,prev_context),1) # batch * (embed + context) size

            ############ LSTM PART ################
            hx_1, cx_1 = self.lstm_cell1(concat_input, (hx1, cx1)) #hx is the query
            hx_2, cx_2 = self.lstm_cell2(hx_1, (hx2, cx2))
            hx_3, cx_3 = self.lstm_cell3(hx_2, (hx3, cx3))

            # ############ ATTENTION PART ################
            
            context_input = self.projection_query(hx_3) #batch*hidden_sp
            context_input = torch.unsqueeze(context_input,dim=1) #batch * 1 * hidden_sp
            energy = torch.bmm(context_input,attention_key) #batch*1*len_seq
            attention = self.softmax(energy) #batch*1*len_seq
            attention_map.append(attention)

            ############ MASKING PART ################
            attention_2d = torch.squeeze(attention,dim=1) #batch*len_seq
            attention_masked = torch.mul(attention_2d,attention_mask) #batch*len_seq
            attention_norm = F.normalize(attention_masked,p=1,dim=1) #batch*len_seq
            attention_norm_3d = torch.unsqueeze(attention_norm,dim=1) #batch*1*len_seq

            ############ CONTEXT PART ################
            context = torch.bmm(attention_norm_3d, attention_val) #batch*1*key_li
            context_2d = torch.squeeze(context,dim=1) #batch*key_li

            output_i = self.projection_vocab(hx_3) #batch*vocab size
            output_list.append(output_i)
            output_softmax = torch.softmax(output_i,dim=1)            
            pred = torch.argmax(output_softmax,dim=1)

            if flag is 'train' or flag is 'eval':
                if (i == (target.shape[0])):
                    # pdb.set_trace()
                    break
            else:
                if pred.item() == 0:
                    break

            pred_list.append(pred)
            prev_output = pred
            prev_context = context_2d

            hx1 = hx_1
            cx1 = cx_1
            hx2 = hx_2
            cx2 = cx_2
            hx3 = hx_3
            cx3 = cx_3

            if flag is not 'test':
                loss_i = self.criterion(output_i,target[i])
                # print('loss {}'.format(loss_i))
                batch_loss.append(loss_i)

        if flag is 'test':
            result = pred_list
        else:
            attention_map_array = torch.stack(attention_map,dim=0) #target_len*B*seq_len
            attention_map_array = torch.squeeze(attention_map_array,2)
            attention_map_array = attention_map_array.permute(1,0,2)
            attention_map_array = attention_map_array[0]

            batch_loss = torch.stack(batch_loss,dim=0)
            # batch_loss = torch.mean(batch_loss,dim=1) #avg the batch loss
            batch_loss = batch_loss*target_mask.float()
            batch_loss_sumseq = torch.sum(batch_loss,dim=0)
            batch_loss_mean = torch.mean(batch_loss_sumseq)

            result = [batch_loss_mean,attention_map_array]

        return result
