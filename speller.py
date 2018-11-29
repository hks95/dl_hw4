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

        if flag is 'train':
            # looping for each time step in target sequence till eos
            for i in range(target.shape[0]): # batch * 1 word

                ############ GUMBEL TRICK FOR TEACHER FORCING ##############3
                if i is 0:
                    y = torch.zeros(target.shape[1]).long().cuda()
                else:
                    prob = torch.randint(0, 100, (1,))
                    if prob>=90:
                        y = prev_output
                        # dist = Gumbel(0, 1)
                        # eps = dist.sample(prev_output.size())
                        # y = torch.argmax(prev_output + eps, -1)  # y is batch size
                    else:
                        y = target[i-1] # first input is sos

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
                prev_context = context_2d

                hx1 = hx_1
                cx1 = cx_1
                hx2 = hx_2
                cx2 = cx_2
                hx3 = hx_3
                cx3 = cx_3

                #target_ignore_idx = target[i] * target_mask[i]
                loss_i = self.criterion(output_i,target[i])
                batch_loss.append(loss_i)
                output_softmax = torch.softmax(output_i,dim=1)
                pred = torch.multinomial(output_softmax, 1).flatten() #to make it 1D
                prev_output = pred

            batch_loss = torch.stack(batch_loss,dim=0)
            # batch_loss = torch.mean(batch_loss,dim=1) #avg the batch loss
            batch_loss = batch_loss*target_mask.float()
            batch_loss_sumseq = torch.sum(batch_loss,dim=0)
            batch_loss_mean = torch.mean(batch_loss_sumseq)
            # batch_loss = torch.sum(batch_loss)

            #output_array = torch.stack(output_list,dim=0)
            #output_array = output_array.reshape(output_array.shape[0],output_array.shape[1],output_array.shape[2])
            #output_array_2d = output_array.view(-1,output_array.shape[2]) #(batch*len seq)*vocab size for ce loss

            result = batch_loss_mean

        else:
            ############ ALWAYS BATCH SIZE 1 ########################
            max_allowed_seq_length = 300
            prev_output = None
            attention_map = []
            output_list = []
            pred_list = []
            batch_loss = []

            for i in range(max_allowed_seq_length):  # 1 word
                if i is 0:
                    y = torch.zeros(1).long().cuda()
                else:
                    if flag is 'dev':
                        y = target[i-1]
                    else:
                        y = prev_output.long()  # first input is sos

                ############ EMBEDDING PART ################
                embed = self.embedding(y)  # 1 * embed size
                concat_input = torch.cat((embed, prev_context), 1)  # 1 * (embed + context) size

                ############ LSTM PART ################
                hx_1, cx_1 = self.lstm_cell1(concat_input, (hx1, cx1))  # hx is the query
                hx_2, cx_2 = self.lstm_cell2(hx_1, (hx2, cx2))
                hx_3, cx_3 = self.lstm_cell3(hx_2, (hx3, cx3))

                # ############ ATTENTION PART ################
                
                context_input = self.projection_query(hx_3) #batch*hidden_sp
                context_input = torch.unsqueeze(context_input,dim=1) #batch * 1 * hidden_sp
                energy = torch.bmm(context_input,attention_key) #batch*1*len_seq
                attention = self.softmax(energy) #batch*1*len_seq
                attention_map.append(attention)

                ############ MASKING PART ################
                attention_2d = torch.squeeze(attention, dim=1)  # 1*len_seq
                attention_masked = torch.mul(attention_2d,attention_mask) #batch*len_seq
                attention_norm = F.normalize(attention_masked, p=1, dim=1)  # 1*len_seq
                attention_norm_3d = torch.unsqueeze(attention_norm, dim=1)  # 1*1*len_seq

                ############ CONTEXT PART ################
                context = torch.bmm(attention_norm_3d, attention_val)  # 1*1*key_li
                context_2d = torch.squeeze(context, dim=1)  # 1*key_li
                
                output_i = self.projection_vocab(hx_3)  # 1*vocab size
                output_list.append(output_i)

                output_softmax = torch.softmax(output_i.flatten(),dim=0)
                pred = torch.argmax(output_softmax)
                pred = torch.unsqueeze(pred,0)
                
                if pred.item() == 0:
                    break
                pred_list.append(pred)

                prev_context = context_2d
                prev_output = pred

                hx1 = hx_1
                cx1 = cx_1
                hx2 = hx_2
                cx2 = cx_2
                hx3 = hx_3
                cx3 = cx_3

                if flag is 'dev':
                    loss_i = self.criterion(output_i,target[i])
                    batch_loss.append(loss_i)

            if flag is 'test':
                result = pred_list
            else:
                attention_map_array = torch.stack(attention_map,dim=0) #target_len*B*seq_len
                attention_map_array = torch.squeeze(attention_map_array,2)
                attention_map_array = attention_map_array.permute(1,0,2)
                attention_map_array = attention_map_array[0]

                output_array = torch.stack(output_list,dim=0)
                output_array = output_array[:target.shape[0],:,:]
                output_array_2d = torch.squeeze(output_array,1)
                # if target.flatten().shape[0] is 201:
                #     pdb.set_trace()
                # print('output shape {} target shape {}'.format(output_array_2d.shape,target.flatten().shape))
                batch_loss = self.criterion(output_array_2d,target.flatten())

                # batch_loss = torch.mean(batch_loss,dim=1) #avg the batch loss
                # batch_loss = batch_loss*target_mask.float() #batch size always 1
                batch_loss_sumseq = torch.sum(batch_loss,dim=0)
                batch_loss_mean = torch.mean(batch_loss_sumseq)

                result = [batch_loss_mean,attention_map_array]

        return result
