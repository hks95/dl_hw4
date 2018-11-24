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

        self.embedding = nn.Embedding(vocab_size, self.embed_size)  # Embedding layer
        self.lstm_cell1 = nn.LSTMCell(self.embed_size + self.context_size, self.hidden_size)
        self.lstm_cell2 = nn.LSTMCell(self.hidden_size, self.hidden_size)
        self.lstm_cell3 = nn.LSTMCell(self.hidden_size, self.hidden_size)

        self.projection1 = nn.Linear(self.hidden_size,self.context_size)
        self.projection2 = nn.Linear(self.hidden_size,self.vocab_size)

        self.softmax = nn.Softmax(dim=2)

        # self.init_weights()

    def forward(self, target, target_lengths, attention_key, attention_val, attention_mask):

        # target is seq * batch size
        hx1 = torch.zeros(target.shape[1], self.hidden_size)
        cx1 = torch.zeros(target.shape[1], self.hidden_size)

        hx2 = torch.zeros(target.shape[1], self.hidden_size)
        cx2 = torch.zeros(target.shape[1], self.hidden_size)

        hx3 = torch.zeros(target.shape[1], self.hidden_size)
        cx3 = torch.zeros(target.shape[1], self.hidden_size)

        prev_context = torch.zeros(target.shape[1], self.context_size)

        # inner product requires this change to satisfy dim
        attention_key = attention_key.reshape(attention_key.shape[0],attention_key.shape[2],attention_key.shape[1])

        # output_array = torch.zeros(target.shape[0],target.shape[1],self.vocab_size) #first output is sos
        output_list = []
        output_list.append(torch.zeros(target.shape[1],self.vocab_size))
        prev_output = output_list[0]
        # looping for each time step in target sequence till eos
        for i in range(target.shape[0]-1): # batch * 1 word

            ############ GUMBEL TRICK FOR TEACHER FORCING ##############3
            prob = torch.randint(0, 100, (1,))
            if prob>=90:
                dist = Gumbel(0, 1)
                eps = dist.sample(prev_output.size())
                y = torch.argmax(prev_output + eps, -1)  # y is batch size
            else:
                y = target[i]

            ############ EMBEDDING PART ################
            embed = self.embedding(y) # batch * embed size
            concat_input = torch.cat((embed,prev_context),1) # batch * (embed + context) size

            ############ LSTM PART ################
            hx_1, cx_1 = self.lstm_cell1(concat_input, (hx1, cx1)) #hx is the query
            hx_2, cx_2 = self.lstm_cell2(hx_1, (hx2, cx2))
            hx_3, cx_3 = self.lstm_cell3(hx_2, (hx3, cx3))

            # ############ ATTENTION PART ################
            context_input = torch.unsqueeze(hx_3,dim=1) #batch * 1 * hidden_sp
            context_input_proj = self.projection1(context_input) #batch*1*key_li
            energy = torch.bmm(context_input_proj,attention_key) #batch*1*len_seq
            attention = self.softmax(energy) #batch*1*len_seq

            ############ MASKING PART ################
            attention_2d = torch.squeeze(attention,dim=1) #batch*len_seq
            attention_masked = attention_2d * attention_mask #batch*len_seq
            attention_norm = F.normalize(attention_masked,p=1,dim=1) #batch*len_seq
            attention_norm_3d = torch.unsqueeze(attention_norm,dim=1) #batch*1*len_seq

            ############ CONTEXT PART ################
            context = torch.bmm(attention_norm_3d, attention_val) #batch*1*key_li
            context_2d = torch.squeeze(context,dim=1) #batch*key_li

            output_i = self.projection2(hx_3) #batch*vocab size
            output_list.append(output_i)
            prev_context = context_2d
            prev_output = output_i

            hx1 = hx_1
            cx1 = cx_1
            hx2 = hx_2
            cx2 = cx_2
            hx3 = hx_3
            cx3 = cx_3

        output_array = torch.stack(output_list,dim=0)
        # output_array = output_array.reshape(output_array.shape[0],output_array.shape[1],output_array.shape[2])
        output_array_2d = output_array.view(-1,output_array.shape[2]) #(batch*len seq)*vocab size for ce loss
        return output_array_2d