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


class listenerModel(nn.Module):

    def __init__(self, input_size, hidden_size, output_size, embed_drop=0, lock_dropi=0, lock_droph=0, lock_dropo=0):
        super(listenerModel, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.rnns = nn.ModuleList(
            [nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=1, batch_first=False, bidirectional=True),
             nn.LSTM(input_size=hidden_size*4, hidden_size=hidden_size, num_layers=1, batch_first=False, bidirectional=True),
             nn.LSTM(input_size=hidden_size*4, hidden_size=hidden_size, num_layers=1, batch_first=False, bidirectional=True)]
        )

        # layer size *2 bcs bidirectional, again *2 because of pyramid

        self.projection_key = nn.Linear(in_features=hidden_size*4, out_features=output_size) #mebe 256 as per nihar's recit
        self.projection_val = nn.Linear(in_features=hidden_size*4, out_features=output_size)

        # self.init_weights()

    def forward(self, seq_list, input_length):

        batch_size = seq_list.shape[1]
        orig_input_length = input_length

        # if self.lock_dropi is not 0:
        #     seq_list = self.lock_dropout(seq_list, dropout=self.lock_dropi)

        # self.drop = nn.Dropout(p=0.1)

        # DO PACKING ONLY FOR LSTM
        ##############################################
        padded_input = seq_list
        hidden = None
        for i, rnn_i in enumerate(self.rnns):
            # pdb.set_trace()
            packed_input = rnn.pack_padded_sequence(padded_input, input_length, batch_first=False)  # packed version
            output_packed, hidden = rnn_i(packed_input)  # N*L*H
            output_padded = rnn.pad_packed_sequence(output_packed, batch_first=False)

            output_padded_data = output_padded[0]
            output_padded_length = output_padded[1]

            #output_padded_reshaped = output_padded_data.reshape(batch_size,output_padded_data.shape[0],output_padded_data.shape[2])
            output_padded_reshaped = output_padded_data.permute(1,0,2)
            n2 = output_padded_reshaped.shape[1]
            n2_even = n2
            if n2 % 2 is not 0:
                n2_even = n2-1
            n3 = output_padded_reshaped.shape[2]

            output_padded_reshaped_new = output_padded_reshaped[:,0:n2_even,:]
            output_padded_reshaped_new2 = output_padded_reshaped_new.contiguous().view(batch_size,int(n2_even/2),int(n3*2))
            #padded_input = output_padded_reshaped_new2.reshape(-1,batch_size,int(n3*2))
            input_length = output_padded_length/2
            padded_input = output_padded_reshaped_new2.permute(1,0,2)

        listener_features = padded_input.permute(1,0,2)
        attention_key = self.projection_key(listener_features)
        attention_val = self.projection_val(listener_features)

        # create attention mask based on the input lengths
        attention_mask = np.zeros((batch_size,input_length[0]))
        for i in range(batch_size):
            attention_mask[i][:input_length[i]] = 1
        attention_mask = torch.from_numpy(attention_mask).float().cuda()
        attention_mask.requires_grad = False
        # pdb.set_trace()
        return attention_key, attention_val, attention_mask

    def init_weights(self):
        for l in self.scoring:
            torch.nn.init.xavier_normal_(l.weight)
            l.bias.data.zero_()

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        return [(weight.new(1, batch_size, self.hidden_size if l != self.nlayers - 1 else (
            self.embed_size if self.tie_weights else self.hidden_size)).zero_(),
                 weight.new(1, batch_size, self.hidden_size if l != self.nlayers - 1 else (
                     self.embed_size if self.tie_weights else self.hidden_size)).zero_())
                for l in range(self.nlayers)]
