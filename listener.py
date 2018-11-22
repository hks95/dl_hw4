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

    def __init__(self, input_size, hidden_size, embed_drop=0, lock_dropi=0, lock_droph=0, lock_dropo=0):
        super(listenerModel, self).__init__()
        self.input_size = 40
        self.hidden_size = 256
        # self.output_size = 47
        self.nlayers = 3

        self.embed_drop = embed_drop
        self.lock_dropi = lock_dropi
        self.lock_droph = lock_droph
        self.lock_dropo = lock_dropo


        self.rnns = nn.ModuleList(
            [nn.LSTM(input_size=40, hidden_size=256, num_layers=1, batch_first=False, bidirectional=True),
             nn.LSTM(input_size=1024, hidden_size=256, num_layers=1, batch_first=False, bidirectional=True),
             nn.LSTM(input_size=1024, hidden_size=256, num_layers=1, batch_first=False, bidirectional=True)]
        )

        # layer size *2 bcs bidirectional, again *2 because of pyramid

        # self.scoring = nn.ModuleList([
        #     nn.Linear(in_features=1024, out_features=512),
        #     nn.Linear(in_features=512, out_features=256),
        #     nn.Linear(in_features=256, out_features=128),
        #     nn.Linear(in_features=128, out_features=47)])

        # self.init_weights()

    def forward(self, seq_list, input_length):

        batch_size = seq_list.shape[1]
        # pdb.set_trace()

        # if self.lock_dropi is not 0:
        #     seq_list = self.lock_dropout(seq_list, dropout=self.lock_dropi)

        # self.drop = nn.Dropout(p=0.1)

        # DO PACKING ONLY FOR LSTM
        ##############################################
        packed_input = rnn.pack_padded_sequence(seq_list, input_length, batch_first=False)  # packed version
        hidden = None
        for i, rnn_i in enumerate(self.rnns):
            output_packed, hidden = rnn_i(packed_input)  # N*L*H
            output_padded = rnn.pad_packed_sequence(output_packed, batch_first=False)

            output_padded_data = output_padded[0]
            output_padded_length = output_padded[1]

            output_padded_reshaped = output_padded_data.reshape(batch_size,output_padded_data.shape[0],output_padded_data.shape[2])

            n2 = output_padded_reshaped.shape[1]
            if n2 % 2 is not 0:
                n2 = n2-1
            n3 = output_padded_reshaped.shape[2]

            output_padded_reshaped = output_padded_reshaped[:,0:n2,:]
            output_padded_reshaped = output_padded_reshaped.reshape(batch_size,int(n2/2),int(n3*2))
            output_padded_reshaped = output_padded_reshaped.reshape(-1,batch_size,int(n3*2))
            output_padded_length = output_padded_length/2
            packed_input = rnn.pack_padded_sequence(output_padded_reshaped, output_padded_length, batch_first=False)

            # if i != self.nlayers - 1:
            #     if self.lock_droph is not 0:
            #         output_packed = self.lock_dropout(output_packed, self.lock_droph)

                    # output padded becomes frames*batchsize*40

        # output_padded, _ = rnn.pad_packed_sequence(output_packed, batch_first=False)  # unpacked output (padded)

        # if self.lock_dropo is not 0:
        #     output_padded = self.lock_dropout(output_padded, self.lock_dropo)
        ##############################################

        listener_features = output_padded_reshaped

        # for i, linear in enumerate(self.scoring):
        #     scores = linear(scores)  # N*L*H
        #     if i is not 3:
                # print('l {}'.format(i))
                # scores = self.drop(scores)

        return listener_features  # return concatenated logits

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


    def embed_dropout(self, embed, words, dropout):
        mask = embed.weight.data.new().resize_((embed.weight.size(0), 1)).bernoulli_(1 - dropout).expand_as(
            embed.weight) / (1 - dropout)
        masked_embed_weight = mask * embed.weight
        X = nn.functional.embedding(words, masked_embed_weight,
                                    -1, embed.max_norm, embed.norm_type,
                                    embed.scale_grad_by_freq, embed.sparse
                                    )
        return X

    def lock_dropout(self, x, dropout):
        if not self.training or not dropout:
            return x
        m = x.data.new(1, x.size(1), x.size(2)).bernoulli_(1 - dropout)  # try swapping to 1 and 2
        mask = Variable(m, requires_grad=False) / (1 - dropout)
        mask = mask.expand_as(x)
        return mask * x
