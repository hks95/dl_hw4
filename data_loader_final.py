import torch
from torch.utils.data import DataLoader  #to leverage multi processing
import numpy as np
import pdb
import random
import os
from wsj_loader import WSJ
import transcript_gen

class ctc_Dataset(DataLoader):
    'Characterizes a dataset for PyTorch'
    def __init__(self,flag,batch_size):
        'Initialization'
        
        loader = WSJ()
        self.flag = flag
        self.batch_size = batch_size

        if flag is 'train':
            self.input, self.labels = loader.train
            self.labels_dict = transcript_gen.get_transcript(self.labels)
        elif flag is 'dev':
            self.input, self.labels = loader.dev
            # _,self.labels_string = transcript_gen.get_transcript(self.labels)
        else:
            self.input, _ = loader.test

        self.num_utterances = self.input.shape[0]
  
        print('total_samples {}'.format(self.num_utterances))

    def __getitem__(self, item):
        data  = self.input[item]
        # if self.flag is 'test':
        #     labels = 0
        # else:
        #     labels = self.labels_string[item]
        return data

    def __len__(self):
        return self.num_utterances

# Collate function. Transform a list of sequences into a batch. Passed as an argument to the DataLoader.
# Returns data on the format seq_len x batch_size
def collate(seq_list):
    
    batch_size = len(seq_list)
    lens = [len(seq) for seq in seq_list] #seq[0] in input
    seq_order = sorted(range(len(lens)), key=lens.__getitem__, reverse=True) #maintains list of indices in sorted order
    longest_seq_length = len(seq_list[seq_order[0]])
    padded_inputs = np.zeros((longest_seq_length,batch_size,40))

    print('longest seq length {}'.format(longest_seq_length))
    data = [seq_list[i] for i in seq_order]
    for i,x in enumerate(data):
        padded_inputs[:x.shape[0], i, :] = x

    # targets = []
    input_length = []
    # targets_length = []
    for i in seq_order:
        # targets.append(seq_list[i][1]) #warp ctc requires 1 to n as labels
        input_length.append(len(seq_list[i]))
        # targets_length.append(len(seq_list[i][1]))
    
    input_length = np.array(input_length)
    # targets_length = np.array(targets_length)

    # return padded_inputs,targets,input_length,targets_length
    return padded_inputs,input_length

def test_collate(seq_list):

    batch_size = len(seq_list)
    input_length = []
    input_length.append(len(seq_list[0][0]))
    input_length = np.array(input_length)

    inputs = seq_list[0][0]

    return inputs,input_length
