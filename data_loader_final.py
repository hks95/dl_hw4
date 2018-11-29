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
            self.input, self.labels,self.labels_dict = loader.train
            self.labels_dict = self.labels_dict.tolist()
            print('total vocab {}'.format(len(self.labels_dict)))
            self.vocab_size = len(self.labels_dict)

        elif flag is 'dev':
            self.input, self.labels,self.labels_dict = loader.dev
            self.labels_dict = self.labels_dict.tolist()
            print('total vocab {}'.format(len(self.labels_dict)))
            self.vocab_size = len(self.labels_dict)
        
        elif flag is 'test':
            self.input, _ = loader.test

        self.num_utterances = self.input.shape[0]
        
        print('total_samples {}'.format(self.num_utterances))


    def __getitem__(self, item):
        data = self.input[item]
        if self.flag is 'test':
            labels = 0
            return data,labels
        else:
            labels = self.labels[item]
            labels_dict = self.labels_dict
            return data,labels,labels_dict
        

    def __len__(self):
        return self.num_utterances

# Collate function. Transform a list of sequences into a batch. Passed as an argument to the DataLoader.
# Returns data on the format seq_len x batch_size
def collate(seq_list):
    
    batch_size = len(seq_list)
    #pdb.set_trace()
    # find decreasing order for inputs
    lens = [len(seq[0]) for seq in seq_list] #seq[0] in input
    seq_order = sorted(range(len(lens)), key=lens.__getitem__, reverse=True) #maintains list of indices in sorted order
    longest_seq_length = len(seq_list[seq_order[0]][0])
    padded_inputs = np.zeros((longest_seq_length,batch_size,40))
    # print('longest seq length {}'.format(longest_seq_length))

    # find decreasing order for labels
    lens = [len(seq[1]) for seq in seq_list] #seq[0] in input
    seq_order_labels = sorted(range(len(lens)), key=lens.__getitem__, reverse=True) #maintains list of indices in sorted order
    longest_seq_length = len(seq_list[seq_order_labels[0]][1])
    padded_targets = np.full((longest_seq_length,batch_size),1)
    target_mask = np.full((longest_seq_length,batch_size),0)

    data = [seq_list[i][0] for i in seq_order]
    for i,x in enumerate(data):
        padded_inputs[:x.shape[0], i, :] = x

    labels = [seq_list[i][1] for i in seq_order] #rearrange labels based on input decreasing order
    for i,x in enumerate(labels):
        padded_targets[:x.shape[0],i] = x
        target_mask[:x.shape[0],i] = 1

    input_length = [len(seq_list[i][0]) for i in seq_order]
    targets_length = [len(seq_list[i][1]) for i in seq_order]

    input_length = np.array(input_length)
    targets_length = np.array(targets_length)
    labels_dict = seq_list[0][2]

    return padded_inputs,padded_targets,input_length,targets_length,target_mask,labels_dict
    # return padded_inputs,input_length

def test_collate(seq_list):

    # batch_size = len(seq_list)
    input_length = []
    input_length.append(len(seq_list[0][0]))
    input_length = np.array(input_length)

    inputs = seq_list[0][0]

    return inputs,input_length
