import numpy as np
import os
import pdb
import string

def main():
    print('generating transcript')

    labels = np.load('./data/train_transcripts.npy')
    label_dict = {}
    label_dict['start'] = 0
    label_dict['end'] = 0
    label_dict[' '] = 1

    label_list = []

    for idx1,transcript_item in enumerate(labels):
        # pdb.set_trace()
        # temp_label_array = np.empty(transcript_item.shape[0],dtype=object)
        # temp_label_array = np.zeros(transcript_item.shape[0])
        temp_label_list = []
        for idx2,word in enumerate(transcript_item):
            word_string = word.decode("utf-8")
            char_list = list(word_string)
            for char in char_list:
                if char not in label_dict:
                    label_dict[char] = len(label_dict)-1
                temp_label_list.append(label_dict[char])
            temp_label_list.append(label_dict[' '])

        temp_label_list[-1] = 0 #remove last space and put end
        #print('label array {}'.format(temp_label_array))
        label_list.append(np.array(temp_label_list))
        
    label_array = np.array(label_list)
    # np.save('./data/train_labels.npy', label_array) 

    label_dict_inv = {v: k for k, v in label_dict.items()}
    pdb.set_trace()
    # np.save('./data/train_label_dict.npy', label_dict_inv) 
    # read_dictionary = np.load('./data/label_dict.npy').item()

if __name__ == '__main__':
        main()
