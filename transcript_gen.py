import numpy as np
import os

def get_transcript(labels):
    print('generating transcript')
    # path = os.getcwd() + '/data/'
    # full_labels = np.load(os.path.join(path, 'train_transcripts.npy'), encoding='bytes')

    label_dict = {}
    label_dict['start'] = 0
    label_dict['end'] = 1

    # label_array = [] #np.zeros(labels.shape[0],labels.shape[1])

    for idx1,transcript_item in enumerate(labels):
        # temp_array =
        for idx2,word in enumerate(transcript_item):
            word_string = word.decode("utf-8")
            if word not in label_dict:
                word_string = word.decode("utf-8")
                label_dict[word_string] = len(label_dict)-1
            # label_array[idx1][idx2] = label_dict[word_string]


    return label_dict