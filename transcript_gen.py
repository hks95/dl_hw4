import numpy as np
import os

def get_transcript(labels):
    print('generating transcript')
    # path = os.getcwd() + '/data/'
    # full_labels = np.load(os.path.join(path, 'train_transcripts.npy'), encoding='bytes')

    label_dict = {}
    label_dict['start'] = 0
    label_dict['end'] = 1

    label_list = []

    for idx1,transcript_item in enumerate(labels):
        # temp_label_array = np.empty(transcript_item.shape[0],dtype=object)
        temp_label_array = np.zeros(transcript_item.shape[0])
        for idx2,word in enumerate(transcript_item):
            word_string = word.decode("utf-8")
            # temp_label_array[idx2] = word_string
            if word not in label_dict:
                word_string = word.decode("utf-8")
                label_dict[word_string] = len(label_dict)-1

            temp_label_array[idx2] = label_dict[word_string]
        label_list.append(temp_label_array)

    label_array = np.array(label_list)


    return label_dict,label_array