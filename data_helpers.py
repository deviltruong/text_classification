# -*- encoding: utf-8 -*-

import numpy as np
import re
import  my_map
import os, sys
import unicodedata
from io import open
def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def load_data_and_labels(positive_data_file, negative_data_file):
    """
    Loads MR polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    # Load data from files
    positive_examples = list(open(positive_data_file, "r").readlines())
    positive_examples = [s.strip() for s in positive_examples]
    negative_examples = list(open(negative_data_file, "r").readlines())
    negative_examples = [s.strip() for s in negative_examples]
    # Split by words
    x_text = positive_examples + negative_examples
    x_text = [clean_str(sent) for sent in x_text]
    # Generate labels
    positive_labels = [[0, 1] for _ in positive_examples]
    negative_labels = [[1, 0] for _ in negative_examples]
    y = np.concatenate([positive_labels, negative_labels], 0)
    return [x_text, y]



def load_dataset_from_disk(dataset):
    x = []
    y = []
    list_samples = {k: [] for k in my_map.name2label.keys()}
    print(list_samples)
    print 'load_data in ' + dataset

    # return list file and folder in dir
    stack = os.listdir(dataset)

    while len(stack) > 0 :
        file_name = stack.pop()
        file_path = os.path.join(dataset, file_name)
        # where is file_path
        if (os.path.isdir(file_path)):
            push_data_to_stack(stack, file_path, file_name)
        else :
            print('%s' % file_path)
            sys.stdout.flush()
            with open(file_path, 'r', encoding='utf-16') as fp:

                content = unicodedata.normalize('NFKC', fp.read())
                arr = content.split()
                if len(arr) > 300:
                    arr = arr[:300]
                content = (" ").join(arr)

                dir_name = get_dir_name(file_path)

                x.append(content)

                y.append(my_map.name2onehot[dir_name])
    y = np.array(y)
    return x,y

def push_data_to_stack(stack, file_path, file_name):
    sub_folder = os.listdir(file_path)
    for element in sub_folder:
        element = file_name + '/' + element
        stack.append(element)
def get_dir_name(path):
    x = os.path.dirname(path)
    return os.path.basename(x)

def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]

if __name__ == '__main__':
    x,y =load_dataset_from_disk('dataset_small/train')
    print(x)
