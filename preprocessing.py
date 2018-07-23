# -*- encoding: utf-8 -*-

import regex
import os, sys
import my_map
import utils
from io import open
import unicodedata
from tokenizer.tokenizer import Tokenizer

r = regex.regex()
tokenizer = Tokenizer()

def load_dataset_from_disk(dataset):
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
            utils.push_data_to_stack(stack, file_path, file_name)
        else :
            print('%s' % file_path)
            sys.stdout.flush()
            with open(file_path, 'r', encoding='utf-16') as fp:

                content = unicodedata.normalize('NFKC', fp.read())

                #tokenizer content
                content = r.run(tokenizer.predict(content))
                #dir name of file_path
                dir_name = utils.get_dir_name(file_path)
                list_samples[dir_name].append(content)
    print('')
    return list_samples

def token(doc):
    doc = unicodedata.normalize('NFKC', doc)
    new_doc = r.run(tokenizer.predict(doc))
    return new_doc
def load_dataset_from_list(list_samples):
    result = []
    for sample in list_samples:
        sample = r.run(tokenizer.predict(sample))
        result.append(sample)
    return result




if __name__ == '__main__':
    load_dataset_from_disk('dataset/train')