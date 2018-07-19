# -*- encoding: utf-8 -*-

import regex
import os, sys
import my_map
import utils
from io import open
import unicodedata
from tokenizer.tokenizer import Tokenizer

tokenizer = Tokenizer()
spliter = tokenizer.spliter


r = regex.regex()


def load_dataset_from_disk(dataset, max_length):
    total_words = set([])
    samples = {k:[] for k in my_map.name2label.keys()}
    stack = os.listdir(dataset)
    print 'loading data in ' + dataset
    while (len(stack) > 0):
        file_name = stack.pop()
        file_path = os.path.join(dataset, file_name)
        if (os.path.isdir(file_path)):
            utils.push_data_to_stack(stack, file_path, file_name)
        else:
            print('\r%s' % (file_path)),
            sys.stdout.flush()
            with open(file_path, 'r', encoding='utf-16') as fp:
                content = unicodedata.normalize('NFKC', fp.read())
                sentences = filter(lambda s: len(s) > 0, spliter.split(content))
                sentences = map(lambda s: r.run(tokenizer.predict(s)), sentences)
                content = u'\n'.join(sentences).lower()
                words = content.split()
                words = words[:max_length]
                total_words.update(words)

                dir_name = utils.get_dir_name(file_path)
                samples[dir_name].append(u' '.join(words[:max_length]))

    print('')
    print('there are %d words' % (len(total_words)))
    return samples, len(total_words)


def load_dataset_from_list(list_samples, max_length):
    result = []
    for sample in list_samples:
        sentences = filter(lambda s: len(s) > 0, spliter.split(sample))
        sentences = map(lambda s: r.run(tokenizer.predict(s)), sentences)
        sample = u'\n'.join(sentences).lower()
        words = sample.split()
        result.append(words[:max_length])
    return result




if __name__ == '__main__':
    load_dataset_from_disk('dataset_small/train', 500)