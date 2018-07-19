# -*- encoding: utf-8 -*-
import numpy as np
import pickle
import sys


vector_dir = 'embedding/vectors.npy'
word_dir = 'embedding/words.pl'

embedd_vectors = np.load(vector_dir)
with open(word_dir, 'rb') as handle:
    embedd_words = pickle.load(handle)
embedd_dim = np.shape(embedd_vectors)[1]

print(embedd_dim)
# gen embedding vector for unknown token
unknown_embedd = np.random.uniform(-0.01, 0.01, (1, embedd_dim))

def get_embedding(word):
    w = word.lower()
    try:
        embedd = embedd_vectors[embedd_words.index(w)]
    except:
        embedd = unknown_embedd
    return embedd


def construct_tensor_word(docs, max_length):


    print('construct tensor word for %d docs' % (len(docs)))
    X = np.empty([len(docs), max_length, embedd_dim])
    for i in range(len(docs)):
        words = docs[i]

        list_words = words.split(" ")

        length = len(list_words)
        for j in range(length):
            if j >= max_length:
                break;
            word = list_words[j].lower()
            try:
                embedd = embedd_vectors[embedd_words.index(word)]
            except:
                embedd = unknown_embedd
            X[i, j] = embedd
        # Zero out X after the end of the sequence <=> ZERO_PADDING
        X[i, length:] = np.zeros([1, embedd_dim])
        print('\rconstructed for document %d-th' % (i),
        sys.stdout.flush())
    return X

# result = construct_tensor_word("anh truong dep trai"*100, 3)
#
# print(result.ndim)