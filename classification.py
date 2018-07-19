# -*- encoding: utf-8 -*-

import utils
import os
import my_map
import preprocessing
from sklearn.externals import joblib
from io import open
import numpy as np
import embedding
import network
from keras.models import load_model
from keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score, confusion_matrix
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences

class classification:
    def __init__(self, root_dir='.'):
        self.model = None
        self.model_name = None
        self.root_dir = root_dir
        self.result_dir = os.path.join(self.root_dir,'result')
        self.max_length = 300
        self.patience = 3

    def load(self, model):
        print('loading %s ...'% model)
        if os.path.isfile(model):
            return joblib.load(model)
        else:
            return None
    def load_model(self, name):
        try:
            self.model = load_model(name)
        except:
            self.model = None

    def load_training_vector(self):
        samples_train = self.load('model/samples_train.pkl')
        y_train = self.load('model/y_train.pkl')
        samples_val = self.load('model/samples_val.pkl')
        y_val = self.load('model/y_val.pkl')
        vocab_size = self.load('model/vocab_size.pkl')
        return samples_train, y_train, samples_val, y_val, vocab_size
        return X_train,y_train

    def load_testing_vector(self):
        X_test = self.load('model/X_test.pkl')
        y_test = self.load('model/y_test.pkl')
        return X_test, y_test

    def save(self, model, path):
        print('saving %s ...'% (path))
        utils.mkdir('model')
        joblib.dump(model, path, compress= True)
        return
    def save_model(self, name):
        utils.mkdir('model')
        self.model.save(name)

    def save_training_vector(self, samples_train, y_train, samples_val, y_val, vocab_size):
        utils.mkdir('model')
        self.save(samples_train, 'model/samples_train.pkl')
        self.save(y_train, 'model/y_train.pkl')
        self.save(samples_val, 'model/samples_val.pkl')
        self.save(y_val, 'model/y_val.pkl')
        self.save(vocab_size, 'model/vocab_size.pkl')

    def save_testing_vector(self, samples_test, y_test):
        utils.mkdir('model')
        self.save(samples_test, 'model/X_test.pkl')
        self.save(y_test, 'model/y_test.pkl')

    # return X,y from dataset
    def prepare_data(self, samples, vocab_size):
        encoded_samples = [];
        y = []
        for name, list_content in samples.items():
            label = my_map.name2label[name]
            for content in list_content:
                y.append(label)
                encoded_samples.append(one_hot(content, vocab_size))
        padded_samples = pad_sequences(encoded_samples, maxlen=self.max_length, padding='post')
        return padded_samples, y
    def split_validation(self, samples_train):

        samples_val = {}
        for cat in samples_train:
            samples = samples_train[cat]
            boundary = int(round(0.9 * len(samples)))
            samples_val.update({cat: samples[boundary:]})
            samples_train[cat] = samples[: boundary]
        return samples_val
    def training(self, data_train, data_test):
        n_labels = len(my_map.label2name)
        padded_train, y_train, padded_val, y_val, vocab_size = self.load_training_vector()

        if padded_train is None or y_train is None or vocab_size is None \
                or padded_val is None or y_val is None:
            samples_train, vocab_size = preprocessing.load_dataset_from_disk(data_train, self.max_length)
            samples_val = self.split_validation(samples_train)

            padded_train, y_train = self.prepare_data(samples_train, vocab_size)
            padded_val, y_val = self.prepare_data(samples_val, vocab_size)

            y_train = utils.convert_list_to_onehot(y_train, n_labels)
            y_val = utils.convert_list_to_onehot(y_val, n_labels)
            self.save_training_vector(padded_train, y_train, padded_val, y_val, vocab_size)

        self.fit(padded_train, y_train, padded_val, y_val, vocab_size)



        self.save_model(self.model_name)


    def fit(self, padded_train, y_train, padded_val, y_val, vocab_size):
        print('build model...')
        # build network
        num_lstm_layer = 2
        num_hidden_node = 32
        dropout = 0.2
        embedding_size = 32
        self.model = network.building_network(vocab_size, embedding_size,
                                              num_lstm_layer, num_hidden_node,
                                              dropout,
                                              self.max_length,
                                              len(my_map.label2name))
        print 'Model summary...'
        print self.model.summary()
        print 'Training model...'
        early_stopping = EarlyStopping(patience=self.patience)
        self.model.fit(padded_train, y_train, batch_size=128, epochs=100,
                       validation_data=(padded_val, y_val),
                       callbacks=[early_stopping])


    def evaluation(self, X, y):
        print(X)
        exit()
        # shape (None,300)
        y_pred = self.model.predict_classes(X, batch_size=128)
        print(y_pred)
        exit()
        y_pred = self.model.predict_classes(X, batch_size=128)
        y = utils.convert_onehot_to_list(y)

        print(y_pred)
        print(y)
        exit()
        accuracy = accuracy_score(y, y_pred)
        print('Accuracy = %.5f' % (accuracy))
        confusion = confusion_matrix(y, y_pred)
        print(confusion)


    def run(self, data_train, data_test,model_name):
        self.load_model(model_name)
        self.model_name = model_name
        if self.model == None:
            self.training(data_train, data_test)
        padded_train, y_train, padded_val, y_val, vocab_size = self.load_training_vector()

        samples_test, y_test = self.load_testing_vector()
        if samples_test is None or y_test is None:


            samples_test, _ = preprocessing.load_dataset_from_disk(data_test, self.max_length)
            padded_test, y_test = self.prepare_data(samples_test, vocab_size)

            self.save_testing_vector(padded_test, y_test)
        self.evaluation(samples_test, y_test)

    def predict(self, list_document):
        # docs = preprocessing.load_dataset_from_list(list_document)
        # X = self.feature_extraction(docs)
        # return self.model.predict(X)
        pass


    def save_to_dir(self, list_document, labels):
        utils.mkdir(self.result_dir)
        _ = map(lambda x: utils.mkdir(os.path.join(self.result_dir, x)), my_map.name2label.keys())
        for i in range(len(labels)):
            output_dir = os.path.join(self.result_dir, my_map.label2name[labels[i]])
            with open(os.path.join(output_dir, utils.id_generator()), 'w', encoding='utf-8') as fw:
                fw.write(unicode(list_document[i]))


if __name__ == '__main__':
    c = classification()
    c.run('dataset_small/train', 'dataset_small/test','model/model_embedlayer.h5')