# -*- encoding: utf-8 -*-

import utils
import os
import my_map
import preprocessing
from sklearn.externals import joblib
from io import open
import embedding
import network
from keras.models import load_model
from keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score, confusion_matrix

class classification:
    def __init__(self, root_dir='.'):
        self.model = None
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
    def load_model(self):
        try:
            self.model = load_model('model/model.h5')
        except:
            self.model = None

    def load_training_vector(self):
        X_train = self.load('model/X_train.pkl')
        y_train = self.load('model/y_train.pkl')
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
    def save_model(self):
        utils.mkdir('model')
        self.model.save('model/model.h5')

    def save_training_vector(self, X_train, y_train):
        utils.mkdir('model')
        self.save(X_train, 'model/X_train.pkl')
        self.save(y_train, 'model/y_train.pkl')

    def save_testing_vector(self, X_test, y_test):
        utils.mkdir('model')
        self.save(X_test, 'model/X_test.pkl')
        self.save(y_test, 'model/y_test.pkl')

    def prepare_data(self, dataset):
        X = []; y = []
        for name, list_content in dataset.items():
            label = my_map.name2label[name]
            for content in list_content:
                y.append(label)
                X.append(content)
        return X, y
    def split_validation(self, samples_train):
        samples_val = {}
        for cat in samples_train:
            samples = samples_train[cat]
            boundary = int(round(0.9 * len(samples)))
            samples_val.update({cat : samples[boundary : ]})
            samples_train[cat] = samples[: boundary]
        return samples_val

    def training(self, data_train, data_test):
        n_labels = len(my_map.label2name)
        X_train, y_train = self.load_training_vector()
        if X_train is None or y_train is None:
            samples_train = preprocessing.load_dataset_from_disk(data_train)
            samples_val = self.split_validation(samples_train)

            X_train, y_train = self.prepare_data(samples_train)
            X_val, y_val = self.prepare_data(samples_val)

            X_train = embedding.construct_tensor_word(X_train, self.max_length)
            y_train = utils.convert_list_to_onehot(y_train, n_labels)

            X_val = embedding.construct_tensor_word(X_val, self.max_length)
            y_val = utils.convert_list_to_onehot(y_val, n_labels)
            self.save_training_vector(X_train, y_train)
        self.fit(X_train, y_train,  X_val, y_val)

        # X_test, y_test = self.load_testing_vector()
        # if X_test is None or y_test is None:
        #     samples_test = preprocessing.load_dataset_from_disk(data_test)
        #     X_test, y_test = self.prepare_data(samples_test)
        #     X_test = embedding.construct_tensor_word(X_test, self.max_length)
        #     y_test = utils.convert_list_to_onehot(y_test, n_labels)
        #     self.save_testing_vector(X_test, y_test)
        # self.evaluation(X_test, y_test)
        self.save_model()


    def fit(self, X, y, X_val, y_val):
        print('build model...')
        # build network
        num_lstm_layer = 2
        num_hidden_node = 32
        dropout = 0.2
        self.model = network.building_network(num_lstm_layer, num_hidden_node, dropout,
                                              self.max_length, embedding.embedd_dim,
                                              len(my_map.label2name))
        print 'Model summary...'
        print self.model.summary()
        print 'Training model...'
        early_stopping = EarlyStopping(patience=self.patience)
        self.model.fit(X, y, batch_size=128, epochs=100,
                           validation_data=(X_val, y_val),
                           callbacks=[early_stopping])


    def evaluation(self, X, y):
        y_pred = self.model.predict_classes(X, batch_size=128)
        y = utils.convert_onehot_to_list(y)


        accuracy = accuracy_score(y, y_pred)
        print('Accuracy = %.5f' % (accuracy))
        confusion = confusion_matrix(y, y_pred)
        print(confusion)


    def run(self, data_train, data_test):
        self.load_model()
        if self.model == None:
            self.training(data_train, data_test)

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
    c.run('dataset_small/train', 'dataset_small/test')