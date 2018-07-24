# -*- encoding: utf-8 -*-


from pyfasttext import FastText
from sklearn.externals import joblib
from pyvi import ViTokenizer
import unicodedata
from sklearn.metrics import accuracy_score, confusion_matrix

class classification:
    def __init__(self):
        self.model = None
    def load_model(self):
        try:
            model = FastText('model/fasttext_not_process.bin')
        except:
            model = None
        if model == None:
            model = FastText()
            model.supervised(input= 'data_test/train_not_process.txt', output='model/fasttext_not_process')
        self.model = model
    def predict(self, content):
        self.load_model()
        model = self.model
        return model.predict_proba([content])
    def testing(self,file_name):
        self.load_model()
        labels = []
        contents = []
        with open(file_name, 'r') as f:
            for line in f:

                if len(line) < 10:
                    continue
                label, content = line.split(" ", 1)
                label_process = label.split("__",2)
                print(label_process)
                label = label_process[2]
                print(label)
                labels.append(label)
                contents.append(content)

        f.close()
        self.evaluation(contents, labels)


    def evaluation(self, X, y):
        y_pred = self.model.predict_proba(X)
        y_pred = map(lambda s: s[0][0], y_pred)
        print(y_pred)
        accuracy = accuracy_score(y, y_pred)
        print('Accuracy = %.5f' % (accuracy))
        confusion = confusion_matrix(y, y_pred)
        print(confusion)

if __name__ == '__main__':
    c = classification()
    c.testing('data_test/test_not_process.txt')

# accuracy = 0.905