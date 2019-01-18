import scipy
import arff
import pandas as pd
import numpy as np
from pandas import Series
import logging
import lda
from skmultilearn.problem_transform import ClassifierChain
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import libsvm
from skmultilearn.problem_transform import BinaryRelevance, ClassifierChain, LabelPowerset
from skmultilearn.adapt.mlknn import MLkNN
from skmultilearn.ensemble.rakelo import RakelO, RakelD
from skmultilearn.neurofuzzy.MLARAMfast import MLARAM
from sklearn.metrics import accuracy_score, hamming_loss, average_precision_score, f1_score
from skmultilearn.adapt import MLkNN
from sklearn.model_selection import KFold
import scipy.sparse as sp
import os

base_dir = "./dataset/"
dir = os.path.join(base_dir, 'medical')
num_label = 45


class Multi_Learning:

    # load train_test_data
    def get_train_test(self):
        dataset = arff.load(open(os.path.join(dir, "medical-train.arff")), encode_nominal=True)
        dataset = np.array(dataset.get("data"))

        m, n = dataset.shape
        X_train = dataset[:int(0.2 * m), :-num_label]
        y_train = dataset[:int(0.2 * m), -num_label:]

        dataset = arff.load(open(os.path.join(dir, "medical-test.arff")), encode_nominal=True)
        dataset = np.array(dataset.get("data"))

        X_test = dataset[:, :-num_label]
        y_test = dataset[:, -num_label:]

        return np.array(X_train), np.array(y_train), np.array(X_test), np.array(y_test)

    # load train_test with label topics.
    def get_train_test_lda(self, topic):

        # get training set
        dataset = arff.load(open(os.path.join(dir, "medical-train.arff")), encode_nominal=True)
        dataset = np.array(dataset.get("data"))

        X_train = dataset[:, :-num_label]
        y_train = dataset[:, -num_label:]

        # get test set
        dataset = arff.load(open(os.path.join(dir, "medical-test.arff")), encode_nominal=True)
        dataset = np.array(dataset.get("data"))

        X_test = dataset[:, :-num_label]
        y_test = dataset[:, -num_label:]

        for k in topic:
            X_iter = X_train.astype(np.int64)

            # get training_data feature topics
            model = lda.LDA(n_topics=k, n_iter=1000)
            model.fit(X_iter)
            doc_topic_x = model.doc_topic_

            # get training data label topics
            model_label = lda.LDA(n_topics=k, n_iter=1000)
            model_label.fit(y_train)
            doc_topic_y = model_label.doc_topic_

            # concat feature-topic and label topic
            x = np.hstack((doc_topic_x, doc_topic_y))

            # discretize the topics
            x = self.discretization_doc_topic(x)
            X_train = np.hstack((X_train, x))

            # multi-label learning to get test_data label topics and feature topics
            classifier = BinaryRelevance(RandomForestClassifier())
            classifier.fit(X_iter, x)
            x = np.array(sp.csr_matrix(classifier.predict(X_test)).toarray())

            X_test = np.hstack((X_test, x))

        return np.array(X_train), np.array(y_train), np.array(X_test), np.array(y_test)

    # discretization_doc_topic
    def discretization_doc_topic(self, theta):
        n, k = theta.shape
        Y = np.zeros((n, k))
        for i in range(n):
            MAX = np.max(theta[i])
            MIN = np.max(theta[i])
            for j in range(k):
                if (MAX - theta[i][j] < 1.0 / k):
                    # if(theta[i][j] > MAX - 1.0/k):
                    Y[i][j] = 1
                else:
                    Y[i][j] = 0
        return Y

    def multi_label_model(self, model_name=None, topics=[2, 3, 5]):

        if model_name is None:
            X_train, y_train, X_test, y_test = self.get_train_test()
        else:
            X_train, y_train, X_test, y_test = self.get_train_test_lda(topics)

        hammingloss = []
        f1_micro = []
        f1_macro = []
        f1_examples = []

        f1_max = 0
        best_classifier = None

        # 10-fold validation
        iters = 0
        kf = KFold(n_splits=10, shuffle=True, random_state=1)
        for train_index, val_index in kf.split(X_train):
            iters += 1
            print("start to %d training..." % (iters))
            train_X = X_train[train_index]
            val_X = X_train[val_index]
            train_y = y_train[train_index]
            val_y = y_train[val_index]

            if model_name == "BR":
                classifier = BinaryRelevance(RandomForestClassifier())
            elif model_name == "CC":
                classifier = ClassifierChain(RandomForestClassifier())
            elif model_name == "LP":
                classifier = LabelPowerset(RandomForestClassifier())
            elif model_name == "RAKEL":
                classifier = RakelO(RandomForestClassifier(), model_count=20, labelset_size=num_label)
            elif model_name == "MLKNN":
                classifier = MLkNN()
            else:
                classifier = BinaryRelevance(RandomForestClassifier())

            classifier.fit(train_X, train_y)

            predictions = classifier.predict(val_X)

            result0 = hamming_loss(val_y, predictions)
            result1 = f1_score(val_y, predictions, average='micro')
            result2 = f1_score(val_y, predictions, average='macro')
            result3 = f1_score(val_y, predictions, average='samples')

            hammingloss.append(result0)
            f1_micro.append(result1)
            f1_macro.append(result2)
            f1_examples.append(result3)

            if result1 > f1_max:
                f1_max = result1
                best_classifier = classifier

        print("10-fold results as follows:")
        print("hm: mean is %f, std is %f." % (np.mean(np.array(hammingloss)), np.std(np.array(hammingloss))))
        print("f1-micro: mean is %f, std is %f." % (np.mean(np.array(f1_micro)), np.std(np.array(f1_micro))))
        print("f1-macro: mean is %f, std is %f." % (np.mean(np.array(f1_macro)), np.std(np.array(f1_macro))))
        print("f1-samples: mean is %f, std is %f." % (np.mean(np.array(f1_examples)), np.std(np.array(f1_examples))))

        test_pred = best_classifier.predict(X_test)
        h1 = hamming_loss(y_test, test_pred)
        f1micro = f1_score(y_test, test_pred, average='micro')
        f1macro = f1_score(y_test, test_pred, average='macro')
        f1example = f1_score(y_test, test_pred, average='samples')

        print("test results as follows:")
        print("hamming loss: %f" % h1)
        print("f1-micro: %f" % f1micro)
        print("f1-macro: %f" % f1macro)
        print("f1-example: %f" % f1example)


def plot(x, y1, y2, y3):
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.interpolate import spline

    xnew1 = np.linspace(x.min(), x.max(), 100)
    ynew1 = spline(x, y1, xnew1)
    ynew2 = spline(x, y2, xnew1)
    ynew3 = spline(x, y3, xnew1)

    # 平滑处理后曲线
    plt.plot(xnew1, ynew1, 'r', label="Micro-F")
    plt.plot(xnew1, ynew2, 'g', label="Macro-F")
    plt.plot(xnew1, ynew3, 'b', label="Example-F")
    # 设置x,y轴代表意思
    plt.xlabel("weight")
    plt.ylabel("probability")
    # 设置标题
    plt.title("The content similarity of different distance")
    # 设置x,y轴的坐标范围
    plt.xlim(0, 1, 0.1)
    plt.ylim(0.5, 0.8)

    plt.legend()
    plt.show()


if __name__ == "__main__":
    test = Multi_Learning()
    test.multi_label_model(model_name="BR", topics=[2,3,5])

