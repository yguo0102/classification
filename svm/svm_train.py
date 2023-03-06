import sys
import pandas as pd
import numpy as np
import pickle

from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score, accuracy_score, classification_report
import statistics as stats
import feature_generator

from datetime import datetime
import pickle

import random

random.seed(0)

def main(train_file, dev_file, test_file, out_file, out_model):

    metrics = {}

    # get features
    train_X, train_classes, dev_X, dev_classes = \
            feature_generator.fit_transform(train_file, dev_file)

    # training with grid search
    c_params = [2, 4, 6, 8, 10]
    kernel_params = ['linear', 'rbf']
    parameters = {'kernel':kernel_params, 'C':c_params}

    svc = svm.SVC(cache_size=200, gamma='auto', probability=True)

    clf = GridSearchCV(svc, parameters, cv=5)
    clf.fit(train_X, train_classes)
    print('best params:', clf.best_params_)
    print('best score:', clf.best_score_)
    metrics['best_score'] = clf.best_score_

    # write dev predictions
    predictions = clf.predict(dev_X)
    acc = accuracy_score(dev_classes, predictions)
    print('evaluation on dev acc:{0:.2f}'.format(acc))
    metrics['eval_score'] = acc

    with open(out_model, 'wb') as handle:
        pickle.dump(clf, handle, protocol=pickle.HIGHEST_PROTOCOL)

    test_X = feature_generator.transform(test_file)
    predictions = clf.predict(test_X)
    df = pd.DataFrame({'index':[i for i in range(len(predictions))], 'prediction':predictions})
    df.to_csv(out_file, index=False, sep='\t')

    test_df = pd.read_csv(test_file)
    test_classes = test_df['label'].to_numpy()

    print(classification_report(test_classes, predictions))
    print('micro f1:{0:.2f}'.format(f1_score(test_classes, predictions, average='micro')))

    return df, metrics


def strfmt_prob(prob_array):
    ret = []
    n_samples, n_classes = prob_array.shape
    for i in range(n_samples):
        str_prob = ','.join([str(prob_array[i][j]) for j in range(n_classes)])
        ret.append(str_prob)
    return ret


if __name__ == '__main__':
    # load the data
    data_dir = sys.argv[1]
    train_file = f'{data_dir}/train.csv'
    dev_file = f'{data_dir}/dev.csv'
    test_file = f'{data_dir}/test.csv'
    out_file = f'{data_dir}/svm_test_result.txt'
    out_model = f'{data_dir}/svm_model.pickle'
    main(train_file, dev_file, test_file, out_file, out_model)
