import pandas as pd
from collections import OrderedDict
import itertools
import numpy as np
import re

from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_selection import SelectKBest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer


abs_vectorizer = TfidfVectorizer(ngram_range=(1, 4), max_features=1000, stop_words='english')

def load_data(f_path):
    '''
        Given a path, loads a data set and puts it into a dataframe
        - simplified mechanism
    '''
    #df = pd.read_csv(f_path, dtype=str, lineterminator='\n')
    df = pd.read_csv(f_path)
    return df

def fit_transform(train_file,
            test_file=None,
            feature_selection_dim=0):

    if isinstance(train_file , pd.DataFrame):
        train_data = train_file
    else:
        train_data = load_data(train_file)
    train_classes = train_data['label'].to_numpy()

    # vectorize
    max_features = 1000
    train_X_vec = abs_vectorizer.fit_transform(train_data['text']).toarray()
    train_X_tfidf = pd.DataFrame(train_X_vec, columns=[f"tfidf_{i}" for i in range(max_features)])
    train_X = train_X_tfidf

    if test_file is not None:
        if isinstance(test_file , pd.DataFrame):
            test_data = test_file
        else:
            test_data = load_data(test_file)
        print(test_data)
        test_classes = test_data['label'].to_numpy()

        test_X_vec = abs_vectorizer.transform(test_data['text']).toarray()
        test_X_tfidf = pd.DataFrame(test_X_vec, columns=[f"tfidf_{i}" for i in range(max_features)])
        test_X = test_X_tfidf
        return train_X, train_classes, test_X, test_classes

    else:
        return train_X, train_classes, None, None


def transform(test_file):

    if isinstance(test_file , pd.DataFrame):
        test_data = test_file
    else:
        test_data = load_data(test_file)
    test_classes = test_data['label'].to_numpy()

    # vectorize
    max_features = 1000
    test_X_vec = abs_vectorizer.transform(test_data['text']).toarray()
    test_X = pd.DataFrame(test_X_vec, columns=[f"tfidf_{i}" for i in range(max_features)])

    #print(test_X)
    return test_X

if __name__ == '__main__':
    import sys
    out = fit_transform('train.csv', 'dev.csv')
    print(out)

