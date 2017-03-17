'''
PROJECT: Document Classification Model
AUTHOR: Juan Pablo Centeno
DATE: 03/15/2017
PROGRAMMING LANGUAGE: Python 2.7.12
PYTHON TOOLS: TfidfVectorizer and RandomForestClassifier
TASK: classifies documents in the form of text extracted from OCRd .tif files by using a supervised approach: a representative selection of documents are labeled and fed into the model for fitting. The model predicts on unseen documents.
'''


import os
import shutil
import numpy as np
import pandas as pd
from time import time

# sklearn tools
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# classifier
from sklearn.ensemble import RandomForestClassifier


def lower_filename(directory):
    '''
    INPUT: main directory where data resides
    OUTPUT: None
    TASK: lowers all file names to have consistent labels
    '''
    for path, _, files in os.walk(directory):
        for f in files:
            os.rename(os.path.join(path, f), os.path.join(path, f.lower()))


def extract_label(directory):
    '''
    INPUT: path to files
    OUTPUT: list of labels
    TASK: extract label from file name
    '''
    labels = []
    for path, _, files in os.walk(directory):
        for f in files:
            if f[-3:] == 'txt':
                labels.append(f.rsplit('-', 2)[-2])

    return labels


def extract_data(directory):
    '''
    TASK: creates list of documents from OCRd images to be used as input data for model.
    '''
    data = []
    for path, _, files in os.walk(directory):
        for f in files:
            if f[-3:] == 'txt':
                with open(os.path.join(path, f)) as t:
                    data.append(t.read())

    return data


def report(results, n_top=3):
    '''
    TASK: utility function to report best scores from GridSearchCV. Code modified from Sklearn.
    '''
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print 'Model with rank: {0}'.format(i)
            print 'Mean validation score: {0:.3f} (std: {1:.3f})'.format(results['mean_test_score'][candidate], results['std_test_score'][candidate])
            print 'Parameters: {0}'.format(results['params'][candidate])
            print('')


def output_classification(origin, destination, untrained_pred):
    '''
    TASK: send labeled documents to supervised_results folder
    '''
    idx = 0
    for path, _, files in os.walk(origin):
        for f in files:
            if f[-3:] == 'tif':
                old = os.path.join(path, f)
                new_f = f.rsplit('-', 2)
                new_f[-2] = untrained_pred[idx]
                new_f = '-'.join(new_f)
                shutil.copyfile(os.path.join(path, f), os.path.join(destination, new_f))
                idx += 1


def scores_df(test_labels, scores_lst):
    '''
    INPUT: labels used for testing the model and scores_lst
    OUTPUT: dataframe of scores per label
    TASK: returns dataframe of scores per label
    '''
    data = np.vstack([np.unique(test_labels), scores_lst])
    cols = 'labels precision recall fscore support'.split()
    scores = pd.DataFrame(data).T
    scores.columns = cols

    return scores


if __name__ == '__main__':
    train_path = 'data/train_supervised'
    # lower_filename(train_path)

    labels = extract_label(train_path)
    data = extract_data(train_path)

    data_train, data_test, y_train, y_test = train_test_split(data, labels, test_size=0.33, stratify=labels, random_state=42)

    vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5,
                                 stop_words='english')
    X_train = vectorizer.fit_transform(data_train)
    X_test = vectorizer.transform(data_test)

    # # use a full grid over all parameters
    # param_grid = {'oob_score': [True, False],
    #               'max_depth': [3, None],
    #               'max_features': [1, 3, 10],
    #               'min_samples_split': [1, 3, 10],
    #               'min_samples_leaf': [1, 3, 10],
    #               'bootstrap': [True, False],
    #               'criterion': ['gini', 'entropy']}

    # # run grid search
    # grid_search = GridSearchCV(clf, param_grid=param_grid)
    # start = time()
    # grid_search.fit(X_train, y_train)
    #
    # print 'GridSearchCV took %.2f seconds for %d candidate parameter settings.' % (time() - start, len(grid_search.cv_results_['params']))
    #
    # report(grid_search.cv_results_)

    # best classifier parameters based on GridSearchCV
    clf = RandomForestClassifier(n_estimators=400, max_depth=20, max_features='sqrt', bootstrap=False, n_jobs=-1, random_state=1)

    # fit model and predict
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)

    # calculate scores for each label (document category)
    precision, recall, fscore, support = precision_recall_fscore_support(y_test, pred)

    # generate dataframe to explore model scores
    scores_lst = [precision, recall, fscore, support]
    scores_frame = scores_df(y_test, scores_lst)

    # calculate and print model accuracy
    accuracy = accuracy_score(y_test, pred)
    print 'overall model accuracy: {}'.format(accuracy)

    # print additional output regarding the model
    print ''
    print 'dataframe "scores_frame" available to explore model scores'

    # # use trained model to predict on unseen documents
    # to_pred_path = 'data/Wells'
    # data_to_pred = extract_data(to_pred_path)
    # X_to_pred = vectorizer.transform(data_to_pred)
    #
    # untrained_pred = clf.predict(X_to_pred)
    # destination = 'data/supervised_results'
    #
    # output_classification(to_pred_path, destination, untrained_pred)
