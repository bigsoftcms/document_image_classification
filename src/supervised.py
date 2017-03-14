import os
import numpy as np

from time import time
from scipy.stats import randint as sp_randint

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score

# classifiers
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.svm import SVC


def lower_filename(directory):
    '''
    INPUT: main directory where data resides
    OUTPUT: None
    TASK: removes white space from file names since shell to open files in other functions don't recognize white space.
    '''
    for path, _, files in os.walk(directory):
        for f in files:
            os.rename(os.path.join(path, f), os.path.join(path, f.lower()))


def extract_label(directory):
    labels = []
    for path, _, files in os.walk(directory):
        for f in files:
            if f[-3:] == 'txt':
                labels.append(f.rsplit('-', 2)[-2])

    return labels


def extract_data(directory):
    data = []
    for path, _, files in os.walk(file_path):
        for f in files:
            if f[-3:] == 'txt':
                with open(os.path.join(path, f)) as t:
                    data.append(t.read())

    return data


# Utility function to report best scores
def report(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print 'Model with rank: {0}'.format(i)
            print 'Mean validation score: {0:.3f} (std: {1:.3f})'.format(results['mean_test_score'][candidate], results['std_test_score'][candidate])
            print 'Parameters: {0}'.format(results['params'][candidate])
            print('')


if __name__ == '__main__':
    file_path = 'data/supervised'
    # lower_filename(file_path)

    labels = extract_label(file_path)

    data_tokenized = extract_data(file_path)

    data_train, data_test, y_train, y_test = train_test_split(data_tokenized, labels, test_size=0.33, random_state=42)

    vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5,
                                 stop_words='english')
    X_train = vectorizer.fit_transform(data_train)
    X_test = vectorizer.transform(data_test)

    # best parameters based on GridSearchCV
    clf = RandomForestClassifier(n_estimators=400, max_depth=20, max_features='sqrt', bootstrap=False, n_jobs=-1, random_state=1)


    # # use a full grid over all parameters
    # param_grid = {'oob_score': [True, False],
    #               "max_depth": [3, None],
    #               "max_features": [1, 3, 10],
    #               "min_samples_split": [1, 3, 10],
    #               "min_samples_leaf": [1, 3, 10],
    #               "bootstrap": [True, False],
    #               "criterion": ["gini", "entropy"]}

    # # run grid search
    # grid_search = GridSearchCV(clf, param_grid=param_grid)
    # start = time()
    # grid_search.fit(X_train, y_train)
    #
    # print 'GridSearchCV took %.2f seconds for %d candidate parameter settings.' % (time() - start, len(grid_search.cv_results_['params']))
    #
    # report(grid_search.cv_results_)

    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)

    score = accuracy_score(y_test, pred)
    print 'accuracy: {}'.format(score)
