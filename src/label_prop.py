import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

from sklearn import datasets
from sklearn.semi_supervised import label_propagation
from sklearn.metrics import classification_report, confusion_matrix


import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

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


def lower_filename(directory):
    '''
    INPUT: main directory where data resides
    OUTPUT: None
    TASK: removes white space from file names since shell to open files in other functions don't recognize white space.
    '''
    for path, _, files in os.walk(directory):
        for f in files:
            os.rename(os.path.join(path, f), os.path.join(path, f.lower()))


file_path = 'data/supervised'
lower_filename(file_path)
labels = np.array(extract_label(file_path))
data = extract_data(file_path)

vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5,
                             stop_words='english')
data_vect = vectorizer.fit_transform(data)
# X_test = vectorizer.transform(data_test)


# digits = datasets.load_digits()
rng = np.random.RandomState(0)
indices = np.arange(len(data))
rng.shuffle(indices)

X = data_vect[indices[:250]].toarray()
le = LabelEncoder()
y = le.fit_transform(labels[indices[:250]])
# images = digits.images[indices[:330]]

n_total_samples = len(y)
n_labeled_points = 30

unlabeled_indices = np.arange(n_total_samples)[n_labeled_points:]
# f = plt.figure()

for i in range(5):
    y_train = np.copy(y)
    y_train[unlabeled_indices] = -1

    lp_model = label_propagation.LabelSpreading(kernel='knn', n_neighbors=7, max_iter=30, n_jobs=-1)
    lp_model.fit(X, y_train)

    predicted_labels = lp_model.transduction_[unlabeled_indices]
    true_labels = y[unlabeled_indices]

    cm = confusion_matrix(true_labels, predicted_labels,
                          labels=lp_model.classes_)

    print('Iteration %i %s' % (i, 70 * '_'))
    print("Label Spreading model: %d labeled & %d unlabeled (%d total)"
          % (n_labeled_points, n_total_samples - n_labeled_points, n_total_samples))

    print(classification_report(true_labels, predicted_labels))

    print("Confusion matrix")
    print(cm)

    # compute the entropies of transduced label distributions
    pred_entropies = stats.distributions.entropy(
        lp_model.label_distributions_.T)

    # select five digit examples that the classifier is most uncertain about
    uncertainty_index = uncertainty_index = np.argsort(pred_entropies)[-5:]

    # keep track of indices that we get labels for
    delete_indices = np.array([])

    # f.text(.05, (1 - (i + 1) * .183),
    #        "model %d\n\nfit with\n%d labels" % ((i + 1), i * 5 + 10), size=10)
    # for index, image_index in enumerate(uncertainty_index):
    #     image = images[image_index]

        # sub = f.add_subplot(5, 5, index + 1 + (5 * i))
        # sub.imshow(image, cmap=plt.cm.gray_r)
        # sub.set_title('predict: %i\ntrue: %i' % (
        #     lp_model.transduction_[image_index], y[image_index]), size=10)
        # sub.axis('off')

        # labeling 5 points, remote from labeled set
        # delete_index, = np.where(unlabeled_indices == image_index)
        # delete_indices = np.concatenate((delete_indices, delete_index))

    unlabeled_indices = np.delete(unlabeled_indices, delete_indices)
    n_labeled_points += 10

# f.suptitle("Active learning with Label Propagation.\nRows show 5 most "
#            "uncertain labels to learn with the next model.")
# plt.subplots_adjust(0.12, 0.03, 0.9, 0.8, 0.2, 0.45)
# plt.show()
