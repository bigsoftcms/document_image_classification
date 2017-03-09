from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer
from sklearn.pipeline import make_pipeline
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics.pairwise import cosine_similarity, pairwise_distances_argmin

from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt


if __name__ == '__main__':
    # Kmeans approach
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

    vectorizer = TfidfVectorizer(max_df=.5)
    X = vectorizer.fit_transform(lemmatized_corpus)
    svd = TruncatedSVD(n_components=100)
    normalizer = Normalizer(copy=False)
    lsa = make_pipeline(svd, normalizer)
    X = lsa.fit_transform(X)

    km = MiniBatchKMeans(n_clusters=30, init='k-means++', n_init=1, init_size=500, batch_size=100, random_state=1)
    km.fit(X)

    original_space_centroids = svd.inverse_transform(km.cluster_centers_)
    order_centroids = original_space_centroids.argsort()[:, ::-1]

    terms = vectorizer.get_feature_names()
    for i in range(30):
        print 'Cluster {}:'.format(i)
        for ind in order_centroids[i, :10]:
            print terms[ind]
        print


    fig = plt.figure(figsize=(8, 3))
    # fig.subplots_adjust(left=0.02, right=0.98, bottom=0.05, top=0.9)
    colors = ['#4EACC5', '#FF9C34', '#4E9A06']

    # MiniBatchKMeans and the KMeans algorithm.
    mbk_means_cluster_centers = np.sort(km.cluster_centers_, axis=0)
    mbk_means_labels = pairwise_distances_argmin(X, mbk_means_cluster_centers)

    # MiniBatchKMeans
    ax = fig.add_subplot(1, 3, 1)
    for k, col in zip(range(30), colors):
        my_members = mbk_means_labels == k
        cluster_center = mbk_means_cluster_centers[k]
        ax.plot(X[my_members, 0], X[my_members, 1], 'w',
                markerfacecolor=col, marker='.')
        ax.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col, markeredgecolor='k', markersize=6)
    ax.set_title('KMeans')
    ax.set_xticks(())
    ax.set_yticks(())


    frequency = defaultdict(int)
    for doc in lemmatized_corpus:
        for token in doc.split():
            frequency[token] += 1

    processed_corpus = [[token for token in text if frequency[token] > 1] for text in texts]


    generate tfidf matrix and vectorizer
    tfidf_mat, vectorizer = tfidf_(txt_paths)
