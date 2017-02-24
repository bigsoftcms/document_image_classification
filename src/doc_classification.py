import os
import numpy as np
from collections import Counter, defaultdict
from string import punctuation, ascii_lowercase, digits
import operator
from itertools import chain, product
import logging
import matplotlib.pyplot as plt
from unidecode import unidecode

import subprocess
import multiprocessing as mp
from timeit import timeit
import time

import spacy

import seaborn as sns; sns.set()

from PIL import Image, ImageSequence

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity, pairwise_distances_argmin
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn.cluster import KMeans, MiniBatchKMeans

from gensim import corpora, models, similarities
from gensim.parsing.preprocessing import STOPWORDS


# run using %run -i <runfile.py> to avoid re-loading nlp as it takes time to complete
if not 'nlp' in locals():
    print 'Loading English Module...'
    nlp = spacy.load('en')
    print 'Completed Loading English Module.'


def remove_filename_spaces(directory):
    '''
    INPUT: main directory where data resides
    OUTPUT: None
    TASK: removes white space from file names since shell to open files in other functions don't recognize white space.
    '''
    for path, _, files in os.walk(directory):
        for f in files:
            if ' ' in f:
                os.rename(os.path.join(path, f), os.path.join(path, f.replace(' ', '')))


def doc_cnts_paths(data_path):
    '''
    INPUT: path to data repository
    OUTPUT: (8) specific file counts and paths based on file extensions (i.e. .tif, .xml, .txt)
    '''
    tif_cnt, xml_cnt, txt_cnt, misc_cnt = 0, 0, 0, 0
    tif_paths, xml_paths, txt_paths, misc_paths = [], [], [], []
    for dirpath, dirnames, filenames in os.walk(data_path):
        for file in filenames:
            if file.endswith('.tif'):
                tif_cnt += 1
                tif_paths.append(os.path.join(dirpath, file))
            elif file.endswith('.xml'):
                xml_cnt += 1
                xml_paths.append(os.path.join(dirpath, file))
            elif file.endswith('.txt'):
                txt_cnt += 1
                txt_paths.append(os.path.join(dirpath, file))
            else:
                misc_cnt += 1
                misc_paths.append(os.path.join(dirpath, file))
    return tif_cnt, xml_cnt, txt_cnt, misc_cnt, tif_paths, xml_paths, txt_paths, misc_paths


def img_info(tif_paths):
    '''
    INPUT: absolute paths to .tif documents
    OUTPUT: (1) prints counts of different compressions and dpi ranges for all .tif documents. Returns 'info' list containing absolute path to image, compression format, and dpi
    '''
    info = []
    comp_cnt, dpi_cnt = Counter(), Counter()
    for path in tif_paths:
        img = Image.open(path)
        info.append((path, img.info['compression'], img.info['dpi']))
    for desc in info:
        comp_cnt[desc[1]] += 1
        dpi_cnt[desc[2]] += 1
    print 'Compression Counts: {0} \nDPI Counts: {1}'.format(comp_cnt, dpi_cnt)
    return info


def shell_tesseract(path):
    '''
    INPUT: absolute path to .tif document
    TASK: performs OCR using tesseract from the shell. Creates a text file from the OCRd document using the same name and location as the .tif document.
    OUTPUT: None
    '''
    # tesseract automatically adds a .txt extension to the OCRd document. Name of new document is 3rd argument + .txt added by tesseract
    subprocess.call(['tesseract', path, path[:-4]])


def parallelize_OCR(tif_paths):
    '''
    INPUT: paths to .tif files.
    OUTPUT: time taken to complete task
    TASK: parallelize OCR of .tif files by calling shell_tesseract and using multiprocessing Pool.
    ISSUES: not tested as a function yet. Would like to print a progress report every 15 to 30 minutes.
    '''
    # parallelize OCR processing and time it
    pool = mp.Pool(processes=4)
    task = pool.map(shell_tesseract, tif_paths)
    return timeit(lambda: task, number=1)


def lemmatize_string(doc, stop_words):
    '''
    INPUT: string that corresponds to a document in a raw corpus and english stop words. Gensim's package STOPWORDS has 337 words.
    OUTPUT: (1) a lemmatized string that corresponds to a corpus document. The string's punctuation is removed, words are lemmatized (words brought back to their 'base' form), and stop_words are removed.
    PACKAGE USED: spaCy
    '''
    # First remove punctuation from string
    # .translate is a string operation
    # spaCy expects a unicode object
    # doc_rmv_uni = doc.decode('unicode_escape').encode('ascii', 'ignore')
    doc_for_lemmatization = unidecode(' '.join(doc.translate(None, punctuation).translate(None, digits).replace('\n', ' ').split()).decode('utf-8')).decode('utf-8')

    # Run the doc through spaCy
    doc_spacy = nlp(doc_for_lemmatization)

    # Lemmatize and lower text
    tokens = [token.lemma_.lower() for token in doc_spacy]

    return ' '.join(w for w in tokens if w not in stop_words)


def lemmatize_corpus(txt_paths):
    '''
    INPUT: paths to OCRd .tif files that are in .txt format.
    OUTPUT: (1) lemmatized corpus
    TASK: uses 'lemmatize_string' function to create a list of lemmatized documents that correspond to all the documents in the raw corpus.
    '''
    raw_corpus = []
    for path in txt_paths:
        with open(path) as file:
            raw_corpus.append(file.read())

    stop_two_letters = [''.join(cb) for cb in product(ascii_lowercase, ascii_lowercase)]
    stop_specific = ['wattenberg', 'yes', 'na', '----', '4n', 'n2', 'acre', "'s", 'pm', '--', 'number', "''", 'ii', 'iii', 'um', 'mu', 'mm', 'mum', 'nwse', 'swne']

    stoplist = STOPWORDS.union([c for c in ascii_lowercase]).union([p for p in punctuation]).union([d for d in digits]).union(stop_specific).union(stop_two_letters)

    lemmatized_corpus = [lemmatize_string(doc, stoplist) for doc in raw_corpus]

    return lemmatized_corpus


def parallel_corpus_lemmatization(txt_paths):
    '''
    INPUT: paths to OCRd .tif files that are in .txt format.
    OUTPUT: (1) lemmatized corpus
    TASK: use multiprocessing Pool to parallelize task using all cores on machine.
    ISSUES: Has some overlap with the 'lemmatize_corpus' function, which could be inneficient.
    '''
    cores = mp.cpu_count()
    n = len(txt_paths)/cores

    txt_paths_chunks = [txt_paths[i:i + n] for i in xrange(0, len(txt_paths), n)]

    pool = mp.Pool(processes=4)

    return list(chain(*pool.map(lemmatize_corpus, txt_paths_chunks)))


def bow_and_dict(lemmatized_corpus, no_below, no_above=0.5):
    '''
    INPUT: lemmatized_corpus. 'no_below' helps with filtering out tokens that appear in less than the 'no_below' number of documents specified. 'no_above' is a fraction of the total corpus and it helps with filtering out tokens that appear in more than the 'no_above' fraction of documents specified. Basically, helps to filter out ubiquitous words that were not caught by stop_words.
    OUTPUT: (1) dictionary, which is a collection of all the unique tokens in the corpus. (2) Bag of words corpus, which represents each document in the corpus as a list of tuples with two elements - token id (referenced to the dictionary) and token frequency.
    TASK: tokenizes documents, creates dictionary from tokens, reduces size of dictionary based on 'no_below' and 'no_above' parameters.
    PACKAGE USED: gensim
    '''
    doc_tokens = [doc.split() for doc in lemmatized_corpus]
    dictionary = corpora.Dictionary(doc_tokens)

    # words appearing in less than 'no_below' documents to be excluded from dictionary
    dictionary.filter_extremes(no_below=no_below)
    bow_corpus = [dictionary.doc2bow(text) for text in doc_tokens]

    return dictionary, bow_corpus


def inspect_classification(bow_corpus, model):
    '''
    INPUT: bow_corpus, model (trained).
    OUTPUT: list of .tif paths to the documents grouped under 'topic' by the model
    TASK: passing a document from the corpus into the model returns a list of the top topics with their corresponding probabilities. Sort and select topic with highest probability for that document. Aggregate results in a defaultdict with key='topic' and value='document number that references .txt/.tif path'.
    ISSUES: Trying to group documents into specific topics and order by decreasing amount of documents per topic; instead of using sparse code to populate insp_cnt list in name/main block.
    '''
    # top_topics_lst is a defaultdict of the form key=topic, value=documents whose most likely topic is that key
    top_topics_lst = defaultdict(list)
    for file_num, doc in enumerate(bow_corpus):
        top_topics_lst[sorted(model[doc], key=lambda x: x[1], reverse=True)[0][0]].append(file_num)

    # files_by_topic is a defaultdict of the form key=topic, value=file paths to images corresponding to the key
    files_by_topic = defaultdict(list)
    for topic in range(model.num_topics):
        try:
            files_by_topic[topic].append([path[:-3]+'tif' for path in np.array(txt_paths)[np.array(top_topics_lst[topic])]])
        except:
            pass

    return top_topics_lst, files_by_topic


def plot_doc_topics(bow_corpus, model):
    '''
    INPUT: bow_corpus, trained model
    OUPUT: doc_topics matrix with shape -> rows=number of documents, columns=number of topics. Values are the probabilities of each topic being represented in the document. Also displays and saves the matrix as a seaborn heatmap with the probability bar highlighted on the right side of the plot.
    '''
    doc_topics = np.zeros((len(bow_corpus), model.num_topics))
    for i, doc in enumerate(bow_corpus):
        tops, probas = zip(*model.get_document_topics(doc))
        for j, top in enumerate(tops):
            doc_topics[i,top] = probas[j]
    sns.heatmap(doc_topics, yticklabels=50)
    plt.yticks(rotation=0)
    plt.ylabel('Document Number', fontsize=14)
    plt.xlabel('Topic Number', fontsize=14)
    plt.title('Probability of Topic for each Document', fontsize=14)
    plt.show()
    plt.savefig('figures/document_topics_heatmap.png')
    return doc_topics


def tfidf_vect(lemm_corpus):
    '''
    INPUT: absolute paths to .txt documents
    OUTPUT: tfidf matrix and vectorizer
    '''
    vectorizer = TfidfVectorizer(min_df=1)
    vectorizer.fit_transform(lemm_corpus)
    return vectorizer


def doc_sim(txt_paths, path, tfidf_matrix):
    '''
    INPUT: paths to OCRd text files, path to .txt document of interest, tfidf_matrix obtained from tfidf_ function
    TASK: display the document of interest and its 5 top most similar documents
    ISSUES: G4 compression TIF documents can't be displayed with PIL.Image.open('file_name').show()
    OUTPUT: None
    '''
    txt_arr = np.array(txt_paths)
    txt_idx = np.where(txt_arr == path)[0][0]
    sim = cosine_similarity(tfidf_matrix[txt_idx], tfidf_matrix)[0]
    desc_idx_5 = np.argsort(sim)[::-1][:5]
    sim_txt_paths = txt_arr[desc_idx_5]
    sim_tif_paths = [path[:-4]+'.tif' for path in sim_txt_paths]
    for path in sim_tif_paths:
        try:
            img = Image.open(path)
            img.show()
        except:
            pass


if __name__ == '__main__':
    data_path = '/Users/jpc/Documents/data_science_inmersive/document_image_classification/data/Wells'

    # allows display of gensim LDA results as the model is being fitted
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    # store counts and paths for files of interest
    tif_cnt, xml_cnt, txt_cnt, misc_cnt, tif_paths, xml_paths, txt_paths, misc_paths = doc_cnts_paths(data_path)

    # lemmatized_corpus = lemmatize_corpus(txt_paths)
    lemmatized_corpus = parallel_corpus_lemmatization(txt_paths)

    # # 'no_below' changes the size of the dictionary
    # # adjust for different classification results
    dictionary, bow_corpus = bow_and_dict(lemmatized_corpus, no_below=75, no_above=0.9)

    # just tried 40 passes. waiting to review results
    lda = models.LdaMulticore(bow_corpus, id2word=dictionary, num_topics=25, passes=40, chunksize=500, random_state=1, workers=4)

    corpora.MmCorpus.serialize('src/lda_mod/well_docs.mm', bow_corpus)
    dictionary.save('src/lda_mod/well_docs.dict')
    lda.save('src/lda_mod/well_docs.model')

    top_topics_lst, files_by_topic = inspect_classification(bow_corpus, lda)

    sorted(top_topics_lst.iteritems(),key=lambda (k,v): len(v),reverse=True)

    # lda.show_topics(-1, formatted=False)
    #
    #
    # for path in files_by_topic[34][0][:30]:
    #     !open {path}







##################################################
    # # Kmeans approach
    # logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    #
    # vectorizer = TfidfVectorizer(max_df=.5)
    # X = vectorizer.fit_transform(lemmatized_corpus)
    # svd = TruncatedSVD(n_components=100)
    # normalizer = Normalizer(copy=False)
    # lsa = make_pipeline(svd, normalizer)
    # X = lsa.fit_transform(X)
    #
    # km = MiniBatchKMeans(n_clusters=30, init='k-means++', n_init=1, init_size=500, batch_size=100, random_state=1)
    # km.fit(X)
    #
    # original_space_centroids = svd.inverse_transform(km.cluster_centers_)
    # order_centroids = original_space_centroids.argsort()[:, ::-1]
    #
    # terms = vectorizer.get_feature_names()
    # for i in range(30):
    #     print 'Cluster {}:'.format(i)
    #     for ind in order_centroids[i, :10]:
    #         print terms[ind]
    #     print
    #
    # # Adjust code!!
    # fig = plt.figure(figsize=(8, 3))
    # # fig.subplots_adjust(left=0.02, right=0.98, bottom=0.05, top=0.9)
    # colors = ['#4EACC5', '#FF9C34', '#4E9A06']
    #
    # # We want to have the same colors for the same cluster from the
    # # MiniBatchKMeans and the KMeans algorithm. Let's pair the cluster centers per
    # # closest one.
    # mbk_means_cluster_centers = np.sort(km.cluster_centers_, axis=0)
    # mbk_means_labels = pairwise_distances_argmin(X, mbk_means_cluster_centers)
    #
    # # MiniBatchKMeans
    # # Adjust code!!
    # ax = fig.add_subplot(1, 3, 1)
    # for k, col in zip(range(30), colors):
    #     my_members = mbk_means_labels == k
    #     cluster_center = mbk_means_cluster_centers[k]
    #     ax.plot(X[my_members, 0], X[my_members, 1], 'w',
    #             markerfacecolor=col, marker='.')
    #     ax.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col, markeredgecolor='k', markersize=6)
    # ax.set_title('KMeans')
    # ax.set_xticks(())
    # ax.set_yticks(())
    #
    #
    #
    # frequency = defaultdict(int)
    # for doc in lemmatized_corpus:
    #     for token in doc.split():
    #         frequency[token] += 1
    #
    # processed_corpus = [[token for token in text if frequency[token] > 1] for text in texts]
    #




    # generate tfidf matrix and vectorizer
    # tfidf_mat, vectorizer = tfidf_(txt_paths)
