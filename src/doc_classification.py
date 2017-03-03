import modules.ocr_input_processing as oip
import modules.image_ocr as io
import modules.main_corpus_processing as mcp

import os
import random
import numpy as np
import pandas as pd
import subprocess
from collections import Counter, defaultdict
from itertools import chain
import dbfread
import utm

import logging
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import seaborn as sns; sns.set()

from PIL import Image

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity, pairwise_distances_argmin
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, silhouette_samples

from gensim import corpora, models, similarities


random.seed(1)


def train_lda(paths, dict_no_below, cores, n_topics):
    '''
    INPUT: paths to .txt files to train model. dict_no_below - remove token that is only present in less than the dict_no_below number of docs supplied. cores - how many are available for processing. n_topics determined to train the model.
    OUTPUT: (4) tokenized_corpus, dictionary, bow_corpus, LDA model
    '''
    tokenized_corpus = mcp.parallel_corpus_lemm_tokenization(paths)

    # no_below = 500 for a first LDA pass to filter out low token count documents, such as maps, photos, or poor quality scans
    dictionary, bow_corpus = mcp.bow_and_dict(tokenized_corpus, no_below=dict_no_below, no_above=0.9)

    chunk = len(paths)/cores

    # train the LDA model
    lda = models.LdaMulticore(bow_corpus, id2word=dictionary, num_topics=n_topics, passes=30, chunksize=chunk, random_state=1, workers=cores)

    return tokenized_corpus, dictionary, bow_corpus, lda


def remove_poor_quality_ocr(file_removal_idx):
    '''
    INPUT: file_removal_idx - index to .txt/.tif paths to be removed
    OUTPUT: (8) specific file counts and paths based on file extensions (i.e. .tif, .xml, .txt)
    TASK: MAKE COPY OF ENTIRE DATA SET TO DESKTOP BEFORE PROCEDING. Look for documents inside bow_corpus that might not have any tokens (maps, logs, rotated images, poor scans). If found, remove their corresponding .txt/.tif paths to then re-run lemm-tok-dict-bow process.
    '''
    if file_removal_idx:
        for i in file_removal_idx:
            print 'removing document indexed at: {0}'.format(i)
            try:
                os.rename(txt_paths[i], os.path.join('/Users/jpc/Documents/data_science_inmersive/document_image_classification/data/ignore_general',txt_paths[i].rsplit('/', 1)[-1]))
                os.rename(tif_paths[i], os.path.join('/Users/jpc/Documents/data_science_inmersive/document_image_classification/data/ignore_general',tif_paths[i].rsplit('/', 1)[-1]))
            except:
                print 'file not found'
    else:
        print 'All documents in bow_corpus have tokens.'

    return oip.doc_cnts_paths(data_path)


def inspect_bow_corpus(bow_corpus):
    '''
    INPUT: bow_corpus
    OUTPUT: list of indices to documents that contribute no tokens to bow_corpus
    '''
    file_removal_idx = []
    for idx, doc in enumerate(bow_corpus):
        if not doc:
            file_removal_idx.append(idx)

    return file_removal_idx


def inspect_classification(bow_corpus, model, txt_paths):
    '''
    INPUT: bow_corpus, model (trained), paths to .txt files used to train model.
    OUTPUT: list of .tif paths to the documents grouped under 'topic' by the model
    TASK: passing a document from the corpus into the model returns a list of the top topics with their corresponding probabilities. Sort and select topic with highest probability for that document. Aggregate results in a defaultdict with key='topic' and value='document number that references .txt/.tif path'.
    ISSUES: Trying to group documents into specific topics and order by decreasing amount of documents per topic; instead of using sparse code to populate insp_cnt list in name/main block.
    '''
    # top_topics_dict is a dictionary of the form key=topic, value=documents (with topic's highest probability) whose most likely topic is that key. top_topics_defdict is used to collect the information that is finalized in top_topics_dict
    top_topics_defdict = defaultdict(list)
    for file_num, doc in enumerate(bow_corpus):
        try:
            probs = np.array(model.get_document_topics(doc))
            top_topics_defdict[probs[probs[:,1].argsort()][::-1][0][0]].append((file_num, probs[probs[:,1].argsort()][::-1][0][1]))
        except:
            pass

    top_topics_dict = {k: np.array(v) for k, v in top_topics_defdict.items()}

    # files_by_topic_defdict is a defaultdict of the form key=topic, value=file paths to images corresponding to the key
    files_by_topic_defdict = defaultdict(list)
    for topic in range(model.num_topics):
        try:
            files_by_topic_defdict[topic].append([path[:-3]+'tif' for path in np.array(txt_paths)[top_topics_dict[topic][:,0].astype(int)]])
        except:
            pass

    doc_count = {}
    for topic, docs in top_topics_dict.items():
        doc_count[topic] = len(docs)

    print 'Documents per Topic: {}'.format(doc_count)

    return top_topics_dict, files_by_topic_defdict


def count_docs_per_topic(top_topics_dict):
    '''
    INPUT: top_topics_dict, which is a dict of highest probability of a topic for a document.
    OUTPUT: Dictionary with a count of documents per topic.
    '''
    doc_count = {}
    for topic, docs in top_topics_dict.items():
        doc_count[topic] = len(docs)

    return doc_count


def display_docs(files_by_topic_defdict, topic_number, low=0, high=30):
    '''
    INPUT: files_by_topic_defdict (defaultdict with documents belonging to specific topic). Topic number to be inspected. Range of documents to be displayed (low to high).
    OUTPUT: None
    TASK: uses subprocess to call 'open' on bash and display documents of interest.
    '''
    for path in files_by_topic_defdict[topic_number][0][low:high]:
        subprocess.call(['open', path])


def display_masked_docs(topic_masks, topic):
    '''
    INPUT: masked docs per topic
    OUTPUT: None
    TASK: check masked topics before blocking them from further processing
    '''
    for path in np.array(tif_paths)[topic_masks[topic]]:
        subprocess.call(['open', path])


def topic_token_dist(model, dictionary, num_words=10):
    '''
    INPUT: trained LDA model, dictionary, top num_words to include for each topic.
    OUTPUT: pandas dataframe of topic, top tokens, token dictionary id, and probability of token in a topic.
    '''
    tok_prob_lst = zip(*model.show_topics(-1, formatted=False, num_words=num_words))[1]
    topic_lst = []
    tok_lst = []
    prob_lst = []
    for t in range(model.num_topics):
        topic_lst.append(np.repeat(t, num_words).tolist())
        tok_lst.append(zip(*tok_prob_lst[t])[0])
        prob_lst.append(zip(*tok_prob_lst[t])[1])

    tokid_lst = [dictionary.token2id[i] for i in np.array(tok_lst).ravel()]

    df = pd.DataFrame({'topic': np.array(topic_lst).ravel(), 'token': np.array(tok_lst).ravel(), 'token_id': tokid_lst, 'token_prob': np.array(prob_lst).ravel()})

    return df


def plot_topic_token_dist():
    df_50 = topic_token_dist(lda, dictionary, num_words=50)
    p = sns.swarmplot(data=df_50, x='topic', y='token_prob', size=6)
    p.set(ylabel='', xlabel='')
    plt.text(-1.3, 0.06, 'Token Probability', va='center', rotation='vertical', size=14)
    plt.text(2.5, -0.037, 'Topic', ha='center', size=14)

    df_10 = topic_token_dist(lda, dictionary)
    grid = sns.FacetGrid(data=df_10, row='topic', hue='topic', sharey=False, size=1.75, aspect=2)
    grid.map(sns.barplot, 'token_prob', 'token')
    grid.set(ylabel='', xlabel='')
    plt.text(-0.07, -28, 'Tokens', va='center', rotation='vertical', size=14)
    plt.text(0.07, 13, 'Token Probability', ha='center', size=14)


def docs_topic_mask(model, top_topics_dict):
    '''
    INPUT: trained model, top_topics_dict
    OUTPUT: topic_masks to retrieve document numbers (.txt or .tif) for each topic created by the model.
    '''
    topic_masks = []
    for n in range(model.num_topics):
        topic_masks.append(top_topics_dict[n][:,0].astype(int))

    return topic_masks


def doc_topics_matrix(bow_corpus, model):
    '''
    INPUT: bow_corpus, trained model
    OUPUT: doc_topics matrix with shape -> rows=number of documents, columns=number of topics. Values are the probabilities of each topic being represented in the document.
    '''
    doc_topics = np.zeros((len(bow_corpus), model.num_topics))
    for i, doc in enumerate(bow_corpus):
        tops, probas = zip(*model.get_document_topics(doc))
        for j, top in enumerate(tops):
            doc_topics[i,top] = probas[j]

    return doc_topics


def plot_doc_topics(doc_topics):
    '''
    INPUT: doc_topics matrix with shape -> rows=number of documents, columns=number of topics. Values are the probabilities of each topic being represented in the document.
    OUPUT: None
    TASK: Displays and saves the matrix as a seaborn heatmap with the probability bar highlighted on the right side of the plot.
    '''
    sns.heatmap(doc_topics, yticklabels=50)
    plt.yticks(rotation=0)
    plt.ylabel('Document Number', fontsize=14)
    plt.xlabel('Topic Number', fontsize=14)
    plt.title('Probability of Topic for each Document', fontsize=14)
    plt.show()


def test_num_topics(bow_corpus, dictionary, n_topics_lst, chunksize, workers):
    '''
    INPUT: n_topics_lst is a list with number of topics to iterate through. chunksize is calculated as number of documentes divided by number of cores. workers are how many cores are available for computation.
    OUTPUT: list of trained LDA models
    TASK: 'grid search' for optimal number of topics which is determined by visual inspection of LDAvis (uniform size circles, minimum overlap).
    NOTES: By previous iteration of this fuction, determined optimal number of passes was at least 30 and that minimum_probability had no effect on results. To optimize use of multiple cores, chunksize will be set to number of documents in corpus divided by cores.
    '''
    lda_models = defaultdict(list)
    for t in n_topics_lst:
        lda_models['n_topics: {0}, chunksize: {1}, workers: {2}'.format(t, chunksize, workers)].append(models.LdaMulticore(bow_corpus, id2word=dictionary, num_topics=t, passes=30, chunksize=chunksize, random_state=1, workers=workers))

    return lda_models


## docstring needed
def pca_doc_topics(bow_corpus, lda_models):
    doc_topics_lst, X_lst = [], []
    for mod in lda_models:
        doc_topics = doc_topics_matrix(bow_corpus, mod)
        doc_topics_lst.append(doc_topics)
        X_lst.append(PCA(n_components=2).fit_transform(doc_topics))

    return doc_topics_lst, X_lst


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
    # allows display of gensim LDA results as the model is being trained
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    data_path = '/Users/jpc/Documents/data_science_inmersive/document_image_classification/data/Wells'

    # store counts and paths for files of interest
    tif_cnt, xml_cnt, txt_cnt, misc_cnt, tif_paths, xml_paths, txt_paths, misc_paths = oip.doc_cnts_paths(data_path)

    # tokens appearing in less than 15% of documents not to be included
    no_below = int(txt_cnt * 0.15)

    # from inspection (LDAvis, topic-token distribution plots), 6 topics separate well the data
    tokenized_corpus, dictionary, bow_corpus, lda = train_lda(paths=txt_paths, dict_no_below=no_below, cores=4, n_topics=6)

    # find index of documents to txt_paths that contribute no tokens to bow_corpus
    file_removal_idx = inspect_bow_corpus(bow_corpus)

    # # MAKE COPY OF ENTIRE DATA SET TO DESKTOP BEFORE PROCEDING
    # # inspect bow_corpus for documents with no tokens and return paths excluding documents with no tokens in bow_corpus
    # tif_cnt, xml_cnt, txt_cnt, misc_cnt, tif_paths, xml_paths, txt_paths, misc_paths = remove_poor_quality_ocr(file_removal_idx)

    # saving corpus, dictionary, model for LDAvis inspection (use a jupyter-notebook)
    corpora.MmCorpus.serialize('src/LDAvis_choose_n_topics/bow_corpus_6.mm', bow_corpus)
    dictionary.save('src/LDAvis_choose_n_topics/dict_6.dict')
    lda.save('src/LDAvis_choose_n_topics/lda_6.model')

    # inspection of the classification
    top_topics_dict, files_by_topic_defdict = inspect_classification(bow_corpus, lda, txt_paths)

    lda.show_topics(-1, formatted=False)

    display_docs(files_by_topic_defdict, 0, high=200)

    # calculate token variance inside topics using all tokens in dictionary
    df = topic_token_dist(model=lda, dictionary=dictionary, num_words=len(dictionary.values()))
    x, y = [], []
    for t in range(lda.num_topics):
        x.append(t)
        y.append(np.var(df[df['topic'] == t]['token_prob']))

    # calculate mean token_id per topic to identify mean of known topic labels, such as permits
    token_dist_df = topic_token_dist(model=lda, dictionary=dictionary, num_words=10)
    m = []
    for t in range(lda.num_topics):
        m.append(np.mean(token_dist_df[token_dist_df['topic'] == t]['token_id']))

    df_top_var = pd.DataFrame({'topic': x, 'token_prob_var': y, 'top10_token_id_mean': m})

    # use token_prob_var < 0.00015 to find most consistent topics in terms of document content. Use topic visual inspection to connect  top10_token_id_mean with topic label.
    df_top_var['topic_label'] = ''
    df_top_var.loc[0, 'topic_label'] = 'completions'
    df_top_var.loc[5, 'topic_label'] = 'permits'

    p = sns.barplot(x, y)
    plt.text(-1.3, 0.00015, 'Token Probability Variance', va='center', rotation='vertical', size=14)
    plt.text(2.5, -0.000025, 'Topic', ha='center', size=14)


    # topic_id distribution per topic
    b = sns.boxplot(data=token_dist_df, y='topic', x='token_id', orient='h')
    b.axes.set_title('Distribution of Token ID per Topic',fontsize=16)
    b.set_xlabel('Token ID',fontsize=14)
    b.set_ylabel('Topic',fontsize=14)

    # if topic variance is under 0.00015, set document paths as classified and exclude from further modeling
    classified_topic = []
    for t in range(lda.num_topics):
        if y[t] < 0.00015:
            classified_topic.append(t)

    classified_txt_paths = defaultdict(list)
    for t in classified_topic:
        classified_txt_paths[t].append(np.array(txt_paths)[topic_masks[t]])

    to_ignore_txt_paths = []
    for t in classified_topic:
        to_ignore_txt_paths = list(chain.from_iterable([x[0] for x in classified_txt_paths.values()]))

    top_num = []
    path_lst = []
    county_lst = []
    for t, v in classified_txt_paths.items():
        top_num.append(np.repeat(t, len(v[0])))
        path_lst.append(v[0])
        for lst in v:
            for path in lst:
                county_lst.append(path.rsplit('/', 1)[-1].rsplit('-')[1])

    topic_lst = [item for sublist in top_num for item in sublist]
    path_lst = [item for sublist in path_lst for item in sublist]

    df_classified = pd.DataFrame({'topic_num': topic_lst, 'txt_path': path_lst, 'county_code': county_lst})
    df_classified['FIPS_code'] = '0500000US08123'

    # API numbers for all files in train set
    path_api = []
        for path in txt_paths:
            path_api.append((path, path.rsplit('/', 1)[-1].rsplit('-', 3)[0]))

    unique_api = set(np.array(path_api)[:,1])

    # COGCC well location data by API number
    dbf = dbfread.DBF('/Users/jpc/Documents/data_science_inmersive/document_image_classification/data/WELLS_SHP/Wells.dbf')
    frame = pd.DataFrame(iter(dbf))

    weld_county_wells = frame[frame['API_Label'].isin(unique_api)]

    wells_utms = weld_county_wells.loc[:, ['API_Label', 'Utm_X', 'Utm_Y']]

    lat_long = []
    for i in range(wells_utms.shape[0]):
        lat_long.append(utm.to_latlon(wells_utms.iloc[i,1], wells_utms.iloc[i,2], 13, 'S'))
    wells_utms['latitude'], wells_utms['longitude'] = zip(*lat_long)

    wells_utms.to_csv('data/wells_lat_long.csv')



    # ******
    # tokenized_corpus = mcp.parallel_corpus_lemm_tokenization(txt_paths)
    #
    # dictionary, bow_corpus = mcp.bow_and_dict(tokenized_corpus, no_below=150)
    #
    # dictionary.merge_with(corpora.dictionary.Dictionary(['north east latitude longitude azi inc vertical section bottom hole degrees degree deg angle ft survey directional curvature curve sea level coordinate md true build inclination azimuth toolface meridian magnetic declination dip lat long turn dogleg compass mwd anticollision'.split()]))
    #
    # bow_corpus = [dictionary.doc2bow(text) for text in tokenized_corpus]
    #
    # chunk = txt_cnt/4
    #
    # lda = models.LdaMulticore(bow_corpus, id2word=dictionary, num_topics=2, passes=30, chunksize=chunk, random_state=1, workers=4)
    #
    #
    #
    # corpora.MmCorpus.serialize('src/lda_mod/well_docs.mm', bow_corpus1)
    # dictionary1.save('src/lda_mod/well_docs.dict')
    # lda1.save('src/lda_mod/well_docs.model')
    #
    #



    # sub_items = []
    # sub_tokenized_corpus_lst = defaultdict(list)
    # sub_dictionary_lst = defaultdict(list)
    # sub_bow_corpus_lst = defaultdict(list)
    # sub_lda = defaultdict(list)
    #
    #
    # topic_masks = docs_topic_mask(lda, top_topics_dict)
    #
    # sub_items = []
    # sub_txt_paths = []
    # for t in range(lda.num_topics):
    #     sub_txt_paths.append(np.array(txt_paths)[topic_masks[t]])
    #
    #     sub_items.append(train_lda(paths=sub_txt_paths[t], dict_no_below=50, cores=4, n_topics=2))
    #
    #
    #

    # # ******
    # # number of topics to train models below
    # n_topics_lst = [6, 7, 8, 9, 10, 14, 16]
    #
    # lda_models = test_num_topics(bow_corpus, dictionary, n_topics_lst, chunksize=264, workers=4)
    #
    # # save models from 'grid search' to become available for LDAvis
    # for key, mod in lda_models.items():
    #     try:
    #         mod[0].save('src/lda_models/'+key+'.model')
    #     except:
    #         print 'model {0} was not saved'.format(key)
    # # ******


    # # ******
    # # create sub-corpuses from original corpus, splitting on topics from first trained model. Test on topic 6: directional surveys bin
    # topic_masks = docs_topic_mask(lda, top_topics_dict)
    # sub_corpus_txt_paths = []
    # for t in range(lda.num_topics):
    #     sub_corpus_txt_paths.append(np.array(txt_paths)[topic_masks[t]])
    #
    # tokenized_sub_corpus = scp.parallel_corpus_lemm_tokenization(sub_corpus_txt_paths[0])
    #
    # sub_dictionary, bow_sub_corpus = cp.bow_and_dict(tokenized_sub_corpus, no_below=55, no_above=0.5)
    #
    # # update dictionary after observing results
    # sub_dictionary.merge_with(corpora.dictionary.Dictionary(['north east latitude longitude azi inc vertical section bottom hole degrees degree deg angle ft survey directional curvature curve sea level coordinate md true build inclination azimuth toolface meridian magnetic declination dip lat long turn dogleg compass mwd anticollision'.split()]))
    #
    # # update bow_sub_corpus with updated dictionary
    # bow_sub_corpus = [sub_dictionary.doc2bow(text) for text in tokenized_sub_corpus]
    #
    # # move documents that provide no tokens to bow_sub_corpus
    # file_removal_idx = inspect_bow_corpus(bow_sub_corpus)
    #
    # topic_docs_removal_idx = topic_masks[1][file_removal_idx]
    #
    # if topic_docs_removal_idx:
    #     for i in topic_docs_removal_idx:
    #         print 'removing document indexed at: {0}'.format(i)
    #         try:
    #             os.rename(txt_paths[i], os.path.join('/Users/jpc/Documents/data_science_inmersive/document_image_classification/data/ignore_general/sub_topics',txt_paths[i].rsplit('/', 1)[-1]))
    #             os.rename(tif_paths[i], os.path.join('/Users/jpc/Documents/data_science_inmersive/document_image_classification/data/ignore_general/sub_topics',tif_paths[i].rsplit('/', 1)[-1]))
    #         except:
    #             print 'file not found'
    # else:
    #     print 'All documents in bow_sub_corpus have tokens.'
    #
    #
    # lda_sub = models.LdaMulticore(bow_sub_corpus, id2word=sub_dictionary, num_topics=2, passes=25, chunksize=29, random_state=1, workers=4)
    #
    # corpora.MmCorpus.serialize('src/lda_sub_models/sub_well_docs.mm', bow_sub_corpus)
    # sub_dictionary.save('src/lda_sub_models/sub_well_docs.dict')
    # lda_sub.save('src/lda_sub_models/sub_well_docs.model') # to save a single model for inspection
    #
    # sub_top_topics_dict, sub_files_by_topic_defdict = inspect_classification(bow_sub_corpus, lda_sub, sub_corpus_txt_paths[1])
    #
    # sub_docs_per_topic = count_docs_per_topic(sub_top_topics_dict)
    #
    # lda_sub.show_topics(-1, formatted=False)
    #
    # display_docs(sub_files_by_topic_defdict, 0, high=218)
    # # ******


    ## find distribution of tokens per topic
    ## plots all topic token distributions in one plot
    # topic_masks = docs_topic_mask(lda, top_topics_dict)
    # tok_per_topic = []
    # for n in range(lda.num_topics):
    #     tok_per_topic.append(np.array(bow_corpus)[topic_masks[n]])
    #
    #
    #
    #     cnt = Counter()
    #     for doc, tok in enumerate(tok_per_topic[0]):
    #         for tup in tok:
    #             cnt[tup[0]] += tup[1]
    #
    #     labels, values = zip(*cnt.items())
    #     sns.distplot(values)
    # plt.xlabel('token counts per topic')
        # sns.barplot(values, labels, orient='h')


    # ## create dataframe: topic, doc#, dict_id, token, token count
    #
    # top_col_len = defaultdict(list)
    # for t in range(lda.num_topics):
    #     for doc in tok_per_topic[t]:
    #         top_col_len[t].append(len(doc))
    #
    #
    # # key is topic. values are arrays corresponding to each document in the topic. content of arrays is topic number repeated the number of tokens in that document.
    # topics_col = defaultdict(list)
    # for t in range(lda.num_topics):
    #     for doc in range(len(tok_per_topic[t])):
    #         topics_col[t].append(np.repeat(t, top_col_len[t][doc]))
    #
    # doc_num_col = defaultdict(list)
    # for t in range(lda.num_topics):
    #     for doc_id, doc in enumerate(tok_per_topic[t]):
    #         doc_num_col[t].append(np.repeat(topic_masks[t][doc_id], len(doc)))
    #
    # dict_id = defaultdict(list)
    # for t in range(lda.num_topics):
    #     for doc in tok_per_topic[t]:
    #         dict_id[t].append(zip(*doc))
    #
    #
    # topics_col_idx = []
    # for t in range(lda.num_topics):
    #     topics_col_idx.append(np.concatenate(topics_col[t]).ravel())
    # x1 = np.concatenate(topics_col_idx).ravel()
    #
    # docs_col_idx = []
    # for t in range(lda.num_topics):
    #     docs_col_idx.append(np.concatenate(doc_num_col[t]).ravel())
    # x2 = np.concatenate(docs_col_idx).ravel()
    #
    # dict_col_idx = []
    # cnt_empty = 0
    # for t in range(lda.num_topics):
    #     for i, d in enumerate(tok_per_topic[t]):
    #         if not d:
    #             # tok_per_topic[t][i] = [(0,0)]
    #             tok_per_topic[t][i] = ''
    #             cnt_empty+=1
    #     dict_col_idx.append(np.vstack(tok_per_topic[t]))
    # # x3 = np.concatenate(dict_col_idx)



    # ## perplexity measure per topic
    # for n in range(lda.num_topics):
    #     print 'log_perplexity for topic {0}: {1}'.format(n, lda.log_perplexity(np.array(bow_corpus)[topic_masks[n]]))
    #
    # mask = top_topics_dict[0][:,0].astype(int)
    # tok_per_topic = np.array(bow_corpus)[mask]
    #
    # topic_docs_tok_dict = {}
    # for idx, doc_num in enumerate(mask):
    #     topic_docs_tok_dict[doc_num] = tok_per_topic[idx]


    #
    #

    #
    #

    #
    # topic_nums, topic_probas, topic_docs = [], [], []
    # for k, v in top_topics_lst.items():
    #     topic_nums = np.hstack((topic_nums, np.repeat(k, len(v))))
    #     topic_probas = np.hstack((topic_probas, np.array(v)[:,1]))
    #     topic_docs =  np.hstack((topic_docs, np.array(v)[:,0]))
    #
    #
    # ## violin plots for n_topics vs passes
    # topics_passes_mods = []
    # for key, mod in lda_models.items():
    #     if isinstance(key, str):
    #         if 'chunksize: 277, min probability: 0.01' in key:
    #             topics_passes_mods.append(mod[0])
    #
    # topic_nums, topic_probas, topic_docs = [], [], []
    # for mod in topics_passes_mods:
    #     top_topics_dict, files_by_topic_defdict = inspect_classification(bow_corpus, mod)
    #
    #     for k, v in top_topics_lst.items():
    #         try:
    #             topic_nums = np.hstack((topic_nums, np.repeat(k, len(v))))
    #             topic_probas = np.hstack((topic_probas, np.array(v)[:,1]))
    #             topic_docs =  np.hstack((topic_docs, np.array(v)[:,0]))
    #         except:
    #             pass
    #


    # lda.show_topics(-1, formatted=False)
    # #

    # top_topics_dict, files_by_topic_defdict = inspect_classification(bow_corpus, lda)



    # open files of interest. Subset by document number
    # !open {tif_paths[981]}

    #
    #
    # sorted(top_topics_lst.iteritems(),key=lambda (k,v): len(v),reverse=True)


    ## silhouette plots
    # silhouette scores: average and per sample
    # doc_topics = doc_topics_matrix(bow_corpus, lda)
    #
    # X = PCA(n_components=2).fit_transform(doc_topics)

    # lst = []
    # for k in top_topics_dict.keys():
    #     for v in top_topics_lst[k]:
    #         lst.append((v,k))
    #
    # doc_top_lst = np.array(sorted(lst))
    #
    # ordered_labels = doc_top_lst[:,1]
    #
    # n_topics = lda.num_topics

















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
