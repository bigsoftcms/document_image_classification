from string import punctuation, ascii_lowercase, digits
from itertools import chain
from unidecode import unidecode
import multiprocessing as mp

import spacy
from gensim import corpora
from gensim.parsing.preprocessing import STOPWORDS
from nltk.corpus import stopwords as sw

'''
Lemmatizes and tokenizes raw corpus in parallel to output a dictionary and bow_corpus. Uses spaCy to lemmatize.
'stop_specific' list of stop-words is available for modification under lemm_tokenize_doc function.
'''

# run using %run -i <runfile.py> to avoid re-loading nlp as it takes time to complete
if not 'nlp' in locals():
    print 'Loading English Module...'
    nlp = spacy.load('en')
    print 'Completed Loading English Module.'


def lemm_tokenize_doc(doc):
    '''
    INPUT: string that corresponds to a document in a raw corpus and a list of stop words.
    OUTPUT: (1) a list of tokens that corresponds to a corpus document. Strings are byte decoded, punctuation, digits, and newlines removed, words are lowered and lemmatized (words brought back to their 'base' form), only nouns are kept, non-words and stop-words are removed.
    PACKAGE USED: spaCy
    '''
    # decode bytes to utf-8 from doc
    ascii_doc = unidecode(doc.decode('utf-8'))

    # remove punctuation, digits, newlines, and lower the text
    clean_doc = ascii_doc.translate(None, punctuation).translate(None, digits).replace('\n', '').lower()

    # spaCy expects a unicode object
    spacy_doc = nlp(clean_doc.decode('utf-8'))

    # lemmatize, only keep nouns, transform to ascii as will no longer use spaCy
    noun_tokens = [unidecode(token.lemma_) for token in spacy_doc if token.pos_ == 'NOUN']

    # keep tokens longer than 2 characters
    long_tokens = [token for token in noun_tokens if len(token) >= 3 and len(token) < 15]

    # remove tokens that have 3 equal consecutive characters
    triples = [''.join(triple) for triple in zip(ascii_lowercase, ascii_lowercase, ascii_lowercase)]
    good_tokens = [token for token in long_tokens if not [triple for triple in triples if triple in token]]

    # remove tokens that are present in stoplist
    stop_specific = ['wattenberg', 'yes', 'acre', 'number', 'mum', 'nwse', 'swne', 'lease', 'rule', 'drilling', 'permit', 'application', 'form', 'felfwl', 'fnlfsl', 'fnl', 'fsl', 'page', 'file', 'date', 'state', 'surface', 'location', 'oil', 'operator', 'commission', 'colorado', 'conservation', 'prod']

    NLTKstopwords = sw.words('english')

    stoplist = STOPWORDS.union(NLTKstopwords).union(stop_specific)

    final_tokens = [token for token in good_tokens if token not in stoplist]

    return final_tokens


def process_corpus(corpus_chunk):
    '''
    INPUT: equally sized chunks of raw corpus for pre-processing
    OUTPUT: (1) lemmatized and tokenized documents for the chunk of corpus supplied to the function.
    TASK: uses 'lemm_tokenize_doc' function to create a list of lemm-tokenized documents that correspond to all the documents in the chunk of raw corpus supplied.
    '''
    return [lemm_tokenize_doc(doc) for doc in corpus_chunk]


def parallel_corpus_lemm_tokenization(txt_paths):
    '''
    INPUT: paths to OCRd .tif files that are in .txt format.
    OUTPUT: (1) lemmatized and tokenized corpus
    TASK: use multiprocessing Pool to parallelize task using all cores on machine.
    '''
    raw_corpus = []
    for path in txt_paths:
        with open(path) as file:
            raw_corpus.append(file.read())

    cores = mp.cpu_count()
    n = len(txt_paths)/cores

    corpus_chunks = [raw_corpus[i:i + n] for i in xrange(0, len(raw_corpus), n)]

    pool = mp.Pool(processes=4)

    return list(chain(*pool.map(process_corpus, corpus_chunks)))


def bow_and_dict(tokenized_corpus, no_below=5, no_above=0.5, keep_n=100000):
    '''
    INPUT: lemmatized_corpus. 'no_below' helps with filtering out tokens that appear in less than the 'no_below' number of documents specified. 'no_above' is a fraction of the total corpus and it helps with filtering out tokens that appear in more than the 'no_above' fraction of documents specified. Basically, helps to filter out ubiquitous words that were not caught by stop_words.
    OUTPUT: (1) dictionary, which is a collection of all the unique tokens in the corpus. (2) Bag of words corpus, which represents each document in the corpus as a list of tuples with two elements - token id (referenced to the dictionary) and token frequency.
    TASK: tokenizes documents, creates dictionary from tokens, reduces size of dictionary based on 'no_below' and 'no_above' parameters.
    PACKAGE USED: gensim
    '''
    dictionary = corpora.Dictionary(tokenized_corpus)

    # words appearing in less than 'no_below' documents to be excluded from dictionary
    dictionary.filter_extremes(no_below=no_below, no_above=no_above, keep_n=keep_n)

    bow_corpus = [dictionary.doc2bow(text) for text in tokenized_corpus]

    return dictionary, bow_corpus
