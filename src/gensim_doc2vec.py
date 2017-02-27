from gensim.models.doc2vec import LabeledSentence

class LabeledLineSentence(object):
    def __init__(self, doc_list, labels_list):
       self.labels_list = labels_list
       self.doc_list = doc_list
    def __iter__(self):
        for idx, doc in enumerate(self.doc_list):
            yield LabeledSentence(words=doc.split(),labels=[self.labels_list[idx]])

def lab_sents(doc_lst, labels_lst):
    for idx, doc in enumerate(doc_lst):
        yield LabeledSentence(doc, labels_lst[idx])
