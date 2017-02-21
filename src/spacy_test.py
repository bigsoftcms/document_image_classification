import spacy

from string import punctuation

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# Load the spacy.en module if it hasn't been loaded already
# When in ipython, execute the script using %run -i my_file.py to avoid
# repeatedly loading in the english module
if not 'nlp' in locals():
    print 'Loading English Module...'
    nlp = spacy.load('en')


def lemmatize_string(doc, stop_words):
    # First remove punctuation from string
    # .translate is a string operation
    # spaCy expects a unicode object
    doc = ' '.join(doc.translate(None, punctuation).replace('\n', ' ').split()).decode('utf-8')

    # Run the doc through spaCy
    doc = nlp(doc)

    # Lemmatize and lower text
    tokens = [token.lemma_.lower() for token in doc]

    return ' '.join(w for w in tokens if w not in stop_words)


def processed_sent(parsedData):
    # Let's look at the sentences
    sents = []
    # the "sents" property returns spans
    # spans have indices into the original string
    # where each index value represents a token
    for span in parsedData.sents:
        # go from the start to the end of each span, returning each token in the sentence
        # combine each token using join()
        sent = ''.join(parsedData[i].string for i in range(span.start, span.end)).strip()
        sents.append(sent)

    return sents


if __name__ == '__main__':
    lemmatized_corpus = [lemmatize_string(doc, ENGLISH_STOP_WORDS) for doc in raw_corpus]

    #
    # for sentence in sents:
    #     print sentence
