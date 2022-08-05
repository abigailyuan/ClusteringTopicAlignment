import gzip
import pickle
import re
import time

import nltk

from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from gensim.matutils import corpus2csc


def parse_wsj_corpus(filename, directory='ProcessedWSJ/'):
    """WSJ collection is parsed into a list of strings.
    The result files include an index file and a pure document file."""
    f = gzip.open(filename, 'r')
    docs = f.read().decode("utf-8")
    f.close()
    docs = docs.split('</DOC>')

    indexes = []
    corpus = []

    for doc in docs:
        docno = doc[doc.find('<DOCNO>') + 7:doc.find('</DOCNO>')].strip()
        text = doc[doc.find('<TEXT>') + 6:doc.find('</TEXT>')].strip()
        indexes.append(docno)
        corpus.append(text)

    pickle.dump(indexes, open(directory + 'wsj_index.pkl', 'wb'))
    pickle.dump(corpus, open(directory + 'wsj_raw.pkl', 'wb'))

    print('WSJ corpus is parsed successfully.')
    print('Number of documents:', len(indexes))

    return corpus


def scrape_doc(doc):
    return re.sub(r'<[^>]+>', ' ', doc)


def clean_doc(doc):
    doc = re.sub(r'[^\s]+@[^\s]+', ' ', doc)  # email
    doc = re.sub(r'(http)?([^\s@]+(\.))+[^\s]+', ' ', doc)  # url, assuming emails have been removed.
    doc = re.sub(r'[^0-9A-Za-z\s]', " ", doc)
    doc = re.sub("[0-9]+", " ", doc)
    doc = re.sub(r"\s+", " ", doc)
    return doc


def clean_corpus(corpus, directory='ProcessedWSJ/'):
    cleaned_corpus = []
    for i in range(len(corpus)):
        cleaned = clean_doc(scrape_doc(corpus[i]))
        cleaned_corpus.append(cleaned)

    pickle.dump(cleaned_corpus, open(directory + 'wsj_cleaned.pkl', 'wb'))
    print('WSJ corpus is cleaned successfully.')

    return cleaned_corpus


def tokenize_corpus(corpus):
    start = time.perf_counter()
    for i in range(len(corpus)):
        doc = corpus[i]
        doc = word_tokenize(doc)
        corpus[i] = doc
    end = time.perf_counter()

    print('Tokenization completed.')
    print('time used:', int(end - start))

    return corpus


def remove_stopwords(corpus, directory='ProcessedWSJ/'):
    stopwords = nltk.corpus.stopwords.words('english')
    stopwords.append('r')

    start = time.perf_counter()
    for i in range(len(corpus)):
        doc = corpus[i]
        doc = [i for i in doc if i not in stopwords]
        corpus[i] = doc
    end = time.perf_counter()

    print('Stopwords removed.')
    print('time used:', int(end - start))

    pickle.dump(corpus, open(directory + 'wsj_tokenized.pkl', 'wb'))

    return corpus


def lemmatize_corpus(corpus, directory='ProcessedWSJ/'):
    # initialise stemmer
    lemmatizer = WordNetLemmatizer()
    # stemmer = SnowballStemmer('english')
    stopwords = nltk.corpus.stopwords.words('english')

    #add new stopwords
    stopwords.append('mr')
    stopwords.append('would')
    stopwords.append('r')

    start = time.perf_counter()
    stemmed_corpus = []
    for doc in corpus:
        # doc = [stemmer.stem(lemmatizer.lemmatize(token, pos='v')) for token in doc]
        doc = [lemmatizer.lemmatize(token.lower(), pos='n') for token in doc]
        doc = [i for i in doc if i not in stopwords]
        stemmed_corpus.append(doc)

    end = time.perf_counter()

    print('WSJ corpus is lemmatised.')
    print('time used:', int(end - start))

    pickle.dump(stemmed_corpus, open(directory + 'wsj_lemmatised.pkl', 'wb'))

    return stemmed_corpus


def generate_subcollections(corpus, len_threshold=100, directory='ProcessedWSJ/'):
    wsj_large = []
    wsj_small = []
    for i in range(len(corpus)):
        if len(corpus[i]) > len_threshold:
            wsj_large.append(corpus[i])
        else:
            wsj_small.append(corpus[i])

    pickle.dump(wsj_large, open(directory + 'wsj_large.pkl', 'wb'))
    pickle.dump(wsj_small, open(directory + 'wsj_small.pkl', 'wb'))

    print('Subcollections created.')

# def convert_to_sparse_matrix(bow):
#     return corpus2csc(bow)
#
# corpus = 'ProcessedWSJ/bow.pkl'
# fp = open(corpus,'rb')
# bow = pickle.load(fp)
# fp.close()
#
# csc = convert_to_sparse_matrix(bow)
# fp = open('ProcessedWSJ/sparse_bow.pkl','wb')
# pickle.dump(csc, fp)
# fp.close()