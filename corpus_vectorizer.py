import pickle
import gensim
import time
from gensim.models import TfidfModel
from gensim.matutils import corpus2dense
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from numpy import asarray


def create_dictionary(filename, no_below=15, no_above=0.5, keep_n=10000, directory='ProcessedWSJ/'):
    '''create dictionary and BOW from stemmed corpus'''
    corpus = pickle.load(open(filename, 'rb'))

    dictionary = gensim.corpora.Dictionary(corpus)
    dictionary.filter_extremes(no_below=no_below, no_above=no_above, keep_n=keep_n)
    pickle.dump(dictionary, open(directory + 'dictionary.pkl', 'wb'))

    bow = [dictionary.doc2bow(doc) for doc in corpus]
    pickle.dump(bow, open(directory + 'bow.pkl', 'wb'))

    return dictionary, bow


def tfidf_vectorize(filename, dictionary, directory='ProcessedWSJ'):
    corpus = pickle.load(open(filename, 'rb'))
    dictionary = pickle.load(open(dictionary, 'rb'))

    start = time.perf_counter()
    model = TfidfModel(corpus=corpus, dictionary=dictionary)
    tfidf_corpus = [model[i] for i in corpus]
    end = time.perf_counter()
    print('Corpus is vectorized with TFIDF.')
    print('time used:', int(end - start))

    pickle.dump(tfidf_corpus, open(directory + 'tfidf_corpus.pkl', 'wb'))

    start = time.perf_counter()
    num_terms = len(dictionary)
    corpus_size = len(tfidf_corpus)
    dense_corpus = corpus2dense(tfidf_corpus, num_terms, corpus_size)
    end = time.perf_counter()
    print('Corpus is converted to dense corpus.')
    print('time used:', int(end - start))

    pickle.dump(dense_corpus.T, open(directory + 'dense_corpus.pkl', 'wb'))

    return 0


def doc2vec_vectorize(filename, vector_size=500, window=2, min_count=15, max_vocab_size=10000,
                      directory='ProcessedWSJ/'):
    corpus = pickle.load(open(filename, 'rb'))

    start = time.perf_counter()
    documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(corpus)]
    end = time.perf_counter()
    print('Corpus is tagged.')
    print('time used:', int(end - start), '\n')

    start = time.perf_counter()
    model = Doc2Vec(documents, vector_size=vector_size, window=window, min_count=min_count,
                    max_vocab_size=max_vocab_size,
                    workers=8)  # was 100,000,0
    model.save(directory + 'doc2vec.model')
    end = time.perf_counter()
    print('Doc2vec model created.')
    print('time used:', int(end - start))

    corpus = []
    for i in range(len(model.docvecs)):
        corpus.append(model.docvecs[i])

    corpus = asarray(corpus)
    pickle.dump(corpus, open(directory + 'wsj_doc2vec.pkl', 'wb'))

    return 0
