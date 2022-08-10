import pickle
import gensim
import time
from gensim.models import TfidfModel
from gensim.matutils import corpus2dense
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from numpy import asarray
from sklearn.feature_extraction.text import CountVectorizer


def create_dictionary(filename, no_below=15, no_above=0.5, keep_n=10000, directory='ProcessedWSJ/'):
    """create dictionary and BOW from stemmed corpus"""
    corpus = pickle.load(open(filename, 'rb'))

    dictionary = gensim.corpora.Dictionary(corpus)
    dictionary.filter_extremes(no_below=no_below, no_above=no_above, keep_n=keep_n)
    pickle.dump(dictionary, open(directory + 'dictionary_lemma.pkl', 'wb'))

    bow = [dictionary.doc2bow(doc) for doc in corpus]
    pickle.dump(bow, open(directory + 'bow_lemma.pkl', 'wb'))

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


def bow_vectorize(filename, dictionary, directory='ProcessedWSJ'):
    corpus = pickle.load(open(filename, 'rb'))
    dictionary = pickle.load(open(dictionary, 'rb'))

    start = time.perf_counter()
    model = CountVectorizer(vocabulary=dictionary)
    vectorized_corpus = model.fit_transform(corpus)
    end = time.perf_counter()
    print('Corpus is vectorized with CountBased Vectorizer.')
    print('time used:', int(end - start))

    pickle.dump(vectorized_corpus, open(directory + 'count_corpus_lemma.pkl', 'wb'))

    num_terms = len(dictionary)
    corpus_size = len(vectorized_corpus)
    dense_corpus = corpus2dense(vectorized_corpus, num_terms, corpus_size)
    pickle.dump(dense_corpus.T, open(directory + 'dense_corpus_lemma.pkl', 'wb'))

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
    model.save(directory + 'doc2vec'+str(vector_size)+'.model')
    end = time.perf_counter()
    print('Doc2vec model created.')
    print('time used:', int(end - start))

    corpus = []
    for i in range(len(model.docvecs)):
        corpus.append(model.docvecs[i])

    corpus = asarray(corpus)
    pickle.dump(corpus, open(directory + 'wsj_doc2vec'+str(vector_size)+'.pkl', 'wb'))

    return 0
