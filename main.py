import preprocessing
import corpus_vectorizer
import pickle

import KMeansGenerator
import LDAGenerator
import Visualisation


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press ⌘F8 to toggle the breakpoint.


def work_pipeline():
    # generate WSJ collection

    # corpus_name = '../datasets/wsj.gz'
    # corpus = preprocessing.parse_wsj_corpus(corpus_name)
    # corpus = preprocessing.clean_corpus(corpus)
    # corpus = preprocessing.tokenize_corpus(corpus)
    # corpus = preprocessing.remove_stopwords(corpus)
    # corpus = preprocessing.lemmatize_corpus(corpus)

    # vectorize corpus
    # corpus = 'ProcessedWSJ/wsj_stemmed.pkl'
    # bow_corpus = 'ProcessedWSJ/bow.pkl'
    # directory = 'ProcessedWSJ/'
    # corpus_vectorizer.create_dictionary(filename=corpus, directory=directory)
    # dictionary = directory+'dictionary.pkl'
    # corpus_vectorizer.tfidf_vectorize(bow_corpus, dictionary, directory=directory)
    #
    # corpus_vectorizer.doc2vec_vectorize(corpus, directory=directory)

    # order of comparison
    order = 40

    # generate K-Means clustering run
    cid = 4
    directory = 'ClusterResults/'
    bow_corpus = 'ProcessedWSJ/bow.pkl'
    dense_corpus = 'ProcessedWSJ/dense_corpus.pkl'
    dictionary = 'ProcessedWSJ/dictionary.pkl'
    KMeansGenerator.generate_k_means(dense_corpus=dense_corpus, run_id=cid, directory=directory, k=order)
    KMeansGenerator.predict_cluster_labels(run_id=cid, directory=directory)
    KMeansGenerator.generate_cluster_keywords(run_id=cid, corpus=dense_corpus, bow=bow_corpus, dictionary=dictionary,
                                              mode='centroid', directory=directory)
    KMeansGenerator.generate_cluster_keywords(run_id=cid, corpus=dense_corpus, bow=bow_corpus, dictionary=dictionary,
                                              mode='cluster', directory=directory)

    # generate topic models
    tid = 4
    directory = 'LDAResults/'
    corpus = 'ProcessedWSJ/tfidf_corpus.pkl'
    dictionary = 'ProcessedWSJ/dictionary.pkl'
    LDAGenerator.generate_lda(corpus=corpus, run_id=tid, num_topics=order, dictionary=dictionary, directory=directory)
    LDAGenerator.predict_topic_labels(run_id=tid, corpus=corpus, directory=directory)
    LDAGenerator.generate_topic_keywords(run_id=tid, num_keywords=10, num_topics=order, directory=directory)

    # generate figures
    # cid = 2
    # tid = 2
    clustering = 'ClusterResults/' + str(cid) + '/model'
    topic_model = 'LDAResults/' + str(tid) + '/model'
    corpus = 'ProcessedWSJ/tfidf_corpus.pkl'
    directory = 'figures/'
    clusters, cluster_topic_matrix = Visualisation.compare_cluster_topic(clustering, topic_model, corpus=corpus,
                                                                         order=order)
    Visualisation.topic_distribution_visualise(clusters, cluster_topic_matrix, cid=cid, tid=tid, order=order,
                                               directory=directory)

    return 0


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    work_pipeline()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
