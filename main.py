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
    #corpus_vectorizer.create_dictionary(filename=corpus, directory=directory)
    #dictionary = directory+'dictionary.pkl'
    #corpus_vectorizer.tfidf_vectorize(bow_corpus, dictionary, directory=directory)

    #corpus_vectorizer.doc2vec_vectorize(corpus, vector_size=10000, directory=directory)

    # order of comparison
    order = 20

    # generate K-Means clustering run
    # cid = 8
    # directory = 'ClusterResults/'
    # bow_corpus = 'ProcessedWSJ/bow.pkl'
    # dense_corpus = 'ProcessedWSJ/dense_corpus.pkl'
    # doc2vec_corpus = 'ProcessedWSJ/wsj_doc2vec10000.pkl'
    # dictionary = 'ProcessedWSJ/dictionary.pkl'
    # KMeansGenerator.generate_k_means(dense_corpus=dense_corpus, run_id=cid,algorithm='auto', directory=directory, k=order)
    # KMeansGenerator.predict_cluster_labels(run_id=cid, directory=directory)
    # KMeansGenerator.generate_cluster_keywords(run_id=cid, corpus=dense_corpus, bow=bow_corpus, dictionary=dictionary,
    #                                           mode='centroid', directory=directory)
    # KMeansGenerator.generate_cluster_keywords(run_id=cid, corpus=dense_corpus, bow=bow_corpus, dictionary=dictionary,
    #                                           mode='cluster', directory=directory)

    # directory = 'ClusterResults/'
    # bow_corpus = 'ProcessedWSJ/bow.pkl'
    # dense_corpus = 'ProcessedWSJ/dense_corpus.pkl'
    # doc2vec_corpus = 'ProcessedWSJ/wsj_doc2vec10000.pkl'
    # dictionary = 'ProcessedWSJ/dictionary.pkl'
    # for cid in range(9,14):
    #     KMeansGenerator.generate_k_means(dense_corpus=dense_corpus, run_id=cid, algorithm='full',directory=directory, k=order)
    #     KMeansGenerator.predict_cluster_labels(run_id=cid, directory=directory)
    #     KMeansGenerator.generate_cluster_keywords(run_id=cid, corpus=dense_corpus, bow=bow_corpus, dictionary=dictionary,
    #                                               mode='centroid', directory=directory)
    #     KMeansGenerator.generate_cluster_keywords(run_id=cid, corpus=dense_corpus, bow=bow_corpus, dictionary=dictionary,
    #                                               mode='cluster', directory=directory)
    #     print('clustering run: '+str(cid)+' generated.')
    #
    # for cid in range(14,19):
    #     KMeansGenerator.generate_k_means(dense_corpus=dense_corpus, run_id=cid, algorithm='elkan', directory=directory,
    #                                      k=order)
    #     KMeansGenerator.predict_cluster_labels(run_id=cid, directory=directory)
    #     KMeansGenerator.generate_cluster_keywords(run_id=cid, corpus=dense_corpus, bow=bow_corpus,
    #                                               dictionary=dictionary,
    #                                               mode='centroid', directory=directory)
    #     KMeansGenerator.generate_cluster_keywords(run_id=cid, corpus=dense_corpus, bow=bow_corpus,
    #                                               dictionary=dictionary,
    #                                               mode='cluster', directory=directory)
    #     print('clustering run: ' + str(cid) + ' generated.')

    # generate topic models
    #tid = 5
    # directory = 'LDAResults/'
    # corpus = 'ProcessedWSJ/tfidf_corpus.pkl'
    # dictionary = 'ProcessedWSJ/dictionary.pkl'
    # LDAGenerator.generate_lda(corpus=corpus, run_id=tid, num_topics=order, dictionary=dictionary, directory=directory)
    # LDAGenerator.predict_topic_labels(run_id=tid, corpus=corpus, directory=directory)
    # LDAGenerator.generate_topic_keywords(run_id=tid, num_keywords=10, num_topics=order, directory=directory)

    # directory = 'LDAResults/'
    # corpus = 'ProcessedWSJ/tfidf_corpus.pkl'
    # dictionary = 'ProcessedWSJ/dictionary.pkl'
    # for tid in range(6,11):
    #     LDAGenerator.generate_lda(corpus=corpus, run_id=tid, num_topics=order, dictionary=dictionary,update_every=1,
    #                               directory=directory)
    #     LDAGenerator.predict_topic_labels(run_id=tid, corpus=corpus, directory=directory)
    #     LDAGenerator.generate_topic_keywords(run_id=tid, num_keywords=10, num_topics=order, directory=directory)
    #     print('LDA run: ' + str(tid) + ' generated.')
    #
    # for tid in range(11,16):
    #     LDAGenerator.generate_lda(corpus=corpus, run_id=tid, num_topics=order, dictionary=dictionary,update_every=0,
    #                               directory=directory)
    #     LDAGenerator.predict_topic_labels(run_id=tid, corpus=corpus, directory=directory)
    #     LDAGenerator.generate_topic_keywords(run_id=tid, num_keywords=10, num_topics=order, directory=directory)
    #     print('LDA run: ' + str(tid) + ' generated.')

    # generate figures
    # cid = 9
    # tid = 1
    # clustering = 'ClusterResults/' + str(cid) + '/model'
    # topic_model = 'LDAResults/' + str(tid) + '/model'
    # corpus = 'ProcessedWSJ/tfidf_corpus.pkl'
    # directory = 'figures/'
    # clusters, cluster_topic_matrix = Visualisation.compare_cluster_topic(clustering, topic_model, corpus=corpus,
    #                                                                      order=order, mode='distribution')
    # Visualisation.topic_distribution_visualise(clusters, cluster_topic_matrix, cid=cid, tid=tid, order=order,
    #                                            directory=directory, mode='distribution')
    #
    # clusters, cluster_topic_matrix = Visualisation.compare_cluster_topic(clustering, topic_model, corpus=corpus,
    #                                                                      order=order, mode='label')
    # Visualisation.topic_distribution_visualise(clusters, cluster_topic_matrix, cid=cid, tid=tid, order=order,
    #                                            directory=directory, mode='label')

    # vector similarity
    # cid = 9
    # tid = 1
    # Visualisation.visualise_vecter_similarity(tid,cid, directory='figures/',norm='l1',figname='row_l1')

    # corpus = 'ProcessedWSJ/tfidf_corpus.pkl'
    # cid = 9
    # tid = 1
    # for c in range(20):
    #     for t in range(20):
    #         dist = Visualisation.get_topic_distribution(corpus=corpus, cid=cid, tid=tid, c=c,t=t, mode='c')
    #         Visualisation.hist_plot(topic_dist=dist,c=c, t=t,tid=1, directory='figures/c9t1/')


    #measure skewness of c9t1
    # cid =9
    # tid = 1
    # corpus = 'ProcessedWSJ/tfidf_corpus.pkl'
    # print('Skewness per cluster\n\n')
    # for c in range(20):
    #     for t in range(20):
    #         dist = Visualisation.get_topic_distribution(corpus=corpus, cid=cid, tid=tid, c=c, t=t, mode='c')
    #         skewness = Visualisation.skewness_measure(dist)
    #         print(f"c{c}t{t}:  {skewness}")
    #
    #
    # print("Skewness of whole corpus")
    # for t in range(20):
    #     dist = Visualisation.get_topic_distribution(corpus=corpus, cid=cid, tid=tid, t=t, mode='all')
    #     skewness = Visualisation.skewness_measure(dist)
    #     print(f"t{t}:  {skewness}")


    #testing
    cid = 9
    tid = 1
    clustering = 'ClusterResults/' + str(cid) + '/model'
    topic_model = 'LDAResults/' + str(tid) + '/model'
    corpus = 'ProcessedWSJ/tfidf_corpus.pkl'
    directory = 'figures/'
    clusters, cluster_topic_matrix = Visualisation.compare_cluster_topic(clustering, topic_model, corpus=corpus,
                                                                         order=order, mode='distribution')
    Visualisation.cluster_topic_dist(clusters, cluster_topic_matrix, 0)

    return 0


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    work_pipeline()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
