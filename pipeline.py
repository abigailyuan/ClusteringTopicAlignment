import numpy as np

import preprocessing
import corpus_vectorizer
import pickle

import KMeansGenerator
import LDAGenerator
import Visualisation


# order = 20
# cid = 9
# tid = 1
#
# clustering = 'ClusterResults/' + str(cid) + '/model'
# topic_model = 'LDAResults/' + str(tid) + '/model'
# corpus = 'ProcessedWSJ/tfidf_corpus.pkl'
# directory = 'figures/'
# clusters, cluster_topic_matrix = Visualisation.compare_cluster_topic(clustering, topic_model, corpus=corpus,
#                                                                          order=order, mode='distribution')
#
# centroids = KMeansGenerator.get_cluster_vectors(cid)
# topics = LDAGenerator.get_topic_vectors(tid)
# cos_sim_matrix = Visualisation.vector_similarity(centroids, topics, norm='')
#
# result = np.multiply(cluster_topic_matrix, cos_sim_matrix)
# print('Cluster scores:')
# print(np.sum(result, 1))
#
# print('Topic scores:')
# print(np.sum(result, 0))
#
#
# # min value = 0
# print('Overall matching score:')
# print(np.sum(np.sum(result, 1)))


def match_score(cid, tid, order, directory='figures/'):
    clustering = 'ClusterResults/' + str(cid) + '/model'
    topic_model = 'LDAResults/' + str(tid) + '/model'
    corpus = 'ProcessedWSJ/tfidf_corpus.pkl'
    clusters, cluster_topic_matrix = Visualisation.compare_cluster_topic(clustering, topic_model, corpus=corpus,
                                                                         order=order, mode='distribution')

    centroids = KMeansGenerator.get_cluster_vectors(cid)
    topics = LDAGenerator.get_topic_vectors(tid)
    cos_sim_matrix = Visualisation.vector_similarity(centroids, topics, norm='')

    result = np.multiply(cluster_topic_matrix, cos_sim_matrix)
    cluster_score = np.sum(result, 1)
    #print('Cluster scores:')
    #print(cluster_score)

    topic_score = np.sum(result, 0)
    #print('Topic scores:')
    #print(topic_score)

    # min value = 0
    score = np.sum(topic_score)
    #print('Overall matching score:')
    #print(score)

    return cluster_score,topic_score,score

# results = np.zeros((10,10))
#
# for cid in range(9,19):
#     for tid in range(6,16):
#         cluster,topic,score = match_score(cid, tid, order=20,directory='figures/')
#         results[cid-9][tid-6] = score
#
# print(results)

corpus = 'ProcessedWSJ/tfidf_corpus.pkl'
Visualisation.get_doc_range(9, 5)
Visualisation.get_topic_distribution(corpus=corpus, cid=9, tid=1, c=0, t=0, mode='all')