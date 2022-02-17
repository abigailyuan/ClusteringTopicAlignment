import numpy as np

import preprocessing
import corpus_vectorizer
import pickle

import KMeansGenerator
import LDAGenerator
import Visualisation


order = 20
cid = 9
tid = 1

clustering = 'ClusterResults/' + str(cid) + '/model'
topic_model = 'LDAResults/' + str(tid) + '/model'
corpus = 'ProcessedWSJ/tfidf_corpus.pkl'
directory = 'figures/'
clusters, cluster_topic_matrix = Visualisation.compare_cluster_topic(clustering, topic_model, corpus=corpus,
                                                                         order=order, mode='distribution')

centroids = KMeansGenerator.get_cluster_vectors(cid)
topics = LDAGenerator.get_topic_vectors(tid)
cos_sim_matrix = Visualisation.vector_similarity(centroids, topics, norm='')

result = np.multiply(cluster_topic_matrix, cos_sim_matrix)
print('Cluster scores:')
print(np.sum(result, 1))

print('Topic scores:')
print(np.sum(result, 0))


# min value = 0
print('Overall matching score:')
print(np.sum(np.sum(result, 1)))
