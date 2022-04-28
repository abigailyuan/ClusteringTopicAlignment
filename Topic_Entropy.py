import Visualisation
from scipy.stats import entropy


def get_topic_entropy(cluster_topic_matrix):
    entropies = []
    for i in range(len(cluster_topic_matrix)):
        topic = cluster_topic_matrix.iloc[:, i]
        entropies.append(entropy(topic))
    return entropies

cid = 9
tid = 7
clustering = 'ClusterResults/' + str(cid) + '/model'
topic_model = 'LDAResults/' + str(tid) + '/model'
corpus = 'ProcessedWSJ/tfidf_corpus.pkl'
directory = 'figures/'
clusters, cluster_topic_matrix = Visualisation.compare_cluster_topic(clustering, topic_model, corpus=corpus,c_order=20, t_order=20, mode='distribution')

entropies = get_topic_entropy(cluster_topic_matrix)
for i in entropies:
    print(i)