import pickle
from gensim.models import LdaModel
import numpy as np


def cluster_keyword_occurrence(topic, corpus, cluster):
    count = 0
    for doc in cluster:
        for word in topic:
            doc_bow = corpus[doc]
            for w, c in doc_bow:
                if word == w:
                    count += c
    return count


def keyword_occurrence(topic, corpus):
    count = 0
    for word in topic:
        for doc in corpus:
            for w, c in doc:
                if word == w:
                    count += c
    return count


def prob_cluster_given_topic(cluster, topic, corpus):
    # get N_c_k
    N_c_k = cluster_keyword_occurrence(topic, corpus, cluster)

    # get N_d_k
    N_d_k = keyword_occurrence(topic, corpus)

    # probability
    p_c_k = N_c_k / N_d_k

    return p_c_k


def get_keywords(model, n_words=10):
    # get word probability matrix per topic
    word_prob_matrix = model.get_topics()

    # extract top 10 keyword ids for each topic
    topics = []
    for topic in word_prob_matrix:
        keywords = sorted(zip(list(range(len(topic))), topic), key=lambda x: x[1])[-n_words:]
        keywords = [x[0] for x in keywords]
        topics.append(keywords)
    return topics


def compute_cluster_entropy(clusters, model, corpus, n_words=10):
    topics = get_keywords(model, n_words)
    corpus_bow = pickle.load(open(corpus, 'rb'))

    entropies = []
    for topic in topics:
        entropy = 0
        for cluster in clusters:
            p_c_k = prob_cluster_given_topic(cluster, topic, corpus_bow)
            entropy += p_c_k * np.log2(p_c_k)
        entropies.append(entropy)
    return entropies


def get_clusters(clustering):
    labels = pickle.load(open(clustering, 'rb'))
    clusters = {}
    for i in range(len(labels)):
        label = labels[i]
        if label not in clusters.keys():
            clusters[label] = [i]
        else:
            clusters[label].append(i)

    clsuters_lst = []
    for cluster in clusters:
        clsuters_lst.append(clusters[cluster])
    return clsuters_lst


# directory = 'LDAResults/'
# corpus = 'ProcessedWSJ/bow.pkl'
# run_id = 17
# model = LdaModel.load(directory + str(run_id) + '/model')
# clustering = 'ClusterResults/18/labels'
# clusters = get_clusters(clustering)
# entropies = compute_cluster_entropy(clusters, model, corpus, n_words=10)
# fp = open(directory + str(run_id) + '/cluster_entropy.lst', 'wb')
# pickle.dump(entropies, fp)
# fp.close()
# print(entropies)
