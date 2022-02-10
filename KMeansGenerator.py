from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import time
from collections import defaultdict as dd
import numpy as np
import os


def generate_k_means(k=10, dense_corpus=None, run_id=0, directory='/ClusterResults/'):
    """generate KMeans and save results to the directory specified."""

    dense_corpus = pickle.load(open(dense_corpus, 'rb'))
    # start to generate KMeans clustering
    start = time.perf_counter()
    kmeans = KMeans(n_clusters=k, max_iter=500, n_init=20).fit(dense_corpus)
    clustering = kmeans.predict(dense_corpus)
    clustering_result = dd(list)
    for i in range(len(clustering)):
        clustering_result[clustering[i]].append(i)
    end = time.perf_counter()

    # print results to screen
    print('time used:', int(end - start))

    # save to directory
    os.mkdir(directory + str(run_id))
    pickle.dump(kmeans, open(directory + str(run_id) + '/' + 'model', 'wb'))

    return clustering_result


def predict_cluster_labels(run_id, directory):
    """predict cluster label for documents in corpus,
    then save the results as a list in to the directory specified"""
    clustering = pickle.load(open(directory + str(run_id) + '/' + 'model', 'rb'))
    labels = clustering.labels_
    pickle.dump(labels, open(directory + str(run_id) + '/' + 'labels', 'wb'))
    return labels


# noinspection PyTypeChecker
def generate_cluster_keywords(run_id, corpus, bow, dictionary, mode='centroid', num_docs=10, num_keywords=10,
                              directory='/ClusterResults/'):
    """modes = ['centroid','cluster']
        generate keywords for clusters with mode option and save results to directory specified."""
    corpus = pickle.load(open(corpus, 'rb'))
    bow = pickle.load(open(bow, 'rb'))
    dictionary = pickle.load(open(dictionary, 'rb'))
    clustering = pickle.load(open(directory + str(run_id) + '/' + 'model', 'rb'))
    clusters = dd(list)
    labels = clustering.labels_
    for i in range(len(labels)):
        cluster = labels[i]
        clusters[cluster].append(i)

    if mode == 'centroid':
        # find the central documents
        central_docs_clusters = dd(list)
        for c in range(len(clusters)):
            docs_id = clusters[c]
            centroid = np.asarray([clustering.cluster_centers_[c]])
            docs = np.asarray([corpus[i] for i in docs_id])
            distances = abs(cosine_similarity(docs, centroid))
            docs_pair = [(distances[i], i) for i in range(len(distances))]
            central_docs_cluster = [v for k, v in sorted(docs_pair, key=lambda x: x[0])[:num_docs]]
            central_docs_clusters[c] = central_docs_cluster

        clusters = central_docs_clusters

    # compute word frequency per cluster
    clusters_word_dist = dd(dict)
    for c in clusters.keys():
        word_dist = dd(float)
        for doc in clusters[c]:
            document = bow[doc]
            for token in document:
                word_dist[token[0]] += token[1]
        clusters_word_dist[c] = word_dist

    # sort the word frequency per cluster
    for c in clusters_word_dist.keys():
        word_dist = clusters_word_dist[c]
        word_lst = sorted([(v, k) for k, v in word_dist.items()], reverse=True)
        clusters_word_dist[c] = word_lst

    # get the list of keywords
    keywords = []
    for c in range(len(clusters_word_dist)):
        c_id = 'cluster ' + str(c)
        keyword = [dictionary[v] for k, v in clusters_word_dist[c][:num_keywords]]

        keywords.append((c_id, keyword))

    fp = open(directory + str(run_id) + '/' + mode + '_keywords.txt', 'w')

    for row in keywords:
        print(row[0])
        print(row[1])
        fp.write(row[0])
        fp.write('\n')
        fp.write(str(row[1]))
        fp.write('\n')
    fp.close()

    return None


def get_cluster_vectors(cid, directory='ClusterResults/'):

    kmeans = pickle.load(open(directory+str(cid)+'/model', 'rb'))

    return kmeans.cluster_centrers_

