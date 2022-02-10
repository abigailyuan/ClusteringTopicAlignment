import pickle

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from gensim.models import LdaModel
from collections import defaultdict as dd
from sklearn.metrics.pairwise import cosine_similarity
import KMeansGenerator
import LDAGenerator


def compare_cluster_topic(clustering, topic_model, corpus, order=10):
    # load topic model
    corpus = pickle.load(open(corpus, 'rb'))
    lda_model = LdaModel.load(topic_model)
    documents_topics = lda_model.get_document_topics(corpus)
    topics_r1 = [sorted(i, key=lambda x: x[1], reverse=True)[0][0] for i in
                 documents_topics]  # find the most likely topic per document

    # load clustering
    kmeans = pickle.load(open(clustering, 'rb'))
    labels = kmeans.labels_
    clusters = dd(int)
    for i in labels:
        clusters[i] += 1

    # create dataframe
    k = order
    d = {'cluster': labels, 'topic_r1': topics_r1}
    # print(len(topics_r1))
    # print(len(labels))
    corpus_labels = pd.DataFrame(d)

    data = np.zeros((k, k), dtype=float)
    for row in range(corpus_labels.shape[0]):
        cluster = corpus_labels.iloc[row, 0]
        topic = corpus_labels.iloc[row, 1]
        data[cluster][topic] += 1

    np.set_printoptions(suppress=True)
    index = ['c' + str(i) for i in range(k)]
    columns = ['t' + str(i) for i in range(k)]
    cluster_topic_matrix = pd.DataFrame(data=data, index=index, columns=columns)
    cluster_topic_matrix.head()

    # get the percentage per bar
    for row in range(cluster_topic_matrix.shape[0]):
        cluster = cluster_topic_matrix.iloc[row]
        cluster_size = cluster.sum()
        for col in range(cluster_topic_matrix.shape[1]):
            cluster_topic_matrix.iloc[row][col] /= (cluster_size / 100)  # making percentage

    return clusters, cluster_topic_matrix


def topic_distribution_visualise(clusters, cluster_topic_matrix, cid=0, tid=0, order=10, directory='figures/'):
    # set plot title
    # plot
    num_topic = order

    crun = str(cid)
    trun = str(tid)

    plt.clf()
    plt.figure()

    ax = cluster_topic_matrix.plot.barh(stacked=True, figsize=(20, 10))

    patterns = [' '] * int(num_topic / 2)
    patterns_back = ['///', '\\\\\\'] * int(num_topic / 4)
    patterns.extend(patterns_back)
    bars = ax.patches
    hatches = []  # list for hatches in the order of the bars
    for h in patterns:  # loop over patterns to create bar-ordered hatches
        for i in range(int(len(bars) / len(patterns))):
            hatches.append(h)
    for bar, hatch in zip(bars, hatches):  # loop over bars and hatches to set hatches in correct order
        bar.set_hatch(hatch)

    cluster_sizes = [clusters[i] for i in range(len(clusters))]

    print(cluster_sizes)
    new_yticklable = []
    i = 0
    for ticklabel in ax.get_yticklabels():
        org_txt = ticklabel.get_text()
        new_yticklable.append(f'{org_txt} ({cluster_sizes[i]:,d})')
        i += 1
    ax.set_yticklabels(new_yticklable)
    ax.set_xlabel('percentage of topics', fontsize=18)
    ax.set_ylabel('cluster (cluster-size)', fontsize=18)
    ax.grid(which='both')
    ax.grid(which='minor', alpha=0.2, linestyle='--')
    ax.legend(loc=1)
    # change figure name and save
    figname = f'{directory}c{crun}t{trun}.pdf'
    plt.savefig(figname)

    return 0


def vector_similarity(centroids, topics):

    cos_sim_matrix = cosine_similarity(centroids, topics)
    return cos_sim_matrix

def visualise_vecter_similarity(tid, cid, directory='figures/'):

    centroids = KMeansGenerator.get_cluster_vectors(cid)
    topics = LDAGenerator.get_topic_vectors(tid)
    cos_sim_matrix = vector_similarity(centroids, topics)
    order = cos_sim_matrix.shape[0]
    clusters = ['c'+str(i) for i in range(order)]
    topics = ['t'+str(i) for i in range(order)]

    fig, ax = plt.subplots()
    im = ax.imshow(cos_sim_matrix)

    # Show all ticks and label them with the respective list entries
    ax.set_xticks(np.arange(len(topics)), labels=topics)
    ax.set_yticks(np.arange(len(clusters)), labels=clusters)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    for i in range(len(clusters)):
        for j in range(len(topics)):
            text = ax.text(j, i, cos_sim_matrix[i, j],
                           ha="center", va="center", color="w")

    fig.tight_layout()
    figname = f'{directory}c{cid}t{tid}heatmap.pdf'
    plt.savefig(directory+figname)

