import pickle

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from gensim.models import LdaModel
from collections import defaultdict as dd
from sklearn.metrics.pairwise import cosine_similarity
import KMeansGenerator
import LDAGenerator
import seaborn as sns
import sklearn
from scipy.stats import skew


def create_topic_rows(document_topics, order):
    topic_rows = np.zeros((len(document_topics), order))

    for doc_id in range(len(document_topics)):
        doc = document_topics[doc_id]
        for topic_id, prob in doc:
            topic_rows[doc_id][topic_id] = prob

    return topic_rows


def compare_cluster_topic(clustering, topic_model, corpus, order=10, mode='label'):
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

    if mode == 'label':
        print('this is label run.')
        # create dataframe
        k = order
        d = {'cluster': labels, 'topic_r1': topics_r1}
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

    elif mode == 'distribution':
        print('this is distribution run.')
        # create dataframe
        topic_rows = create_topic_rows(documents_topics, order=order)
        d = {'cluster':labels}
        for i in range(order):
            col = topic_rows[:, i]
            d['topic_'+str(i)] = col

        corpus_labels = pd.DataFrame(d)

        cluster_topic_matrix = corpus_labels.groupby(['cluster']).sum()

        cluster_topic_matrix = cluster_topic_matrix.div(cluster_topic_matrix.sum(axis=1), axis=0)
        #print(cluster_topic_matrix)


    return clusters, cluster_topic_matrix


def topic_distribution_visualise(clusters, cluster_topic_matrix, cid=0, tid=0, order=10, directory='figures/',mode='label'):
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

    #print(cluster_sizes)
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
    if mode == 'label':
        figname = f'{directory}c{crun}t{trun}.pdf'
    else:
        figname = f'{directory}c{crun}t{trun}_{mode}.pdf'
    plt.savefig(figname)

    return 0


def vector_similarity(centroids, topics, norm='l1',axis=1):

    cos_sim_matrix = cosine_similarity(centroids, topics)
    if norm=='l1':
        cos_sim_matrix = sklearn.preprocessing.normalize(cos_sim_matrix, norm, axis)
    #print(cos_sim_matrix)
    return cos_sim_matrix

def visualise_vecter_similarity(tid, cid, directory='figures/',norm='none', figname=''):

    centroids = KMeansGenerator.get_cluster_vectors(cid)
    topics = LDAGenerator.get_topic_vectors(tid)
    cos_sim_matrix = vector_similarity(centroids, topics,norm)
    order = cos_sim_matrix.shape[0]
    clusters = ['c'+str(i) for i in range(order)]
    topics = ['t'+str(i) for i in range(order)]

    fig, ax = plt.subplots(figsize=(16,10))

    ax = sns.heatmap(cos_sim_matrix,annot=True, annot_kws={'fontsize':"small"})
    ax.set_xlabel("Topic")
    ax.set_ylabel("Cluster")
    if figname:
        figname = f'{directory}c{cid}t{tid}heatmap_{figname}.pdf'
    else:
        figname = f'{directory}c{cid}t{tid}heatmap.pdf'
    plt.savefig(figname)

def get_doc_range(c_id, c):
    doc_range = []
    labels = pickle.load(open('ClusterResults/'+str(c_id)+'/labels','rb'))
    for i in range(len(labels)):
        if labels[i] == c:
            doc_range.append(i)

    return doc_range

def topic_distribution(document_topics, t, doc_range=None):
    if doc_range:
        dist = []
        for i in doc_range:
            try:
                dist.append(document_topics[i][t][1])
            except:
                dist.append(0)

    else:
        dist = []
        for i in document_topics:
            dist.append(i[t][1])
    return dist

def get_topic_distribution(corpus=None, cid=0, tid=0, c=0, t=0, mode='all'):
    corpus = pickle.load(open(corpus, 'rb'))
    labels = pickle.load(open('ClusterResults/'+str(cid)+'/labels','rb'))
    lda_model = LdaModel.load('LDAResults/'+str(tid)+'/model')
    documents_topics = lda_model.get_document_topics(corpus)
    if mode == 'all':
        topic_dist = topic_distribution(documents_topics, t)
    else:
        doc_range = get_doc_range(cid, c)
        topic_dist = topic_distribution(documents_topics, t, doc_range)

    return topic_dist

def hist_plot(topic_dist, t, c, tid, directory):
    fig, ax = plt.subplots(figsize=(16, 10))
    ax = sns.histplot(data=topic_dist, kde=True)
    ax.set_xlabel("topic percentage")
    ax.set_ylabel("count")

    figname = directory+'c'+str(c)+'t'+str(t)+'_prob_density.pdf'
    plt.savefig(figname)

def skewness_measure(dist):
    return skew(dist)