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

def to_labels(queries):
    docs = []
    for q in queries:
        for d in queries[q]:
            docs.append((d,q))

    docs = sorted(docs)

    return docs

def compare_cluster_topic(clustering, topic_model, corpus, c_order=20, t_order=20, mode='label', normalised = False):
    # load topic model
    corpus = pickle.load(open(corpus, 'rb'))
    lda_model = LdaModel.load(topic_model)
    documents_topics = lda_model.get_document_topics(corpus, minimum_probability=0.0005)
    topics_r1 = [sorted(i, key=lambda x: x[1], reverse=True)[0][0] for i in
                 documents_topics]  # find the most likely topic per document

    # load clustering
    kmeans = pickle.load(open(clustering, 'rb'))
    labels = kmeans.labels_
    clusters = dd(int)
    for i in labels:
        clusters[i] += 1

    # for qrel version
    # clusters = clustering
    # docs = to_labels(clusters)
    # labels = [v for k,v in docs]


    if mode == 'label':
        print('this is label run.')
        # create dataframe

        d = {'cluster': labels, 'topic_r1': topics_r1}
        corpus_labels = pd.DataFrame(d)

        data = np.zeros((c_order, t_order), dtype=float)
        for row in range(corpus_labels.shape[0]):
            cluster = corpus_labels.iloc[row, 0]
            topic = corpus_labels.iloc[row, 1]
            data[cluster][topic] += 1

        np.set_printoptions(suppress=True)
        index = ['c' + str(i) for i in range(c_order)]
        columns = ['t' + str(i) for i in range(t_order)]
        cluster_topic_matrix = pd.DataFrame(data=data, index=index, columns=columns)

        # drop dominant topics here
        # cluster_topic_matrix = cluster_topic_matrix.drop(['t9','t13'], axis='columns')

        # get the percentage per bar
        for row in range(cluster_topic_matrix.shape[0]):
            cluster = cluster_topic_matrix.iloc[row]
            cluster_size = cluster.sum()
            for col in range(cluster_topic_matrix.shape[1]):
                cluster_topic_matrix.iloc[row][col] /= (cluster_size / 100)  # making percentage

    elif mode == 'distribution':
        print('this is distribution run.')
        # create dataframe
        topic_rows = create_topic_rows(documents_topics, order=t_order)
        d = {'cluster':labels}

        # for qrel
        # selected_rows = [k for k,v in docs]
        for i in range(t_order):
            #col = topic_rows[selected_rows, i]
            col = topic_rows[:,i]
            d['t'+str(i)] = col

        corpus_labels = pd.DataFrame(d)

        cluster_topic_matrix = corpus_labels.groupby(['cluster']).sum()

        #drop dominant topics here
        # cluster_topic_matrix = cluster_topic_matrix.drop(['t9','t13'], axis='columns')

        # normalise the topic distributions with corpus topic probability
        if normalised:
            corpus_topics = {}
            for doc in documents_topics:
                for topic, val in doc:
                    corpus_topics[topic] = corpus_topics.get(topic, 0) + val

            corpus_sum = sum([v for v in corpus_topics.values()])

            for k in corpus_topics.keys():
                corpus_topics[k] /= corpus_sum

            for col in range(cluster_topic_matrix.shape[1]):
                colume = cluster_topic_matrix.iloc[:, col] / corpus_topics[col]
                cluster_topic_matrix['t' + str(col)] = colume

        for row in range(cluster_topic_matrix.shape[0]):
            curr_row = cluster_topic_matrix.iloc[row, :]
            row_sum = curr_row.sum()
            for col in range(cluster_topic_matrix.shape[1]):
                cluster_topic_matrix.iat[row, col] /= row_sum

    return clusters, cluster_topic_matrix


def topic_distribution_visualise(ax, clusters, cluster_topic_matrix, cid=0, tid=0, c_order=30,t_order=20, directory='figures/',mode='label'):
    # set plot title
    # plot
    num_topic = t_order

    crun = str(cid)
    trun = str(tid)

    # plt.clf()
    # plt.figure()

    try:
        ax = cluster_topic_matrix.plot.barh(ax = ax, stacked=True, figsize=(30,30))
    except:
        print(cluster_topic_matrix)
        return 0

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

    # for qrel
    # cluster_sizes = [len(clusters[i]) for i in sorted(list(clusters.keys()))]
    cluster_sizes = [clusters[i] for i in range(len(clusters))]

    new_yticklable = []
    i = 0
    for ticklabel in ax.get_yticklabels():
        org_txt = ticklabel.get_text()
        new_yticklable.append(f'{org_txt} ({cluster_sizes[i]:,d})')
        i += 1
    ax.set_yticklabels(new_yticklable)
    ax.set_xlabel('percentage of topics', fontsize=18)
    ax.set_ylabel('query (# retrieved docs)', fontsize=18)
    ax.grid(which='both')
    ax.grid(which='minor', alpha=0.2, linestyle='--')
    ax.legend(loc=1)
    ax.set_title("Topic Distribution Per Cluster", fontsize = 18)
    # change figure name and save
    if mode == 'label':
        figname = f'{directory}c{crun}t{trun}_no913.pdf'
    else:
        figname = f'{directory}c{crun}t{trun}_{mode}_normalised.pdf'
    # plt.savefig(figname)

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
            for k,v in i:
                if k == t:
                    dist.append(v)
    return dist

def get_topic_distribution(corpus=None, cid=0, tid=0, c=0, t=0, mode='all'):
    corpus = pickle.load(open(corpus, 'rb'))
    labels = pickle.load(open('ClusterResults/'+str(cid)+'/labels','rb'))
    lda_model = LdaModel.load('LDAResults/'+str(tid)+'/model')
    documents_topics = lda_model.get_document_topics(bow=corpus, minimum_probability=0.000)
    if mode == 'all':
        topic_dist = topic_distribution(documents_topics, t)
    else:
        doc_range = get_doc_range(cid, c)
        topic_dist = topic_distribution(documents_topics, t, doc_range)

    return topic_dist


# displot
def hist_plot(ax, topic_dist, t, c, tid, directory):
    ax = sns.histplot(data=topic_dist, bins=100, kde=True,cumulative=False, ax=ax)
    # ax.set_xlim(0, 1)
    ax.set_xlabel("topic percentage", fontsize=18)
    ax.set_ylabel("count", fontsize = 18)
    return ax

    # #figname = directory+'c'+str(c)+'t'+str(t)+'_prob_density.pdf'
    # figname = directory+'c'+str(c)+'t'+str(t)+'_prob_density.pdf'
    # plt.savefig(figname)
    # plt.close()


def cluster_topic_dist(clusters, cluster_topic_matrix, t):
    return cluster_topic_matrix.iloc[:, t]


def skewness_measure(dist):
    return skew(dist)