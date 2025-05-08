import os
import pickle
import numpy as np
from collections import defaultdict as dd
from scipy.cluster.hierarchy import dendrogram
import matplotlib.pyplot as plt
from scipy import  spatial
from gensim.models import LdaModel
from sklearn.random_projection import GaussianRandomProjection

from model_creation.HierarchicalClusteringGenerator import agglomaritive
from model_creation.LDAGenerator import generate_lda, generate_topic_keywords
from model_creation.LDAGenerator import get_document_topic_matrix
from mappings.GreedyMapping import greedy_mapping, global_mapping, feature_mapping
from model_creation import KMeansGenerator

import hdbscan
from sentence_transformers import SentenceTransformer
from Specificity.specificity import calculate_Mi


dir = 'ProcessedWiki/'
corpus = dir+'bow_lemma.pkl'
dictionary = dir+'id2word.dict'

run_id = 3
n_dims = 50

# lda = generate_lda(corpus=corpus, run_id=3, num_topics=50, dictionary=dictionary, update_every=1, directory='Results/wiki/')
#
# generate_topic_keywords(run_id=3,num_topics=50, directory='Results/wiki/' )
#
# # random projection
# doc2vec_corpus = pickle.load(open(dir+'wsj_doc2vec500.pkl', 'rb'))
#
# rj = GaussianRandomProjection(n_components=n_dims)
# new_x = rj.fit_transform(X=doc2vec_corpus)
# pickle.dump(new_x, open(f'Results/wiki/rj.{n_dims}.pkl', 'wb'))


# # hierarchical clustering
rj = pickle.load(open(f'Results/wiki/rj.{n_dims}.pkl', 'rb'))
clustering = agglomaritive(rj.T)
pickle.dump(clustering, open(f'Results/wiki/hierarchical/rj_{n_dims}.pkl', 'wb'))

# # calculate specificity
# lda = LdaModel.load('Results/wiki/3/model')
bow = pickle.load(open(corpus, 'rb'))
# Mis = calculate_Mi(model=lda, corpus=bow, mode='lda', bi_mode='GMM', myui_mode='sqrt')
# pickle.dump(Mis, open('Results/wiki/3/specificity.lst', 'wb'))

Mis = pickle.load(open('Results/wiki/3/specificity.lst', 'rb'))

# mapping
def print_mapping(mapping):
    topic_mapping = dd(list)
    for k, v in mapping:
        topic_mapping[v].append(k)
    return topic_mapping

topics = get_document_topic_matrix(bow, tid=str(run_id))
mapping = feature_mapping(list(rj.T), list(topics.T))
topic_mapping = print_mapping(mapping)
for k,v in topic_mapping.items():
    print(f'Topic {k}: {v}')


def plot_specificity_vs_mapping(Mis, topic_mapping):
    fig, ax = plt.subplots()

    x_mapped = [i for i in range(50) if i in topic_mapping.keys()]
    y_mapped = [Mis[i] for i in x_mapped]
    l1 = ax.scatter(x_mapped, y_mapped, c='red')

    x_unmapped = [i for i in range(50) if not (i in topic_mapping.keys())]
    y_unmapped = [Mis[i] for i in x_unmapped]
    l2 = ax.scatter(x_unmapped, y_unmapped, c='blue')

    plt.xlabel('Topic ID')
    plt.ylabel('Topic Specificity')

    fig.legend((l1,l2), ('mapped topic', 'unmapped topic'), loc='upper right')


    # annotate
    for t,v in topic_mapping.items():
        text = str(v)
        x = t
        y = Mis[t]
        ax.annotate(text, xy=(x,y), xytext=(x, y+0.01), fontsize=10)
    plt.show()

# plot_specificity_vs_mapping(Mis, topic_mapping)

def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)


model = clustering
plt.title("Agglomerative Clustering of Random Projection Features (Ward linkage)")
# plot the top three levels of the dendrogram
plot_dendrogram(model, truncate_mode="level", p=30)
plt.xlabel("Number of points in node (or index of point if no parenthesis).")
plt.show()



# c_doc2vec = KMeansGenerator.generate_k_means(k=5, dense_corpus='ProcessedWiki/wsj_doc2vec500.pkl',run_id=5, directory='Results/')
# c_rp = KMeansGenerator.generate_k_means(k=5, dense_corpus='Results/wiki/rj.20.pkl', run_id=6,directory='Results/')

# c_d = pickle.load(open('Results/5/model', 'rb'))
# c_r = pickle.load(open('Results/6/model', 'rb'))



# KMeansGenerator.generate_cluster_keywords(run_id=0, corpus=dir+'wsj_doc2vec500.pkl', bow=corpus, dictionary=dictionary, num_docs=100,num_keywords=20, directory='Results/')

