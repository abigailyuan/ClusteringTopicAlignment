import os
import pickle
from collections import defaultdict as dd
from gensim.models.doc2vec import Doc2Vec
from numpy import asarray
import numpy as np
from gensim.models.ldamodel import LdaModel
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from seaborn import heatmap
from scipy.spatial import distance
import matplotlib.pyplot as plt

from model_creation.KMeansGenerator import generate_k_means, predict_cluster_labels
from model_creation.LDAGenerator import generate_lda, generate_topic_keywords
from pre_processing import corpus_vectorizer
from sklearn.random_projection import GaussianRandomProjection

def get_topic_distribution(model, corpus):
    num_topics = model.get_topics().shape[0]
    corpus_dist = np.ndarray((len(corpus), num_topics)) # number of topics is 23
    for i in range(len(corpus)):
        doc = corpus[i]
        dist = model.get_document_topics(doc, minimum_probability=0)
        for topic, prob in dist:
            corpus_dist[i][topic] = prob
    return corpus_dist


# os.mkdir('Results/wiki/')
dir = 'ProcessedWiki/'
corpus = dir+'bow_lemma.pkl'
dictionary = dir+'id2word.dict'

# os.mkdir(dir+'kmeans/')

# lda = generate_lda(corpus=corpus, run_id=0, num_topics=20, dictionary=dictionary, update_every=1, directory='Results/wiki/')
#
# generate_topic_keywords(0,num_topics=20, directory='Results/wiki/' )

# generate_k_means(k=20, dense_corpus=dir+'wsj_doc2vec500.pkl', directory=dir+'kmeans/')

# predict_cluster_labels(0, dir+'kmeans/')

# random projection

doc2vec_corpus = pickle.load(open(dir+'wsj_doc2vec500.pkl', 'rb'))

rj = GaussianRandomProjection(n_components=20)
new_x = rj.fit_transform(X=doc2vec_corpus)
pickle.dump(new_x, open('Results/wiki/rj.20.pkl', 'wb'))