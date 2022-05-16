import time

from gensim.models import LdaModel
from collections import defaultdict as dd
import pickle
import pandas as pd

import LanguageModel
from rel_to_clustering import get_query_docs
import numpy as np
import matplotlib.pyplot as plt

def get_collection_topic_distribution(tid, corpus_file):

    # load topic model and corpus
    topic_model = 'LDAResults/' + str(tid) + '/model'
    lda = LdaModel.load(topic_model)
    corpus = pickle.load(open(corpus_file, 'rb'))

    # predict topic distribution
    documents_topics = lda.get_document_topics(corpus, minimum_probability=0.00)


    # aggregate distributions and normalise
    topics = dd(list)
    for doc in documents_topics:
        for topic, val in doc:
            topics[topic].append(val)

    corpus_topics = pd.DataFrame(topics, index=[i for i in range(len(documents_topics))])

    normalised_topics = corpus_topics.sum(axis=0) / corpus_topics.sum(axis=0).sum()
    return normalised_topics, corpus_topics

def get_subset_topic_distribution(corpus, query):
    subset = corpus.loc[query, :]
    normalised_topics = subset.sum(axis=0) / subset.sum().sum()

    return normalised_topics


def KL_divergence(reference, target):
    # P(x) * log P(x) / Q(x)
    elementwise_vals = target * np.log2(target / reference)
    return elementwise_vals.sum()

def get_domain(query_file):
    # TODO get domain list for topics and compare with KL divergence value
    fp = open(query_file,'r')
    text = fp.read()
    fp.close()
    domains = dd(list)
    queries_dict = {}
    queries = text.split('</top>')
    for i in range(len(queries[:-1])):
        queries[i] = queries[i].split('\n')
        query_dict = {}
        num=0
        for row in queries[i]:
            if '<dom>' in row:
                query_dict['domain'] = row[14:].strip()
            if '<num> ' in row:
                num = int(row[14:])
            if '<title>' in row:
                query_dict['title'] = row[15:]
        queries_dict[num] = query_dict
    return queries_dict


def get_topic_word_distribution(lda, topic_id):
    term_topic_matrix = lda.get_topics()
    return term_topic_matrix[topic_id]

def get_tfidf_weights(docs):
    tfidf_weights = np.zeros(10000)
    for i in range(len(tfidf_weights)):
        tfidf_weights[i] = 0.1
    for d in docs:
        for word, score in d:
            tfidf_weights[word] += score
    s = np.sum(tfidf_weights)
    for i in range(len(tfidf_weights)):
        tfidf_weights[i] /= s
    return tfidf_weights

# topic_model = 'LDAResults/' + '16' + '/model'
# lda = LdaModel.load(topic_model)
# print(len(get_topic_word_distribution(lda, 0)))
#
# corpus = 'ProcessedWSJ/tfidf_corpus.pkl'
# dictionary = 'ProcessedWSJ/dictionary.pkl'
# dict = pickle.load(open(dictionary, 'rb'))
# bow_corpus = pickle.load((open('ProcessedWSJ/bow.pkl','rb')))
# print(len(dict))
# print(list(dict.keys())[:10])
# print(list(dict.values())[:10])
# tfidf = pickle.load(open(corpus,'rb'))
# print(tfidf[0])
# print(sum([i[1] for i in tfidf[0]]))
# queries = get_domain('topics.101-150')
# domains = dd(list)
# for i in queries:
#     domains[queries[i]['domain']].append(i)
# for k in domains:
#     print(k)
# print(len(domains))
# print(domains['International Politics'])
# print(domains['Military'])
# print(domains['International Economics'])
# print(domains['Medical & Biological'])
# queries = get_query_docs()
# normalised_distribution, corpus_topics = get_collection_topic_distribution(tid=1, corpus_file='ProcessedWSJ/tfidf_corpus.pkl')
# query_kl = dd(float)
# for query in queries.keys():
#     # get query subset topic distribution
#     query_topics = get_subset_topic_distribution(corpus_topics, queries[query])
#
#     # compute KL divergence regarding to corpus distribution
#     kl_divergence = KL_divergence(normalised_distribution, query_topics)
#     query_kl[query] = kl_divergence
#
# query_kl_lst = sorted([(k,v) for k,v in query_kl.items()], key=lambda x:x[1])
# print(query_kl_lst[:5])
# x = [x[0] for x in query_kl_lst]
# y = [x[1] for x in query_kl_lst]
#
# fig, ax = plt.subplots(figsize=(20,10))
# ax.bar(np.arange(101,201), y, width=1, edgecolor="white")
# ax.set_xticks(np.arange(101,201),x)
# ax.set_xlabel('TREC topics', fontsize=18)
# ax.set_ylabel('KL Divergence', fontsize=18)
# ax.set_title("KL Divergence with reference to collection topic distribution", fontsize = 18)
# plt.xticks(rotation=90, fontsize=10)
# plt.savefig("figures/KL_divergence.pdf")
# plt.clf()
#
# fig, ax = plt.subplots(figsize=(20,10))
# x = [k for k,v in query_kl.items()]
# y = [v for k,v in query_kl.items()]
# ax.bar(x,y, width=1, edgecolor="white")
# ax.set_xlabel('TREC topics', fontsize=18)
# ax.set_ylabel('KL Divergence', fontsize=18)
# ax.set_xticks(np.arange(101,201),x)
# ax.set_title("KL Divergence with reference to collection topic distribution", fontsize = 18)
# plt.xticks(rotation=90, fontsize=10)
# plt.savefig("figures/KL_divergence_unsorted.pdf")

def word_based_KL():
    # word based KL diverence (query subset vs. topics)
    topic_model = 'LDAResults/' + '16' + '/model'
    lda = LdaModel.load(topic_model)
    bow_corpus = pickle.load((open('ProcessedWSJ/bow.pkl','rb')))

    # create topic word distributions
    word_topic_matrix = []
    for i in range(20):
        prob = get_topic_word_distribution(lda, i)
        word_topic_matrix.append(prob)

    # create query subset unigram language models
    queries = get_query_docs()
    query_LMs = []
    for query in queries.keys():
        # get query subset LM
        docs = LanguageModel.get_documents(queries[query], bow_corpus)
        LM = LanguageModel.unigram_model(docs,0.1)
        word_prob = LanguageModel.to_word_probabilities(LM)
        query_LMs.append(word_prob)

    start = time.perf_counter()
    # compute query-topic KL Divergence
    topic_query_kl_matrix = []
    for topic in word_topic_matrix:
        query_kls = []
        for query_LM in query_LMs:
            kl_divergence = KL_divergence(topic, query_LM)
            query_kls.append(kl_divergence)
        topic_query_kl_matrix.append(query_kls)

    # convert to a numpy ndarray
    topic_query_kl_matrix = np.array(topic_query_kl_matrix)

    print(topic_query_kl_matrix)
    end = time.perf_counter()
    print('time used:', int(end - start))
    pickle.dump(topic_query_kl_matrix, open('topic_query_KL_matrix.pkl','wb'))


def tfidf_based_KL():
    # word based KL diverence (query subset vs. topics)
    topic_model = 'LDAResults/' + '1' + '/model'
    lda = LdaModel.load(topic_model)
    tfidf_corpus = pickle.load(open('ProcessedWSJ/tfidf_corpus.pkl','rb'))

    # create topic word distributions
    word_topic_matrix = []
    for i in range(20):
        prob = get_topic_word_distribution(lda, i)
        word_topic_matrix.append(prob)

    # create query subset normalised TFIDF word weights
    queries = get_query_docs()
    tfidf_weights = []
    for query in queries.keys():
        # get query subset LM
        docs = LanguageModel.get_documents(queries[query], tfidf_corpus)
        tfidf_weight = get_tfidf_weights(docs)
        tfidf_weights.append(tfidf_weight)

    start = time.perf_counter()
    # compute query-topic KL Divergence
    topic_query_kl_matrix = []
    for topic in word_topic_matrix:
        query_kls = []
        for query_tfidf in tfidf_weights:
            kl_divergence = KL_divergence(topic, query_tfidf)
            query_kls.append(kl_divergence)
        topic_query_kl_matrix.append(query_kls)

    # convert to a numpy ndarray
    topic_query_kl_matrix = np.array(topic_query_kl_matrix)

    print(topic_query_kl_matrix)
    end = time.perf_counter()
    print('time used:', int(end - start))
    pickle.dump(topic_query_kl_matrix, open('tfidf_topic_query_KL_matrix.pkl', 'wb'))

tfidf_based_KL()
