from gensim.models import LdaModel
from collections import defaultdict as dd
import pickle
import pandas as pd
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
    queries = text.split('</top>')


queries = get_query_docs()
normalised_distribution, corpus_topics = get_collection_topic_distribution(tid=1, corpus_file='ProcessedWSJ/tfidf_corpus.pkl')
query_kl = dd(float)
for query in queries.keys():
    # get query subset topic distribution
    query_topics = get_subset_topic_distribution(corpus_topics, queries[query])

    # compute KL divergence regarding to corpus distribution
    kl_divergence = KL_divergence(normalised_distribution, query_topics)
    query_kl[query] = kl_divergence

query_kl_lst = sorted([(k,v) for k,v in query_kl.items()], key=lambda x:x[1])
print(query_kl_lst[:5])
x = [x[0] for x in query_kl_lst]
y = [x[1] for x in query_kl_lst]

fig, ax = plt.subplots(figsize=(20,10))
ax.bar(np.arange(101,201), y, width=1, edgecolor="white")
ax.set_xticks(np.arange(101,201),x)
ax.set_xlabel('TREC topics', fontsize=18)
ax.set_ylabel('KL Divergence', fontsize=18)
ax.set_title("KL Divergence with reference to collection topic distribution", fontsize = 18)
plt.xticks(rotation=90, fontsize=10)
plt.savefig("figures/KL_divergence.pdf")
plt.clf()

fig, ax = plt.subplots(figsize=(20,10))
x = [k for k,v in query_kl.items()]
y = [v for k,v in query_kl.items()]
ax.bar(x,y, width=1, edgecolor="white")
ax.set_xlabel('TREC topics', fontsize=18)
ax.set_ylabel('KL Divergence', fontsize=18)
ax.set_xticks(np.arange(101,201),x)
ax.set_title("KL Divergence with reference to collection topic distribution", fontsize = 18)
plt.xticks(rotation=90, fontsize=10)
plt.savefig("figures/KL_divergence_unsorted.pdf")