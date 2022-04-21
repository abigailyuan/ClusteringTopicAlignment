import re
from collections import defaultdict as dd
import pickle
import matplotlib.pyplot as plt

# parse qrels and extract wsj qrels
import Visualisation


def get_query_doc(qrel):
    qrel = qrel.split(' ')
    query = int(qrel[0])-51
    doc = qrel[2]
    rel = qrel[-1][0]
    return query,doc,rel

def read_qrels(filename, queries):
    fp = open(filename, 'r')
    rows = fp.readlines()
    fp.close()

    for row in rows:
        query, doc, rel = get_query_doc(row)
        if re.match(r'WSJ', doc) and rel == '1':
            queries[query].append(doc)

    return queries

def build_index(index):
    indexes = {}
    for i,doc in enumerate(index):
        indexes[doc] = i
    return indexes

def convert_to_id(queries):
    index = pickle.load(open("ProcessedWSJ/wsj_index.pkl",'rb'))
    index = build_index(index)
    for q in queries:
        docs = queries[q]

        doc_ids = []
        for d in docs:
            if d in index.keys():
                doc_ids.append(index[d])

        queries[q] = doc_ids

def to_labels(queries):
    docs = []
    for q in queries:
        for d in queries[q]:
            docs.append((d,q))

    docs = sorted(docs)

    return docs

def get_run_doc(row):
    row = row.split(' ')
    topic = int(row[0])
    doc = row[2]
    return topic, doc

def read_run(filename, queries):
    fp = open(filename, 'r')
    rows = fp.readlines()
    fp.close()

    for row in rows:
        query, doc = get_run_doc(row)
        if re.match(r'WSJ', doc):
            queries[query].append(doc)

    return queries

# split qrels into clusters where a topic is a cluster

# save them as clusters
# queries = dd(list)
# read_qrels("wsj_qrels/qrels.051-100.trec1.adhoc", queries)
# read_qrels("wsj_qrels/qrels.101-150.trec2.adhoc", queries)
# read_qrels("wsj_qrels/qrels.151-200.trec3.adhoc", queries)
# read_qrels("wsj_qrels/qrels.201-250.trec4.adhoc", queries)
# read_qrels("wsj_qrels/qrels.251-300.trec5.adhoc", queries)
#
# convert_to_id(queries)
# clusters_dict = {}
# cid = 99
# for step in range(0,241,20):
#     for i in range(20):
#         clusters_dict[i] = queries[i+step]
#
#
#     cid += 1
#     tid = 1
#     clustering = clusters_dict
#     topic_model = 'LDAResults/' + str(tid) + '/model'
#     corpus = 'ProcessedWSJ/tfidf_corpus.pkl'
#     directory = 'figures/'
#     clusters, cluster_topic_matrix = Visualisation.compare_cluster_topic(clustering, topic_model, corpus=corpus,
#                                                                          c_order=20,t_order=20, mode='distribution')
#
#     Visualisation.topic_distribution_visualise(clusters, cluster_topic_matrix, cid=cid, tid=tid, c_order=20, t_order=20,
#                                                directory=directory, mode='distribution')
#
#     # clusters, cluster_topic_matrix = Visualisation.compare_cluster_topic(clustering, topic_model, corpus=corpus,
#     #                                                                      c_order=20, t_order=20, mode='label')
#     # Visualisation.topic_distribution_visualise(clusters, cluster_topic_matrix, cid=cid, tid=tid, c_order=20, t_order=20,
#     #                                           directory=directory, mode='label')
#
# print(len(queries.keys()))
# for i,k in enumerate(queries.keys()):
#     print(i,k)

queries = dd(list)
read_run("trec2/input.citri1", queries)
read_run("trec3.adhoc/input.citri1", queries)
read_run("trec4.adhoc/input.citri1", queries)

convert_to_id(queries)
query_keys = []
new_queries = {}
for q in queries:
    if queries[q] != []:
        new_queries[q] = queries[q]
    else:
        print(q)
print(len(queries))
print(len(new_queries))
# fig, axes = plt.subplots(5,1, figsize=(80, 160))
#
# cid = 199
# ax_i = 0
# for i in range(101,201,20):
#     clusters_dict = {}
#     for query in range(i,i+20):
#         clusters_dict[query] = new_queries[query]
#
#     ax = axes[ax_i]
#     ax_i += 1
#     cid += 1
#     tid = 1
#     clustering = clusters_dict
#     topic_model = 'LDAResults/' + str(tid) + '/model'
#     corpus = 'ProcessedWSJ/tfidf_corpus.pkl'
#     directory = 'figures/'
#     clusters, cluster_topic_matrix = Visualisation.compare_cluster_topic(clustering, topic_model, corpus=corpus,
#                                                                          c_order=20,t_order=20, mode='distribution', normalised=False)
#     Visualisation.topic_distribution_visualise(ax, clusters, cluster_topic_matrix, cid=cid, tid=tid, c_order=20, t_order=20,
#                                                directory=directory, mode='distribution')
#
#     # clusters, cluster_topic_matrix = Visualisation.compare_cluster_topic(clustering, topic_model, corpus=corpus,
#     #                                                                      c_order=20, t_order=20, mode='label')
#     # Visualisation.topic_distribution_visualise(clusters, cluster_topic_matrix, cid=cid, tid=tid, c_order=20, t_order=20,
#     #                                           directory=directory, mode='label')
#
#     plt.savefig("retrieved_docs_topic_distribution.svg")