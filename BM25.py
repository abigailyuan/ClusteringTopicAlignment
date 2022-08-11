import pickle
from rank_bm25 import BM25Okapi

def retrieve_BM25(corpus, queries, n_docs = 100):
    bm25 = BM25Okapi(corpus)

    for query in queries:
        bm25.get_top_n(query, corpus, n=n_docs)


file = 'ProcessedWSJ/wsj_lemmatised.pkl'
fp = open(file,'rb')
corpus = pickle.load(fp)
fp.close()

fp = open('tokenized_queries.pkl','rb')
queries = pickle.load(fp)
fp.close()


print(len(queries))
for q in queries[:10]:
    print(q)


