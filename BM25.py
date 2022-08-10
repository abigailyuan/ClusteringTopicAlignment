from rank_bm25 import BM25Okapi

def build_bm25(tokenised_corpus):
    bm25 = BM25Okapi(tokenised_corpus)
    return bm25

