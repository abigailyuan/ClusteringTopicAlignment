#!/usr/bin/env python
# lda_train.py

import os
import re
import pickle
import argparse

import nltk
from nltk.corpus import stopwords

import spacy
from gensim.utils import simple_preprocess
from gensim.models.phrases import Phrases, Phraser
from gensim.corpora import Dictionary
from gensim.models.ldamodel import LdaModel
from gensim.models.ldamulticore import LdaMulticore

from sklearn.datasets import fetch_20newsgroups

# ────────────────────────────────────────────────────────────────────────────────
# Setup NLTK & SpaCy
# ────────────────────────────────────────────────────────────────────────────────
nltk.download('stopwords', quiet=True)
EN_STOP = set(stopwords.words('english'))

# extra generic stopwords you often see in news/forums
GENERIC_EXTRA = {'said', 'would', 'could', 'also'}

# per-collection high-freq noise terms
COLLECTION_EXTRAS = {
    'wsj':  {'mr', 'ms', 'company', 'new', 'york'},
    '20ng': {'subject', 'lines', 'organization', 'writes'},
    'wiki': {'edit', 'redirect', 'page', 'wikipedia', 'category'},
}

# load spaCy model for lemmatization
try:
    nlp = spacy.load('en_core_web_sm', disable=['parser','ner'])
except OSError:
    import spacy.cli
    spacy.cli.download("en_core_web_sm")
    nlp = spacy.load('en_core_web_sm', disable=['parser','ner'])


# ────────────────────────────────────────────────────────────────────────────────
# 1. Collection-specific preprocessing
# ────────────────────────────────────────────────────────────────────────────────
def preprocess_docs(docs, collection: str):
    """
    docs        : list of raw document strings
    collection  : one of 'wsj','20ng','wiki'
    returns     : list of token lists
    """
    coll = collection.lower()
    if coll not in COLLECTION_EXTRAS:
        raise ValueError(f"Unknown collection {collection!r}, choose from {list(COLLECTION_EXTRAS)}")

    # build stop-list
    stop_words = EN_STOP | GENERIC_EXTRA | COLLECTION_EXTRAS[coll]

    # (1) tokenize & basic cleanup
    tokenized = []
    for doc in docs:
        text = doc.lower()
        text = re.sub(r'\S+@\S+', ' ', text)       # strip emails
        text = re.sub(r'http\S+', ' ', text)       # strip URLs
        text = re.sub(r'\d+', ' ', text)           # strip digits
        toks = simple_preprocess(text, deacc=True, min_len=3)
        toks = [t for t in toks if t not in stop_words]
        tokenized.append(toks)

    # (2) detect bigrams & trigrams
    bigram = Phraser(Phrases(tokenized, min_count=20, threshold=10))
    trigram = Phraser(Phrases(bigram[tokenized], min_count=10, threshold=5))
    tokenized = [trigram[bigram[doc]] for doc in tokenized]

    # (3) lemmatize, keep only content words
    lemmatized = []
    for doc in tokenized:
        sp_doc = nlp(" ".join(doc))
        lemmas = [
            token.lemma_ for token in sp_doc
            if token.pos_ in {'NOUN','VERB','ADJ','ADV'}
               and token.lemma_ not in stop_words
        ]
        lemmatized.append(lemmas)

    return lemmatized


# ────────────────────────────────────────────────────────────────────────────────
# 2. Collection loader
# ────────────────────────────────────────────────────────────────────────────────
def load_collection(name, path=None):
    """
    Returns a list of raw document-strings.
    - wsj:  WSJ().load(path)
    - 20ng: fetch_20newsgroups(subset='all')
    - wiki: reads all .txt files under `path`
    """
    if name == "wsj":
        return pickle.load(open('ProcessedWSJ/wsj_preprocessed.pkl','rb'))

    elif name == "20ng":
        return pickle.load(open('Processed20NG/20ng_preprocessed.pkl','rb'))

    elif name == "wiki":
        return pickle.load(open('ProcessedWIKI/wiki_preprocessed.pkl','rb'))

    else:
        raise ValueError(f"Unknown collection '{name}'")


# ────────────────────────────────────────────────────────────────────────────────
# 3. Train LDA + compute coherence
# ────────────────────────────────────────────────────────────────────────────────
def train_lda(texts, num_topics=50, passes=10,
              no_below=5, no_above=0.5, multicore=False):
    # build dictionary & filter extremes
    dictionary = Dictionary(texts)
    dictionary.filter_extremes(no_below=no_below, no_above=no_above)

    # bag-of-words corpus
    corpus = [dictionary.doc2bow(text) for text in texts]
    
    if multicore:
            lda = LdaMulticore(
                corpus=corpus,
                id2word=dictionary,
                num_topics=num_topics,
                passes=passes,
                workers=8,                # match --cpus-per-task
                chunksize=2000,           # tune for your corpus size
                random_state=42
            )
    else:
        lda = LdaModel(
            corpus=corpus,
            id2word=dictionary,
            num_topics=num_topics,
            passes=passes,
            random_state=42
        )

    return lda, dictionary, corpus


# ────────────────────────────────────────────────────────────────────────────────
# 4. Topic–doc matrix
# ────────────────────────────────────────────────────────────────────────────────
def build_topic_doc_matrix(lda, corpus, num_topics):
    import numpy as np
    num_docs = len(corpus)
    mat = np.zeros((num_topics, num_docs), dtype=float)
    for j, bow in enumerate(corpus):
        for topic_id, prob in lda.get_document_topics(bow, minimum_probability=0):
            mat[topic_id, j] = prob
    return mat


# ────────────────────────────────────────────────────────────────────────────────
# 5. CLI entrypoint
# ────────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Train LDA on WSJ, 20NG, or Wiki with enhanced preprocessing"
    )
    parser.add_argument("--dataset", choices=["wsj","20ng","wiki"],
                        required=True, help="Which collection to use")
    parser.add_argument("--path", type=str, default=None,
                        help="Path for WSJ pickle or Wiki .txt dir")
    parser.add_argument("--num_topics", type=int, default=50,
                        help="Number of LDA topics")
    parser.add_argument("--passes", type=int, default=10,
                        help="Number of training passes")
    parser.add_argument("--no_below", type=int, default=5,
                        help="Min doc-freq for dictionary")
    parser.add_argument("--no_above", type=float, default=0.5,
                        help="Max doc-freq (proportion) for dictionary")
    parser.add_argument("--multicore", action="store_true",
                        help="Use LdaMulticore instead of LdaModel")
    parser.add_argument("--output_dir", type=str, default="Results/LDA",
                        help="Where to save models & matrices")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # 1. load and preprocess
    raw = load_collection(args.dataset, args.path)
    texts = preprocess_docs(raw, args.dataset)
    print(f"[1] Preprocessed {len(texts)} docs from '{args.dataset}'.")

    # 2. train & coherence
    lda, dictionary, corpus = train_lda(
        texts,
        num_topics=args.num_topics,
        passes=args.passes,
        no_below=args.no_below,
        no_above=args.no_above,
        multicore=args.multicore
    )
    print(f"[2] LDA training complete ({'multicore' if args.multicore else 'single'}).")

    # 3. save artifacts
    prefix = args.dataset
    lda.save(os.path.join(args.output_dir, f"{prefix}_lda{args.num_topics}.model"))
    dictionary.save(os.path.join(args.output_dir, f"{prefix}_dictionary.dict"))
    with open(os.path.join(args.output_dir, f"{prefix}_corpus.pkl"), "wb") as f:
        pickle.dump(corpus, f)

    # 4. build & save topic–doc matrix
    td_mat = build_topic_doc_matrix(lda, corpus, args.num_topics)
    with open(os.path.join(args.output_dir, f"{prefix}_topic_doc_matrix.pkl"), "wb") as f:
        pickle.dump(td_mat, f)
    print(f"[3] Topic–doc matrix saved; shape = {td_mat.shape}")

    # 5. sample output
    print("Sample topic–doc matrix (first 5 topics × 5 docs):")
    print(td_mat[:5, :5])


if __name__ == "__main__":
    main()
