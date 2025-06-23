#!/usr/bin/env python
# lda_train.py

import os
import sys
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
import numpy as np

# ────────────────────────────────────────────────────────────────────────────────
# 0. Setup NLTK & SpaCy
# ────────────────────────────────────────────────────────────────────────────────
nltk.download('stopwords', quiet=True)
EN_STOP = set(stopwords.words('english'))

GENERIC_EXTRA = {'said', 'would', 'could', 'also'}
COLLECTION_EXTRAS = {
    'wsj':  {'mr', 'ms', 'company', 'new', 'york'},
    '20ng': {'subject', 'lines', 'organization', 'writes'},
    'wiki': {'edit', 'redirect', 'page', 'wikipedia', 'category'},
}

# Load spaCy model for lemmatization
try:
    nlp = spacy.load('en_core_web_sm', disable=['parser','ner'])
except OSError:
    import spacy.cli
    spacy.cli.download('en_core_web_sm')
    nlp = spacy.load('en_core_web_sm', disable=['parser','ner'])


def load_preprocessed(dataset: str):
    processed_dir = f"Processed{dataset.upper()}"
    pre_path = os.path.join(processed_dir, f"{dataset}_preprocessed.pkl")
    if not os.path.exists(pre_path):
        sys.exit(f"[ERROR] Preprocessed file not found: {pre_path}")
    with open(pre_path, 'rb') as f:
        tokens = pickle.load(f)
    if not isinstance(tokens, list) or not all(isinstance(doc, list) for doc in tokens):
        sys.exit(f"[ERROR] Expected list of token lists in {pre_path}")
    return tokens


def train_lda(texts, num_topics, passes, no_below, no_above, multicore):
    dictionary = Dictionary(texts)
    dictionary.filter_extremes(no_below=no_below, no_above=no_above)
    corpus = [dictionary.doc2bow(text) for text in texts]
    if multicore:
        lda = LdaMulticore(corpus=corpus, id2word=dictionary, num_topics=num_topics,
                            passes=passes, workers=8, chunksize=2000, random_state=42)
    else:
        lda = LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics,
                        passes=passes, random_state=42)
    return lda, dictionary, corpus


def build_topic_doc_matrix(lda, corpus, num_topics):
    num_docs = len(corpus)
    mat = np.zeros((num_topics, num_docs), dtype=float)
    for j, bow in enumerate(corpus):
        for topic_id, prob in lda.get_document_topics(bow, minimum_probability=0):
            mat[topic_id, j] = prob
    return mat


def main():
    parser = argparse.ArgumentParser(description="Train or load LDA and (re)generate topic-doc matrix.")
    parser.add_argument("--dataset", choices=["wsj","20ng","wiki"], required=True)
    parser.add_argument("--num_topics", type=int, required=True)
    parser.add_argument("--passes", type=int, default=10)
    parser.add_argument("--no_below", type=int, default=5)
    parser.add_argument("--no_above", type=float, default=0.5)
    parser.add_argument("--multicore", action='store_true')
    parser.add_argument("--output_dir", type=str, default="Results/LDA")
    parser.add_argument("--force", action='store_true', help="Force retrain even if model exists.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    prefix = args.dataset
    n_topics = args.num_topics

    # Paths include n_topics for unique matrices
    model_path = os.path.join(args.output_dir, f"{prefix}_lda{n_topics}.model")
    corpus_path = os.path.join(args.output_dir, f"{prefix}_corpus.pkl")
    mat_path   = os.path.join(args.output_dir, f"{prefix}_topic_doc_matrix_{n_topics}.pkl")

    # Load preprocessed tokens
    texts = load_preprocessed(prefix)

    # Train or load model
    if os.path.exists(model_path) and not args.force:
        print(f"[SKIP TRAIN] Loading existing LDA model: {model_path}")
        lda = LdaModel.load(model_path)
        if os.path.exists(corpus_path):
            corpus = pickle.load(open(corpus_path, 'rb'))
        else:
            # If corpus missing, rebuild via preprocessing
            corpus = [Dictionary(load_preprocessed(prefix)).doc2bow(doc) for doc in texts]
    else:
        print(f"[TRAIN] Training LDA with {n_topics} topics...")
        lda, dictionary, corpus = train_lda(
            texts, n_topics, args.passes, args.no_below, args.no_above, args.multicore)
        # Save dictionary, corpus, and model
        dict_path = os.path.join(args.output_dir, f"{prefix}_dictionary.dict")
        dictionary.save(dict_path)
        print(f"[SAVED] Dictionary at {dict_path}")
        with open(corpus_path, 'wb') as f:
            pickle.dump(corpus, f)
        print(f"[SAVED] Corpus at {corpus_path}")
        lda.save(model_path)
        print(f"[SAVED] LDA model at {model_path}")

    # Always regenerate unique matrix
    print(f"[BUILD MATRIX] Building topic-doc matrix for {n_topics} topics...")
    td_mat = build_topic_doc_matrix(lda, corpus, n_topics)
    with open(mat_path, 'wb') as f:
        pickle.dump(td_mat, f)
    print(f"[SAVED] Topic-doc matrix at {mat_path}, shape = {td_mat.shape}")

if __name__ == "__main__":
    main()
