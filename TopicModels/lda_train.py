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
    spacy.cli.download("en_core_web_sm")
    nlp = spacy.load('en_core_web_sm', disable=['parser','ner'])


def preprocess_docs(docs, collection: str):
    """
    Collection-specific preprocessing using the same steps you had:
      1) Tokenize & basic cleanup (remove emails, URLs, digits, stopwords).
      2) Detect bigrams/trigrams.
      3) Lemmatize, keep only NOUN/VERB/ADJ/ADV, remove stopwords.
    Returns: list of token lists.
    """
    coll = collection.lower()
    if coll not in COLLECTION_EXTRAS:
        raise ValueError(f"Unknown collection {collection!r}; choose from {list(COLLECTION_EXTRAS)}")

    # Build stop-list
    stop_words = EN_STOP | GENERIC_EXTRA | COLLECTION_EXTRAS[coll]

    # (1) Tokenize & basic cleanup
    tokenized = []
    for doc in docs:
        text = doc.lower()
        text = simple_preprocess(text, deacc=True, min_len=3)  # initial tokenize & lowercase
        # Remove any tokens in stop_words or purely digits
        toks = [t for t in text if t not in stop_words and not t.isdigit()]
        tokenized.append(toks)

    # (2) Bigrams & trigrams
    bigram = Phraser(Phrases(tokenized, min_count=20, threshold=10))
    trigram = Phraser(Phrases(bigram[tokenized], min_count=10, threshold=5))
    tokenized = [trigram[bigram[doc]] for doc in tokenized]

    # (3) Lemmatize, keep only content words
    lemmatized = []
    for doc in tokenized:
        sp_doc = nlp(" ".join(doc))
        lemmas = [
            token.lemma_ for token in sp_doc
            if token.pos_ in {'NOUN','VERB','ADJ','ADV'} and token.lemma_ not in stop_words
        ]
        lemmatized.append(lemmas)

    return lemmatized


def load_preprocessed(dataset: str):
    """
    Load a pickled list of token lists from Processed{DATASET}/{dataset}_preprocessed.pkl
    """
    processed_dir = f"Processed{dataset.upper()}"
    pre_path = os.path.join(processed_dir, f"{dataset}_preprocessed.pkl")
    if not os.path.exists(pre_path):
        sys.exit(f"[ERROR] Preprocessed file not found: {pre_path}")
    with open(pre_path, 'rb') as f:
        tokens = pickle.load(f)
    # We expect tokens to be a list of lists of strings
    if not isinstance(tokens, list) or not all(isinstance(doc, list) for doc in tokens):
        sys.exit(f"[ERROR] Expected a list of token lists in {pre_path}")
    return tokens


def train_lda(texts, num_topics=50, passes=10, no_below=5, no_above=0.5, multicore=False):
    """
    Given tokenized texts (list of token lists), build a dictionary, filter extremes,
    produce a BOW corpus, and train an LDA model.
    Returns: (lda_model, dictionary, corpus)
    """
    dictionary = Dictionary(texts)
    dictionary.filter_extremes(no_below=no_below, no_above=no_above)
    corpus = [dictionary.doc2bow(text) for text in texts]

    if multicore:
        lda = LdaMulticore(
            corpus=corpus,
            id2word=dictionary,
            num_topics=num_topics,
            passes=passes,
            workers=8,
            chunksize=2000,
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


def build_topic_doc_matrix(lda, corpus, num_topics):
    """
    Create a (num_topics × num_docs) matrix where entry (i,j) =
    probability of topic i in document j.
    """
    import numpy as np
    num_docs = len(corpus)
    mat = np.zeros((num_topics, num_docs), dtype=float)
    for j, bow in enumerate(corpus):
        for topic_id, prob in lda.get_document_topics(bow, minimum_probability=0):
            mat[topic_id, j] = prob
    return mat


def main():
    parser = argparse.ArgumentParser(
        description="Train LDA on a preprocessed corpus."
    )
    parser.add_argument(
        "--dataset",
        choices=["wsj", "20ng", "wiki"],
        required=True,
        help="Which collection to use (wsj, 20ng, or wiki)."
    )
    parser.add_argument(
        "--num_topics",
        type=int,
        required=True,
        help="Number of LDA topics."
    )
    parser.add_argument(
        "--passes",
        type=int,
        default=10,
        help="Number of training passes (default: 10)."
    )
    parser.add_argument(
        "--no_below",
        type=int,
        default=5,
        help="Min doc-frequency for dictionary (default: 5)."
    )
    parser.add_argument(
        "--no_above",
        type=float,
        default=0.5,
        help="Max doc-frequency (proportion) for dictionary (default: 0.5)."
    )
    parser.add_argument(
        "--multicore",
        action="store_true",
        help="Use LdaMulticore instead of LdaModel."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="Results/LDA",
        help="Directory to save models & matrices (default: Results/LDA)."
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    prefix = args.dataset
    n_topics = args.num_topics

    # 1) Load preprocessed tokens
    texts = load_preprocessed(prefix)
    print(f"[1] Loaded {len(texts)} preprocessed documents for '{prefix}'.")

    # 2) Train LDA
    lda, dictionary, corpus = train_lda(
        texts,
        num_topics=n_topics,
        passes=args.passes,
        no_below=args.no_below,
        no_above=args.no_above,
        multicore=args.multicore
    )
    mode = "multicore" if args.multicore else "single-core"
    print(f"[2] LDA training complete ({mode}) with num_topics={n_topics}.")

    # 3) Save dictionary & corpus
    dict_path = os.path.join(args.output_dir, f"{prefix}_dictionary.dict")
    dictionary.save(dict_path)
    print(f"[SAVED] Dictionary at {dict_path}")

    corpus_path = os.path.join(args.output_dir, f"{prefix}_corpus.pkl")
    with open(corpus_path, "wb") as f:
        pickle.dump(corpus, f)
    print(f"[SAVED] Corpus (BOW) at {corpus_path}")

    # 4) Save LDA model
    model_path = os.path.join(args.output_dir, f"{prefix}_lda{n_topics}.model")
    lda.save(model_path)
    print(f"[SAVED] LDA model at {model_path}")

    # 5) Build & save topic–document matrix (always named {prefix}_topic_doc_matrix.pkl)
    td_mat = build_topic_doc_matrix(lda, corpus, n_topics)
    td_path = os.path.join(args.output_dir, f"{prefix}_topic_doc_matrix.pkl")
    with open(td_path, "wb") as f:
        pickle.dump(td_mat, f)
    print(f"[SAVED] Topic–doc matrix at {td_path}, shape = {td_mat.shape}")

    # Sample output
    print("Sample topic–doc matrix (first 5 topics × 5 docs):")
    print(td_mat[:5, :5])


if __name__ == "__main__":
    main()
