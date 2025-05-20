#!/usr/bin/env python
# bertopic_train.py

import os
import re
import pickle
import argparse
import nltk
from nltk.corpus import stopwords
import spacy
from gensim.utils import simple_preprocess
from sklearn.datasets import fetch_20newsgroups

from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer

# ────────────────────────────────────────────────────────────────────────────────
nltk.download('stopwords', quiet=True)
EN_STOP = set(stopwords.words('english'))

GENERIC_EXTRA = {'said', 'would', 'could', 'also'}
COLLECTION_EXTRAS = {
    'wsj':  {'mr', 'ms', 'company', 'new', 'york'},
    '20ng': {'subject', 'lines', 'organization', 'writes'},
    'wiki': {'edit', 'redirect', 'page', 'wikipedia', 'category'},
}

try:
    nlp = spacy.load('en_core_web_sm', disable=['parser','ner'])
except OSError:
    import spacy.cli
    spacy.cli.download("en_core_web_sm")
    nlp = spacy.load('en_core_web_sm', disable=['parser','ner'])

# ────────────────────────────────────────────────────────────────────────────────
def preprocess_docs(docs, collection: str):
    coll = collection.lower()
    if coll not in COLLECTION_EXTRAS:
        raise ValueError(f"Unknown collection {collection!r}, choose from {list(COLLECTION_EXTRAS)}")

    stop_words = EN_STOP | GENERIC_EXTRA | COLLECTION_EXTRAS[coll]

    tokenized = []
    for doc in docs:
        text = doc.lower()
        text = re.sub(r'\S+@\S+', ' ', text)
        text = re.sub(r'http\S+', ' ', text)
        text = re.sub(r'\d+', ' ', text)
        toks = simple_preprocess(text, deacc=True, min_len=3)
        toks = [t for t in toks if t not in stop_words]
        sp_doc = nlp(" ".join(toks))
        lemmas = [token.lemma_ for token in sp_doc if token.pos_ in {'NOUN','VERB','ADJ','ADV'}]
        tokenized.append(" ".join(lemmas))

    return tokenized

# ────────────────────────────────────────────────────────────────────────────────
def load_collection(name, path=None):
    if name == "wsj":
        return pickle.load(open('ProcessedWSJ/wsj_raw.pkl','rb'))
    elif name == "20ng":
        return pickle.load(open('Processed20NG/20ng_raw.pkl','rb'))
    elif name == "wiki":
        return pickle.load(open('ProcessedWIKI/wiki_raw.pkl','rb'))
    else:
        raise ValueError(f"Unknown collection '{name}'")

# ────────────────────────────────────────────────────────────────────────────────
def train_bertopic(docs, n_topics=None):
    vectorizer_model = CountVectorizer(stop_words='english', max_features=3000)
    model = BERTopic(nr_topics=n_topics, vectorizer_model=vectorizer_model, calculate_probabilities=True)
    topics, probs = model.fit_transform(docs)
    return model, topics, probs

# ────────────────────────────────────────────────────────────────────────────────
def build_topic_doc_matrix(probs):
    import numpy as np
    topic_doc_matrix = np.array(probs).T
    return topic_doc_matrix

# ────────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Train BERTopic on WSJ, 20NG, or Wiki")
    parser.add_argument("--dataset", choices=["wsj","20ng","wiki"], required=True)
    parser.add_argument("--path", type=str, default=None)
    parser.add_argument("--n_topics", type=int, default=None)
    parser.add_argument("--output_dir", type=str, default="Results/BERTopic")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    raw = load_collection(args.dataset, args.path)
    texts = preprocess_docs(raw, args.dataset)
    print(f"[1] Preprocessed {len(texts)} docs from '{args.dataset}'.")

    model, topics, probs = train_bertopic(texts, args.n_topics)
    print(f"[2] BERTopic training complete with {len(set(topics))} topics.")

    prefix = args.dataset
    model.save(os.path.join(args.output_dir, f"{prefix}_bertopic_model"))

    with open(os.path.join(args.output_dir, f"{prefix}_topics.pkl"), "wb") as f:
        pickle.dump(topics, f)

    with open(os.path.join(args.output_dir, f"{prefix}_topic_doc_matrix.pkl"), "wb") as f:
        td_mat = build_topic_doc_matrix(probs)
        pickle.dump(td_mat, f)
        print(f"[3] Topic–doc matrix saved; shape = {td_mat.shape}")

    print("Sample topic–doc matrix (first 5 topics × 5 docs):")
    print(td_mat[:5, :5])

if __name__ == "__main__":
    main()
