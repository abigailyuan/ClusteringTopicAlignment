#!/usr/bin/env python
"""
bertopic_train.py

Train a BERTopic model on a preprocessed corpus, using a specified number of topics.
This mirrors the interface of lda_train.py, taking --dataset and --num_topics.

Usage (example):
    python bertopic_train.py --dataset wiki --num_topics 80 --output_dir Results/BERTOPIC
"""

import os
import sys
import pickle
import argparse

# Ensure project root is on path to import calculate_specificity_bertopic
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from bertopic import BERTopic

# Lazy-import for BERTopic specificity:
def import_bertopic_specificity():
    try:
        from TopicSpecificityBerTopic.topic_specificity_bertopic import calculate_specificity_bertopic
        return calculate_specificity_bertopic
    except ImportError:
        sys.exit("[ERROR] Cannot import calculate_specificity_bertopic from TopicSpecificityBerTopic/")


def load_preprocessed(dataset: str):
    """
    Load a pickled list of token lists from Processed{DATASET}/{dataset}_preprocessed.pkl.
    Returns: list of token lists.
    """
    processed_dir = f"Processed{dataset.upper()}"
    pre_path = os.path.join(processed_dir, f"{dataset}_preprocessed.pkl")
    if not os.path.exists(pre_path):
        sys.exit(f"[ERROR] Preprocessed file not found: {pre_path}")
    with open(pre_path, 'rb') as f:
        tokens = pickle.load(f)
    if not isinstance(tokens, list) or not all(isinstance(doc, list) for doc in tokens):
        sys.exit(f"[ERROR] Expected a list of token lists in {pre_path}")
    return tokens


def build_docs_from_tokens(token_lists):
    """
    Given a list of token-lists, return a list of whitespace-joined strings (one per document).
    """
    return [" ".join(tokens) for tokens in token_lists]


def main():
    parser = argparse.ArgumentParser(
        description="Train BERTopic on a preprocessed corpus."
    )
    parser.add_argument(
        '--dataset',
        required=True,
        choices=['wsj', '20ng', 'wiki'],
        help="Collection name (wsj, 20ng, or wiki)."
    )
    parser.add_argument(
        '--num_topics',
        type=int,
        required=True,
        help="Number of topics (nr_topics) to request from BERTopic."
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default="Results/BERTOPIC",
        help="Directory to save the BERTopic model and matrices (default: Results/BERTOPIC)."
    )
    parser.add_argument(
        '--calculate_specificity',
        action='store_true',
        help="If set, compute specificity scores after training and save them."
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    prefix = args.dataset
    k = args.num_topics

    # ① Load tokens, build raw-text docs
    token_lists = load_preprocessed(prefix)
    docs = build_docs_from_tokens(token_lists)
    print(f"[1] Loaded {len(docs)} documents (reconstructed from tokens) for '{prefix}'.")

    # ② Train BERTopic with nr_topics = k
    print(f"[2] Training BERTopic on '{prefix}' with nr_topics={k} ...")
    model = BERTopic(nr_topics=k, calculate_probabilities=True)
    topics, probs = model.fit_transform(docs)
    print(f"[2] BERTopic training complete. Model has {len(model.get_topic_freq())} total topics.")

    # ③ Save the model
    model_path = os.path.join(args.output_dir, f"{prefix}_bertopic_{k}.model")
    model.save(model_path)
    print(f"[SAVED] BERTopic model at {model_path}")

    # ④ Build and save topic–document matrix (num_topics × num_docs)
    import numpy as np
    num_docs = len(docs)
    if probs is None:
        _, probs = model.transform(docs)

    if isinstance(probs, np.ndarray):
        # probs shape: (n_docs, n_topics)
        td_mat = probs.T
        n_topics_actual = td_mat.shape[0]
    else:
        # list-of-lists of (topic, prob)
        valid_topics = sorted([t for t in set(topics) if t >= 0])
        n_topics_actual = len(valid_topics)
        td_mat = np.zeros((n_topics_actual, num_docs), dtype=float)
        for doc_idx, doc_topics in enumerate(probs):
            for t_id, p in doc_topics:
                if t_id >= 0:
                    row = valid_topics.index(t_id)
                    td_mat[row, doc_idx] = p

    td_path = os.path.join(args.output_dir, f"{prefix}_topic_doc_matrix.pkl")
    with open(td_path, "wb") as f:
        pickle.dump(td_mat, f)
    print(f"[SAVED] BERTopic topic–doc matrix at {td_path}, shape = {td_mat.shape}")

    # ⑤ Optionally compute & save specificity scores
    if args.calculate_specificity:
        calculate_specificity_bertopic = import_bertopic_specificity()
        print(f"[3] Calculating specificity scores for k={k} ...")
        specificity_scores = calculate_specificity_bertopic(
            model,
            threshold_mode='gmm',
            specificity_mode='diff'
        )
        spec_path = os.path.join(args.output_dir, f"{prefix}_bertopic_{k}_specificity.pkl")
        with open(spec_path, "wb") as f:
            pickle.dump(specificity_scores, f)
        print(f"[SAVED] Specificity scores at {spec_path}")

if __name__ == "__main__":
    main()
