#!/usr/bin/env python
"""
bertopic_grid_search.py

Grid‐search over a range of `nr_topics` for BERTopic, computing specificity for each model.
Attempts to use GPU‐accelerated UMAP from RAPIDS (cuml). Falls back to CPU UMAP if not available.
Always reduces to 2D.
"""

import os
import sys
import pickle
import argparse

from bertopic import BERTopic

# Ensure project root is on path to import calculate_specificity_bertopic
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from TopicSpecificityBerTopic.topic_specificity_bertopic import calculate_specificity_bertopic

# ─── Try to import GPU‐accelerated UMAP from RAPIDS (cuml) ──────────────────────
try:
    from cuml.manifold import UMAP as GPU_UMAP

    def make_umap():
        # GPU UMAP, reduces to 2 dimensions
        return GPU_UMAP(
            n_components=2,
            n_neighbors=15,
            min_dist=0.1,
            metric='cosine',
            random_state=42
        )
    print("[INFO] Using cuml UMAP (GPU).")
except ImportError:
    from umap import UMAP as CPU_UMAP

    def make_umap():
        # CPU UMAP fallback, also 2 dimensions
        return CPU_UMAP(
            n_components=2,
            n_neighbors=15,
            min_dist=0.1,
            metric='cosine',
            random_state=42
        )
    print("[INFO] cuml not found; using CPU UMAP from umap-learn.")


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
    Given a list of token‐lists, return a list of whitespace‐joined strings (one per document).
    """
    return [" ".join(tokens) for tokens in token_lists]


def grid_search_bertopic(
    docs,         # list of document‐strings
    collection,
    start_topics=10,
    end_topics=200,
    step=10,
    output_dir='Results/BERTOPIC'
):
    """
    For each k in range(start_topics, end_topics+1, step):
      - Train BERTopic(nr_topics=k) with a 2D UMAP (GPU if available)
      - Compute specificity via calculate_specificity_bertopic
      - Save model to {output_dir}/{collection}_bertopic_{k}.model
    Returns: dict {k: specificity_scores_list}
    """
    os.makedirs(output_dir, exist_ok=True)
    results = {}

    # Create a single UMAP instance (GPU or CPU) with n_components=2
    umap_model = make_umap()

    for k in range(start_topics, end_topics + 1, step):
        print(f"\n[GRID] Training BERTopic (nr_topics={k}) on '{collection}' using 2D UMAP …")

        model = BERTopic(
            nr_topics=k,
            calculate_probabilities=True,
            umap_model=umap_model
        )
        topics, probs = model.fit_transform(docs)

        model_filename = f"{collection}_bertopic_{k}.model"
        model_path = os.path.join(output_dir, model_filename)
        model.save(model_path)
        print(f"[SAVED] BERTopic model at {model_path}")

        print(f"[GRID] Calculating specificity for k={k} …")
        specificity_scores = calculate_specificity_bertopic(
            model,
            threshold_mode='gmm',
            specificity_mode='diff'
        )
        results[k] = specificity_scores
        print(f"[GRID] Specificity scores @ k={k}: {specificity_scores}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Grid search BERTopic: GPU/CPU UMAP → 2D, range of nr_topics, specificity calculation"
    )
    parser.add_argument(
        '--dataset', choices=['wiki', '20ng', 'wsj'], required=True,
        help="Collection name"
    )
    parser.add_argument(
        '--start', type=int, default=10,
        help="Starting number of topics (default: 10)"
    )
    parser.add_argument(
        '--end', type=int, default=200,
        help="Ending number of topics (default: 200)"
    )
    parser.add_argument(
        '--step', type=int, default=10,
        help="Step size between topic counts (default: 10)"
    )
    parser.add_argument(
        '--output_pickle',
        help="Pickle file path to save specificity dictionary (default: Results/BERTOPIC/{collection}_bertopic_specificity.pkl)",
        default=None
    )
    args = parser.parse_args()

    collection = args.dataset

    token_lists = load_preprocessed(collection)
    docs = build_docs_from_tokens(token_lists)
    print(f"[1] Loaded {len(docs)} documents (from tokens) for '{collection}'.")

    specificity_dict = grid_search_bertopic(
        docs=docs,
        collection=collection,
        start_topics=args.start,
        end_topics=args.end,
        step=args.step,
        output_dir=f"Results/BERTOPIC"
    )

    default_pickle = f"Results/BERTOPIC/{collection}_bertopic_specificity.pkl"
    pickle_path = args.output_pickle or default_pickle
    os.makedirs(os.path.dirname(pickle_path), exist_ok=True)
    with open(pickle_path, 'wb') as pf:
        pickle.dump(specificity_dict, pf)
    print(f"\n[SAVED] Specificity dictionary saved to {pickle_path}")


if __name__ == '__main__':
    main()
