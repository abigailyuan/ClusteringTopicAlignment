#!/usr/bin/env python
"""
mms.py

Library and CLI to compute the average improvement of specificity scores
for mapped topic features, supporting both BERTopic and LDA models.

This module lives under the `Metrics` package as `Metrics/mms.py`.

Provides:
  - average_mapped_improvement(): importable function
  - CLI entrypoint for script usage
"""
import os
import pickle
import numpy as np
import sys
import argparse
from bertopic import BERTopic
from gensim.models import LdaModel


def import_bertopic_specificity():
    try:
        from TopicSpecificityBerTopic.topic_specificity_bertopic import calculate_specificity_bertopic
        return calculate_specificity_bertopic
    except ImportError:
        sys.exit("[ERROR] Cannot import calculate_specificity_bertopic")


def import_lda_specificity():
    try:
        from topic_specificity import calculate_specificity_for_all_topics
        return calculate_specificity_for_all_topics
    except ImportError:
        sys.exit("[ERROR] Cannot import calculate_specificity_for_all_topics for LDA specificity")


def load_mapping(mapping_path: str) -> list:
    """Load a mapping object from a pickle file (list of (feature_id, topic_id))."""
    with open(mapping_path, 'rb') as f:
        mapping = pickle.load(f)
    if not isinstance(mapping, (list, tuple)):
        sys.exit(f"[ERROR] Mapping must be a list or tuple, got {type(mapping)}")
    return mapping


def calculate_bertopic_scores(
    model_path: str,
    threshold_mode: str,
    specificity_mode: str,
    n_topics: int
) -> dict:
    """Compute specificity scores for a BERTopic model at the given path, with caching."""
    if not os.path.exists(model_path):
        sys.exit(f"[ERROR] BERTopic model not found: {model_path}")

    # Cache lookup
    cache_dir  = os.path.join('Results', 'BERTOPIC')
    cache_file = f"bertopic_{n_topics}_specificity_scores.pkl"
    cache_path = os.path.join(cache_dir, cache_file)
    if os.path.exists(cache_path):
        with open(cache_path, 'rb') as f:
            return pickle.load(f)

    # Compute scores
    model   = BERTopic.load(model_path)
    calc_fn = import_bertopic_specificity()
    scores  = calc_fn(model,
                      threshold_mode=threshold_mode,
                      specificity_mode=specificity_mode)
    if isinstance(scores, dict):
        out = scores
    else:
        arr = np.array(scores, dtype=float)
        out = {i: float(arr[i]) for i in range(arr.shape[0])}

    # Save to cache
    os.makedirs(cache_dir, exist_ok=True)
    with open(cache_path, 'wb') as f:
        pickle.dump(out, f)

    return out


def calculate_lda_scores(
    model_path: str,
    corpus_path: str,
    threshold_mode: str,
    specificity_mode: str,
    n_topics: int
) -> dict:
    """Compute specificity scores for an LDA model and corpus at the given paths, with caching."""
    if not os.path.exists(model_path):
        sys.exit(f"[ERROR] LDA model not found: {model_path}")
    if not os.path.exists(corpus_path):
        sys.exit(f"[ERROR] Corpus not found: {corpus_path}")

    # Cache lookup
    cache_dir  = os.path.join('Results', 'LDA')
    cache_file = f"lda_{n_topics}_specificity_scores.pkl"
    cache_path = os.path.join(cache_dir, cache_file)
    if os.path.exists(cache_path):
        with open(cache_path, 'rb') as f:
            return pickle.load(f)

    # Compute scores
    lda     = LdaModel.load(model_path)
    corpus  = pickle.load(open(corpus_path, 'rb'))
    calc_fn = import_lda_specificity()
    scores  = calc_fn(model=lda,
                      corpus=corpus,
                      mode='lda',
                      threshold_mode=threshold_mode,
                      specificity_mode=specificity_mode)
    if isinstance(scores, dict):
        out = scores
    else:
        arr = np.array(scores, dtype=float)
        out = {i: float(arr[i]) for i in range(arr.shape[0])}

    # Save to cache
    os.makedirs(cache_dir, exist_ok=True)
    with open(cache_path, 'wb') as f:
        pickle.dump(out, f)

    return out


def _compute_improvements_from_specs(specs: dict, mapping: list) -> tuple:
    """
    Given per-topic specificity scores and a mapping, compute:
      - per-cluster relative improvement over overall average
      - average of those relative improvements
    """
    overall_avg = float(np.mean(list(specs.values())))
    clusters    = {}
    for feat_id, cluster_id in mapping:
        clusters.setdefault(cluster_id, []).append(feat_id)

    improvements = {}
    for cluster_id, feats in clusters.items():
        valid = [fid for fid in feats if fid in specs]
        if not valid:
            continue
        cluster_avg = float(np.mean([specs[fid] for fid in valid]))
        improvements[cluster_id] = (cluster_avg - overall_avg) / overall_avg

    if not improvements:
        sys.exit("[ERROR] No valid mapped clusters to compute improvements.")
    avg_improvement = float(np.mean(list(improvements.values())))
    return avg_improvement, improvements


def average_mapped_improvement(
    topic_model: str,
    dataset: str,
    n_topics: int,
    mapping: list,
    threshold_mode: str = 'gmm',
    specificity_mode: str = 'diff'
) -> float:
    """
    Parameters
    ----------
    topic_model : 'bertopic' or 'lda'
    dataset     : dataset name (e.g., 'wiki', '20ng', 'wsj')
    n_topics    : number of topics used in the model
    mapping     : list of (feature_id, topic_id) tuples
    threshold_mode, specificity_mode : passed to specificity calculations

    Returns
    -------
    avg_improvement : float
        average relative improvement over all clusters
    """
    base_dir = os.path.join('Results', topic_model.upper())
    if topic_model == 'bertopic':
        model_path = os.path.join(base_dir, f"{dataset}_bertopic_{n_topics}.model")
        specs      = calculate_bertopic_scores(
            model_path,
            threshold_mode,
            specificity_mode,
            n_topics
        )
    elif topic_model == 'lda':
        model_path  = os.path.join(base_dir, f"{dataset}_lda{n_topics}.model")
        corpus_path = os.path.join(base_dir, f"{dataset}_corpus.pkl")
        specs       = calculate_lda_scores(
            model_path,
            corpus_path,
            threshold_mode,
            specificity_mode,
            n_topics
        )
    else:
        raise ValueError("topic_model must be 'bertopic' or 'lda'")

    avg_imp, _ = _compute_improvements_from_specs(specs, mapping)
    return avg_imp


# CLI entrypoint
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Compute avg specificity improvement for mapped topics."
    )
    parser.add_argument(
        '--topic_model', required=True, choices=['bertopic', 'lda'],
        help="Which topic model to evaluate"
    )
    parser.add_argument(
        '--dataset', required=True,
        help="Dataset name (e.g., 'wiki', '20ng', 'wsj')"
    )
    parser.add_argument(
        '--n_topics', type=int, required=True,
        help="Number of topics used in the model"
    )
    parser.add_argument(
        '--mapping', required=True,
        help="Path to pickle of mapping object (list of (feature_id, topic_id))."
    )
    parser.add_argument(
        '--threshold_mode', default='gmm',
        help="Threshold mode for specificity (default: gmm)"
    )
    parser.add_argument(
        '--specificity_mode', default='diff',
        help="Specificity mode for calculation (default: diff)"
    )
    parser.add_argument(
        '--output', default=None,
        help="Optional path to save result (pickle with avg_improvement)"
    )
    args = parser.parse_args()

    mapping = load_mapping(args.mapping)
    avg_imp = average_mapped_improvement(
        args.topic_model,
        args.dataset,
        args.n_topics,
        mapping,
        args.threshold_mode,
        args.specificity_mode
    )
    print(f"Average improvement: {avg_imp:.6f}")

    if args.output:
        with open(args.output, 'wb') as f:
            pickle.dump({'avg_improvement': avg_imp}, f)
        print(f"Saved average improvement to {args.output}")
