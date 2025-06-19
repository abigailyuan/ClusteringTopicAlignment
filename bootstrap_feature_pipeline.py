#!/usr/bin/env python3
import os
import pickle
import random
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np

from Mapping.map_topic_embedding import map_embeddings
from Metrics.mms import average_mapped_improvement

# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------

COLLECTIONS    = ['wiki', '20ng', 'wsj']
LANG_MODELS    = ['sbert', 'doc2vec', 'repllama']
TOPIC_MODELS   = ['lda', 'bertopic']

# Optimal topic counts per collection for each model
OPTIMAL_TOPICS = {
    'lda':     {'wsj': 50, 'wiki': 80, '20ng': 70},
    'bertopic':{'wsj': 50, 'wiki': 80, '20ng': 40}
}

N_BOOTSTRAPS  = 100        # bootstrap samples per feature-set
FEATURE_STEP  = 10        # step size when sweeping feature counts
MAX_FEATURES  = 400   # upper cap on features to test

# -----------------------------------------------------------------------------
# UTILITIES
# -----------------------------------------------------------------------------

def sample_columns(arr: np.ndarray, n: int, random_state: int = None) -> np.ndarray:
    """
    Randomly sample `n` columns from a 2D array `arr`.
    """
    docs, dims = arr.shape
    if not (1 <= n <= dims):
        raise ValueError(f"n must be between 1 and {dims}, got {n}")
    rng = np.random.default_rng(random_state)
    cols = rng.choice(dims, size=n, replace=False)
    return arr[:, cols]


def sampling_pipeline(
    corpus: str,
    lang_model: str,
    topic_model: str,
    n_bootstraps: int = N_BOOTSTRAPS,
    max_features: int = MAX_FEATURES,
    step: int = FEATURE_STEP
) -> Dict[int, List[float]]:
    """
    Sweep over feature counts for (corpus, lang_model, topic_model)
    and return bootstrap specificity improvements.
    """
    # ——— Load embeddings ———
    emb_path = f'Processed{corpus.upper()}/{corpus}_{lang_model}_corpus_embeddings.pkl'
    embedding = pickle.load(open(emb_path, 'rb'))
    if hasattr(embedding, 'cpu'):
        embedding = embedding.cpu().numpy()

    # ——— Load topic matrix ———
    n_topics = OPTIMAL_TOPICS[topic_model][corpus]
    tm_dir   = 'LDA' if topic_model == 'lda' else 'BERTOPIC'
    tm_path  = f'Results/{tm_dir}/{corpus}_topic_doc_matrix_{n_topics}.pkl'
    topics   = pickle.load(open(tm_path, 'rb'))

    # ——— Determine sweep upper bound ———
    true_max = min(max_features, embedding.shape[1])

    results = {}
    for n_features in range(step, true_max + 1, step):
        imps = []
        for i in range(n_bootstraps):
            sample = sample_columns(embedding, n_features, random_state=i)
            mapping = map_embeddings(
                dataset=corpus,
                lang_model=lang_model,
                topic_model=topic_model,
                dim=n_topics,
                features=sample,
                topics=topics,
                output_dir='Results',
                save=False
            )
            avg_imp = average_mapped_improvement(
                topic_model=topic_model,
                dataset=corpus,
                n_topics=n_topics,
                mapping=mapping,
                threshold_mode='gmm',
                specificity_mode='diff'
            )
            imps.append(avg_imp)
        results[n_features] = imps
        print(f"[{corpus}/{lang_model}/{topic_model}] "
              f"features={n_features} → mean imp={np.mean(imps):.4f}")
    return results


def plot_specificity_improvement(corpus: str, lang_model: str, topic_model: str, results: Dict[int, List[float]], title: str) -> None:
    """
    Plot number of features vs. average mean specificity improvement,
    with error bars showing one standard deviation.
    """
    x = sorted(results.keys())
    means = [np.mean(results[n]) for n in x]
    stds  = [np.std(results[n], ddof=1) for n in x]  # sample standard deviation

    plt.figure(figsize=(10, 6))
    plt.errorbar(
        x, means, yerr=stds,
        marker='o', linestyle='-',
        capsize=5,              # small horizontal cap on the error bars
        elinewidth=1,           # line width of error bars
        markeredgewidth=1
    )
    plt.title(title)
    plt.xlabel("Number of Features")
    plt.ylabel("Average Mean Specificity Improvement")
    plt.grid(True)
    plt.tight_layout()
    
    out_path = f"Results/bootstrap_plots/{corpus}_{lang_model}_{topic_model}.pdf"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=300)
    plt.close()


# -----------------------------------------------------------------------------
# ENTRY POINT
# -----------------------------------------------------------------------------

def main():
    for corpus in COLLECTIONS:
        for lang_model in LANG_MODELS:
            for topic_model in TOPIC_MODELS:
                print(f"\n=== {corpus.upper()} | {lang_model} | {topic_model.upper()} ===")
                results = sampling_pipeline(corpus, lang_model, topic_model)
                plot_specificity_improvement(
                    corpus,
                    lang_model,
                    topic_model,
                    results,
                    title=f"{corpus.upper()} • {lang_model} • {topic_model}"
                )


if __name__ == "__main__":
    main()
