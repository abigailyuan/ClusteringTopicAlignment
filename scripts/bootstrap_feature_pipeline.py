#!/usr/bin/env python3
import os
import pickle
import random
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np

from Mapping.map_topic_embedding import map_embeddings
from Metrics.mms import average_mapped_improvement
from Embeddings.embed_utils import random_project

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
FEATURE_STEP  = 10         # step size when sweeping feature counts
MAX_FEATURES  = 101       # upper cap on features to test

# -----------------------------------------------------------------------------
# UTILITIES
# -----------------------------------------------------------------------------

def generate_sample(arr: np.ndarray, n: int, random_state: int = None) -> np.ndarray:
    """
    Generate a random projection of input embedding with n dimensions using random seed provided.
    """
    docs, dims = arr.shape
    if not (1 <= n <= dims):
        raise ValueError(f"n must be between 1 and {dims}, got {n}")
    return random_project(arr, n, random_state)

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
    print(f"\n→ Starting sampling pipeline for {corpus.upper()} | {lang_model} | {topic_model.upper()}")

    # ——— Load embeddings ———
    emb_path = f'Processed{corpus.upper()}/{corpus}_{lang_model}_corpus_embeddings.pkl'
    print(f"   • Loading embeddings from {emb_path}...")
    embedding = pickle.load(open(emb_path, 'rb'))
    if hasattr(embedding, 'cpu'):
        embedding = embedding.cpu().numpy()
    print(f"     → Embeddings loaded, shape = {embedding.shape}")

    # ——— Load topic matrix ———
    n_topics = OPTIMAL_TOPICS[topic_model][corpus]
    tm_dir   = 'LDA' if topic_model == 'lda' else 'BERTOPIC'
    tm_path  = f'Results/{tm_dir}/{corpus}_topic_doc_matrix_{n_topics}.pkl'
    print(f"   • Loading topic-document matrix ({n_topics} topics) from {tm_path}...")
    topics   = pickle.load(open(tm_path, 'rb'))
    print(f"     → Topic matrix loaded, shape = {getattr(topics, 'shape', 'unknown')}")

    # ——— Determine sweep upper bound ———
    true_max = min(max_features, embedding.shape[1])
    print(f"   • Sweeping features from {step} to {true_max} in steps of {step}")

    results = {}
    for n_features in range(step, true_max + 1, step):
        print(f"\n   → Sampling with {n_features} features ({n_bootstraps} bootstraps)...")
        imps = []
        for i in range(n_bootstraps):
            # you can uncomment below for per-iteration logs, but it will be very chatty:
            # print(f"      · Bootstrap {i+1}/{n_bootstraps}")
            random_state = random.randint(0, 2**32 - 1)
            sample = generate_sample(embedding, n_features, random_state=random_state)
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
        mean_imp = np.mean(imps)
        results[n_features] = imps
        print(f"     → Completed {n_bootstraps} bootstraps: mean specificity improvement = {mean_imp:.4f}")

    print(f"→ Finished sampling pipeline for {corpus.upper()} | {lang_model} | {topic_model.upper()}")
    return results

def plot_specificity_improvement(
    corpus: str,
    lang_model: str,
    topic_model: str,
    results: Dict[int, List[float]],
    title: str
) -> None:
    """
    Plot number of features vs. average mean specificity improvement,
    with error bars showing one standard deviation.
    """
    print(f"   • Plotting results for {corpus.upper()} | {lang_model} | {topic_model.upper()}...")
    x = sorted(results.keys())
    means = [np.mean(results[n]) for n in x]
    stds  = [np.std(results[n], ddof=1) for n in x]

    plt.figure(figsize=(10, 6))
    plt.errorbar(
        x, means, yerr=stds,
        marker='o', linestyle='-',
        capsize=5, elinewidth=1, markeredgewidth=1
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
    print(f"     → Saved plot to {out_path}")

# -----------------------------------------------------------------------------
# ENTRY POINT
# -----------------------------------------------------------------------------

def main():
    print("=== BEGIN BOOTSTRAP SAMPLING & PLOTTING ===")
    for corpus in COLLECTIONS:
        for lang_model in LANG_MODELS:
            for topic_model in TOPIC_MODELS:
                print(f"\n### Processing: {corpus.upper()} → {lang_model} → {topic_model.upper()} ###")
                results = sampling_pipeline(corpus, lang_model, topic_model)
                plot_specificity_improvement(
                    corpus,
                    lang_model,
                    topic_model,
                    results,
                    title=f"{corpus.upper()} • {lang_model} • {topic_model}"
                )
    print("\n=== ALL DONE ===")

if __name__ == "__main__":
    main()
