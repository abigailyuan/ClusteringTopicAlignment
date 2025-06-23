#!/usr/bin/env python3
import os
import pickle
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

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
FEATURE_STEP  = 10         # step size when sweeping feature counts
MAX_FEATURES  = 101        # upper cap on features to test

# -----------------------------------------------------------------------------
# UTILITIES
# -----------------------------------------------------------------------------

def generate_pca(arr: np.ndarray, n: int, random_state: int = None) -> np.ndarray:
    """
    Reduce input embedding to n dimensions using PCA with given random_state.
    """
    docs, dims = arr.shape
    if not (1 <= n <= dims):
        raise ValueError(f"n must be between 1 and {dims}, got {n}")
    pca = PCA(n_components=n, random_state=random_state)
    # fit PCA on the entire embedding and transform
    reduced = pca.fit_transform(arr)
    return reduced


def sampling_pipeline(
    corpus: str,
    lang_model: str,
    topic_model: str,
    n_bootstraps: int = N_BOOTSTRAPS,
    max_features: int = MAX_FEATURES,
    step: int = FEATURE_STEP
) -> Dict[int, List[float]]:
    """
    Sweep over PCA-reduced feature counts for (corpus, lang_model, topic_model)
    and return bootstrap specificity improvements.
    """
    print(f"\n→ Starting PCA pipeline for {corpus.upper()} | {lang_model} | {topic_model.upper()}")

    # Load embeddings
    emb_path = f'Processed{corpus.upper()}/{corpus}_{lang_model}_corpus_embeddings.pkl'
    print(f"   • Loading embeddings from {emb_path}...")
    embedding = pickle.load(open(emb_path, 'rb'))
    if hasattr(embedding, 'cpu'):
        embedding = embedding.cpu().numpy()
    print(f"     → Embeddings loaded, shape = {embedding.shape}")

    # Load topic matrix
    n_topics = OPTIMAL_TOPICS[topic_model][corpus]
    tm_dir   = 'LDA' if topic_model == 'lda' else 'BERTOPIC'
    tm_path  = f'Results/{tm_dir}/{corpus}_topic_doc_matrix_{n_topics}.pkl'
    print(f"   • Loading topic matrix ({n_topics} topics) from {tm_path}...")
    topics   = pickle.load(open(tm_path, 'rb'))
    print(f"     → Topic matrix loaded")

    # Determine sweep bounds
    true_max = min(max_features, embedding.shape[1])
    print(f"   • Sweeping PCA components from {step} to {true_max} in steps of {step}")

    results = {}
    for n_features in range(step, true_max + 1, step):
        print(f"\n   → PCA reduction to {n_features} dims ({n_bootstraps} bootstraps)...")
        imps = []
        for i in range(n_bootstraps):
            reduced = generate_pca(embedding, n_features, random_state=i)
            mapping = map_embeddings(
                dataset=corpus,
                lang_model=lang_model,
                topic_model=topic_model,
                dim=n_topics,
                features=reduced,
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

    print(f"→ Finished PCA pipeline for {corpus.upper()} | {lang_model} | {topic_model.upper()}")
    return results


def plot_specificity_improvement(
    corpus: str,
    lang_model: str,
    topic_model: str,
    results: Dict[int, List[float]],
    title: str
) -> None:
    """
    Plot number of PCA components vs. average mean specificity improvement.
    """
    print(f"   • Plotting PCA results for {corpus.upper()} | {lang_model} | {topic_model.upper()}...")
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
    plt.xlabel("Number of PCA Components")
    plt.ylabel("Average Mean Specificity Improvement")
    plt.grid(True)
    plt.tight_layout()

    out_path = f"Results/bootstrap_plots/pca_{corpus}_{lang_model}_{topic_model}.pdf"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"     → Saved PCA plot to {out_path}")

# -----------------------------------------------------------------------------
# ENTRY POINT
# -----------------------------------------------------------------------------

def main():
    print("=== BEGIN PCA-BASED BOOTSTRAP SAMPLING & PLOTTING ===")
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
                    title=f"{corpus.upper()} • {lang_model} • {topic_model} (PCA)"
                )
    print("\n=== ALL PCA JOBS DONE ===")

if __name__ == "__main__":
    main()
