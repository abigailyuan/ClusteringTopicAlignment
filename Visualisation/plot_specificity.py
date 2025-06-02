#!/usr/bin/env python
import pickle
import numpy as np
import matplotlib.pyplot as plt
import argparse


def plot_specificity(results_dict_path):
    # Load results dictionary
    with open(results_dict_path, 'rb') as f:
        results = pickle.load(f)

    n_topics = sorted(results.keys())
    means = [np.mean(results[n]) for n in n_topics]
    mins = [np.min(results[n]) for n in n_topics]
    maxs = [np.max(results[n]) for n in n_topics]

    lower_error = [m - mn for m, mn in zip(means, mins)]
    upper_error = [mx - m for mx, m in zip(maxs, means)]

    plt.figure()
    plt.errorbar(
        n_topics,
        means,
        yerr=[lower_error, upper_error],
        fmt='o-',
        capsize=5,
        label='All Topics'
    )
    plt.xlabel("Number of Topics")
    plt.ylabel("Mean Specificity Score")
    # plt.title("Specificity vs. Number of Topics")
    plt.grid(True)
    plt.legend()


def plot_top10_specificity(results_dict_path):
    # Load results dictionary
    with open(results_dict_path, 'rb') as f:
        results = pickle.load(f)

    n_topics = sorted(results.keys())
    means = []
    mins = []
    maxs = []

    for n in n_topics:
        scores = results[n]
        top_scores = sorted(scores, reverse=True)[:10]
        means.append(np.mean(top_scores))
        mins.append(min(top_scores))
        maxs.append(max(top_scores))

    lower_error = [m - mn for m, mn in zip(means, mins)]
    upper_error = [mx - m for mx, m in zip(maxs, means)]

    plt.figure()
    plt.errorbar(
        n_topics,
        means,
        yerr=[lower_error, upper_error],
        fmt='s--',
        capsize=5,
        label='Top-10 Topics'
    )
    plt.xlabel("Number of Topics")
    plt.ylabel("Mean Specificity of Top-10 Topics")
    plt.title("Top-10 Topic Specificity vs. Number of Topics")
    plt.grid(True)
    plt.legend()


def main():
    parser = argparse.ArgumentParser(
        description="Plot specificity metrics vs. number of topics."
    )
    parser.add_argument(
        "--results", required=True,
        help="Path to pickle file containing specificity dict {n_topics: [scores]}"
    )
    parser.add_argument(
        "--mode", choices=['all','top10','both'], default='all',
        help="Which plot to generate: 'all' for all topics, 'top10' for top-10, or 'both'"
    )
    args = parser.parse_args()

    if args.mode in ['all', 'both']:
        plot_specificity(args.results)
    if args.mode in ['top10', 'both']:
        plot_top10_specificity(args.results)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()