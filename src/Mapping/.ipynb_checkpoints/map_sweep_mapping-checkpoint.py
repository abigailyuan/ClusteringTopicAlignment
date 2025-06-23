#!/usr/bin/env python
"""
map_sweep_mapping.py

Map projected embeddings to topic-document matrices for a sweep of topic counts.
This script mirrors map_topic_embedding.py but adds a --num_topics parameter to output
mapping files with that suffix, e.g. wiki_sbert_lda_20_mapping.pkl.

Usage:
    python map_sweep_mapping.py \
      --dataset wiki --lang_model sbert --topic_model lda --dim 20 --num_topics 20 \
      --output_dir Results
"""
import os
import sys
import pickle
import argparse
import torch
import numpy as np

# Insert project root into path for local imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from Mapping.GreedyMapping import feature_mapping


def load_array(path):
    """
    Load a pickle file and return a NumPy array.
    Handles torch.Tensor or NumPy arrays.
    """
    arr = pickle.load(open(path, 'rb'))
    if torch.is_tensor(arr):
        return arr.cpu().numpy()
    return np.asarray(arr)


def main():
    parser = argparse.ArgumentParser(
        description="Map projected embeddings to topic-document matrices, with sweep support."
    )
    parser.add_argument('--dataset', choices=['wsj','20ng','wiki'], required=True)
    parser.add_argument('--lang_model', choices=['doc2vec','sbert','repllama'], required=True)
    parser.add_argument('--topic_model', choices=['lda','bertopic'], required=True)
    parser.add_argument('--dim', type=int, required=True,
                        help="Feature dimension for projected features.")
    parser.add_argument('--num_topics', type=int, required=True,
                        help="Number of topics, for naming the mapping output.")
    parser.add_argument('--features', type=str, default=None,
                        help="Override path to projected features pickle.")
    parser.add_argument('--topics', type=str, default=None,
                        help="Override path to topic-document matrix pickle.")
    parser.add_argument('--output_dir', type=str, default='Results',
                        help="Directory to save mapping result.")
    args = parser.parse_args()

    ds = args.dataset.lower()
    dim = args.dim
    k = args.num_topics

    proc_dir = f"Processed{ds.upper()}"

    # Determine features path
    feats_path = args.features or os.path.join(
        proc_dir, f"{ds}_{args.lang_model}_{dim}_projected_features.pkl"
    )

    # Determine topics matrix path
    tops_path = args.topics or os.path.join(
        'Results', args.topic_model.upper(),
        f"{ds}_topic_doc_matrix_{k}.pkl" if args.topic_model=='lda' else f"{ds}_topic_doc_matrix.pkl"
    )

    if not os.path.exists(feats_path):
        sys.exit(f"[ERROR] Projected features not found: {feats_path}")
    if not os.path.exists(tops_path):
        sys.exit(f"[ERROR] Topic–document matrix not found: {tops_path}")

    print(f"Loading projected features from {feats_path}")
    features = load_array(feats_path)
    print(f"Loading topic–document matrix from {tops_path}")
    topics = load_array(tops_path)

    # Align shapes
    if features.ndim==2 and features.shape[0]==topics.shape[1]:
        features = features.T
        print(f"Transposed features to {features.shape}")

    mapping = feature_mapping(features, topics)

    os.makedirs(args.output_dir, exist_ok=True)
    out_fname = f"{ds}_{args.lang_model}_{args.topic_model}_{k}_mapping.pkl"
    out_path = os.path.join(args.output_dir, out_fname)
    with open(out_path, 'wb') as f:
        pickle.dump(mapping, f)
    print(f"[SAVED] Sweep mapping saved to {out_path}")

if __name__=='__main__':
    main()
