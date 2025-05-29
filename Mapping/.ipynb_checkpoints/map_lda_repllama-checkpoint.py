#!/usr/bin/env python
# map_lda_repllama_general.py

import os
import argparse
import pickle
import torch
import numpy as np
from mappings.GreedyMapping import feature_mapping


def load_array(path):
    """
    Load an array-like object from pickle, handling torch.Tensor or numpy.
    """
    arr = pickle.load(open(path, 'rb'))
    if torch.is_tensor(arr):
        return arr.cpu().numpy()
    return np.asarray(arr)


def main():
    parser = argparse.ArgumentParser(
        description="Map projected features to LDA topic distributions for WSJ, 20NG, or Wiki"
    )
    parser.add_argument(
        '--dataset', choices=['wsj', '20ng', 'wiki'], required=True,
        help="Which dataset mapping to perform"
    )
    parser.add_argument(
        '--features', type=str, default=None,
        help="Path to projected_features pickle (default based on dataset)"
    )
    parser.add_argument(
        '--topics', type=str, default=None,
        help="Path to topic–doc matrix pickle (default based on dataset)"
    )
    parser.add_argument(
        '--output_dir', type=str, default='Results',
        help="Directory to save the mapping result"
    )
    args = parser.parse_args()

    # Set default paths if none provided
    ds = args.dataset.lower()
    if args.features:
        feats_path = args.features
    else:
        feats_path = os.path.join('ProcessedWSJ' if ds == 'wsj' else f'Processed{ds.upper()}',
                                  f'{ds}_projected_features.pkl')

    if args.topics:
        tops_path = args.topics
    else:
        tops_path = os.path.join('Results', 'LDA', f'{ds}_topic_doc_matrix.pkl')

    # Load arrays
    print(f"Loading projected features from {feats_path}")
    projected_np = load_array(feats_path)
    print(f"Loading topic–document matrix from {tops_path}")
    topics_np = load_array(tops_path)

    print(f"projected_features shape: {projected_np.shape}")
    print(f"topics matrix shape: {topics_np.shape}")

    # Compute mapping
    mapping = feature_mapping(projected_np, topics_np)

    # Prepare output
    os.makedirs(args.output_dir, exist_ok=True)
    out_fname = f"{ds}_repllama_lda_mapping.pkl"
    out_path = os.path.join(args.output_dir, out_fname)
    pickle.dump(mapping, open(out_path, 'wb'))

    print(f"Mapping saved to {out_path}")
    print(type(mapping))
    print(mapping)

if __name__ == '__main__':
    main()
