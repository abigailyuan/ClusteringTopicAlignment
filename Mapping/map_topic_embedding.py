import os
import sys
import argparse
import pickle
import torch
import numpy as np

# Insert project root into path for local imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from mappings.GreedyMapping import feature_mapping


def load_array(path):
    """
    Load a pickle file and return a NumPy array.
    Handles torch.Tensor or numpy arrays.
    """
    arr = pickle.load(open(path, 'rb'))
    if torch.is_tensor(arr):
        return arr.cpu().numpy()
    return np.asarray(arr)


def main():
    parser = argparse.ArgumentParser(
        description="Map projected embeddings to topic-document matrices."
    )
    parser.add_argument(
        '--dataset', choices=['wsj', '20ng', 'wiki'], required=True,
        help="Dataset name."
    )
    parser.add_argument(
        '--lang_model', choices=['doc2vec', 'sbert', 'repllama'], required=True,
        help="Embedding method used."
    )
    parser.add_argument(
        '--topic_model', choices=['lda', 'bertopic'], required=True,
        help="Topic model used."
    )
    parser.add_argument(
        '--features', type=str, default=None,
        help="Path to projected features pickle (overrides default)."
    )
    parser.add_argument(
        '--topics', type=str, default=None,
        help="Path to topic-document matrix pickle (overrides default)."
    )
    parser.add_argument(
        '--output_dir', type=str, default='Results',
        help="Directory to save the mapping result."
    )
    args = parser.parse_args()

    ds = args.dataset.lower()
    # Processed directory contains embeddings
    proc_dir = f"Processed{ds.upper()}"

    # Determine features path
    if args.features:
        feats_path = args.features
    else:
        feats_path = os.path.join(proc_dir, f"{ds}_{args.lang_model}_projected_features.pkl")

    # Determine topics matrix path
    if args.topics:
        tops_path = args.topics
    else:
        tops_path = os.path.join('Results', args.topic_model.upper(), f"{ds}_topic_doc_matrix.pkl")

    # Validate paths
    if not os.path.exists(feats_path):
        sys.exit(f"[ERROR] Projected features not found: {feats_path}")
    if not os.path.exists(tops_path):
        sys.exit(f"[ERROR] Topic-document matrix not found: {tops_path}")

    # Load data
    print(f"Loading projected features from {feats_path}")
    features = load_array(feats_path)
    print(f"Loading topic–document matrix from {tops_path}")
    topics = load_array(tops_path)

    print(f"projected_features shape: {features.shape}")
    print(f"topics matrix shape:       {topics.shape}")
    
    # Align feature and topic dimensions
    # Features should be (n_features, n_docs), Topics should be (n_topics, n_docs)
    if features.ndim == 2 and features.shape[0] == topics.shape[1]:
        features = features.T
        print(f"Transposed features to {features.shape}")

    # Compute mapping
    mapping = feature_mapping(features, topics)

    # Save mapping
    os.makedirs(args.output_dir, exist_ok=True)
    out_fname = f"{ds}_{args.lang_model}_{args.topic_model}_mapping.pkl"
    out_path = os.path.join(args.output_dir, out_fname)
    with open(out_path, 'wb') as f:
        pickle.dump(mapping, f)

    print(f"[SAVED] Mapping saved to {out_path}")


if __name__ == '__main__':
    main()