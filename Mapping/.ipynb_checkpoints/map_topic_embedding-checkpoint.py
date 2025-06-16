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
    Handles torch.Tensor or NumPy arrays.
    """
    arr = pickle.load(open(path, 'rb'))
    if torch.is_tensor(arr):
        return arr.cpu().numpy()
    return np.asarray(arr)


def get_default_paths(dataset, lang_model, topic_model, dim,
                      features_path=None, topics_path=None,
                      output_dir='Results'):
    """
    Determine default file paths for features, topics, and mapping output.
    """
    ds = dataset.lower()
    proc_dir = f"Processed{ds.upper()}"

    # Projected features path
    if features_path:
        feats_path = features_path
    else:
        feats_path = os.path.join(
            proc_dir,
            f"{ds}_{lang_model}_{dim}_projected_features.pkl"
        )

    # Topic-document matrix path
    if topics_path:
        tops_path = topics_path
    else:
        tops_path = os.path.join(
            output_dir,
            topic_model.upper(),
            f"{ds}_topic_doc_matrix.pkl"
        )

    # Mapping output path
    out_fname = f"{ds}_{lang_model}_{topic_model}_mapping.pkl"
    out_path = os.path.join(output_dir, out_fname)

    return feats_path, tops_path, out_path


def map_embeddings(dataset,
                   lang_model,
                   topic_model,
                   dim,
                   features=None,
                   topics=None,
                   features_path=None,
                   topics_path=None,
                   output_dir='Results',
                   save=False):
    """
    Compute mapping between embeddings and topic matrix.

    Parameters:
    - dataset (str): 'wsj', '20ng', or 'wiki'
    - lang_model (str): 'doc2vec', 'sbert', or 'repllama'
    - topic_model (str): 'lda' or 'bertopic'
    - dim (int): projection dimension
    - features (np.ndarray or torch.Tensor, optional): pre-loaded feature array
    - topics (np.ndarray or torch.Tensor, optional): pre-loaded topic matrix
    - features_path (str, optional): path to features pickle (if features not provided)
    - topics_path (str, optional): path to topic pickle (if topics not provided)
    - output_dir (str): where to save mapping pickle
    - save (bool): if True, write mapping to disk

    Returns:
    - mapping: result of feature_mapping()
    """
    # If arrays not provided, resolve file paths
    if features is None and topics is None:
        feats_path, tops_path, out_path = get_default_paths(
            dataset, lang_model, topic_model, dim,
            features_path, topics_path, output_dir
        )

    # Load or use provided features
    if features is None:
        if not os.path.exists(feats_path):
            raise FileNotFoundError(f"Projected features not found: {feats_path}")
        features_arr = load_array(feats_path)
    else:
        # Accept torch tensor or numpy array
        if torch.is_tensor(features):
            features_arr = features.cpu().numpy()
        else:
            features_arr = np.asarray(features)

    # Load or use provided topics
    if topics is None:
        if not os.path.exists(tops_path):
            raise FileNotFoundError(f"Topic-document matrix not found: {tops_path}")
        topics_arr = load_array(tops_path)
    else:
        if torch.is_tensor(topics):
            topics_arr = topics.cpu().numpy()
        else:
            topics_arr = np.asarray(topics)

    # Handle transposition if needed
    if features_arr.ndim == 2 and features_arr.shape[0] == topics_arr.shape[1]:
        features_arr = features_arr.T

    # Compute mapping
    mapping = feature_mapping(features_arr, topics_arr)

    # Optionally save mapping
    if save:
        os.makedirs(output_dir, exist_ok=True)
        with open(out_path, 'wb') as f:
            pickle.dump(mapping, f)

    return mapping


def main():
    parser = argparse.ArgumentParser(
        description="Map projected embeddings to topic-document matrices."
    )
    parser.add_argument('--dataset', choices=['wsj', '20ng', 'wiki'], required=True,
                        help="Dataset name (wsj, 20ng, wiki)")
    parser.add_argument('--lang_model', choices=['doc2vec', 'sbert', 'repllama'], required=True,
                        help="Embedding method used")
    parser.add_argument('--topic_model', choices=['lda', 'bertopic'], required=True,
                        help="Topic model used")
    parser.add_argument('--dim', type=int, required=True,
                        help="Number of projection features used")
    parser.add_argument('--features', type=str, default=None,
                        help="Override path to projected features pickle")
    parser.add_argument('--topics', type=str, default=None,
                        help="Override path to topic-document matrix pickle")
    parser.add_argument('--output_dir', type=str, default='Results',
                        help="Directory to save mapping result")
    parser.add_argument('--no-save', action='store_true',
                        help="Compute mapping without saving to disk")

    args = parser.parse_args()

    mapping = map_embeddings(
        dataset=args.dataset,
        lang_model=args.lang_model,
        topic_model=args.topic_model,
        dim=args.dim,
        features=None,
        topics=None,
        features_path=args.features,
        topics_path=args.topics,
        output_dir=args.output_dir,
        save=not args.no_save
    )

    if not args.no_save:
        ds = args.dataset.lower()
        out_fname = f"{ds}_{args.lang_model}_{args.topic_model}_mapping.pkl"
        out_path = os.path.join(args.output_dir, out_fname)
        print(f"[SAVED] Mapping saved to {out_path}")
    else:
        print("Mapping computed (not saved). Use the returned object as needed.")

if __name__ == '__main__':
    main()
