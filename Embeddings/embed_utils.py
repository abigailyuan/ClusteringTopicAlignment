import os
import pickle
import numpy as np
from typing import List, Any
from sklearn.random_projection import GaussianRandomProjection

OPTIMAL_LDA_TOPICS = {
    'wsj':50,
    'wiki':80,
    '20ng':70
    }

def get_processed_dir(collection: str) -> str:
    """Return the processed directory for a given collection."""
    return f"Processed{collection.upper()}"


def get_raw_path(collection: str) -> str:
    """Return the path to the raw documents pickle file."""
    return os.path.join(get_processed_dir(collection), f"{collection}_raw.pkl")


def ensure_dir(path: str):
    """Create directory if it doesn't exist."""
    os.makedirs(path, exist_ok=True)


def load_documents(collection: str) -> List[str]:
    """
    Load and return the list of documents for the given collection.
    Raises FileNotFoundError if the raw file is missing.
    """
    raw_path = get_raw_path(collection)
    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"Raw documents not found: {raw_path}")
    with open(raw_path, 'rb') as f:
        docs = pickle.load(f)
    return docs


def save_pickle(obj: Any, path: str):
    """Save any object to a pickle file at `path`."""
    ensure_dir(os.path.dirname(path))
    with open(path, 'wb') as f:
        pickle.dump(obj, f)


def save_embeddings(embeddings: np.ndarray, collection: str, filename: str):
    """
    Save embeddings array to the processed directory with the given filename.
    """
    processed_dir = get_processed_dir(collection)
    ensure_dir(processed_dir)
    out_path = os.path.join(processed_dir, filename)
    save_pickle(embeddings, out_path)


def random_project(embeddings: np.ndarray, n_features: int = 50, seed: int = 42) -> np.ndarray:
    """
    Apply Gaussian random projection to reduce `embeddings` to `n_features` dimensions.
    """
    projector = GaussianRandomProjection(n_components=n_features, random_state=seed)
    return projector.fit_transform(embeddings)
