#!/usr/bin/env python
import os
import argparse
from sentence_transformers import SentenceTransformer

from embed_utils import (
    load_documents,
    save_embeddings,
    random_project,
    get_processed_dir,
    OPTIMAL_LDA_TOPICS
)
from preprocess import prepare_sbert


def embed_sbert(collection: str, dim: int = None, model_name: str = 'all-MiniLM-L6-v2'):
    method = 'sbert'
    # If --dim provided, use it; otherwise use OPTIMAL_LDA_TOPICS[collection]
    n_features = dim if (dim is not None) else OPTIMAL_LDA_TOPICS[collection]

    processed_dir = get_processed_dir(collection)
    raw_path = os.path.join(processed_dir, f"{collection}_{method}_corpus_embeddings.pkl")
    proj_path = os.path.join(processed_dir, f"{collection}_{method}_{n_features}_projected_features.pkl")

    # Skip if both raw and projected exist
    if os.path.exists(raw_path) and os.path.exists(proj_path):
        print(f"[SKIP] {collection} embeddings with {method} at dim={n_features} exist.")
        return

    raw_docs = load_documents(collection)
    docs = [prepare_sbert(d) for d in raw_docs]

    model = SentenceTransformer(model_name)
    embeddings = model.encode(docs, batch_size=32, show_progress_bar=True)

    # Save raw embeddings
    save_embeddings(embeddings, collection, f"{collection}_{method}_corpus_embeddings.pkl")

    # Project to n_features
    projected = random_project(embeddings=embeddings, n_features=n_features)
    save_embeddings(projected, collection, f"{collection}_{method}_{n_features}_projected_features.pkl")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Embed documents using Sentence-BERT (with optional projection dim).")
    parser.add_argument(
        '--collection',
        required=True,
        choices=['wiki', '20ng', 'wsj'],
        help="Name of the collection (wiki, 20ng, or wsj)."
    )
    parser.add_argument(
        '--dim',
        type=int,
        help="Number of features for random projection. If omitted, uses OPTIMAL_LDA_TOPICS[collection]."
    )
    parser.add_argument(
        '--model_name',
        type=str,
        default='all-MiniLM-L6-v2',
        help="Sentence-BERT model name (default: all-MiniLM-L6-v2)."
    )
    args = parser.parse_args()

    embed_sbert(args.collection, dim=args.dim, model_name=args.model_name)
