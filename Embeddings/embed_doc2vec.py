#!/usr/bin/env python
import os
import argparse
import numpy as np
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

from embed_utils import (
    load_documents,
    save_embeddings,
    random_project,
    get_processed_dir,
    OPTIMAL_LDA_TOPICS
)
from preprocess import clean_doc


def embed_doc2vec(collection: str, dim: int = None, vector_size: int = 300, epochs: int = 40):
    method = 'doc2vec'
    # If --dim was supplied, use that; otherwise use OPTIMAL_LDA_TOPICS[collection]
    n_features = dim if (dim is not None) else OPTIMAL_LDA_TOPICS[collection]

    processed_dir = get_processed_dir(collection)
    raw_path = os.path.join(processed_dir, f"{collection}_{method}_corpus_embeddings.pkl")
    proj_path = os.path.join(processed_dir, f"{collection}_{method}_{n_features}_projected_features.pkl")

    # Skip if both the raw and projected embeddings already exist
    if os.path.exists(raw_path) and os.path.exists(proj_path):
        print(f"[SKIP] {collection} embeddings with {method} at dim={n_features} exist.")
        return

    # Load and clean documents
    docs = load_documents(collection)
    tagged = [TaggedDocument(words=clean_doc(doc), tags=[str(i)]) for i, doc in enumerate(docs)]

    # Train Doc2Vec
    model = Doc2Vec(vector_size=vector_size, min_count=2, epochs=epochs, workers=4)
    model.build_vocab(tagged)
    model.train(tagged, total_examples=model.corpus_count, epochs=model.epochs)

    # Infer raw embeddings (size = vector_size)
    embeddings = np.vstack([model.infer_vector(doc.words) for doc in tagged])
    save_embeddings(embeddings, collection, f"{collection}_{method}_corpus_embeddings.pkl")

    # Random projection down to n_features
    projected = random_project(embeddings=embeddings, n_features=n_features)
    save_embeddings(projected, collection, f"{collection}_{method}_{n_features}_projected_features.pkl")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Embed documents using Doc2Vec (with optional projection dim).")
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
    args = parser.parse_args()

    embed_doc2vec(args.collection, dim=args.dim)
