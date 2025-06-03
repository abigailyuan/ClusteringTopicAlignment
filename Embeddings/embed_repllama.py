#!/usr/bin/env python
import os
import argparse
import torch
import time

from repllama_embedder import ReplLlamaEmbedder
from embed_utils import (
    load_documents,
    save_embeddings,
    random_project,
    get_processed_dir,
    OPTIMAL_LDA_TOPICS
)
from preprocess import prepare_sbert


def embed_repllama(collection: str, dim: int = None, batch_size: int = 8):
    method = 'repllama'
    # Use the provided dim; otherwise use OPTIMAL_LDA_TOPICS[collection]
    n_features = dim if (dim is not None) else OPTIMAL_LDA_TOPICS[collection]

    processed_dir = get_processed_dir(collection)
    raw_path = os.path.join(processed_dir, f"{collection}_{method}_corpus_embeddings.pkl")
    proj_path = os.path.join(processed_dir, f"{collection}_{method}_{n_features}_projected_features.pkl")

    # Skip if both raw and projected embeddings already exist
    if os.path.exists(raw_path) and os.path.exists(proj_path):
        print(f"[SKIP] {collection} embeddings with {method} at dim={n_features} exist.")
        return

    # Load and preprocess documents
    raw_docs = load_documents(collection)
    docs = [prepare_sbert(d) for d in raw_docs]

    # Initialize embedder
    embedder = ReplLlamaEmbedder(
        peft_model_name='castorini/repllama-v1-7b-lora-passage',
        llm_model_name='meta-llama/Llama-2-7b-hf'
    )

    if torch.cuda.is_available():
        dev = torch.cuda.current_device()
        print(f"CUDA device: {torch.cuda.get_device_name(dev)}")
    else:
        print("CUDA not available, using CPU.")

    # Embed corpus
    t0 = time.time()
    embeddings = embedder.embed_corpus(docs, batch_size=batch_size)
    print(f"Embedding done in {time.time() - t0:.2f}s")

    # Save raw embeddings
    save_embeddings(embeddings, collection, f"{collection}_{method}_corpus_embeddings.pkl")

    # Project down to n_features
    projected = random_project(embeddings=embeddings, n_features=n_features)
    save_embeddings(projected, collection, f"{collection}_{method}_{n_features}_projected_features.pkl")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Embed documents using RepLLaMA (with optional projection dim).")
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
        '--batch_size',
        type=int,
        default=8,
        help="Batch size for embedding (default: 8)."
    )
    args = parser.parse_args()

    embed_repllama(args.collection, dim=args.dim, batch_size=args.batch_size)
