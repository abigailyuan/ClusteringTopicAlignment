import os
import argparse
import torch
import time

from repllama_embedder import ReplLlamaEmbedder
from embed_utils import (
    load_documents,
    save_embeddings,
    random_project,
    get_processed_dir
)
from preprocess import prepare_sbert


def embed_repllama(collection: str, batch_size: int = 8):
    method = 'repllama'
    processed_dir = get_processed_dir(collection)
    raw_path = os.path.join(processed_dir, f"{collection}_{method}_corpus_embeddings.pkl")
    proj_path = os.path.join(processed_dir, f"{collection}_{method}_projected_features.pkl")

    # Skip if cached
    if os.path.exists(raw_path) and os.path.exists(proj_path):
        print(f"[SKIP] {collection} embeddings with {method} exist.")
        return

    raw_docs = load_documents(collection)
    docs = [prepare_sbert(d) for d in raw_docs]

    embedder = ReplLlamaEmbedder(
        peft_model_name='castorini/repllama-v1-7b-lora-passage',
        llm_model_name='meta-llama/Llama-2-7b-hf'
    )

    if torch.cuda.is_available():
        dev = torch.cuda.current_device()
        print(f"CUDA device: {torch.cuda.get_device_name(dev)}")
    else:
        print("CUDA not available, using CPU.")

    t0 = time.time()
    embeddings = embedder.embed_corpus(docs, batch_size=batch_size)
    print(f"Embedding done in {time.time() - t0:.2f}s")

    save_embeddings(embeddings, collection, f"{collection}_{method}_corpus_embeddings.pkl")

    projected = random_project(embeddings)
    save_embeddings(projected, collection, f"{collection}_{method}_projected_features.pkl")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Embed documents using RepLLaMA.")
    parser.add_argument('--collection', required=True, choices=['wiki', '20ng', 'wsj'])
    args = parser.parse_args()
    embed_repllama(args.collection)
