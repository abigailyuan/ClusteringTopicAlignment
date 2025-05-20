import os
import argparse
from sentence_transformers import SentenceTransformer

from embed_utils import (
    load_documents,
    save_embeddings,
    random_project,
    get_processed_dir
)
from preprocess import prepare_sbert


def embed_sbert(collection: str, model_name: str = 'all-MiniLM-L6-v2'):
    method = 'sbert'
    processed_dir = get_processed_dir(collection)
    raw_path = os.path.join(processed_dir, f"{collection}_{method}_corpus_embeddings.pkl")
    proj_path = os.path.join(processed_dir, f"{collection}_{method}_projected_features.pkl")

    # Skip if cached
    if os.path.exists(raw_path) and os.path.exists(proj_path):
        print(f"[SKIP] {collection} embeddings with {method} exist.")
        return

    raw_docs = load_documents(collection)
    docs = [prepare_sbert(d) for d in raw_docs]

    model = SentenceTransformer(model_name)
    embeddings = model.encode(docs, batch_size=32, show_progress_bar=True)

    save_embeddings(embeddings, collection, f"{collection}_{method}_corpus_embeddings.pkl")
    projected = random_project(embeddings)
    save_embeddings(projected, collection, f"{collection}_{method}_projected_features.pkl")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Embed documents using Sentence-BERT.")
    parser.add_argument('--collection', required=True, choices=['wiki', '20ng', 'wsj'])
    args = parser.parse_args()
    embed_sbert(args.collection)