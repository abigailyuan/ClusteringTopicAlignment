import os
import argparse
import numpy as np
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

from embed_utils import (
    load_documents,
    save_embeddings,
    random_project,
    get_processed_dir
)
from preprocess import clean_doc


def embed_doc2vec(collection: str, vector_size: int = 300, epochs: int = 40):
    method = 'doc2vec'
    processed_dir = get_processed_dir(collection)
    raw_path = os.path.join(processed_dir, f"{collection}_{method}_corpus_embeddings.pkl")
    proj_path = os.path.join(processed_dir, f"{collection}_{method}_projected_features.pkl")

    # Skip if cached
    if os.path.exists(raw_path) and os.path.exists(proj_path):
        print(f"[SKIP] {collection} embeddings with {method} exist.")
        return

    docs = load_documents(collection)
    tagged = [TaggedDocument(words=clean_doc(doc), tags=[str(i)]) for i, doc in enumerate(docs)]

    model = Doc2Vec(vector_size=vector_size, min_count=2, epochs=epochs, workers=4)
    model.build_vocab(tagged)
    model.train(tagged, total_examples=model.corpus_count, epochs=model.epochs)

    embeddings = np.vstack([model.infer_vector(doc.words) for doc in tagged])
    save_embeddings(embeddings, collection, f"{collection}_{method}_corpus_embeddings.pkl")

    projected = random_project(embeddings)
    save_embeddings(projected, collection, f"{collection}_{method}_projected_features.pkl")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Embed documents using Doc2Vec.")
    parser.add_argument('--collection', required=True, choices=['wiki', '20ng', 'wsj'])
    args = parser.parse_args()
    embed_doc2vec(args.collection)