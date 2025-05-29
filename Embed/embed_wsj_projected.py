import os
import time
import os
import pickle
import torch
from Preprocessing.wsj import WSJ
from repllama_embedder import ReplLlamaEmbedder

from huggingface_hub import logging
logging.set_verbosity_debug()

# Configuration
PEFT_MODEL = 'castorini/repllama-v1-7b-lora-passage'
LLM_MODEL = 'meta-llama/Llama-2-7b-hf'
BATCH_SIZE = 8
N_FEATURES = 50
PROCESSED_DIR = 'ProcessedWSJ'
WSJ_PICKLE = 'ProcessedWSJ/wsj_raw.pkl'


# Redirect Hugging Face caches to project GPFS to avoid home quota
os.environ['HF_HUB_CACHE']      = '/data/gpfs/projects/punim2412/hf_cache/hub'
os.environ['HF_DATASETS_CACHE'] = '/data/gpfs/projects/punim2412/hf_cache/datasets'
os.environ['HF_METRICS_CACHE']  = '/data/gpfs/projects/punim2412/hf_cache/metrics'
os.environ['HF_MODULES_CACHE']  = '/data/gpfs/projects/punim2412/hf_cache/modules'


def main():
    # Create output directory
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    # 1. Load WSJ documents
    wsj = WSJ()
    documents = wsj.load(WSJ_PICKLE)
    n_docs = len(documents)
    print(f"Loaded {n_docs} WSJ documents.")

    # 2. Initialize the embedder
    embedder = ReplLlamaEmbedder(
        peft_model_name=PEFT_MODEL,
        llm_model_name=LLM_MODEL
    )
    
    print("CUDA available:", torch.cuda.is_available())
    print("Current device:", torch.cuda.current_device(), torch.cuda.get_device_name(torch.cuda.current_device()))

    # 3. Embed the corpus with timing
    t_start = time.time()
    corpus_embeddings = embedder.embed_corpus(documents, batch_size=BATCH_SIZE)
    t_end = time.time()
    elapsed = t_end - t_start
    per_doc = elapsed / n_docs if n_docs > 0 else 0
    print(f"Embedded {n_docs} docs in {elapsed:.2f}s (avg {per_doc:.4f}s/doc)")

    # 4. Save raw embeddings
    corpus_path = os.path.join(PROCESSED_DIR, 'corpus_embeddings.pkl')
    with open(corpus_path, 'wb') as f:
        pickle.dump(corpus_embeddings, f)
    print(f"Saved corpus embeddings to {corpus_path}")

    # 5. Random projection with timing
    t_proj_start = time.time()
    feature_matrix = embedder.get_feature_matrix(documents, batch_size=BATCH_SIZE)
    projected = embedder.random_projection(feature_matrix, n_features=N_FEATURES, seed=42)
    t_proj_end = time.time()
    proj_time = t_proj_end - t_proj_start
    print(f"Projected to {projected.shape} in {proj_time:.2f}s")

    # 6. Save projected features
    proj_path = os.path.join(PROCESSED_DIR, 'projected_features.pkl')
    with open(proj_path, 'wb') as f:
        pickle.dump(projected, f)
    print(f"Saved projected features to {proj_path}")


if __name__ == '__main__':
    main()
