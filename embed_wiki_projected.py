import os
import time
import pickle
import torch

from repllama_embedder import ReplLlamaEmbedder


# Configuration
PEFT_MODEL    = 'castorini/repllama-v1-7b-lora-passage'
LLM_MODEL     = 'meta-llama/Llama-2-7b-hf'
BATCH_SIZE    = 8
N_FEATURES    = 50

PROCESSED_DIR      = 'ProcessedWIKI'
NEWGROUPS_PICKLE   = os.path.join(PROCESSED_DIR, 'wiki_raw.pkl')

# redirect HF cache to GPFS (or any custom location)
os.environ['HF_HUB_CACHE']      = '/data/gpfs/projects/punim2412/hf_cache/hub'
os.environ['HF_DATASETS_CACHE'] = '/data/gpfs/projects/punim2412/hf_cache/datasets'
os.environ['HF_METRICS_CACHE']  = '/data/gpfs/projects/punim2412/hf_cache/metrics'
os.environ['HF_MODULES_CACHE']  = '/data/gpfs/projects/punim2412/hf_cache/modules'


def main():
    # ensure output folder exists
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    # 1. Load or cache 20NG documents
    if os.path.exists(NEWGROUPS_PICKLE):
        with open(NEWGROUPS_PICKLE, 'rb') as f:
            documents = pickle.load(f)
        print(f"Loaded {len(documents)} cached 20NG documents.")
    else:
        print('ERROR: there is no raw text corpus of WIKI sample!')
        return 0

    n_docs = len(documents)

    # 2. Initialize the embedder
    embedder = ReplLlamaEmbedder(
        peft_model_name=PEFT_MODEL,
        llm_model_name=LLM_MODEL
    )
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        print("Device:", device, torch.cuda.get_device_name(device))

    # 3. Embed the corpus
    t0 = time.time()
    corpus_embeddings = embedder.embed_corpus(documents, batch_size=BATCH_SIZE)
    t1 = time.time()
    total = t1 - t0
    avg = total / n_docs if n_docs else 0
    print(f"Embedded {n_docs} docs in {total:.2f}s ({avg:.4f}s/doc)")

    # 4. Save raw embeddings
    emb_path = os.path.join(PROCESSED_DIR, 'wiki_corpus_embeddings.pkl')
    with open(emb_path, 'wb') as f:
        pickle.dump(corpus_embeddings, f)
    print(f"Saved embeddings to {emb_path}")

    # 5. Feature extraction + random projection
    t2 = time.time()
    feat_mat = embedder.get_feature_matrix(documents, batch_size=BATCH_SIZE)
    projected = embedder.random_projection(feat_mat, n_features=N_FEATURES, seed=42)
    t3 = time.time()
    print(f"Projected to {projected.shape} in {(t3 - t2):.2f}s")

    # 6. Save projected features
    proj_path = os.path.join(PROCESSED_DIR, 'wiki_projected_features.pkl')
    with open(proj_path, 'wb') as f:
        pickle.dump(projected, f)
    print(f"Saved projected features to {proj_path}")


if __name__ == '__main__':
    main()
