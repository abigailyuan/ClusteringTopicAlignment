import subprocess
import argparse
import os
import sys

collections = ['wiki', '20ng', 'wsj']

# Single “source of truth” for optimal topic counts:
OPTIMAL_LDA_TOPICS = {
    'wsj': 50,
    'wiki': 80,
    '20ng': 70
}

# Placeholder for future Bertopic‐specific topic counts
OPTIMAL_BERTOPIC_TOPICS = {
    'wsj': 50,
    'wiki':80,
    '20ng':40
}

# Embedding scripts accept a “--dim <int>”
embedding_scripts = {
    'doc2vec':  'Embeddings/embed_doc2vec.py',
    'sbert':    'Embeddings/embed_sbert.py',
    'repllama': 'Embeddings/embed_repllama.py'
}

topic_train_scripts = {
    'lda':      'TopicModels/lda_train.py',
    'bertopic': 'TopicModels/bertopic_train.py'
}

general_mapping      = 'Mapping/map_topic_embedding.py'
visualisation_script = 'Visualisation/visualise_mapping.py'


def run_pipeline(lang_model, topic_model, force_components=None):
    if force_components is None:
        force_components = []
    force_set = set(force_components)

    # If you force topic, also force embedding
    if 'topic' in force_set:
        force_set |= {'embedding'}
    # If you force preprocessing, also force topic & embedding
    if 'preprocessing' in force_set:
        force_set |= {'topic', 'embedding'}

    force_components = list(force_set)

    print(f"\n===== RUNNING PIPELINE =====")
    print(f"Language Model: {lang_model}")
    print(f"Topic Model:    {topic_model}")
    print(f"Force Steps:    {force_components}\n============================\n")

    # Build dims_list = [wiki_dim, 20ng_dim, wsj_dim]
    if topic_model == 'lda':
        dims_list = [OPTIMAL_LDA_TOPICS.get(coll) for coll in collections]
        if any(d is None for d in dims_list):
            missing = [coll for coll, d in zip(collections, dims_list) if d is None]
            sys.exit(f"[ERROR] Missing entries in OPTIMAL_LDA_TOPICS for: {missing}")
    else:  # bertopic
        dims_list = [OPTIMAL_BERTOPIC_TOPICS.get(coll) for coll in collections]
        if any(d is None for d in dims_list):
            missing = [coll for coll, d in zip(collections, dims_list) if d is None]
            sys.exit(f"[ERROR] Missing entries in OPTIMAL_BERTOPIC_TOPICS for: {missing}")

    dims_arg = ','.join(str(d) for d in dims_list)

    for coll in collections:
        coll_upper = coll.upper()
        n_topics = dims_list[collections.index(coll)]

        # — Step 1: Embedding —
        emb_out = f'Processed{coll_upper}/{coll}_{lang_model}_{n_topics}_projected_features.pkl'
        if 'embedding' not in force_components and os.path.exists(emb_out):
            print(f"[SKIP] Embedding for {coll} + {lang_model} @ dim={n_topics} exists.")
        else:
            script = embedding_scripts[lang_model]
            print(f"[RUNNING] {script} --collection {coll} --dim {n_topics}")
            subprocess.run([
                'python', script,
                '--collection', coll,
                '--dim', str(n_topics)
            ], check=True)

        # — Step 1.5: Preprocessing for Topic Modeling —
        raw_in  = f'Processed{coll_upper}/{coll}_raw.pkl'
        pre_out = f'Processed{coll_upper}/{coll}_preprocessed.pkl'
        if 'preprocessing' not in force_components and os.path.exists(pre_out):
            print(f"[SKIP] Preprocessing for {coll} exists.")
        else:
            print(f"[RUNNING] inline preprocessing for {coll}")
            inline_code = (
                'from Preprocessing.lda_preprocessing import load_raw_corpus, preprocess_corpus; '
                f'raw = load_raw_corpus("{raw_in}"); '
                f'preprocess_corpus(raw, save_path="{pre_out}")'
            )
            subprocess.run(['python', '-c', inline_code], check=True)

        # — Step 2: Topic Modeling —
        if topic_model == 'lda':
            tm_script = topic_train_scripts['lda']
            model_path = f'Results/LDA/{coll}_lda{n_topics}.model'
            if 'topic' not in force_components and os.path.exists(model_path):
                print(f"[SKIP] LDA model for {coll} @ {n_topics} topics exists.")
            else:
                print(f"[RUNNING] {tm_script} --dataset {coll} --num_topics {n_topics}")
                subprocess.run([
                    'python', tm_script,
                    '--dataset', coll,
                    '--num_topics', str(n_topics)
                ], check=True)
        else:
            tm_script = topic_train_scripts['bertopic']
            model_path = f'Results/BERTOPIC/{coll}_bertopic_{n_topics}.model'
            if 'topic' not in force_components and os.path.exists(model_path):
                print(f"[SKIP] BERTopic model for {coll} exists.")
            else:
                print(f"[RUNNING] {tm_script} --dataset {coll} --num_topics {n_topics}")
                subprocess.run([
                    'python', tm_script,
                    '--dataset', coll,
                    '--num_topics', str(n_topics)
                ], check=True)

        # — Step 3: Mapping (always run) —
        print(
            f"[RUNNING] {general_mapping} "
            f"--dataset {coll} --lang_model {lang_model} "
            f"--topic_model {topic_model} --dim {n_topics}"
        )
        subprocess.run([
            'python', general_mapping,
            '--dataset', coll,
            '--lang_model', lang_model,
            '--topic_model', topic_model,
            '--dim', str(n_topics)
        ], check=True)

    # — Step 4: Visualization (always run) —
    print(
        f"[RUNNING] {visualisation_script} "
        f"--lang_model {lang_model} --topic_model {topic_model} --dims {dims_arg}"
    )
    subprocess.run([
        'python', visualisation_script,
        '--lang_model', lang_model,
        '--topic_model', topic_model,
        '--dims', dims_arg
    ], check=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run full pipeline with optional forced reruns.")
    parser.add_argument(
        '--lang_model',
        choices=list(embedding_scripts.keys()),
        required=True,
        help="Embedding model: doc2vec, sbert, or repllama"
    )
    parser.add_argument(
        '--topic_model',
        choices=list(topic_train_scripts.keys()),
        required=True,
        help="Topic model: lda or bertopic"
    )
    parser.add_argument(
        '--force_components',
        nargs='*',
        default=[],
        choices=['embedding', 'preprocessing', 'topic'],
        help="Specify which pipeline components to force rerun (mapping & visualization always run)."
    )
    parser.add_argument(
        '--force_all',
        action='store_true',
        help="Force rerun of embedding, preprocessing, and topic (mapping & visualization always run)."
    )
    args = parser.parse_args()

    if args.force_all:
        args.force_components = ['embedding', 'preprocessing', 'topic']

    run_pipeline(args.lang_model, args.topic_model, force_components=args.force_components)
