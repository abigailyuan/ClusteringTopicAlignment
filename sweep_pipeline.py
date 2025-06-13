#!/usr/bin/env python
"""
run_sweep_pipeline.py

Run mapping and visualization over a sweep of topic counts per collection.
For each collection, iterate over a list of candidate topic counts, train (if needed), map, and then visualize using the new sweep-mapping script.
"""
import subprocess
import argparse
import os
import sys

collections = ['wiki', '20ng', 'wsj']

# Define the ranges you want to test per collection:
TOPIC_SWEEP = {
    'wiki': [20, 40, 60, 80, 100],
    '20ng': [20, 40, 60, 80, 100],
    'wsj':  [20, 40, 60, 80, 100]
}

# Embedding and topic scripts
embedding_scripts = {
    'doc2vec':  'Embeddings/embed_doc2vec.py',
    'sbert':    'Embeddings/embed_sbert.py',
    'repllama': 'Embeddings/embed_repllama.py'
}

topic_train_scripts = {
    'lda':      'TopicModels/lda_train.py',
    'bertopic': 'TopicModels/bertopic_train.py'
}

general_mapping = 'Mapping/map_topic_embedding.py'
# New visualisation script for sweep mappings
sweep_visualisation_script = 'Visualisation/visualise_sweep_mapping.py'


def run_sweep(lang_model, topic_model, force=False):
    for coll in collections:
        # Sweep each topic count for this collection
        for n_topics in TOPIC_SWEEP[coll]:
            print(f"\n=== Processing {coll} with {n_topics} topics ===")
            # Embedding step
            emb_file = f'Processed{coll.upper()}/{coll}_{lang_model}_{n_topics}_projected_features.pkl'
            if force or not os.path.exists(emb_file):
                subprocess.run([
                    'python', embedding_scripts[lang_model],
                    '--collection', coll,
                    '--dim', str(n_topics)
                ], check=True)
            else:
                print(f"[SKIP] Embedding exists: {emb_file}")

            # Preprocessing (only once per collection)
            pre_file = f'Processed{coll.upper()}/{coll}_preprocessed.pkl'
            if force or not os.path.exists(pre_file):
                inline = (
                    'from Preprocessing.lda_preprocessing import load_raw_corpus, preprocess_corpus; '
                    f'raw = load_raw_corpus("Processed{coll.upper()}/{coll}_raw.pkl"); '
                    f'preprocess_corpus(raw, save_path="{pre_file}")'
                )
                subprocess.run(['python', '-c', inline], check=True)
            else:
                print(f"[SKIP] Preprocessing exists: {pre_file}")

            # Topic training
            train_args = [
                'python', topic_train_scripts[topic_model],
                '--dataset', coll,
                '--num_topics', str(n_topics)
            ]
            if topic_model == 'bertopic':
                out_dir = f'Results/BERTOPIC/{coll}_sweep_{n_topics}'
                os.makedirs(out_dir, exist_ok=True)
                train_args.extend(['--output_dir', out_dir])
            if force:
                train_args.append('--calculate_specificity')
            subprocess.run(train_args, check=True)

            # Mapping
            subprocess.run([
                'python', general_mapping,
                '--dataset', coll,
                '--lang_model', lang_model,
                '--topic_model', topic_model,
                '--dim', str(n_topics)
            ], check=True)

        # After sweeping this collection, visualize mappings across n_topics
        dims_arg = ','.join(str(n) for n in TOPIC_SWEEP[coll])
        vis_cmd = [
            'python', sweep_visualisation_script,
            '--collection', coll,
            '--lang_model', lang_model,
            '--topic_model', topic_model,
            '--dims', dims_arg
        ]
        subprocess.run(vis_cmd, check=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Sweep topic counts for mapping experiments using the new sweep-visualisation."
    )
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
        '--force',
        action='store_true',
        help="Force re-run of all steps even if outputs exist."
    )
    args = parser.parse_args()

    run_sweep(args.lang_model, args.topic_model, force=args.force)
