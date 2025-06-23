#!/usr/bin/env python
"""
run_sweep_pipeline.py

Run mapping and visualization over a sweep of topic counts per collection.
For each collection, iterate over a list of candidate topic counts, embed, preprocess, train (always calling the train script), map, and then visualize using the new sweep-mapping script.
Training scripts handle their own skip logic.
"""
import subprocess
import argparse
import os
import sys

collections = ['wiki', '20ng', 'wsj']

# Define the ranges you want to test per collection:
TOPIC_SWEEP = {
    'wiki': [20, 30,40,50, 60,70, 80, 90,100],
    '20ng': [20, 30,40,50, 60,70, 80, 90,100],
    'wsj':  [20, 30,40,50, 60,70, 80, 90,100]
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

general_mapping = 'Mapping/map_sweep_mapping.py'
sweep_visualisation_script = 'Visualisation/visualise_sweep_mapping.py'


def run_sweep(lang_model, topic_model, force=False):
    for coll in collections:
        for n_topics in TOPIC_SWEEP[coll]:
            print(f"\n=== Processing {coll} with {n_topics} topics ===")

            # Embedding
            emb_file = f'Processed{coll.upper()}/{coll}_{lang_model}_{n_topics}_projected_features.pkl'
            if force or not os.path.exists(emb_file):
                subprocess.run([
                    'python', embedding_scripts[lang_model],
                    '--collection', coll,
                    '--dim', str(n_topics)
                ], check=True)
            else:
                print(f"[SKIP] Embedding exists: {emb_file}")

            # Preprocessing (once per collection)
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

            # Topic training (always call train script)
            train_args = [
                'python', topic_train_scripts[topic_model],
                '--dataset', coll,
                '--num_topics', str(n_topics),
            ]
            # direct output_dir for both models
            if topic_model == 'lda':
                train_args.extend(['--output_dir', 'Results/LDA'])
            else:
                train_args.extend(['--output_dir', 'Results/BERTOPIC'])
            if force:
                train_args.append('--force')
            print(f"[RUNNING] {' '.join(train_args)}")
            subprocess.run(train_args, check=True)

            # Mapping (sweep-specific)
            map_cmd = [
                'python', 'Mapping/map_sweep_mapping.py',
                '--dataset', coll,
                '--lang_model', lang_model,
                '--topic_model', topic_model,
                '--dim', str(n_topics),
                '--num_topics', str(n_topics),
                '--output_dir', 'Results'
            ]
            subprocess.run(map_cmd, check=True)

        # Visualize sweep for this collection
        dims_arg = ','.join(str(n) for n in TOPIC_SWEEP[coll])
        subprocess.run([
            'python', sweep_visualisation_script,
            '--collection', coll,
            '--lang_model', lang_model,
            '--topic_model', topic_model,
            '--dims', dims_arg
        ], check=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Sweep topic counts with training scripts handling skip logic and new sweep mapping visuals."
    )
    parser.add_argument('--lang_model', choices=list(embedding_scripts.keys()), required=True)
    parser.add_argument('--topic_model', choices=list(topic_train_scripts.keys()), required=True)
    parser.add_argument('--force', action='store_true', help="Pass --force to training scripts.")
    args = parser.parse_args()
    run_sweep(args.lang_model, args.topic_model, force=args.force)
