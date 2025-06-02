import subprocess
import argparse
import os

collections = ['wiki', '20ng', 'wsj']
OPTIMAL_LDA_TOPICS = {
    'wsj':50,
    'wiki':80,
    '20ng':70
    }

# Step 1.5 preprocessing script
preprocessing_script = 'Preprocessing/preprocess_step_1_5.py'

embedding_scripts = {
    'doc2vec': 'Embeddings/embed_doc2vec.py',
    'sbert':   'Embeddings/embed_sbert.py',
    'repllama':'Embeddings/embed_repllama.py'
}
topic_train_scripts = {
    'lda':     'TopicModels/lda_train.py',
    'bertopic':'TopicModels/bertopic_train.py'
}
general_mapping = 'Mapping/map_topic_embedding.py'


def run_pipeline(lang_model, topic_model, force_components=None):
    if force_components is None:
        force_components = []
    # Determine dependent steps to force
    force_set = set(force_components)
    if 'topic' in force_set:
        force_set |= {'mapping', 'visualization'}
    if 'embedding' in force_set:
        force_set |= {'mapping', 'visualization'}
    if 'preprocessing' in force_set:
        force_set |= {'topic', 'mapping', 'visualization'}
    force_components = list(force_set)

    print(f"\n===== RUNNING PIPELINE =====")
    print(f"Language Model: {lang_model}")
    print(f"Topic Model:    {topic_model}")
    print(f"Force Steps:    {force_components}\n============================\n")

    for coll in collections:
        # Step 1: Embedding
        emb_out = f'Processed{coll}/{coll}_{lang_model}_projected_features.pkl'
        if 'embedding' not in force_components and os.path.exists(emb_out):
            print(f"[SKIP] Embedding for {coll} + {lang_model} exists.")
        else:
            script = embedding_scripts[lang_model]
            print(f"[RUNNING] {script} --collection {coll}")
            subprocess.run(['python', script, '--collection', coll], check=True)

        # Step 1.5: Preprocessing for Topic Modeling
        raw_in = f'Processed{coll}/{coll}_raw.pkl'
        pre_out = f'Processed{coll}/{coll}_preprocessed.pkl'
        if 'preprocessing' not in force_components and os.path.exists(pre_out):
            print(f"[SKIP] Preprocessing for {coll} exists.")
        else:
            print(f"[RUNNING] {preprocessing_script} --input {raw_in} --output {pre_out}")
            subprocess.run([
                'python', preprocessing_script,
                '--input', raw_in,
                '--output', pre_out
            ], check=True)

        # Step 2: Topic Modeling
        if topic_model == 'lda':
            model_flag, model_path = '--dataset', f'Results/LDA/{coll}_lda{OPTIMAL_LDA_TOPICS[coll]}.model'
        else:
            model_flag, model_path = '--dataset', f'Results/BERTOPIC/{coll}_bertopic_model'

        if 'topic' not in force_components and (os.path.exists(model_path) or os.path.isdir(model_path)):
            print(f"[SKIP] {topic_model.upper()} model for {coll} exists.")
        else:
            tm_script = topic_train_scripts[topic_model]
            print(f"[RUNNING] {tm_script} {model_flag} {coll}")
            subprocess.run(['python', tm_script, model_flag, coll], check=True)

        # Step 3: Mapping
        map_out = f'Results/{coll}_{lang_model}_{topic_model}_mapping.pkl'
        if 'mapping' not in force_components and os.path.exists(map_out):
            print(f"[SKIP] Mapping for {coll} + {lang_model}+{topic_model} exists.")
        else:
            print(f"[RUNNING] {general_mapping} --collection {coll} --lang_model {lang_model} --topic_model {topic_model}")
            subprocess.run([
                'python', general_mapping,
                '--dataset', coll,
                '--lang_model', lang_model,
                '--topic_model', topic_model
            ], check=True)

    # Step 4: Visualization
    viz_out = f"Results/{topic_model}_{lang_model}.pdf"
    if 'visualization' not in force_components and os.path.exists(viz_out):
        print(f"[SKIP] Visualization {viz_out} exists.")
    else:
        viz_script = 'Visualisation/visualise_mapping.py'
        print(f"[RUNNING] {viz_script} --lang_model {lang_model} --topic_model {topic_model}")
        subprocess.run(['python', viz_script, '--lang_model', lang_model, '--topic_model', topic_model], check=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run full pipeline with optional forced reruns.")
    parser.add_argument(
        '--lang_model', choices=list(embedding_scripts.keys()), required=True,
        help="Embedding model: doc2vec, sbert, or repllama"
    )
    parser.add_argument(
        '--topic_model', choices=list(topic_train_scripts.keys()), required=True,
        help="Topic model: lda or bertopic"
    )
    parser.add_argument(
        '--force_components', nargs='*', default=[],
        choices=['embedding','preprocessing','topic','mapping','visualization'],
        help="Specify which pipeline components to force rerun"
    )
    parser.add_argument(
        '--force_all', action='store_true',
        help="Force rerun of all pipeline components"
    )
    args = parser.parse_args()

    # Override force_components if force_all is set
    if args.force_all:
        args.force_components = ['embedding','preprocessing','topic','mapping','visualization']

    run_pipeline(args.lang_model, args.topic_model, force_components=args.force_components)
