import subprocess
import argparse

collections = ['wiki', '20ng', 'wsj']

# Mapping of language model to its embedding script
embedding_scripts = {
    'doc2vec': 'Embeddings/embed_doc2vec.py',
    'sbert':   'Embeddings/embed_sbert.py',
    'repllama':'Embeddings/embed_repllama.py'
}

# Mapping of topic model to training scripts
topic_train_scripts = {
    'lda':     'TopicModels/lda_train.py',
    'bertopic':'TopicModels/bertopic_train.py'
}

# Mapping (for repllama only)
mapping_scripts = {
    ('lda', 'repllama'):     'Mapping/map_repllama_lda.py',
    ('bertopic','repllama'): 'Mapping/map_repllama_bertopic.py'
}


def run_pipeline(lang_model, topic_model):
    print(f"\n===== RUNNING PIPELINE =====")
    print(f"Language Model: {lang_model}")
    print(f"Topic Model:    {topic_model}")
    print(f"============================\n")

    for collection in collections:
        # Step 1: Generate embeddings
        print(f"[RUNNING] {embedding_scripts[lang_model]} --collection {collection}")
        subprocess.run(['python', embedding_scripts[lang_model], '--collection', collection], check=True)

        # Step 2: Train topic model
        script = topic_train_scripts[topic_model]
        if topic_model == 'lda':
            print(f"[RUNNING] {script} --collection {collection}")
            subprocess.run(['python', script, '--dataset', collection], check=True)
        else:
            print(f"[RUNNING] {script} --dataset {collection}")
            subprocess.run(['python', script, '--dataset', collection], check=True)

        # Step 3: Run mapping (only for repllama)
        if lang_model == 'repllama':
            mapping_script = mapping_scripts[(topic_model, lang_model)]
            print(f"[RUNNING] {mapping_script} --collection {collection}")
            subprocess.run(['python', mapping_script, '--collection', collection], check=True)

    # Step 4: Visualisation
    print("\n[VISUALISING]")
    print(f"[RUNNING] Visualisation/visualise_mapping.py --model {topic_model}")
    subprocess.run(['python', 'Visualisation/visualise_mapping.py', '--model', topic_model], check=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run full pipeline for embedding → topic modeling → mapping → visualization.")
    parser.add_argument('--lang_model', type=str, choices=['doc2vec', 'sbert', 'repllama'], required=True,
                        help="Embedding model to use.")
    parser.add_argument('--topic_model', type=str, choices=['lda', 'bertopic'], required=True,
                        help="Topic model to use.")

    args = parser.parse_args()
    run_pipeline(args.lang_model, args.topic_model)
