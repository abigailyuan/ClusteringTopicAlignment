#!/usr/bin/env python
"""
visualise_sweep_mapping.py

For a single collection and embedding-method+topic-model, plot specificity vs topic-rank mapping
for multiple topic counts. Each color corresponds to a different n_topics, showing mapped vs unmapped topics.
"""
import os
import sys
import pickle
import argparse
import numpy as np
import matplotlib.pyplot as plt

# Ensure project root
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from gensim.models import LdaModel
from bertopic import BERTopic
from topic_specificity import calculate_specificity_for_all_topics


# ─── Ensure project root is on path ────────────────────────────────────────────
# If this file is in /project/Visualisation/visualise_mapping.py,
# then project_root will be /project
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

# ─── Make sure TopicSpecificityBerTopic is on sys.path ────────────────────────
# TopicSpecificityBerTopic/ is a sibling of run_pipeline.py (i.e. one level above "Visualisation/")
topic_spec_path = os.path.join(project_root, 'TopicSpecificityBerTopic')
if os.path.isdir(topic_spec_path):
    sys.path.insert(0, topic_spec_path)
else:
    sys.exit(f"[ERROR] Expected to find 'TopicSpecificityBerTopic' at {topic_spec_path}")
    
# Lazy‐import for the BERTopic‐specific specificity function
def import_bertopic_specificity():
    try:
        from topic_specificity_bertopic import calculate_specificity_bertopic
        return calculate_specificity_bertopic
    except ImportError:
        sys.exit("[ERROR] Cannot import topic_specificity_bertopic from TopicSpecificityBerTopic/")

def main():
    parser = argparse.ArgumentParser(description="Visualise mapping across topic-count sweep.")
    parser.add_argument('--collection', required=True, choices=['wiki','20ng','wsj'])
    parser.add_argument('--lang_model', required=True, choices=['doc2vec','sbert','repllama'])
    parser.add_argument('--topic_model', required=True, choices=['lda','bertopic'])
    parser.add_argument('--dims', type=str, required=True, help="Comma-separated list of topic counts, e.g. '20,40,60,80'")
    args = parser.parse_args()

    dims_list = [int(x) for x in args.dims.split(',')]
    if not dims_list:
        sys.exit("[ERROR] Provide at least one topic count in --dims.")

    # Prepare specificity function for BERTopic if needed
    if args.topic_model == 'bertopic':
        calculate_bert_spec = import_bertopic_specificity()

    os.makedirs('Results/Visualisation', exist_ok=True)
    plt.figure(figsize=(10,6))

    # Generate a distinct color for each sweep index
    colors = plt.cm.viridis(np.linspace(0,1,len(dims_list)))

    for idx, n_topics in enumerate(dims_list):
        # 1) specificity scores
        if args.topic_model == 'lda':
            model_path = os.path.join('Results','LDA', f"{args.collection}_lda{n_topics}.model")
            if not os.path.exists(model_path):
                sys.exit(f"[ERROR] LDA model not found: {model_path}")
            lda = LdaModel.load(model_path)
            corpus = pickle.load(open(os.path.join('Results','LDA', f"{args.collection}_corpus.pkl"),'rb'))
            scores = calculate_specificity_for_all_topics(
                model=lda,
                corpus=corpus,
                mode='lda',
                threshold_mode='gmm',
                specificity_mode='diff'
            )
        else:
            model_path = os.path.join('Results','BERTOPIC', f"{args.collection}_bertopic_{n_topics}.model")
            if not os.path.exists(model_path):
                sys.exit(f"[ERROR] BERTopic model not found: {model_path}")
            bt = BERTopic.load(model_path)
            scores = calculate_bert_spec(
                bt,
                threshold_mode='gmm',
                specificity_mode='diff'
            )

        # 2) mapping
        map_fname = f"{args.collection}_{args.lang_model}_{args.topic_model}_{n_topics}_mapping.pkl"
        map_path = os.path.join('Results', map_fname)
        if not os.path.exists(map_path):
            sys.exit(f"[ERROR] Mapping file not found: {map_path}")
        mappings = pickle.load(open(map_path,'rb'))
        mapped = {t for _,t in mappings}

        # 3) sort topics by specificity
        sorted_ts = sorted(enumerate(scores), key=lambda x: x[1])
        topics_sorted, scores_sorted = zip(*sorted_ts)
        x = list(range(len(topics_sorted)))
        mapped_x = [x[i] for i,t in enumerate(topics_sorted) if t in mapped]
        mapped_y = [scores_sorted[i] for i,t in enumerate(topics_sorted) if t in mapped]
        unmapped_x = [x[i] for i,t in enumerate(topics_sorted) if t not in mapped]
        unmapped_y = [scores_sorted[i] for i,t in enumerate(topics_sorted) if t not in mapped]

        c = colors[idx]
        plt.scatter(mapped_x, mapped_y, color=c, alpha=1.0, label=f'{n_topics}-mapped', marker='o', s=30)
        plt.scatter(unmapped_x, unmapped_y, color=c, alpha=0.3, label=f'{n_topics}-unmapped', marker='x', s=20)

    plt.xlabel('Topic Rank (sorted by specificity)')
    plt.ylabel('Specificity Score')
    plt.title(f"Mapping Specificity Sweep: {args.collection.upper()} + {args.lang_model.upper()} + {args.topic_model.upper()}")
    plt.legend(bbox_to_anchor=(1.05,1), loc='upper left')
    plt.tight_layout()

    out_path = os.path.join('Results/Visualisation', f"mapping_sweep_{args.collection}_{args.topic_model}_{args.lang_model}.pdf")
    plt.savefig(out_path)
    print(f"[SAVED] Sweep mapping visualization at {out_path}")
    plt.show()

if __name__ == '__main__':
    main()
