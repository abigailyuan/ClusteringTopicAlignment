import os
import sys
import pickle
import argparse
import matplotlib.pyplot as plt

# Ensure project root is on path for imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from gensim.models import LdaModel
from bertopic import BERTopic
from topic_specificity import calculate_specificity_for_all_topics
from TopicSpecificityBerTopic.topic_specificity_bertopic import calculate_specificity_bertopic


def plot_mapping(lang_model, topic_model):
    collections = ['wiki', '20ng', 'wsj']
    colors = ['C0', 'C1', 'C2']

    plt.figure(figsize=(10, 6))

    for i, coll in enumerate(collections):
        # 1) Compute specificity
        if topic_model == 'lda':
            # Load LDA model & corpus
            lda = LdaModel.load(f"Results/LDA/{coll}_lda50.model")
            corpus = pickle.load(open(f"Results/LDA/{coll}_corpus.pkl", 'rb'))
            scores = calculate_specificity_for_all_topics(
                model=lda,
                corpus=corpus,
                mode='lda',
                threshold_mode='gmm',
                specificity_mode='diff'
            )
        else:  # bertopic
            bt = BERTopic.load(f"Results/BERTOPIC/{coll}_bertopic_model")
            scores = calculate_specificity_bertopic(
                bt,
                threshold_mode='gmm',
                specificity_mode='diff'
            )

        # 2) Load mapping
        map_file = f"Results/{coll}_{lang_model}_{topic_model}_mapping.pkl"
        mappings = pickle.load(open(map_file, 'rb'))
        mapped = {t for _, t in mappings}

        # 3) Sort topics by specificity
        ts = list(enumerate(scores))
        sorted_ts = sorted(ts, key=lambda x: x[1])
        topics, sp = zip(*sorted_ts)
        x = list(range(len(topics)))
        mapped_x = [x[j] for j, t in enumerate(topics) if t in mapped]
        mapped_y = [sp[j] for j, t in enumerate(topics) if t in mapped]
        unmapped_x = [x[j] for j, t in enumerate(topics) if t not in mapped]
        unmapped_y = [sp[j] for j, t in enumerate(topics) if t not in mapped]

        plt.scatter(mapped_x, mapped_y, color=colors[i], alpha=1.0,
                    label=f'{coll}-{lang_model}-mapped')
        plt.scatter(unmapped_x, unmapped_y, color=colors[i], alpha=0.3,
                    label=f'{coll}-{lang_model}-unmapped')

    plt.xlabel('Topic Index (sorted by specificity)')
    plt.ylabel('Specificity Score')
    plt.title(f'Specificity: {lang_model.upper()} + {topic_model.upper()}')
    plt.legend()
    plt.tight_layout()

    out_path = f"Results/{topic_model}_{lang_model}.pdf"
    plt.savefig(out_path)
    print(f"[SAVED] Visualization at {out_path}")
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Visualise topic mapping specificity for any embedding+topic model"
    )
    parser.add_argument(
        '--lang_model', required=True,
        choices=['doc2vec', 'sbert', 'repllama'],
        help="Embedding method"
    )
    parser.add_argument(
        '--topic_model', required=True,
        choices=['lda', 'bertopic'],
        help="Topic model type"
    )
    args = parser.parse_args()
    plot_mapping(args.lang_model, args.topic_model)
