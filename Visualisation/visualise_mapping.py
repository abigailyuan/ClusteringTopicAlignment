import os
import sys
import pickle
import argparse
import matplotlib.pyplot as plt

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

from gensim.models import LdaModel
from bertopic import BERTopic
from topic_specificity import calculate_specificity_for_all_topics

# Lazy‐import for the BERTopic‐specific specificity function
def import_bertopic_specificity():
    try:
        from topic_specificity_bertopic import calculate_specificity_bertopic
        return calculate_specificity_bertopic
    except ImportError:
        sys.exit("[ERROR] Cannot import topic_specificity_bertopic from TopicSpecificityBerTopic/")


def plot_mapping(lang_model, topic_model, dims_list):
    collections = ['wiki', '20ng', 'wsj']
    colors = ['C0', 'C1', 'C2']

    # Build a dict: {'wiki': 80, '20ng': 70, 'wsj': 50}, for example
    dims = dict(zip(collections, dims_list))

    plt.figure(figsize=(10, 6))

    for i, coll in enumerate(collections):
        # 1) Compute specificity
        if topic_model == 'lda':
            # Look up exact number of topics from dims dict
            n_topics = dims.get(coll)
            if n_topics is None:
                sys.exit(f"[ERROR] No dim provided for '{coll}'.")

            lda_path = os.path.join('Results', 'LDA', f"{coll}_lda{n_topics}.model")
            if not os.path.exists(lda_path):
                sys.exit(f"[ERROR] Cannot find LDA model: {lda_path}")
            lda = LdaModel.load(lda_path)

            corpus_path = os.path.join('Results', 'LDA', f"{coll}_corpus.pkl")
            if not os.path.exists(corpus_path):
                sys.exit(f"[ERROR] Corpus not found: {corpus_path}")
            corpus = pickle.load(open(corpus_path, 'rb'))

            scores = calculate_specificity_for_all_topics(
                model=lda,
                corpus=corpus,
                mode='lda',
                threshold_mode='gmm',
                specificity_mode='diff'
            )

        else:  # bertopic
            # Lazy‐import calculate_specificity_bertopic
            calculate_specificity_bertopic = import_bertopic_specificity()

            bt_path = os.path.join('Results', 'BERTOPIC', f"{coll}_bertopic_model")
            if not os.path.isdir(bt_path):
                sys.exit(f"[ERROR] BERTopic model directory not found: {bt_path}")
            bt = BERTopic.load(bt_path)

            scores = calculate_specificity_bertopic(
                bt,
                threshold_mode='gmm',
                specificity_mode='diff'
            )

        # 2) Load mapping results for this (coll, lang_model, topic_model)
        map_file = os.path.join('Results', f"{coll}_{lang_model}_{topic_model}_mapping.pkl")
        if not os.path.exists(map_file):
            sys.exit(f"[ERROR] Mapping file not found: {map_file}")
        mappings = pickle.load(open(map_file, 'rb'))
        mapped_topics = {t for _, t in mappings}

        # 3) Sort topics by specificity
        ts = list(enumerate(scores))
        sorted_ts = sorted(ts, key=lambda x: x[1])
        topics_sorted, scores_sorted = zip(*sorted_ts)

        x = list(range(len(topics_sorted)))
        mapped_x   = [x[j] for j, t in enumerate(topics_sorted) if t in mapped_topics]
        mapped_y   = [scores_sorted[j] for j, t in enumerate(topics_sorted) if t in mapped_topics]
        unmapped_x = [x[j] for j, t in enumerate(topics_sorted) if t not in mapped_topics]
        unmapped_y = [scores_sorted[j] for j, t in enumerate(topics_sorted) if t not in mapped_topics]

        plt.scatter(
            mapped_x,
            mapped_y,
            color=colors[i],
            alpha=1.0,
            label=f'{coll}-{lang_model}-mapped'
        )
        plt.scatter(
            unmapped_x,
            unmapped_y,
            color=colors[i],
            alpha=0.3,
            label=f'{coll}-{lang_model}-unmapped'
        )

    plt.xlabel('Topic Rank (sorted by specificity)')
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
        description=(
            "Visualise topic mapping specificity. Pass --dims only when topic_model=lda, "
            "as a comma-separated list [wiki,20ng,wsj], e.g. '80,70,50'."
        )
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
    parser.add_argument(
        '--dims',
        type=str,
        required=True,
        help=(
            "Comma-separated list of topic dims for [wiki,20ng,wsj], e.g. '80,70,50'.\n"
            "Required if --topic_model=lda. Ignored if --topic_model=bertopic."
        )
    )
    args = parser.parse_args()

    # Parse dims into [80, 70, 50], check length == 3
    dims_list = [int(x) for x in args.dims.split(',')]
    if len(dims_list) != 3:
        sys.exit("[ERROR] --dims must have exactly 3 comma-separated integers, e.g. '80,70,50'.")

    plot_mapping(args.lang_model, args.topic_model, dims_list)
