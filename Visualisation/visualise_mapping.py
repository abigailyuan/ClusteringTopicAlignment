import os
import argparse
import pickle
import numpy as np
import matplotlib.pyplot as plt

from gensim.models import LdaModel
from topic_specificity import calculate_specificity_for_all_topics
from topic_specificity_bertopic import calculate_specificity_bertopic
from bertopic import BERTopic


def plot_model(model_type):
    collections = ['wiki', '20ng', 'wsj']
    colors = ['C0', 'C1', 'C2']

    plt.figure(figsize=(8, 5))

    for i, collection in enumerate(collections):
        print(f"[INFO] Processing {collection} using {model_type.upper()}...")

        if model_type == 'lda':
            dictionary = pickle.load(open(f'Results/LDA/{collection}_dictionary.dict', 'rb'))
            corpus = pickle.load(open(f'Results/LDA/{collection}_corpus.pkl', 'rb'))
            lda = LdaModel.load(f"Results/LDA/{collection}_lda50.model")
            specificity_scores = calculate_specificity_for_all_topics(
                model=lda,
                corpus=corpus,
                mode='lda',
                threshold_mode='gmm',
                specificity_mode='sqrt'
            )
            mapping_file = f'Results/{collection}_repllama_lda_mapping.pkl'

        elif model_type == 'bertopic':
            bertopic_model = BERTopic.load(f"Results/BerTopic/{collection}_bertopic_model")
            specificity_scores = calculate_specificity_bertopic(
                bertopic_model,
                threshold_mode='gmm',
                specificity_mode='sqrt'
            )
            mapping_file = f'Results/{collection}_repllama_bertopic_mapping.pkl'

        else:
            raise ValueError(f"Unknown model type: {model_type}")

        mappings = pickle.load(open(mapping_file, 'rb'))

        # Mapped topic indices
        mapped_topics = {topic_id for _, topic_id in mappings}
        topic_scores = list(enumerate(specificity_scores))
        sorted_topic_scores = sorted(topic_scores, key=lambda x: x[1])
        sorted_topics, sorted_scores = zip(*sorted_topic_scores)
        x_positions = list(range(len(sorted_topics)))

        mapped_x = [x for x, t in zip(x_positions, sorted_topics) if t in mapped_topics]
        mapped_y = [s for t, s in zip(sorted_topics, sorted_scores) if t in mapped_topics]
        unmapped_x = [x for x, t in zip(x_positions, sorted_topics) if t not in mapped_topics]
        unmapped_y = [s for t, s in zip(sorted_topics, sorted_scores) if t not in mapped_topics]

        plt.scatter(mapped_x, mapped_y, color=colors[i], alpha=1.0, label=f'{collection}-mapped')
        plt.scatter(unmapped_x, unmapped_y, color=colors[i], alpha=0.3, label=f'{collection}-unmapped')

    plt.xlabel('Topics')
    plt.ylabel('Specificity')
    plt.legend()
    title = f'RepLlMa Embedding mapped {model_type.upper()} topics sorted by Specificity Scores'
    plt.title(title)
    plt.tight_layout()
    output_file = f'Results/{model_type.lower()}_repllama.pdf'
    plt.savefig(output_file)
    plt.show()
    print(f"[DONE] Figure saved as {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualise topic specificity and mappings.")
    parser.add_argument('--model', choices=['lda', 'bertopic'], required=True,
                        help="Specify the topic model to plot (lda or bertopic)")

    args = parser.parse_args()
    plot_model(args.model)
