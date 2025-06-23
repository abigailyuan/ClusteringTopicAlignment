import pickle
import argparse
import os
import multiprocessing
from gensim.corpora import Dictionary
from gensim.models.ldamulticore import LdaMulticore
from topic_specificity import calculate_specificity_for_all_topics


def grid_search_lda(
    corpus_tokens,
    dictionary,
    collection,
    start_topics=10,
    end_topics=200,
    step=10,
    output_dir='Results/LDA',
    workers=None
):
    """
    Train LDA models over a range of topic numbers and compute specificity scores,
    using LdaMulticore for parallel training.
    Saves each model and returns a dict mapping num_topics to list of specificity scores.
    """
    os.makedirs(output_dir, exist_ok=True)
    if workers is None:
        # Reserve one core for system
        workers = max(1, multiprocessing.cpu_count() - 1)

    # Convert documents to BoW corpus
    bow_corpus = [dictionary.doc2bow(doc) for doc in corpus_tokens]

    results = {}
    for num_topics in range(start_topics, end_topics + 1, step):
        print(f"Training LdaMulticore with {num_topics} topics for {collection} using {workers} workers...")
        # Use default symmetric alpha since auto-tuning isn't supported
        lda = LdaMulticore(
            corpus=bow_corpus,
            id2word=dictionary,
            num_topics=num_topics,
            random_state=42,
            passes=10,
            workers=workers,
            per_word_topics=False
        )
        model_filename = f"{collection}_lda{num_topics}.model"
        model_path = os.path.join(output_dir, model_filename)
        lda.save(model_path)

        # Compute specificity scores
        print(f"Calculating specificity for {num_topics} topics...")
        scores = calculate_specificity_for_all_topics(
            model=lda,
            corpus=bow_corpus,
            mode='lda',
            threshold_mode='gmm',
            specificity_mode='diff'
        )
        results[num_topics] = scores
        print(f"Specificity scores @ {num_topics} topics: {scores}\n")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Grid search LDA: range of topics and specificity calculation"
    )
    parser.add_argument(
        '--dataset', choices=['wiki', '20ng', 'wsj'], required=True,
        help="Collection name"
    )
    parser.add_argument(
        '--preprocessed_path',
        help="Path to preprocessed corpus pickle (list of token lists)",
        default=None
    )
    parser.add_argument(
        '--start', type=int, default=10,
        help="Starting number of topics"
    )
    parser.add_argument(
        '--end', type=int, default=200,
        help="Ending number of topics"
    )
    parser.add_argument(
        '--step', type=int, default=10,
        help="Step size"
    )
    parser.add_argument(
        '--output_pickle',
        help="Pickle file path to save specificity dictionary",
        default=None
    )
    parser.add_argument(
        '--workers', type=int, default=None,
        help="Number of worker processes for parallel LDA training"
    )
    args = parser.parse_args()

    collection = args.dataset
    # Determine path to preprocessed docs
    pre_path = args.preprocessed_path or f"Processed{collection.upper()}/{collection}_preprocessed.pkl"

    # Load preprocessed documents (list of token lists)
    with open(pre_path, 'rb') as f:
        corpus_tokens = pickle.load(f)

    # Build dictionary
    dictionary = Dictionary(corpus_tokens)

    # Run grid search with multicore
    specificity_dict = grid_search_lda(
        corpus_tokens=corpus_tokens,
        dictionary=dictionary,
        collection=collection,
        start_topics=args.start,
        end_topics=args.end,
        step=args.step,
        output_dir=f"Results/LDA",
        workers=args.workers
    )

    # Save results dictionary to pickle
    pickle_path = args.output_pickle or f"Results/LDA/{collection}_lda_specificity.pkl"
    os.makedirs(os.path.dirname(pickle_path), exist_ok=True)
    with open(pickle_path, 'wb') as pf:
        pickle.dump(specificity_dict, pf)
    print(f"Specificity dictionary saved to {pickle_path}")


if __name__ == '__main__':
    main()
