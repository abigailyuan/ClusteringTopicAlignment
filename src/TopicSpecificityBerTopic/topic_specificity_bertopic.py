import numpy as np
from sklearn.mixture import GaussianMixture
from bertopic import BERTopic


def calculate_specificity_bertopic(
    topic_model: BERTopic,
    threshold_mode: str = 'gmm',
    specificity_mode: str = 'diff'
):
    """
    Calculate specificity scores for each topic in a BERTopic model.
    Assumes `topic_model.probabilities_` is available from `fit_transform`.

    Args:
        topic_model (BERTopic): A fitted BERTopic instance.
        threshold_mode (str): 'median', 'percentile', or 'gmm'.
        specificity_mode (str): 'diff' or 'sqrt'.

    Returns:
        List[float]: Specificity score for each topic.
    """

    if topic_model.probabilities_ is None:
        raise ValueError("probabilities_ not available. Use `calculate_probabilities=True` during fit_transform.")

    probs = topic_model.probabilities_  # shape: (n_docs, n_topics)
    num_topics = probs.shape[1]
    scores = []

    for t in range(num_topics):
        weights = probs[:, t]

        # Step 1: Threshold selection
        if threshold_mode == 'median':
            bi = np.median(weights)
        elif threshold_mode == 'percentile':
            bi = np.percentile(weights, 96)
        elif threshold_mode == 'gmm':
            reshaped = weights.reshape(-1, 1)
            gmm = GaussianMixture(n_components=2).fit(reshaped)
            means = gmm.means_.flatten()
            stds = np.sqrt(gmm.covariances_).flatten()
            bi = means[0] + 2 * stds[0] if means[0] <= means[1] else means[1] + 2 * stds[1]
        else:
            raise ValueError(f"Invalid threshold_mode: {threshold_mode}")

        # Step 2: Apply threshold and compute specificity
        Di = weights[weights > bi]
        Vi = len(Di)

        if Vi == 0:
            scores.append(0.0)
            continue

        if specificity_mode == 'diff':
            myui = np.mean(Di - bi)
        elif specificity_mode == 'sqrt':
            myui = np.sqrt(np.mean((Di - bi) ** 2))
        else:
            raise ValueError(f"Invalid specificity_mode: {specificity_mode}")

        specificity = myui / (1 - bi) if (1 - bi) != 0 else 0.0
        scores.append(specificity)

    return scores
