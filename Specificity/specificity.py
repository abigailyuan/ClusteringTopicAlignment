import os
import pickle

import numpy as np
from sklearn.mixture import GaussianMixture

def get_topic_weight_median(topic_weights):
    topic_weights.sort()
    return topic_weights[len(topic_weights) // 2]

def get_75_quatile(topic_weights):
    topic_weights.sort()
    return topic_weights[int(len(topic_weights) * 0.96)]

def get_bi_from_mixture(topic_weights):
    mixture = GaussianMixture(n_components=2, covariance_type="full").fit(topic_weights.reshape(-1, 1))
    means_hat = mixture.means_.flatten()
    sds_hat = np.sqrt(mixture.covariances_).flatten()

    # find the background distribution
    if means_hat[0] <= means_hat[1]:
        bi = means_hat[0]+2*sds_hat[0]
    else:
        bi = means_hat[1] + 2 * sds_hat[1]
    return bi

def get_Di(bi, topic_weights):
    return [i for i in topic_weights if i > bi]

def get_Vi(bi, topic_weights):
    return len([i for i in topic_weights if i > bi])

def get_B_i(bi, Vi, topic_weights):

    specificity_score = np.sqrt(sum([(i-bi)**2 for i in topic_weights])) / Vi
    print(sum([(i-bi)**2 for i in topic_weights]))
    print(np.sqrt(sum([(i - bi) ** 2 for i in topic_weights])))
    print(sum([(i-bi) for i in topic_weights]))
    print('------------')
    # specificity_score = sum([(i-bi) for i in topic_weights]) / Vi
    specificity_score = (1 / (1-bi)) * specificity_score
    return specificity_score

def get_topic_distribution(model, corpus):
    num_topics = model.get_topics().shape[0]
    corpus_dist = np.ndarray((len(corpus), num_topics)) # number of topics is 23
    for i in range(len(corpus)):
        doc = corpus[i]
        dist = model.get_document_topics(doc, minimum_probability=0)
        for topic, prob in dist:
            corpus_dist[i][topic] = prob
    return corpus_dist

def get_topic_distribution_hdp(model, corpus, num_topic):
    corpus_dist = np.zeros((len(corpus), num_topic)) # number of topics is 23
    for i in range(len(corpus)):
        doc = model[corpus[i]]
        for topic, prob in doc:
            if topic < num_topic:
                corpus_dist[i][topic] = prob

    return corpus_dist

def get_topic_distribution_lsa(model, corpus, num_topics):
    corpus_dist = np.zeros((len(corpus), num_topics))  # number of topics is 23
    vectorised_corpus = model[corpus]
    for doc_id in range(len(vectorised_corpus)):
        for topic, weight in vectorised_corpus[doc_id]:
            corpus_dist[doc_id][topic] = weight
    # normalise along topics
    global_minimum = corpus_dist.min()
    offset = 0 - global_minimum
    corpus_dist = corpus_dist+offset

    # normalise along documents
    for row in range(len(corpus_dist)):
        doc = corpus_dist[row]
        normalised_doc = normalize_list_numpy(doc)
        corpus_dist[row] = normalised_doc
    return corpus_dist

def normalize_list_numpy(list_numpy):
    list_numpy = list_numpy / sum(list_numpy)
    return list_numpy


# use this one for computing specificity
def calculate_Mi(model, corpus, mode, bi_mode, myui_mode, dist=False):
    M_is = []
    corpus_topic_distribution = None
    distribution_file = f'topic_distribution.pkl'
    dir = f'Results/wiki/3/'
    if type(dist) != bool:
        # print('Using provided corpus inference distribution...')
        corpus_topic_distribution = dist
    elif (distribution_file in os.listdir(dir)):
        corpus_topic_distribution = pickle.load(open(dir + distribution_file, 'rb'))
    else:
        if mode == 'lsa':
            corpus_topic_distribution = get_topic_distribution_lsa(model, corpus)
        if mode == 'lda':
            corpus_topic_distribution = get_topic_distribution(model, corpus)
        if mode == 'hdp':
            corpus_topic_distribution = get_topic_distribution_hdp(model, corpus)
        # pickle.dump(corpus_topic_distribution, open(dir + distribution_file, 'wb'))
    for topic in range(corpus_topic_distribution.shape[1]):
        topic_weights = corpus_topic_distribution[:, topic]
        if bi_mode == 'median':
            b_i = get_topic_weight_median(topic_weights)
        elif bi_mode == 'quartile':
            b_i = get_75_quatile(topic_weights)
        elif bi_mode == 'GMM':
            b_i = get_bi_from_mixture(topic_weights)
        V_i = get_Vi(b_i, topic_weights)
        D_i = get_Di(b_i, topic_weights)
        if myui_mode == 'diff':
            myui = calculate_myui(D_i, b_i, V_i)
        elif myui_mode == 'sqrt':
            myui = calculate_myui_sqrt(D_i, b_i, V_i)
        M_i = myui / (1 - b_i)
        M_is.append(M_i)
    return M_is

def calculate_myui(topic_weights, bi, Vi):
    if Vi:
        myu_i = sum([(i - bi) for i in topic_weights]) / Vi
    else:
        return 0
    return myu_i

def calculate_myui_sqrt(topic_weights, bi, Vi):
    if Vi:
        myui_sqrt = np.sqrt(sum([(i - bi)**2 for i in topic_weights]) / Vi)
    else:
        return 0
    return myui_sqrt

def calculate_variance(bi, myui, Vi, topic_weights):
    var = (1 / (Vi-1))*sum([(i-bi-myui)**2 for i in topic_weights])
    return var

def calculate_Zi(model, corpus, mode, dist=False):
    Z_is = []
    corpus_topic_distribution = None
    distribution_file = f'topic_distribution.pkl'
    dir = f'results/{mode.upper()}/'
    if type(dist) != bool:
        corpus_topic_distribution = dist
    elif (distribution_file in os.listdir(dir)):
        corpus_topic_distribution = pickle.load(open(dir + distribution_file, 'rb'))
    else:
        if mode == 'lsa':
            corpus_topic_distribution = get_topic_distribution_lsa(model, corpus)
        if mode == 'lda':
            corpus_topic_distribution = get_topic_distribution(model, corpus)
        if mode == 'hdp':
            corpus_topic_distribution = get_topic_distribution_hdp(model, corpus)
        pickle.dump(corpus_topic_distribution, open(dir + distribution_file, 'wb'))
    for topic in range(corpus_topic_distribution.shape[1]):
        topic_weights = corpus_topic_distribution[:, topic]
        b_i = get_bi_from_mixture(topic_weights)
        V_i = get_Vi(b_i, topic_weights)
        if V_i < 2:
            Z_i = 'N/A'
        else:
            D_i = get_Di(b_i, topic_weights)
            myui = calculate_myui(D_i, b_i, V_i)
            var = calculate_variance(b_i, myui, V_i, D_i)
            Z_i = myui / np.sqrt(var)
        Z_is.append(Z_i)

    return Z_is