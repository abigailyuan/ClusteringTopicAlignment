import numpy as np
from gensim.models import LdaModel
import os
import pickle
import time


def generate_lda(corpus, run_id, num_topics=10, dictionary=None, update_every=1,directory='/LDAResults/'):
    """generate an LDA model with the selected parameters and
    save to directory specified."""

    corpus = pickle.load(open(corpus, 'rb'))
    dictionary = pickle.load(open(dictionary, 'rb'))
    start = time.perf_counter()
    lda_model = LdaModel(corpus, num_topics=num_topics, id2word=dictionary, alpha='auto', random_state=run_id, update_every=update_every)

    os.mkdir(directory + str(run_id))
    lda_model.save(directory + str(run_id) + '/model')

    end = time.perf_counter()

    print('time used:', int(end - start))
    return lda_model


def predict_topic_labels(run_id, corpus, mode='1', directory='/LDAResults/'):
    """mode = ['1','2','3'] Number represents how many top topics are considered.
    Predict the topic label based on the selected mode and
    save results to specified directory.
    """

    corpus = pickle.load(open(corpus, 'rb'))
    lda_model = LdaModel.load(directory + str(run_id) + '/model')
    documents_topics = lda_model.get_document_topics(corpus)
    topics_r1 = []
    if mode == '1':
        topics_r1 = [sorted(i, key=lambda x: x[1], reverse=True)[0][0] for i in
                     documents_topics]  # find the most likely topic per document
    # TODO label documents with multiple top ranked topics

    return topics_r1


def generate_topic_keywords(run_id, num_keywords=10, num_topics=10, directory='/LDAResults/'):
    """generate topic keywords and save to result directory"""

    lda_model = LdaModel.load(directory + str(run_id) + '/model')
    topics = sorted(lda_model.show_topics(num_topics=num_topics, num_words=num_keywords, formatted=False))

    fp = open(directory + str(run_id) + '/' + 'topic_keywords.txt', 'w')
    for topic in topics:
        tid = str(topic[0])
        keywords = str([k for k, v in topic[1]])
        fp.write('topic ' + tid)
        fp.write('\n')
        fp.write(keywords)
        fp.write('\n')
        # print('topic ' + tid)
        # print(keywords)

    fp.close()
    return topics


def get_topic_vectors(tid, directory='LDAResults/'):

    lda_model = LdaModel.load(directory+str(tid)+'/model')
    return lda_model.get_topics()

def get_document_topic_matrix(bow, tid, dir='Results/wiki/'):
    lda_model = LdaModel.load(dir+tid+'/model')
    topics = lda_model.get_document_topics(bow, minimum_probability=0)
    topic_matrix = []
    for doc in topics:
        topic_dist = [i[1] for i in doc]
        topic_matrix.append(topic_dist)

    topic_matrix = np.asarray(topic_matrix)

    return topic_matrix
