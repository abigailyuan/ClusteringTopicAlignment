from gensim.models import LdaModel
import os
import pickle
import time
import math
from gensim.models.coherencemodel import CoherenceModel


def co_occurence(corpus_bow, word1, word2):
    D_w1_w2 = 0
    for doc in corpus_bow:
        words = [w[0] for w in doc]
        if word1 in words and word2 in words:
            D_w1_w2 += 1

    return D_w1_w2
def occurence(corpus_bow, word):
    D_w = 0
    for doc in corpus_bow:
        for w,count in doc:
            if w == word:
                D_w += 1
    return D_w


def get_keywords(model, n_words=10):
    #get word probability matrix per topic
    word_prob_matrix = model.get_topics()

    #extract top 10 keyword ids for each topic
    topics = []
    for topic in word_prob_matrix:
        keywords = sorted(zip(list(range(len(topic))), topic), key=lambda x:x[1])[-n_words:]
        keywords = [x[0] for x in keywords]
        topics.append(keywords)
    return topics

def pairwise_coherence_topic(topic, corpus_bow, beta):

    coherrence = 0

    for i in range(len(topic)):
        w1 = topic[i]
        D_i = occurence(corpus_bow, w1)
        for j in range(i+1, len(topic)):
            w2 = topic[j]
            D_ij = co_occurence(corpus_bow, w1,w2)

            occurrence_ratio = math.log((D_ij + beta) / D_i)
            coherrence += occurrence_ratio
    return coherrence



def pairwise_coherence(model, corpus, beta=0.1, n_words=10):
    topics = get_keywords(model, n_words)
    corpus_bow = pickle.load(open(corpus, 'rb'))

    coherence_lst = []

    for topic in topics:
        coherrence = pairwise_coherence_topic(topic,corpus_bow,beta)
        coherence_lst.append(coherrence)
    return coherence_lst



#
# directory = 'LDAResults/'
# corpus = 'ProcessedWSJ/bow.pkl'
# run_id = 17
# model = LdaModel.load(directory + str(run_id) + '/model')
# coherences = pairwise_coherence(model,corpus,beta=0.1, n_words=10)
# fp = open('LDAResults/'+str(run_id)+'/pairwise_coherrence.lst','wb')
# pickle.dump(coherences, fp)
# fp.close()



