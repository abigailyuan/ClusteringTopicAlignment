import pickle

from nltk.corpus import wordnet as wn
import matplotlib.pyplot as plt


def word_similarity(w1, w2):
    sum_sim = 0
    num_pairs = 0
    w1_synsets = wn.synsets(w1)
    w2_synsets = wn.synsets(w2)
    for s1 in w1_synsets:
        for s2 in w2_synsets:
            sim = s1.wup_similarity(s2)
            sum_sim += sim
            num_pairs += 1

    return sum_sim / num_pairs if (num_pairs != 0) else 0

def topic_wordnet_coherence(topic):
    # use mean value as coherence
    sum_sim = 0
    num_pairs = 0
    for i in range(len(topic)):
        w1 = topic[i]
        for j in range(i, len(topic)):
            w2 = topic[j]
            sum_sim += word_similarity(w1, w2)
            num_pairs += 1
    return sum_sim / num_pairs

def coherence(model, n_topics=20):
    '''calculate per topic coherence score using WordNet similarity
    return the average score and a list of per topic coherence scores'''

    scores = []
    for i in range(n_topics):
        topic = [x[0] for x in model.show_topic(i)]
        coherence_score = topic_wordnet_coherence(topic)
        scores.append(coherence_score)
    return [sum(scores)/len(scores), scores]

directory = 'LDAResults/'
corpus = 'ProcessedWSJ/bow_lemma.pkl'

# generate topic model
directory = 'LDAResults/'
corpus = 'ProcessedWSJ/bow_lemma.pkl'
dictionary = 'ProcessedWSJ/dictionary_lemma.pkl'
tid = 32
n_topics = 20
# LDAGenerator.generate_lda(corpus=corpus, run_id=tid, num_topics=n_topics, dictionary=dictionary, directory=directory)
# LDAGenerator.predict_topic_labels(run_id=tid, corpus=corpus, directory=directory)
# LDAGenerator.generate_topic_keywords(run_id=tid, num_keywords=10, num_topics=n_topics, directory=directory)
# print('topic model generated.')

# model = LdaModel.load(directory + str(tid) + '/model')
#
# score, scores = coherence(model)
# print(score)
# print(scores)
#
# select optimal number of topics
# tid = 42
# model_coherences = []
# for n_topics in range(100, 501, 100):
#     tid += 1
#     LDAGenerator.generate_lda(corpus=corpus, run_id=tid, num_topics=n_topics, dictionary=dictionary, directory=directory)
#     # LDAGenerator.predict_topic_labels(run_id=tid, corpus=corpus, directory=directory)
#     LDAGenerator.generate_topic_keywords(run_id=tid, num_keywords=10, num_topics=n_topics, directory=directory)
#     print('topic model generated.')
#
#     # calculate coherence score and save
#     model = LdaModel.load(directory + str(tid) + '/model')
#     scores = coherence(model, n_topics=n_topics)
#     fp = open(directory + str(tid) + '/wordnet_coherence.pkl','wb')
#     pickle.dump(scores, fp)
#     fp.close()
#     model_coherences.append((n_topics,scores[0]))
#
# for n, score in model_coherences:
#     print(n, score)

# fp = open('select_ntopics_wordnet_100_500.lst','wb')
# pickle.dump(model_coherences, fp)
# fp.close()
#
# tid = 45
# model = LdaModel.load(directory + str(tid) + '/model')
# scores = coherence(model, n_topics=n_topics)
# fp = open(directory + str(tid) + '/wordnet_coherence.pkl','wb')
# pickle.dump(scores, fp)
# fp.close()
#
n_topics = list(range(10,101,10))
median_coherence = []
mean_coherence = []
for tid in range(33, 43):
    fp = open('LDAResults/'+str(tid)+'/pairwise_coherence.lst','rb')
    scores = pickle.load(fp)
    fp.close()
    entropy = sum(scores) / len(scores)
    # mean_coherence.append(statistics.mean(scores))
    # median_coherence.append(statistics.median(scores))
    mean_coherence.append(entropy)

#
# # for tid in range(44,46):
# #     fp = open('LDAResults/' + str(tid) + '/corpus_KL.lst', 'rb')
# #     scores = pickle.load(fp)
# #     fp.close()
# #     mean_coherence.append(statistics.mean(scores))
# #     #median_coherence.append(statistics.median(scores))
#
for i in range(10):
    # print(n_topics[i], median_coherence[i], mean_coherence[i])
    print(n_topics[i], mean_coherence[i])

plt.plot(n_topics, mean_coherence)



plt.ylabel('Cluster-based Entropy')
# plt.ylim(0,1)
plt.xlabel('Number of Topics')
#plt.savefig('cluster_entropy_topics_10.pdf')
plt.show()


# run_id = 36
# fp = open(directory + str(run_id) + '/wordnet_coherence.pkl', 'rb')
# coherences = pickle.load(fp)
# fp.close()
# scores = sorted(zip(coherences[1], list(range(40))), key=lambda x:x[0], reverse=True)
# fp = open('topic_rank_wordnet.txt','w')
# for score, id in scores:
#     fp.write(str(id)+'  '+str(score))
#     fp.write('\n')
# fp.close()