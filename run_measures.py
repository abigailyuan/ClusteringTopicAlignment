from gensim.models import LdaModel, CoherenceModel

import KLDivergence
import LDAGenerator
import topic_measures
import perplexity
import cluster_entropy
import pickle


def run_measures(n_topics, tid):
    # generate topic model
    directory = 'LDAResults/'
    corpus = 'ProcessedWSJ/bow_lemma.pkl'
    dictionary = 'ProcessedWSJ/dictionary_lemma.pkl'
    # LDAGenerator.generate_lda(corpus=corpus, run_id=tid, num_topics=n_topics, dictionary=dictionary, directory=directory)
    # LDAGenerator.predict_topic_labels(run_id=tid, corpus=corpus, directory=directory)
    # LDAGenerator.generate_topic_keywords(run_id=tid, num_keywords=10, num_topics=n_topics, directory=directory)
    # print('topic model generated.')

    # choose a clustering
    clustering = 'ClusterResults/18/labels'
    clusters = cluster_entropy.get_clusters(clustering)

    # generate KL Divergence
    directory = 'LDAResults/'
    # corpus = 'ProcessedWSJ/bow_lemma.pkl'
    corpus = 'ProcessedWSJ/wsj_lemmatised.pkl'
    run_id = tid
    # model = LdaModel.load(directory + str(run_id) + '/model')
    # KLs = KLDivergence.corpus_KL(model, corpus, n_words=10)
    # fp = open('LDAResults/' + str(run_id) + '/corpus_KL.lst', 'wb')
    # pickle.dump(KLs, fp)
    # fp.close()
    # print('KL:')
    # print(sum(KLs)/n_topics)
    #
    # # generate perplexity
    # corpus_bow = pickle.load(open(corpus, 'rb'))
    # perplexity = model.log_perplexity(corpus_bow)
    # fp = open('LDAResults/' + str(run_id) + '/perplexity.int', 'wb')
    # pickle.dump(perplexity, fp)
    # fp.close()
    # print('perplexity:')
    # print(perplexity)

    # generate coherence
    model = LdaModel.load(directory + str(run_id) + '/model')
    # coherences = topic_measures.pairwise_coherence(model, corpus, beta=0.1, n_words=10)
    # fp = open('LDAResults/' + str(run_id) + '/pairwise_coherence.lst', 'wb')
    # pickle.dump(coherences, fp)
    # fp.close()
    # print('coherence:')
    # print(sum(coherences)/n_topics)

    # cm = CoherenceModel(model=model, texts=corpus, coherence='c_npmi')
    # coherence = cm.get_coherence()
    # coherences = cm.get_coherence_per_topic()
    # print('coherence:', coherence)
    # fp = open('LDAResults/' + str(run_id) + '/npmi_coherence.lst', 'wb')
    # pickle.dump(coherences, fp)
    # fp.close()

    # generate cluster entropy
    entropies = cluster_entropy.compute_cluster_entropy(clusters, model, corpus, n_words=10)
    fp = open(directory + str(run_id) + '/cluster_entropy.lst', 'wb')
    pickle.dump(entropies, fp)
    fp.close()
    print('cluster entropy:')
    print(sum(entropies)/n_topics)


# run_measures(n_topics=10, tid=18)
#
run_measures(n_topics=60, tid=38)

run_measures(n_topics=70, tid=39)
run_measures(n_topics=80, tid=40)
run_measures(n_topics=90, tid=41)
run_measures(n_topics=100, tid=42)


# n_topics = 10
# for tid in range(36,43):
#     print('tid=',tid)
#     run_measures(n_topics=n_topics, tid=tid)
#     n_topics += 10
#
# n_topics=200
# for tid in range(44,46):
#     print('tid=',tid)
#     run_measures(n_topics=n_topics, tid=tid)
#     n_topics += 100

# # rank topics using 40 topics
# directory = 'LDAResults/'
# corpus = 'ProcessedWSJ/bow_lemma.pkl'
# run_id = 36
# fp = open(directory + str(run_id) + '/corpus_KL.lst', 'rb')
# KLs = pickle.load(fp)
# fp.close()
# KLs_id = sorted(zip(KLs, list(range(40))), key=lambda x:x[0], reverse=True)
# fp = open('topic_rank_KL.txt','w')
# for score, id in KLs_id:
#     fp.write(str(id)+'  '+str(score))
#     fp.write('\n')
# fp.close()
#
# fp = open(directory + str(run_id) + '/wordnet_coherence.pkl', 'rb')
# coherences = pickle.load(fp)
# fp.close()
# scores = sorted(zip(coherences[1], list(range(40))), key=lambda x:x[0], reverse=True)
# fp = open('topic_rank_wordnet.txt','w')
# for score, id in scores:
#     fp.write(str(id)+'  '+str(score))
#     fp.write('\n')
# fp.close()
#
# fp = open(directory + str(run_id) + '/pairwise_coherence.lst', 'rb')
# coherences = pickle.load(fp)
# fp.close()
# scores = sorted(zip(coherences, list(range(40))), key=lambda x:x[0], reverse=True)
# fp = open('topic_rank_pairwise.txt','w')
# for score, id in scores:
#     fp.write(str(id)+'  '+str(score))
#     fp.write('\n')
# fp.close()
#
# fp = open(directory + str(run_id) + '/cluster_entropy.lst', 'rb')
# entropy = pickle.load(fp)
# fp.close()
# scores = sorted(zip(entropy, list(range(40))), key=lambda x:x[0], reverse=True)
# fp = open('topic_rank_entropy.txt','w')
# for score, id in scores:
#     fp.write(str(id)+'  '+str(score))
#     fp.write('\n')
# fp.close()