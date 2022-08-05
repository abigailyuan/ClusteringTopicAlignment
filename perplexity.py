from gensim.models import LdaModel
import pickle

# directory = 'LDAResults/'
# corpus = 'ProcessedWSJ/bow.pkl'
# run_id = 17
# model = LdaModel.load(directory + str(run_id) + '/model')
# corpus_bow = pickle.load(open(corpus, 'rb'))
# total_docs = len(corpus_bow)
# chunck = corpus_bow[:int(0.1*total_docs)]
# perplexity = model.log_perplexity(chunck, total_docs)
# print(perplexity)
# # fp = open('LDAResults/'+str(run_id)+'/perplexity.int','wb')
# # pickle.dump(perplexity,fp)
# # fp.close()
# print(type(perplexity))

# generate topic model
# import LDAGenerator

# tid = 30
# n_topics = 20
# directory = 'LDAResults/'
# corpus = 'ProcessedWSJ/bow.pkl'
# dictionary = 'ProcessedWSJ/dictionary.pkl'

#create training set
# fp = open(corpus,'rb')
# bow = pickle.load(fp)
# fp.close()
#
# train_set = bow[:-1000]
# test_set = bow[-1000:]
#
# fp = open('ProcessedWSJ/train.pkl','wb')
# pickle.dump(train_set,fp)
# fp.close()
#
# fp = open('ProcessedWSJ/test.pkl','wb')
# pickle.dump(test_set,fp)
# fp.close()
#
# train = 'ProcessedWSJ/train.pkl'
#
# LDAGenerator.generate_lda(corpus=train, run_id=tid, num_topics=n_topics, dictionary=dictionary, directory=directory)
# LDAGenerator.predict_topic_labels(run_id=tid, corpus=corpus, directory=directory)
# LDAGenerator.generate_topic_keywords(run_id=tid, num_keywords=10, num_topics=n_topics, directory=directory)
# print('topic model generated.')
#
# # calculate perplexity with test set
#
# model = LdaModel.load(directory + str(tid) + '/model')
# corpus_bow = pickle.load(open(corpus, 'rb'))
# total_docs = len(corpus_bow)
# chunck = pickle.load(open('ProcessedWSJ/test.pkl','rb'))
# perplexity = model.log_perplexity(chunck, total_docs)
# print(perplexity)