from gensim.models import LdaModel

def generateLDA(corpus, run_id, num_topics=10, dictionary = None, directory='/LDAResults/'):
    '''generate an LDA model with the selected parameters and
    save to directory specified.'''
    lda_model = LdaModel(corpus, num_topics=num_topics, id2word=dictionary, alpha='auto')

    lda_model.save(directory+str(run_id)+'/model')
    return lda_model

def predictTopicLabels(run_id, corpus, mode='1', directory='/LDAResults/'):
    '''mode = ['1','2','3'] Number represents how many top topics are considered.
    Predict the topic label based on the selected mode and
    save results to specified directory.
    '''

    lda_model = LdaModel.load(directory+str(run_id)+'/model')
    documents_topics = lda_model.get_document_topics(corpus)
    print('first document:')
    print(documents_topics[0])
    topics_r1  =[]
    if mode == '1':

        topics_r1 = [sorted(i, key=lambda x: x[1], reverse=True)[0][0] for i in
                 documents_topics]  # find the most likely topic per document
    #TODO label documents with multiple top ranked topics

    return topics_r1

def generateTopicKeywords(run_id, num_keywords=10, num_topics=10, directory='/LDAResults/'):
    '''generate topic keywords and save to result directory'''
     

    return None