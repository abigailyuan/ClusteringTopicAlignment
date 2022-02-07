# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
import preprocessing
import KMeansGenerator
import LDAGenerator
import Visualisation
import corpus_vectorizer
import pickle

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press ⌘F8 to toggle the breakpoint.


def work_pipeline():


    # generate WSJ collection

    # corpus_name = '../datasets/wsj.gz'
    # corpus = preprocessing.parse_wsj_corpus(corpus_name)
    # corpus = preprocessing.clean_corpus(corpus)
    # corpus = preprocessing.tokenize_corpus(corpus)
    # corpus = preprocessing.remove_stopwords(corpus)
    # corpus = preprocessing.lemmatize_corpus(corpus)


    # vectorize corpus
    # corpus = 'sample1WSJ/samplewsj_stemmed.pkl'
    # bow_corpus = 'sample1WSJ/bow.pkl'
    # directory = 'sample1WSJ/'
    # corpus_vectorizer.create_dictionary(filename=corpus, directory=directory)
    # dictionary = directory+'dictionary.pkl'
    # corpus_vectorizer.tfidf_vectorize(bow_corpus, dictionary, directory=directory)
    #
    # corpus_vectorizer.doc2vec_vectorize(corpus, directory=directory)

    # generate K-Means clustering run
    cid = 0
    directory = 'SampleResults/ClusterResults/'
    corpus = 'sample1WSJ/samplewsj_stemmed.pkl'

    bow_corpus = 'sample1WSJ/bow.pkl'
    #KMeansGenerator.generate_k_means(dense_corpus='sample1WSJ/dense_corpus.pkl',run_id=cid, directory=directory)
    #KMeansGenerator.predict_cluster_labels(run_id=cid, directory=directory)
    KMeansGenerator.generate_cluster_keywords(run_id=cid, corpus=bow_corpus, mode='centroid', directory=directory)
    KMeansGenerator.generate_cluster_keywords(run_id=cid, corpus=bow_corpus, mode='cluster', directory=directory)





    return 0


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    work_pipeline()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
