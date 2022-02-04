# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
import preprocessing
import KMeansGenerator
import LDAGenerator
import Visualisation
import corpus_vectorizer

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
    corpus = 'sample1WSJ/samplewsj_stemmed.pkl'
    directory = 'sample1WSJ/'
    dictionary, bow = corpus_vectorizer.create_dictionary(corpus=corpus, directory=directory)
    corpus_vectorizer.tfidf_vectorize(corpus, dictionary, directory=directory)

    corpus_vectorizer.doc2vec_vectorize(corpus, directory=directory)

    # generate K-Means clustering run





    return 0


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    work_pipeline()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
