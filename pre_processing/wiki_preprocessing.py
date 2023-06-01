# For processing wikipedia dump collection
import json
import os
import pickle
from collections import defaultdict as dd
import preprocessing

# load collection and remove documents less than 200 chars
dir = 'Resources/'
corpus = dd(list)
for file in os.listdir(dir):
    fp = open(dir+file, 'r')
    collection = [d['content'] for d in json.load(fp)['documents'] if len(d['content']) >= 1000]
    corpus[file[:-5]] = collection[-1000:]
# os.mkdir('ProcessedWiki/')
pickle.dump(corpus, open('ProcessedWiki/sample_1000.dict', 'wb'))

# tokenise the documents and save a copy of the labels
corpus = pickle.load(open('ProcessedWiki/sample_1000.dict', 'rb'))
collection = []
for i in corpus:
    collection += corpus[i]

cleaned = preprocessing.clean_corpus(collection, directory='ProcessedWiki/')
tokend = preprocessing.tokenize_corpus(cleaned)
nostop = preprocessing.remove_stopwords(tokend, directory='ProcessedWiki/')
lemmatized = preprocessing.lemmatize_corpus(nostop, directory='ProcessedWiki/')

