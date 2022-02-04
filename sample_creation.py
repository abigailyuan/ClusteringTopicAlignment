import pickle
import os

directory = 'ProcessedWSJ/'
files = ['wsj_cleaned.pkl', 'wsj_index.pkl','wsj_raw.pkl','wsj_stemmed.pkl','wsj_tokenized.pkl']

os.mkdir('sample1WSJ')
for file in files:
    filename = directory+file
    object = pickle.load(open(filename, 'rb'))
    sample = object[:len(object)//100]
    pickle.dump(sample, open(directory+'sampleWSJ/sample'+file, 'wb'))
    print(len(sample))
    print(len(object))


