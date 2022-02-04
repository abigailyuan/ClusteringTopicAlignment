import pickle

directory = 'ProcessedWSJ/'
files = ['wsj_cleaned.pkl', 'wsj_index.pkl','wsj_raw.pkl','wsj_stemmed.pkl','wsj_tokenized.pkl']

for file in files:
    filename = directory+file
    object = pickle.load(open(filename, 'rb'))
    print(file)
    print(type(object))
    print(object[0])
