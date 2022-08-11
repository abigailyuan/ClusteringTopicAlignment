import pickle

from nltk.stem import WordNetLemmatizer

def remove_digits(token):
    new_token = ''
    for c in token:
        if c.isalpha():
            new_token += c
    return new_token


# load queries
fp = open('10000.topics', 'r')
queries = fp.readlines()
fp.close()


# parse it into tokens
processed_queries = []
for query in queries:
    query = query[query.find(':'):]
    query = query.split(' ')

    # remove digits
    query = [remove_digits(token) for token in query]
    processed_queries.append(query)

# lemmatise tokens
lemmatizer = WordNetLemmatizer()
for i in range(len(processed_queries)):
    query = processed_queries[i]
    query = [lemmatizer.lemmatize(token, pos='n') for token in query]
    processed_queries[i] = query

# save as a list
fp = open('tokenized_queries.pkl','wb')
pickle.dump(processed_queries, fp)
fp.close()