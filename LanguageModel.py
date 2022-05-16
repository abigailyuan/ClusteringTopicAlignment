import numpy


def get_documents(query, bow_corpus):
    docs = []
    for d in query:
        docs.append(bow_corpus[d])
    return docs


def unigram_model(docs, k=1):
    # create a langauge model
    LM = {}
    BOW = 0
    for i in range(10000):
        LM[i] = k

    for doc in docs:
        for word, count in doc:
            if word in LM:
                LM[word] += count
            else:
                LM[word] = count
            BOW += count

    # smoothing factor k + M occurance
    M = len(LM)
    for word in LM:
        LM[word] = (LM[word] + k) / (BOW + M)

    return LM


def to_word_probabilities(LM):
    word_probability = numpy.zeros((10000, 1))
    for word in LM:
        word_probability[word] = LM[word]
    return word_probability
