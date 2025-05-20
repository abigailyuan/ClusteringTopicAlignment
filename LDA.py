# lda_wsj_with_matrix.py

import os
import pickle

# Install prerequisites if you haven’t already:
#   pip install gensim nltk

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

nltk.download('punkt_tab')

import numpy as np
import gensim
from gensim.corpora import Dictionary
from gensim.models.ldamodel import LdaModel
from gensim.models.ldamulticore import LdaMulticore

from Preprocessing.wsj import WSJ

def preprocess(docs):
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    stop_words = set(stopwords.words('english'))

    processed = []
    for doc in docs:
        tokens = word_tokenize(doc.lower())
        filtered = [tok for tok in tokens if tok.isalpha() and tok not in stop_words]
        processed.append(filtered)
    return processed

def train_lda(texts, num_topics=50, passes=10, no_below=5, no_above=0.5):
    dictionary = Dictionary(texts)
    dictionary.filter_extremes(no_below=no_below, no_above=no_above)
    corpus = [dictionary.doc2bow(text) for text in texts]
    lda = LdaModel(
        corpus=corpus,
        id2word=dictionary,
        num_topics=num_topics,
        passes=passes,
        random_state=42
    )
    return lda, dictionary, corpus

def build_topic_doc_matrix(lda: LdaModel, corpus: list, num_topics: int) -> np.ndarray:
    """
    Build a matrix of shape (num_topics, num_documents),
    where entry [i,j] is P(topic_i | document_j).
    """
    num_docs = len(corpus)
    mat = np.zeros((num_topics, num_docs), dtype=float)
    for j, bow in enumerate(corpus):
        # get_document_topics with minimum_probability=0 ensures all topics appear
        for topic_id, prob in lda.get_document_topics(bow, minimum_probability=0):
            mat[topic_id, j] = prob
    return mat

def main():
    WSJ_PICKLE  = "ProcessedWSJ/wsj_raw.pkl"
    OUTPUT_DIR  = "Results/LDA/"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. Load and preprocess
    wsj   = WSJ()
    docs  = wsj.load(WSJ_PICKLE)
    texts = preprocess(docs)
    print(f"Loaded & preprocessed {len(texts)} documents.")

    # 2. Train LDA
    NUM_TOPICS = 50
    lda, dictionary, corpus = train_lda(texts, num_topics=NUM_TOPICS, passes=10)
    print("LDA training complete.")

    # 3. Save model and dictionary
    lda.save(os.path.join(OUTPUT_DIR, "wsj_lda50.model"))
    dictionary.save(os.path.join(OUTPUT_DIR, "wsj_dictionary.dict"))
    pickle.dump(corpus, open(os.path.join(OUTPUT_DIR, "wsj_corpus.pkl"), 'wb'))

    # 4. Build topic–document matrix
    td_matrix = build_topic_doc_matrix(lda, corpus, NUM_TOPICS)
    print("Built topic–document matrix:", td_matrix.shape)
    #     → (50, num_documents)

    # 5. Save the matrix
    with open(os.path.join(OUTPUT_DIR, "wsj_topic_doc_matrix.pkl"), "wb") as f:
        pickle.dump(td_matrix, f)
    print("Saved topic–document matrix to Results/LDA/wsj_topic_doc_matrix.pkl")

    # (Optional) print out matrix snippet
    print("Sample (first 5 topics × first 5 docs):")
    print(td_matrix[:5, :5])
    
    print("Gensim version:", gensim.__version__)
    print("Gensim path:   ", gensim.__file__)

    print("Your model type:", type(lda))

    print("Is instance of core LdaModel?    ", isinstance(lda, LdaModel))
    print("Is instance of LdaMulticore?     ", isinstance(lda, LdaMulticore))
    print("Has attribute `state`?           ", hasattr(lda, "state"))
    print("`dir(lda)` snippet:               ", [attr for attr in dir(lda) if attr.startswith("state")])


if __name__ == "__main__":
    main()
