from topic_specificity import calculate_specificity_for_all_topics
import pickle

 # 1. Load and preprocess
wsj   = WSJ()
docs  = wsj.load(WSJ_PICKLE)
texts = preprocess(docs)
print(f"Loaded & preprocessed {len(texts)} documents.")

dictionary = pickle.load(open('Results/LDA/wsj_dictionary.dict','rb'))
corpus = [dictionary.doc2bow(text) for text in texts]

# 2. Load LDA model
lda = pickle.load(open('Results/LDA/wsj_lda50.model','rb'))

spec_scores = calculate_specificity_for_all_topics(
    model=lda,
    corpus=corpus,
    mode='lda',
    threshold_mode='gmm',         # options: 'median', 'percentile', 'gmm'
    specificity_mode='sqrt'       # options: 'diff', 'sqrt'
)

print("Specificity scores:", spec_scores)