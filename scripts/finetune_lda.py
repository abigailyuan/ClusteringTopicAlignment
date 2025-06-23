import pickle
import nltk
import re
from nltk.corpus import stopwords

import spacy
import plotille
from gensim.utils import simple_preprocess
from gensim.models.phrases import Phrases, Phraser
from gensim.corpora import Dictionary
from gensim.models.ldamodel import LdaModel
from gensim.models.ldamulticore import LdaMulticore
from gensim.models.coherencemodel import CoherenceModel

from topic_specificity import calculate_specificity_for_all_topics

nltk.download('stopwords', quiet=True)
EN_STOP = set(stopwords.words('english'))

# extra generic stopwords you often see in news/forums
GENERIC_EXTRA = {'said', 'would', 'could', 'also'}

# per-collection high-freq noise terms
COLLECTION_EXTRAS = {
    'wsj':  {'mr', 'ms', 'company', 'new', 'york'},
    '20ng': {'subject', 'lines', 'organization', 'writes'},
    'wiki': {'edit', 'redirect', 'page', 'wikipedia', 'category'},
}

# load spaCy model for lemmatization
try:
    nlp = spacy.load('en_core_web_sm', disable=['parser','ner'])
except OSError:
    import spacy.cli
    spacy.cli.download("en_core_web_sm")
    nlp = spacy.load('en_core_web_sm', disable=['parser','ner'])



def preprocess_docs(docs, collection: str):
    """
    docs        : list of raw document strings
    collection  : one of 'wsj','20ng','wiki'
    returns     : list of token lists
    """
    coll = collection.lower()
    if coll not in COLLECTION_EXTRAS:
        raise ValueError(f"Unknown collection {collection!r}, choose from {list(COLLECTION_EXTRAS)}")

    # build stop-list
    stop_words = EN_STOP | GENERIC_EXTRA | COLLECTION_EXTRAS[coll]

    # (1) tokenize & basic cleanup
    tokenized = []
    for doc in docs:
        text = doc.lower()
        text = re.sub(r'\S+@\S+', ' ', text)       # strip emails
        text = re.sub(r'http\S+', ' ', text)       # strip URLs
        text = re.sub(r'\d+', ' ', text)           # strip digits
        toks = simple_preprocess(text, deacc=True, min_len=3)
        toks = [t for t in toks if t not in stop_words]
        tokenized.append(toks)

    # (2) detect bigrams & trigrams
    bigram = Phraser(Phrases(tokenized, min_count=20, threshold=10))
    trigram = Phraser(Phrases(bigram[tokenized], min_count=10, threshold=5))
    tokenized = [trigram[bigram[doc]] for doc in tokenized]

    # (3) lemmatize, keep only content words
    lemmatized = []
    for doc in tokenized:
        sp_doc = nlp(" ".join(doc))
        lemmas = [
            token.lemma_ for token in sp_doc
            if token.pos_ in {'NOUN','VERB','ADJ','ADV'}
               and token.lemma_ not in stop_words
        ]
        lemmatized.append(lemmas)

    return lemmatized

def preprocess_docs_tight_phrases(docs, collection='wiki'):
    # 1. do your usual tokenization & stopword removal
    coll = collection.lower()
    if coll not in COLLECTION_EXTRAS:
        raise ValueError(f"Unknown collection {collection!r}, choose from {list(COLLECTION_EXTRAS)}")

    # build stop-list
    stop_words = EN_STOP | GENERIC_EXTRA | COLLECTION_EXTRAS[coll]

    # (1) tokenize & basic cleanup
    tokenized = []
    for doc in docs:
        text = doc.lower()
        text = re.sub(r'\S+@\S+', ' ', text)       # strip emails
        text = re.sub(r'http\S+', ' ', text)       # strip URLs
        text = re.sub(r'\d+', ' ', text)           # strip digits
        toks = simple_preprocess(text, deacc=True, min_len=3)
        toks = [t for t in toks if t not in stop_words]
        tokenized.append(toks)
    
    # 2. phrase detection with tighter settings
    bigram  = Phraser(Phrases(tokenized,          min_count=30, threshold=20))
    trigram = Phraser(Phrases(bigram[tokenized],  min_count=15, threshold=10))
    tokenized = [trigram[bigram[doc]] for doc in tokenized]

    # 3. lemmatize as before…
    lemmatized = []
    for doc in tokenized:
        sp_doc = nlp(" ".join(doc))
        lemmas = [
            t.lemma_ for t in sp_doc
            if t.pos_ in {'NOUN','VERB','ADJ','ADV'}
               and t.lemma_ not in stop_words
        ]
        lemmatized.append(lemmas)

    return lemmatized



# 1. Load raw docs
raw = pickle.load(open('ProcessedWIKI/wiki_raw.pkl','rb'))

# 2. Define your variants
def preprocess_baseline(docs):
    # your existing preprocess_docs(docs, '20ng')
    return preprocess_docs(docs, 'wiki')

def preprocess_variant_a(docs):
    # e.g. increase stopword list dynamically
    texts = preprocess_docs(docs, 'wiki')
    # dynamically remove top-10 most frequent tokens
    from collections import Counter
    freq = Counter(tok for doc in texts for tok in doc)
    top10 = {w for w,_ in freq.most_common(10)}
    return [[t for t in doc if t not in top10] for doc in texts]

def preprocess_variant_b(docs):
    # e.g. tighten phrase thresholds
    # you can copy your preprocess_docs and bump threshold
    return preprocess_docs_tight_phrases(docs, 'wiki')

variants = {
    'baseline': preprocess_baseline,
    'remove_top10': preprocess_variant_a,
    'tighter_phrases': preprocess_variant_b,
}

# 3. Fixed LDA training parameters
lda_kwargs = dict(
    num_topics=50,
    passes=10,
    alpha='asymmetric',
    eta='auto',
    random_state=42
)

results = {}
for name, prep in variants.items():
    # a) preprocess
    texts = prep(raw)
    # b) build dict & corpus
    dictionary = Dictionary(texts)
    dictionary.filter_extremes(no_below=5, no_above=0.5)
    corpus = [dictionary.doc2bow(doc) for doc in texts]
    # c) train LDA
    lda = LdaModel(corpus=corpus, id2word=dictionary, **lda_kwargs)
    # d) compute specificity
    scores = calculate_specificity_for_all_topics(
                model=lda,
                corpus=corpus,
                mode='lda',
                threshold_mode='gmm',
                specificity_mode='diff')
    results[name] = scores
    print(f"{name:15s} → specificity = {scores}")
    

# 4. Compare
for name, scores in results.items():
    xs = list(range(len(scores)))
    ys = scores

    # Create a Figure and set size + labels
    fig = plotille.Figure()
    fig.width  = 60          # characters wide
    fig.height = 15          # characters tall
    fig.x_label = "Topic #"
    fig.y_label = "Specificity"

    # Plot the points
    fig.scatter(xs, ys)

    # Render it
    print(f"\n===== {name} =====")
    print(fig.show())