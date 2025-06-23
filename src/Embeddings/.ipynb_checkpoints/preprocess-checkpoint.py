import re
import spacy
from typing import List

# Load spaCy pipeline once
nlp = spacy.load("en_core_web_sm", disable=["parser","ner","textcat"])
STOP_WORDS = nlp.Defaults.stop_words


def clean_doc(doc: str) -> List[str]:
    """
    Clean and tokenize a document for Doc2Vec.
    - Strips HTML, URLs, JSON-like braces
    - Lowercases, removes stopwords and non-alphabetic tokens
    - Lemmatizes remaining tokens
    """
    text = re.sub(r"<[^>]+>|https?://\S+|\{.*?\}", " ", doc)
    tokens = []
    for tok in nlp(text.lower()):
        if not tok.is_alpha or tok.text in STOP_WORDS:
            continue
        tokens.append(tok.lemma_)
    return tokens


def prepare_sbert(doc: str) -> str:
    """
    Minimal cleaning for SBERT and RepLLaMA embeddings:
    - Strips HTML, URLs, JSON-like braces
    """
    return re.sub(r"<[^>]+>|https?://\S+|\{.*?\}", " ", doc).strip()