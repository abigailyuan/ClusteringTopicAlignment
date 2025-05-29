import re
import spacy
import pickle
from typing import List, Iterable
from pathlib import Path
import argparse

# Load spaCy model once, disabling unneeded components for speed
nlp = spacy.load("en_core_web_sm", disable=["parser", "ner", "textcat"])
STOP_WORDS = set(nlp.Defaults.stop_words).union({
    'r', 'mr', 'would'
})

def clean_and_tokenize(doc: str) -> List[str]:
    """Clean and tokenize a single document."""
    doc = re.sub(r'<[^>]+>', ' ', doc)
    doc = re.sub(r'[^\s]+@[^\s]+', ' ', doc)
    doc = re.sub(r'https?://\S+|www\.\S+', ' ', doc)
    doc = re.sub(r'\{.*?\}', ' ', doc)
    doc = re.sub(r'\s+', ' ', doc).strip()

    tokens = []
    for token in nlp(doc.lower()):
        if (
            token.is_alpha and
            token.text not in STOP_WORDS and
            len(token.text) > 2
        ):
            tokens.append(token.lemma_)
    return tokens


def preprocess_corpus(
    raw_docs: Iterable[str],
    save_path: str = None
) -> List[str]:
    """Clean and tokenize each document, returning list of strings."""
    processed = [clean_and_tokenize(doc) for doc in raw_docs]

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "wb") as f:
            pickle.dump(processed, f)
        print(f"[Step 1.5] Processed corpus saved to {save_path}")

    return processed

def load_raw_corpus(path: str) -> List[str]:
    """Load a raw corpus from a pickled list of strings."""
    with open(path, "rb") as f:
        return pickle.load(f)

def main():
    parser = argparse.ArgumentParser(description="Step 1.5 - Clean and preprocess corpus for LDA.")
    parser.add_argument("--input", required=True, help="Path to raw corpus pickle file (list of strings).")
    parser.add_argument("--output", required=True, help="Path to save cleaned corpus pickle file (list of strings).")
    args = parser.parse_args()

    raw_corpus = load_raw_corpus(args.input)
    preprocess_corpus(raw_corpus, save_path=args.output)

if __name__ == "__main__":
    main()
