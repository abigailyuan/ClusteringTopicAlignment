# Topic Mapping & Specificity Pipeline

This repository provides a unified pipeline to:

1. **Generate embeddings** for multiple text collections using different embedding methods (Doc2Vec, Sentence-BERT, RepLLaMA).
2. **Train topic models** (LDA or BERTopic) with 50 topics on each collection.
3. **Map** embedding features to topic-document distributions.
4. **Calculate specificity** scores for each topic based on embedding-to-topic mappings.
5. **Visualize** mapped vs. unmapped topics sorted by specificity.

---

## 🗂️ Project Structure

```plaintext
project_root/
├── Embeddings/
│   ├── embed_doc2vec.py
│   ├── embed_sbert.py
│   ├── embed_repllama.py
│
├── TopicModels/
│   ├── lda_train.py
│   ├── bertopic_train.py
│
├── Mapping/
│   ├── map_repllama_lda.py
│   ├── map_repllama_bertopic.py
│
├── Visualisation/
│   └── visualise_mapping.py
│
├── embed_utils.py
├── preprocess.py
├── run_pipeline.py
└── README.md
```

**Processed Data** go to:

```
Processedwiki/
Processed20ng/
Processedwsj/
```

Each contains:

* `<collection>_<method>_corpus_embeddings.pkl`
* `<collection>_<method>_projected_features.pkl`

**Results** go to:

```
Results/
├── LDA/
│   ├── wiki_dictionary.dict
│   ├── wiki_corpus.pkl
│   ├── wiki_lda50.model
│   └── ...
│
├── BERTopic/
│   ├── wiki_bertopic_model/   
│   └── ...
│
├── wiki_repllama_lda_mapping.pkl
├── wiki_repllama_bertopic_mapping.pkl
├── ...
├── lda_repllama.pdf
└── bertopic_repllama.pdf
```

---

## ⚙️ Installation & Dependencies

1. **Clone repository**:

   ```bash
   git clone <repo_url>
   cd project_root
   ```

2. **Create Python environment** (recommended):

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

> **requirements.txt** should include:
>
> * `gensim`
> * `sentence-transformers`
> * `bertopic`
> * `scikit-learn`
> * `spacy` (+ `en_core_web_sm` model)
> * `torch`
> * `numpy`
> * `matplotlib`

---

## 🚀 Usage

Run the full pipeline with a single command. Specify the embedding method (`doc2vec`, `sbert`, or `repllama`) and the topic model (`lda` or `bertopic`):

```bash
python run_pipeline.py --lang_model <doc2vec|sbert|repllama> --topic_model <lda|bertopic>
```

**Examples:**

* Doc2Vec + LDA:

  ```bash
  python run_pipeline.py --lang_model doc2vec --topic_model lda
  ```

* SBERT + LDA:

  ```bash
  python run_pipeline.py --lang_model sbert --topic_model lda
  ```

* RepLLaMA + BERTopic:

  ```bash
  python run_pipeline.py --lang_model repllama --topic_model bertopic
  ```

This will:

1. Embed each collection (or skip if cached).
2. Train the selected topic model.
3. Map RepLLaMA features to topics (if `--lang_model repllama`).
4. Generate a scatter plot saved under `Results/`.

---

## 🔬 Experiments Explained

We perform a controlled comparison across **three text collections**:

* **WSJ**: Wall Street Journal news articles.
* **20NG**: 20 Newsgroups posts.
* **Wiki**: Wikipedia article snippets.

**Embedding Methods**:

* **Doc2Vec**: Unsupervised doc embeddings via gensim.
* **SBERT**: Pretrained Sentence-BERT embeddings.
* **RepLLaMA**: LLM-based embeddings from a LoRA-fine-tuned Llama-2 model.

**Topic Models**:

* **LDA** (gensim): Classic probabilistic topic model.
* **BERTopic**: Transformer-based topic extraction with embeddings + clustering.

**Mapping**:

* For **RepLLaMA**, we map each embedding feature to the most aligned topic via a greedy algorithm, saving `*_repllama_lda_mapping.pkl` or `*_repllama_bertopic_mapping.pkl`.

**Specificity Scores**:

* Calculate per-topic scores by thresholding document-topic probabilities (GMM/median/percentile) and measuring how strongly topics concentrate on a subset of docs.

**Visualization**:

* Scatter plots of specificity: mapped features (alpha=1.0) vs. unmapped (alpha=0.3), sorted by score, across all collections.

This setup allows us to compare how different embedding and topic modeling choices affect topic specificity and feature alignment.

---

## 🔄 Extending the Pipeline

* **Add new embeddings**: create `embed_newmethod.py` using `embed_utils.py` + `preprocess.py`.
* **Add more topic models**: implement `TopicModels/newmodel_train.py` and mapping in `Mapping/`.
* **Customize metrics**: modify `topic_specificity.py` or `topic_specificity_bertopic.py`.

---

## 📄 License

[MIT License](LICENSE)
