# NLP-Reco

A hotel recommendation system built around Turkish text. It extracts semantic topics from hotel descriptions using BERTopic, then uses those topic embeddings to find similar hotels, ranked by a popularity signal. The repo ships with a synthetic dataset so you can run the whole pipeline locally without any proprietary data.

---

## Why this is interesting

Most recommendation systems either rely entirely on collaborative filtering (what other users clicked) or on structured metadata (star rating, price, location). This project takes a different approach: it reads the *text* of what guests actually wrote about a hotel and uses that to define similarity.

The harder part is the language. Turkish is agglutinative — a single word like *otellerin* encodes what English needs three words to say ("of the hotels"). If you run standard tokenization without morphological analysis, you end up treating *otelde*, *otelin*, and *otellerden* as completely different tokens when they all stem from *otel*. The `TurkishTextProcessor` uses [Zemberek](https://github.com/ahmetaa/zemberek-nlp) to handle this properly, reducing each word to its dictionary lemma before topic modeling runs.

The recommendation logic itself is a two-stage pipeline that mirrors how production systems at companies like Spotify and Netflix are structured: a fast retrieval step narrows the candidate set using embedding similarity, then a separate ranking step re-orders those candidates using a different signal (here, a popularity score based on review count and image count). Keeping these stages separate makes each one easier to reason about and swap out independently.

---

## Quickstart

Install [uv](https://docs.astral.sh/uv/) if you don't have it:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Clone the repo and install dependencies:

```bash
git clone <repo-url>
cd nlp-reco
uv sync
```

Run the pipeline. No Java required:

```bash
uv run python main.py --processor fallback
```

Output is written to `data/hotel_recommendations.csv`. Each row is a `(source_hotel, recommended_hotel, rank)` triple.

To use the full Turkish morphological processor (requires Java 11+, see below):

```bash
uv run python main.py --processor turkish
```

---

## Text processors

The pipeline supports two backends, selectable at runtime via `--processor`.

| Mode | Flag | What it does | Requires |
|---|---|---|---|
| `turkish` | `--processor turkish` | Zemberek morphological lemmatization, language-aware normalization | Java 11+ |
| `fallback` | `--processor fallback` | HTML cleaning, lowercase, whitespace tokenization, stopword removal | Nothing |

The fallback processor is enough to run the complete pipeline and is the right starting point if you want to try the code quickly or adapt it to a different language. The Turkish processor gives meaningfully better topic coherence because it reduces inflected forms to their stems.

Both share the same abstract interface (`BaseTextProcessor`), so swapping them has no effect on any downstream code.

**Apple Silicon note:** Java 8 on ARM64 (M1/M2/M3) has a known JVM crash in the `jshort_disjoint_arraycopy` stub — this is an Oracle Java 8 / ARM64 JIT issue, not fixable in Python. Use Java 11+:

```bash
brew install --cask temurin@11
```

---

## How the pipeline works

### 1. Feature engineering

Two scores are computed from the hotel metadata and used at ranking time:

**Weighted star rating** — a weighted sum of 8 rating dimensions. General impression carries the most weight (30%), followed by food (25%), then cleaning, location, and service (10% each), with wifi, condition, and price at 5% each. The result is capped at the 99th percentile to handle outliers and normalized to a 1–5 scale.

**Popularity score** — `0.7 * review_count + 0.3 * image_count`, also capped at the 99th percentile and normalized to [0, 1]. This is a proxy for how well-documented a hotel is, not necessarily how good it is — a hotel with 2000 reviews is almost certainly more data-rich than one with 20.

### 2. Text preprocessing

Each hotel has multiple text descriptions that get merged into one document per hotel. Before that merge, the text goes through a cleaning pipeline: HTML tags removed, Turkish normalization, tokenization, stopword filtering, and lemmatization. Two terms — *misafir* (guest) and *km* — are removed after merging because they show up in nearly every document and carry no discriminating information.

### 3. Topic modeling

BERTopic runs on the merged, lemmatized descriptions using a multilingual sentence-transformer model (`paraphrase-multilingual-MiniLM-L12-v2`). Each hotel gets assigned to a topic, and BERTopic produces an embedding vector for each topic centroid. That centroid vector is what we use for similarity — not a per-hotel embedding, but a per-topic one. Hotels in the same topic share an identical embedding.

Hotels that BERTopic assigns to the outlier topic (-1) get no valid topic embedding. Rather than dropping them, the pipeline falls back to ranking them by popularity score alone.

### 4. Recommendations

**Stage 1 — Retrieval:** Cosine similarity is computed across all valid topic embeddings. For each hotel, the 10 most similar hotels are retrieved. Hotels in the same topic will have cosine similarity of 1.0; cross-topic similarity is lower.

**Stage 2 — Ranking:** The 10 candidates are re-ranked by `popularity_score` in descending order. This means among topically similar hotels, the more popular ones surface first.

### 5. Validation

The only validation metric currently implemented is the **city divergence ratio**: what fraction of recommendations point to hotels in a different city than the source hotel. Lower is better if you want geographically coherent recommendations.

On the mock dataset (50 hotels across 6 cities, ~8 per city) this ratio is around 86%, which is close to what random chance predicts — with only 8 same-city candidates out of 49 total, you'd expect `1 - 8/49 ≈ 85.7%` cross-city recommendations even from a random model. On the original private dataset with hundreds of hotels per city, the ratio dropped to around 24%, meaning the topic model was picking up city-specific vocabulary in the descriptions (Cappadocia reviews talk about fairy chimneys and cave hotels; Bodrum reviews talk about yacht harbors and the Aegean).

---

## Project structure

```
nlp-reco/
├── main.py                    # entry point, --processor flag
├── pyproject.toml             # uv-managed dependencies
├── src/
│   ├── config.py              # paths, weights, defaults
│   ├── data_preprocessing.py  # loading, cleaning, weighted rating, popularity score
│   ├── topic_modeling.py      # BERTopic training, embedding extraction
│   ├── recommendation.py      # cosine similarity retrieval, popularity ranking
│   └── validation.py          # city divergence metric
├── utils/
│   └── text_processing.py     # BaseTextProcessor ABC, TurkishTextProcessor,
│                              # FallbackTextProcessor, get_text_processor() factory
├── data/
│   ├── hotel_info.csv         # 50 synthetic Turkish hotels
│   ├── hotel_desc.csv         # ~178 synthetic Turkish reviews
│   ├── session_data.csv       # 500 synthetic booking sessions
│   └── stopwords.txt          # Turkish stopword list
├── scripts/
│   └── generate_mock_data.py  # regenerates the synthetic dataset (seed=42)
├── notebooks/
│   └── demo.ipynb             # step-by-step walkthrough
├── models/                    # BERTopic model is saved here after training
└── jars/
    └── zemberek-full.jar      # only needed for --processor turkish
```

---

## Adapting to a different domain

The pipeline itself is not hotel-specific. To point it at restaurant reviews or product descriptions, three things need to change:

1. **Data** — provide three CSVs with the same column schema that `load_data()` expects (one with item metadata, one with text descriptions, one with session/interaction data).
2. **Text processor** — subclass `BaseTextProcessor` with whatever cleaning and lemmatization makes sense for your language, then add a branch in `get_text_processor()`.
3. **Stopwords** — replace `data/stopwords.txt` with a list appropriate for your domain and language.

The `process_text()` pipeline (clean → normalize → tokenize → remove stopwords → lemmatize → join) is defined on the base class and runs without modification regardless of which processor is active.

---

## Dependencies

Managed with [uv](https://docs.astral.sh/uv/).

```bash
uv sync          # installs core dependencies into .venv
uv sync --dev    # also installs pytest, jupyter, ipykernel
```

Core: `bertopic`, `scikit-learn`, `pandas`, `numpy`, `spacy`, `langdetect`, `tqdm`.
Optional: `jpype1` (bridges Python to the Zemberek Java library — only needed for `--processor turkish`).
