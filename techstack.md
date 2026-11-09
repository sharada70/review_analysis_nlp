# Tech stack — ReviewLab

## Overview

Beginner NLP demo that analyzes synthetic product reviews, trains a small sentiment classifier, and ships a static multi-product dashboard on GitHub Pages.

## Language & runtime

| Layer | Choice |
| --- | --- |
| Analysis / build | Python 3 |
| Hosted UI | Static HTML / CSS / vanilla JavaScript |
| Legacy local UI | Streamlit (`app.py`) |

## Python libraries (`requirements.txt`)

| Package | Role |
| --- | --- |
| **pandas** (`>=2.2,<3`) | Load CSV, filter, aggregate metrics |
| **pyarrow** (`>=18.1,<19`) | Pandas backend (pinned for Windows compatibility) |
| **nltk** (`>=3.8`) | Tokenize reviews, remove stopwords |
| **scikit-learn** (`>=1.4`) | TF-IDF vectorizer + Multinomial Naive Bayes pipeline, cross-validation |
| **matplotlib** (`>=3.8`) | Charts in the Streamlit UI |
| **plotly** (`>=5.20`) | Interactive charts in the Streamlit UI |
| **streamlit** (`>=1.36`) | Optional local dashboard (legacy single-product era) |

## Static site (primary demo)

| Piece | Choice |
| --- | --- |
| Build | `build_site.py` → JSON under `docs/data/` |
| Pages | `docs/index.html`, `docs/styles.css`, `docs/app.js` |
| Charts | [Chart.js](https://www.chartjs.org/) `4.4.8` (CDN) |
| Fonts | Google Fonts — **Sora** (display), **Figtree** (UI) |
| Hosting | GitHub Pages (`docs/` folder, `.nojekyll`) |
| Preview | `python -m http.server` serving `docs/` |

## Data formats

- **Source:** CSV (`data/reviews.csv`) — columns include product, date, reviewer, rating, sentiment, review text
- **Published:** JSON per product + `products.json` for the static frontend
- **Labels:** `Negative` / `Neutral` / `Positive` (supervised sentiment classes)

## Local commands

```text
# Regenerate demo reviews
py generate_reviews.py

# Rebuild GitHub Pages JSON from the NLP pipeline
py build_site.py

# Preview static site
py -m http.server 8080 --directory docs

# Optional Streamlit UI
py -m streamlit run app.py
```

---

## NLP explained (for demos / viva)

### What problem we solve

Customers leave free-text reviews. NLP turns that text into:

1. **Descriptive insights** — how many reviews are positive/negative, which themes show up, how ratings change over months
2. **A classifier** — given a new sentence, predict Negative / Neutral / Positive

We use classic, explainable methods (not deep learning / LLMs), so every step is easy to justify.

### End-to-end pipeline

```text
CSV reviews
   │
   ├─► Cleaning (NLTK tokenize + stopwords)
   │      └─► Word counts, distinctive words, theme categories
   │
   └─► TF-IDF features (scikit-learn)
          └─► Multinomial Naive Bayes
                 ├─► 5-fold CV accuracy + confusion matrix
                 └─► Exported weights → live scoring in the browser
```

### 1. Text preprocessing (NLTK)

**Goal:** turn messy sentences into a clean list of content words.

Steps in `analyze.tokenize()`:

1. Lowercase the review
2. Split into tokens with NLTK `word_tokenize` (falls back to a simple regex if NLTK data is missing)
3. Keep alphabetic tokens longer than 2 characters
4. Drop **stopwords** (common words like *the*, *and*, *is* that add little meaning)
5. Also drop project-specific filler / product names (`earbuds`, `aurora`, `nova`, …) so charts show opinions, not brand mentions

**How to explain it:**  
“Before counting or classifying, we normalize text and remove noise words so the model focuses on opinion-bearing terms like *battery*, *bass*, or *comfortable*.”

### 2. Exploratory analysis (no ML yet)

These are statistics on cleaned tokens / ratings:

| Output | Meaning |
| --- | --- |
| Sentiment counts / shares | How reviews split across Negative / Neutral / Positive |
| Average rating + monthly trend | Whether quality perception improves or drops over time |
| Top words | Most frequent content words after stopword removal |
| Distinctive words (“lift”) | Words that appear more often *inside* one sentiment than outside it |
| Theme categories | Rule-based mapping of keywords → product themes |

**Distinctive-word score (lift):**  
`share of word in this sentiment − share of word in other sentiments`  
A word must appear at least twice so one review cannot dominate.

**How to explain it:**  
“Frequency shows what people mention. Lift shows what is *characteristic* of complaints vs praise.”

### 3. Theme / category extraction (rule-based NLP)

Not every NLP step needs a trained model. We map tokens to product themes with a hand-built dictionary (`CATEGORY_MAP`), e.g.:

| Category | Example keywords |
| --- | --- |
| Battery & Power | battery, charge, drains, lasts |
| Sound Quality | bass, vocals, muddy, crisp |
| Fit & Comfort | fit, comfortable, tips, painful |
| Connectivity | bluetooth, pairing, lag, drops |
| Value & Price | price, expensive, worth, cheap |
| Calls & Mic | call, mic, microphone |
| Noise Control | anc, isolation, transparency |
| Fitness & Health | steps, sleep, heart, gps (Orbit Fit Band) |

Simple plural / `-ing` normalization collapses near-duplicates (`batteries` → `battery`).

Uses:

- Per-review keyword chips + categories
- Theme frequency across the catalog
- **Improve** tab: recent critical reviews + “fix X — reviewers mention Y” opportunities

**How to explain it:**  
“Alongside the ML classifier, we use a lightweight lexicon to group related words into product themes. That is interpretable and useful for product feedback, even without deep learning.”

### 4. Feature extraction — TF-IDF

Computers need numbers, not raw text. We use scikit-learn `TfidfVectorizer`:

| Setting | Value | Why |
| --- | --- | --- |
| `ngram_range=(1, 2)` | unigrams + bigrams | Captures single words *and* short phrases (`battery dies`) |
| `stop_words="english"` | built-in English stop list | Extra cleanup inside the vectorizer |
| `min_df=2` | term must appear in ≥ 2 reviews | Drops ultra-rare noise / typos |

**TF (term frequency):** how often a term appears in *this* review.  
**IDF (inverse document frequency):** down-weights terms that appear in almost every review (less informative).  
**TF-IDF:** high when a term is important in this review *and* relatively rare across the corpus.

**How to explain it:**  
“TF-IDF turns each review into a sparse vector. Words that are common everywhere get less weight; words that stand out in a review get more weight.”

### 5. Classifier — Multinomial Naive Bayes

We use `MultinomialNB` inside a `Pipeline` with the TF-IDF step.

**Idea in plain language:**

- Learn how often each word/phrase appears under each sentiment label
- For a new review, combine those likelihoods with the class prior (how common each sentiment is)
- Pick the label with the highest probability
- “Naive” means it assumes word features are independent given the label — simple, fast, and often strong on short text like reviews

**Why Naive Bayes fits this project:**

- Works well on bag-of-words / TF-IDF text
- Small demo dataset-friendly
- Probabilities are easy to show in the UI
- Weights can be exported and scored in plain JavaScript (no Python on GitHub Pages)

**How to explain it:**  
“We train Multinomial Naive Bayes on TF-IDF features to classify review sentiment. It is a classic baseline for text classification and easy to interpret via top words per class.”

### 6. Evaluation — stratified 5-fold cross-validation

We do **not** report accuracy from predicting the same reviews we trained on.

1. Split data into 5 folds, keeping class proportions similar (`StratifiedKFold`)
2. Train on 4 folds, predict the held-out fold
3. Repeat so every review gets an honest out-of-fold prediction
4. Report overall **accuracy**, per-class **precision / recall / F1**, and a **confusion matrix**
5. Then refit the pipeline on **all** reviews for live scoring of new text

**How to explain it:**  
“Cross-validation measures generalization. The confusion matrix shows which sentiments the model mixes up — for example Neutral vs Positive.”

### 7. Model interpretability — top words per class

From `feature_log_prob_`, we list the highest-weighted unigrams/bigrams for Negative, Neutral, and Positive.

**How to explain it:**  
“This shows what the model ‘listens for.’ If Negative leans on *battery* / *painful* and Positive on *clear* / *comfortable*, the model aligns with human intuition.”

### 8. Live prediction in the browser

`export_model()` saves vocabulary, IDF, class priors, and feature log-probabilities into the product JSON.  
`docs/app.js` rebuilds a TF-IDF-style vector for typed text and scores Naive Bayes locally — so the GitHub Pages demo can classify new sentences without a backend.

### Two NLP tracks (say this clearly)

| Track | Method | Output |
| --- | --- | --- |
| **Insight / themes** | Token counts + lexicon categories | Charts, distinctive words, improvement cues |
| **Sentiment model** | TF-IDF + Multinomial NB | Accuracy, confusion matrix, live classify |

That combination is intentional: descriptive NLP for product understanding, and a supervised model for automated labeling.

### Short script you can speak

> “We start with product reviews labeled Negative, Neutral, or Positive.  
> NLTK cleans the text — tokenize, lowercase, drop stopwords.  
> From cleaned tokens we compute frequencies, distinctive words, and map keywords into themes like Battery and Sound.  
> Separately, scikit-learn builds TF-IDF features (including bigrams) and trains Multinomial Naive Bayes.  
> We evaluate with stratified 5-fold cross-validation, then export the model so the static site can score a new review in the browser.”

### Honest limitations (good to mention if asked)

- Demo data is **synthetic / template-generated**, not scraped from a live store
- Lexicon themes are **hand-crafted** for this catalog (not learned)
- Naive Bayes ignores word order beyond bigrams and assumes feature independence
- Small curated dataset — results illustrate the pipeline more than production accuracy
- No transformers / LLMs by design (clarity and offline static hosting)
