"""Load the demo reviews and turn them into stats plus a small sentiment model.

pandas holds the table and the averages.
nltk splits each review into words and drops words that carry little meaning.
scikit-learn turns the remaining text into numbers and trains Naive Bayes.
"""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

DATA_PATH = Path(__file__).resolve().parent / "data" / "reviews.csv"

SENTIMENT_ORDER = ["Negative", "Neutral", "Positive"]

# Product names and filler words. Kept out so the word charts show opinions.
EXTRA_STOP = {
    "aurora",
    "x1",
    "earbud",
    "earbuds",
    "bud",
    "buds",
    "pair",
    "nova",
    "pro",
    "pulse",
    "mini",
    "orbit",
    "flux",
    "anc",
    "headset",
    "headphones",
    "headphone",
    "speaker",
    "band",
    "really",
    "much",
    "one",
    "also",
    "even",
    "still",
    "get",
    "got",
    "made",
    "make",
    "like",
    # Recent NLTK stopword lists dropped a few of these.
    "would",
    "could",
    "people",
    "want",
    "say",
    "says",
    "said",
    "friends",
    "nothing",
    "something",
}

# Duplicate / near-duplicate tokens collapse into one theme category.
CATEGORY_MAP: dict[str, str] = {
    # Battery & Power
    "battery": "Battery & Power",
    "batteries": "Battery & Power",
    "charge": "Battery & Power",
    "charging": "Battery & Power",
    "charged": "Battery & Power",
    "charges": "Battery & Power",
    "dies": "Battery & Power",
    "died": "Battery & Power",
    "drains": "Battery & Power",
    "drain": "Battery & Power",
    "lasts": "Battery & Power",
    "lasting": "Battery & Power",
    "power": "Battery & Power",
    # Sound Quality
    "sound": "Sound Quality",
    "audio": "Sound Quality",
    "bass": "Sound Quality",
    "vocals": "Sound Quality",
    "voices": "Sound Quality",
    "voice": "Sound Quality",
    "tone": "Sound Quality",
    "crackle": "Sound Quality",
    "crackles": "Sound Quality",
    "muddy": "Sound Quality",
    "tinny": "Sound Quality",
    "distortion": "Sound Quality",
    "clear": "Sound Quality",
    "clarity": "Sound Quality",
    "crisp": "Sound Quality",
    "warm": "Sound Quality",
    "flat": "Sound Quality",
    "volume": "Sound Quality",
    "loud": "Sound Quality",
    "quiet": "Sound Quality",
    "hiss": "Sound Quality",
    "midrange": "Sound Quality",
    "soundstage": "Sound Quality",
    # Fit & Comfort
    "fit": "Fit & Comfort",
    "fits": "Fit & Comfort",
    "comfortable": "Fit & Comfort",
    "comfort": "Fit & Comfort",
    "comfortably": "Fit & Comfort",
    "tips": "Fit & Comfort",
    "painful": "Fit & Comfort",
    "pads": "Fit & Comfort",
    "cushioning": "Fit & Comfort",
    "clamp": "Fit & Comfort",
    "headband": "Fit & Comfort",
    "seal": "Fit & Comfort",
    "hot": "Fit & Comfort",
    "sweaty": "Fit & Comfort",
    "irritates": "Fit & Comfort",
    "skin": "Fit & Comfort",
    "strap": "Fit & Comfort",
    "clasp": "Fit & Comfort",
    # Connectivity
    "bluetooth": "Connectivity",
    "connection": "Connectivity",
    "connect": "Connectivity",
    "connecting": "Connectivity",
    "connected": "Connectivity",
    "pairing": "Connectivity",
    "pairs": "Connectivity",
    "pair": "Connectivity",
    "drops": "Connectivity",
    "drop": "Connectivity",
    "dropped": "Connectivity",
    "lag": "Connectivity",
    "lags": "Connectivity",
    "latency": "Connectivity",
    "sync": "Connectivity",
    "syncs": "Connectivity",
    "multipoint": "Connectivity",
    "wifi": "Connectivity",
    "stable": "Connectivity",
    # Value & Price
    "price": "Value & Price",
    "priced": "Value & Price",
    "expensive": "Value & Price",
    "worth": "Value & Price",
    "value": "Value & Price",
    "cheap": "Value & Price",
    "cheaper": "Value & Price",
    "cost": "Value & Price",
    # Calls & Mic
    "calls": "Calls & Mic",
    "call": "Calls & Mic",
    "mic": "Calls & Mic",
    "microphone": "Calls & Mic",
    "speakerphone": "Calls & Mic",
    # Noise Control
    "anc": "Noise Control",
    "isolation": "Noise Control",
    "noise": "Noise Control",
    "transparency": "Noise Control",
    "rumble": "Noise Control",
    # Fitness & Health (Orbit)
    "steps": "Fitness & Health",
    "sleep": "Fitness & Health",
    "heart": "Fitness & Health",
    "rate": "Fitness & Health",
    "gps": "Fitness & Health",
    "workout": "Fitness & Health",
    "workouts": "Fitness & Health",
    "calories": "Fitness & Health",
    "swim": "Fitness & Health",
    "tracking": "Fitness & Health",
    "tracks": "Fitness & Health",
}

PRODUCT_SLUGS = {
    "Aurora X1 Wireless Earbuds": "aurora-x1",
    "Nova Pro Over-Ear Headphones": "nova-pro",
    "Pulse Mini Bluetooth Speaker": "pulse-mini",
    "Orbit Fit Band": "orbit-fit",
    "Flux ANC Headset": "flux-anc",
}

FALLBACK_STOP = {
    "the", "a", "an", "and", "or", "but", "if", "in", "on", "at", "to", "for",
    "of", "is", "it", "this", "that", "with", "my", "i", "me", "they", "them",
    "was", "are", "be", "have", "has", "not", "no", "so", "too", "very", "just",
    "from", "when", "about", "after", "before", "into", "than", "then", "there",
    "their", "what", "which", "would", "could", "should", "you", "your", "we",
    "our", "he", "she", "his", "her", "as", "by", "do", "did", "does", "been",
    "being", "am", "will", "can", "these", "those", "had", "have",
}

_STOPWORDS: set[str] | None = None


def ensure_nltk() -> None:
    """Download the tokenizer and stopword list the first time they are missing."""
    needed = (
        ("punkt_tab", "tokenizers/punkt_tab"),
        ("stopwords", "corpora/stopwords"),
    )
    for resource, finder in needed:
        try:
            nltk.data.find(finder)
        except LookupError:
            nltk.download(resource, quiet=True)


def stopword_set() -> set[str]:
    global _STOPWORDS
    if _STOPWORDS is not None:
        return _STOPWORDS
    try:
        ensure_nltk()
        words = set(stopwords.words("english"))
    except Exception:
        words = set(FALLBACK_STOP)
    _STOPWORDS = words | EXTRA_STOP
    return _STOPWORDS


def tokenize(text: str) -> list[str]:
    """Lowercase a review, split it into words, and drop stopwords."""
    lowered = text.lower()
    try:
        ensure_nltk()
        raw = word_tokenize(lowered)
    except Exception:
        raw = re.findall(r"[a-z]+", lowered)
    stops = stopword_set()
    return [word for word in raw if word.isalpha() and len(word) > 2 and word not in stops]


def load_reviews(path: Path | None = None) -> pd.DataFrame:
    frame = pd.read_csv(path or DATA_PATH)
    frame["date"] = pd.to_datetime(frame["date"])
    frame["rating"] = frame["rating"].astype(int)
    frame["sentiment"] = frame["sentiment"].astype(str)
    frame["word_count"] = frame["review"].str.split().str.len()
    return frame.sort_values(["date", "review_id"]).reset_index(drop=True)


def filter_reviews(
    frame: pd.DataFrame,
    sentiments: list[str],
    rating_min: int,
    rating_max: int,
) -> pd.DataFrame:
    chosen = frame["sentiment"].isin(sentiments)
    stars = frame["rating"].between(rating_min, rating_max)
    return frame.loc[chosen & stars].copy()


def sentiment_counts(frame: pd.DataFrame) -> pd.DataFrame:
    counts = (
        frame["sentiment"]
        .value_counts()
        .reindex(SENTIMENT_ORDER, fill_value=0)
        .rename_axis("sentiment")
        .reset_index(name="reviews")
    )
    total = counts["reviews"].sum()
    counts["share"] = counts["reviews"] / total if total else 0
    return counts


def summary_stats(frame: pd.DataFrame) -> dict:
    counts = sentiment_counts(frame)
    top_row = counts.sort_values("reviews", ascending=False).iloc[0]
    return {
        "n": int(len(frame)),
        "avg_rating": float(frame["rating"].mean()),
        "rating_std": float(frame["rating"].std(ddof=0)),
        "median_words": float(frame["word_count"].median()),
        "mean_words": float(frame["word_count"].mean()),
        "min_words": int(frame["word_count"].min()),
        "max_words": int(frame["word_count"].max()),
        "counts": counts,
        "top_sentiment": str(top_row["sentiment"]),
        "top_share": float(top_row["share"]),
    }


def monthly_summary(frame: pd.DataFrame) -> pd.DataFrame:
    temp = frame.copy()
    temp["month"] = temp["date"].dt.to_period("M")
    temp["is_negative"] = (temp["sentiment"] == "Negative").astype(int)
    out = (
        temp.groupby("month", sort=True)
        .agg(
            reviews=("review_id", "size"),
            avg_rating=("rating", "mean"),
            negative_reviews=("is_negative", "sum"),
        )
        .reset_index()
    )
    out["month"] = out["month"].dt.strftime("%b %Y")
    out["negative_share"] = out["negative_reviews"] / out["reviews"]
    return out


def _token_counts(reviews: pd.Series) -> Counter:
    counts: Counter = Counter()
    for text in reviews:
        counts.update(tokenize(text))
    return counts


def word_frequencies(frame: pd.DataFrame, n: int = 12) -> pd.DataFrame:
    common = _token_counts(frame["review"]).most_common(n)
    return pd.DataFrame(common, columns=["word", "count"])


def distinctive_words(frame: pd.DataFrame, sentiment: str, n: int = 8) -> pd.DataFrame:
    """Words that take up more of this sentiment than of the other reviews.

    Score = (share of words inside the sentiment) - (share outside it).
    A word needs to appear at least twice so a single review cannot dominate.
    """
    inside = frame[frame["sentiment"] == sentiment]
    outside = frame[frame["sentiment"] != sentiment]
    in_counts = _token_counts(inside["review"])
    out_counts = _token_counts(outside["review"]) if not outside.empty else Counter()
    in_total = sum(in_counts.values()) or 1
    out_total = sum(out_counts.values()) or 1
    only_one_group = outside.empty

    rows = []
    for word, count in in_counts.items():
        if count < 2:
            continue
        share_in = count / in_total
        share_out = 0 if only_one_group else out_counts[word] / out_total
        lift = share_in - share_out
        if not only_one_group and lift <= 0:
            continue
        rows.append({"word": word, "count": count, "lift": lift})

    table = pd.DataFrame(rows)
    if table.empty:
        return table
    return table.sort_values(["lift", "count"], ascending=False).head(n).reset_index(drop=True)


def insights(frame: pd.DataFrame) -> list[str]:
    """Short readings of the table, so the charts are not the only explanation."""
    stats = summary_stats(frame)
    lines = [
        (
            f"{stats['top_share'] * 100:.0f}% of these reviews are "
            f"{stats['top_sentiment'].lower()}."
        )
    ]

    monthly = monthly_summary(frame)
    if len(monthly) >= 2:
        first, last = monthly.iloc[0], monthly.iloc[-1]
        lines.append(
            f"Average rating moved from {first['avg_rating']:.1f} in {first['month']} "
            f"to {last['avg_rating']:.1f} in {last['month']}."
        )
        worst = monthly.loc[monthly["avg_rating"].idxmin()]
        if worst["month"] not in {first["month"], last["month"]}:
            lines.append(
                f"The weakest month in this selection is {worst['month']}, "
                f"at {worst['avg_rating']:.1f} stars."
            )

    negative = distinctive_words(frame, "Negative", n=1)
    if not negative.empty:
        lines.append(
            f"“{negative.iloc[0]['word']}” shows up more in negative reviews "
            f"than in the rest."
        )
    return lines


def train_model(frame: pd.DataFrame) -> dict:
    """Train Naive Bayes and score it on reviews it did not train on.

    Cross-validation splits the demo into five folds. Each review is predicted
    by a model trained on the other four folds. That accuracy is the honest one.
    The returned pipeline is then fit on every demo review so it can score a
    brand-new sentence.
    """
    texts = frame["review"].tolist()
    labels = frame["sentiment"].astype(str).tolist()
    pipeline = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(ngram_range=(1, 2), stop_words="english", min_df=2),
            ),
            ("clf", MultinomialNB()),
        ]
    )
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    predicted = cross_val_predict(pipeline, texts, labels, cv=folds)
    report = classification_report(
        labels,
        predicted,
        labels=SENTIMENT_ORDER,
        output_dict=True,
        zero_division=0,
    )
    pipeline.fit(texts, labels)
    return {
        "pipeline": pipeline,
        "accuracy": float(accuracy_score(labels, predicted)),
        "report": report,
        "matrix": confusion_matrix(labels, predicted, labels=SENTIMENT_ORDER),
        "labels": SENTIMENT_ORDER,
    }


def score_table(report: dict) -> pd.DataFrame:
    rows = []
    for label in SENTIMENT_ORDER:
        item = report[label]
        rows.append(
            {
                "Sentiment": label,
                "Precision": item["precision"],
                "Recall": item["recall"],
                "F1": item["f1-score"],
                "Reviews": int(item["support"]),
            }
        )
    return pd.DataFrame(rows)


def top_model_words(pipeline: Pipeline, n: int = 6) -> pd.DataFrame:
    """Words and pairs of words the trained model leans on for each label."""
    vectorizer: TfidfVectorizer = pipeline.named_steps["tfidf"]
    clf: MultinomialNB = pipeline.named_steps["clf"]
    names = vectorizer.get_feature_names_out()
    rows = []
    for index, label in enumerate(clf.classes_):
        scores = clf.feature_log_prob_[index]
        best = scores.argsort()[-n:][::-1]
        for rank, feature_index in enumerate(best, start=1):
            rows.append(
                {
                    "sentiment": str(label),
                    "rank": rank,
                    "word": names[feature_index],
                }
            )
    return pd.DataFrame(rows)


def predict_review(pipeline: Pipeline, text: str) -> dict:
    probabilities = pipeline.predict_proba([text])[0]
    classes = [str(label) for label in pipeline.classes_]
    best = max(range(len(classes)), key=lambda index: probabilities[index])
    return {
        "label": classes[best],
        "probabilities": [
            {"sentiment": classes[index], "probability": float(probabilities[index])}
            for index in range(len(classes))
        ],
        "tokens": tokenize(text),
    }


def normalize_token(word: str) -> str:
    """Collapse simple plural duplicates onto one stem-like form."""
    if word.endswith("ies") and len(word) > 4:
        return word[:-3] + "y"
    if word.endswith("ing") and len(word) > 5 and word not in {"thing", "something"}:
        base = word[:-3]
        if base.endswith(base[-1:]) and len(base) > 3:
            return base[:-1] if base[-1] == base[-2] else base
        return base
    if word.endswith("es") and len(word) > 4:
        return word[:-2]
    if word.endswith("s") and len(word) > 3 and not word.endswith("ss"):
        return word[:-1]
    return word


def category_for_token(word: str) -> str | None:
    if word in CATEGORY_MAP:
        return CATEGORY_MAP[word]
    stem = normalize_token(word)
    return CATEGORY_MAP.get(stem)


def extract_keywords(text: str) -> dict:
    """Return tokens, mapped categories, and keyword chips for one review."""
    tokens = tokenize(text)
    keyword_counts: Counter = Counter(tokens)
    categories: set[str] = set()
    category_keywords: dict[str, Counter] = {}
    for word in tokens:
        category = category_for_token(word)
        if not category:
            continue
        categories.add(category)
        category_keywords.setdefault(category, Counter())[word] += 1
    return {
        "keywords": [
            {"word": word, "count": int(count)}
            for word, count in keyword_counts.most_common(12)
        ],
        "categories": sorted(categories),
        "category_keywords": {
            category: [
                {"word": word, "count": int(count)}
                for word, count in counts.most_common(6)
            ]
            for category, counts in category_keywords.items()
        },
    }


def enrich_reviews(frame: pd.DataFrame) -> list[dict]:
    """Attach keywords and theme categories to every review row."""
    rows = []
    for _, row in frame.iterrows():
        extracted = extract_keywords(str(row["review"]))
        rows.append(
            {
                "review_id": int(row["review_id"]),
                "date": row["date"].strftime("%Y-%m-%d")
                if hasattr(row["date"], "strftime")
                else str(row["date"]),
                "reviewer": str(row["reviewer"]),
                "rating": int(row["rating"]),
                "sentiment": str(row["sentiment"]),
                "review": str(row["review"]),
                "word_count": int(row["word_count"]),
                "keywords": extracted["keywords"],
                "categories": extracted["categories"],
                "category_keywords": extracted["category_keywords"],
            }
        )
    return rows


def keyword_categories(frame: pd.DataFrame, n: int = 10) -> list[dict]:
    """Group duplicate / near-duplicate keywords into theme categories.

    Each hit is counted once per review per category. Sentiment breakdown
    shows which labels mention that theme most often.
    """
    category_mentions: Counter = Counter()
    category_keywords: dict[str, Counter] = {}
    category_sentiment: dict[str, Counter] = {}
    category_reviews: dict[str, list[dict]] = {}

    for _, row in frame.iterrows():
        extracted = extract_keywords(str(row["review"]))
        found = set(extracted["categories"])
        for category, chips in extracted["category_keywords"].items():
            for chip in chips:
                category_keywords.setdefault(category, Counter())[chip["word"]] += chip["count"]
        for category in found:
            category_mentions[category] += 1
            category_sentiment.setdefault(category, Counter())[str(row["sentiment"])] += 1
            bucket = category_reviews.setdefault(category, [])
            if len(bucket) < 8:
                bucket.append(
                    {
                        "review_id": int(row["review_id"]),
                        "date": row["date"].strftime("%Y-%m-%d")
                        if hasattr(row["date"], "strftime")
                        else str(row["date"]),
                        "reviewer": str(row["reviewer"]),
                        "rating": int(row["rating"]),
                        "sentiment": str(row["sentiment"]),
                        "review": str(row["review"]),
                        "keywords": [
                            chip["word"]
                            for chip in extracted["category_keywords"].get(category, [])
                        ],
                    }
                )

    rows = []
    for category, mentions in category_mentions.most_common(n):
        keywords = category_keywords.get(category, Counter())
        sentiments = category_sentiment.get(category, Counter())
        rows.append(
            {
                "category": category,
                "mentions": int(mentions),
                "keywords": [
                    {"word": word, "count": int(count)}
                    for word, count in keywords.most_common(8)
                ],
                "by_sentiment": {
                    label: int(sentiments.get(label, 0)) for label in SENTIMENT_ORDER
                },
                "sample_reviews": category_reviews.get(category, []),
            }
        )
    return rows


def improvement_feedback(frame: pd.DataFrame, recent_n: int = 12) -> dict:
    """Surface recent critical feedback and category-wise improvement cues."""
    ordered = frame.sort_values("date", ascending=False)
    critical = ordered[ordered["sentiment"].isin(["Negative", "Neutral"])]
    recent_rows = []
    for _, row in critical.head(recent_n).iterrows():
        extracted = extract_keywords(str(row["review"]))
        recent_rows.append(
            {
                "review_id": int(row["review_id"]),
                "date": row["date"].strftime("%Y-%m-%d")
                if hasattr(row["date"], "strftime")
                else str(row["date"]),
                "reviewer": str(row["reviewer"]),
                "rating": int(row["rating"]),
                "sentiment": str(row["sentiment"]),
                "review": str(row["review"]),
                "keywords": [item["word"] for item in extracted["keywords"][:8]],
                "categories": extracted["categories"],
            }
        )

    negative = frame[frame["sentiment"] == "Negative"]
    opportunities = []
    if not negative.empty:
        for item in keyword_categories(negative, n=8):
            neg = item["by_sentiment"].get("Negative", 0)
            if neg <= 0:
                continue
            top_words = ", ".join(chip["word"] for chip in item["keywords"][:4]) or "theme terms"
            opportunities.append(
                {
                    "category": item["category"],
                    "negative_mentions": neg,
                    "keywords": item["keywords"][:6],
                    "action": (
                        f"Improve {item['category'].lower()} — reviewers often mention "
                        f"{top_words}."
                    ),
                    "sample_reviews": item["sample_reviews"][:4],
                }
            )

    return {
        "recent": recent_rows,
        "opportunities": opportunities,
        "recent_count": len(recent_rows),
        "opportunity_count": len(opportunities),
    }


def export_model(pipeline: Pipeline) -> dict:
    """Serialize TF-IDF + MultinomialNB weights for browser-side scoring."""
    vectorizer: TfidfVectorizer = pipeline.named_steps["tfidf"]
    clf: MultinomialNB = pipeline.named_steps["clf"]
    return {
        "vocabulary": {str(term): int(index) for term, index in vectorizer.vocabulary_.items()},
        "idf": [float(value) for value in vectorizer.idf_],
        "ngram_range": list(vectorizer.ngram_range),
        "classes": [str(label) for label in clf.classes_],
        "class_log_prior": [float(value) for value in clf.class_log_prior_],
        "feature_log_prob": [
            [float(value) for value in row] for row in clf.feature_log_prob_
        ],
    }


def product_list(frame: pd.DataFrame) -> list[dict]:
    rows = []
    for product, group in frame.groupby("product", sort=True):
        name = str(product)
        rows.append(
            {
                "name": name,
                "slug": PRODUCT_SLUGS.get(name, re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")),
                "reviews": int(len(group)),
                "avg_rating": float(group["rating"].mean()),
            }
        )
    return rows


if __name__ == "__main__":
    reviews = load_reviews()
    stats = summary_stats(reviews)
    print(f"{stats['n']} reviews, average rating {stats['avg_rating']:.2f}")
    for line in insights(reviews):
        print("-", line)
    for label in SENTIMENT_ORDER:
        print(f"\n{label}")
        print(distinctive_words(reviews, label).to_string(index=False))
    trained = train_model(reviews)
    print(f"\naccuracy {trained['accuracy']:.3f}")
    print(top_model_words(trained["pipeline"]).to_string(index=False))
    print(trained["matrix"])
