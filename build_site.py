"""Build static JSON for the GitHub Pages demo from the NLP pipeline."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from analyze import (
    SENTIMENT_ORDER,
    distinctive_words,
    enrich_reviews,
    export_model,
    improvement_feedback,
    insights,
    keyword_categories,
    load_reviews,
    monthly_summary,
    product_list,
    score_table,
    sentiment_counts,
    summary_stats,
    top_model_words,
    train_model,
    word_frequencies,
)

ROOT = Path(__file__).resolve().parent
DOCS = ROOT / "docs"
DATA_OUT = DOCS / "data"


def _json_ready(value):
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_ready(value.tolist())
    if isinstance(value, (np.floating, float)):
        return float(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y-%m-%d")
    if pd.isna(value):
        return None
    return value


def frame_payload(frame: pd.DataFrame) -> dict:
    stats = summary_stats(frame)
    counts = sentiment_counts(frame)
    monthly = monthly_summary(frame)
    stars = (
        frame["rating"]
        .value_counts()
        .reindex([1, 2, 3, 4, 5], fill_value=0)
        .rename_axis("stars")
        .reset_index(name="reviews")
    )
    model = train_model(frame)
    scores = score_table(model["report"])
    words = {
        label: distinctive_words(frame, label, n=6).to_dict(orient="records")
        for label in SENTIMENT_ORDER
    }
    review_rows = enrich_reviews(frame)
    categories = keyword_categories(frame, n=10)
    improvements = improvement_feedback(frame, recent_n=12)

    return {
        "product": str(frame["product"].iloc[0]),
        "stats": {
            "n": stats["n"],
            "avg_rating": stats["avg_rating"],
            "median_words": stats["median_words"],
            "min_words": stats["min_words"],
            "max_words": stats["max_words"],
            "top_sentiment": stats["top_sentiment"],
            "top_share": stats["top_share"],
        },
        "insights": insights(frame),
        "sentiment_counts": counts.to_dict(orient="records"),
        "star_counts": stars.to_dict(orient="records"),
        "monthly": monthly.to_dict(orient="records"),
        "categories": categories,
        "improvements": improvements,
        "distinctive_words": words,
        "top_words": word_frequencies(frame, n=12).to_dict(orient="records"),
        "reviews": review_rows,
        "model": {
            "accuracy": model["accuracy"],
            "report": scores.to_dict(orient="records"),
            "matrix": model["matrix"].tolist(),
            "labels": SENTIMENT_ORDER,
            "top_words": top_model_words(model["pipeline"], n=6).to_dict(orient="records"),
            "export": export_model(model["pipeline"]),
        },
        "date_range": {
            "start": frame["date"].min().strftime("%b %Y"),
            "end": frame["date"].max().strftime("%b %Y"),
        },
    }


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_ready(payload), indent=2), encoding="utf-8")


def main() -> None:
    frame = load_reviews()
    products = product_list(frame)
    DATA_OUT.mkdir(parents=True, exist_ok=True)
    write_json(DATA_OUT / "products.json", {"products": products})

    for product in products:
        subset = frame[frame["product"] == product["name"]].copy()
        payload = frame_payload(subset)
        write_json(DATA_OUT / f"{product['slug']}.json", payload)
        print(f"Built {product['slug']}.json ({product['reviews']} reviews)")

    print(f"Wrote product index with {len(products)} products to {DATA_OUT}")


if __name__ == "__main__":
    main()
