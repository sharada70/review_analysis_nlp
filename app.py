"""Legacy Streamlit UI (single-product era).

The hosted demo is the static site in docs/ (GitHub Pages).
Build it with: py build_site.py
Preview with:  py -m http.server 8080 --directory docs

From this folder (optional local Streamlit):

    py -m streamlit run app.py
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import streamlit as st

from analyze import (
    SENTIMENT_ORDER,
    distinctive_words,
    filter_reviews,
    insights,
    load_reviews,
    monthly_summary,
    predict_review,
    score_table,
    sentiment_counts,
    summary_stats,
    top_model_words,
    train_model,
    word_frequencies,
)

COLORS = {
    "Positive": "#1F7A4D",
    "Neutral": "#B0893E",
    "Negative": "#B64032",
}

EXAMPLES = {
    "Battery complaint": (
        "The battery dies in an hour and the audio drops whenever I walk."
    ),
    "Happy with them": (
        "Clear sound, a comfortable fit, and a stable connection all day."
    ),
    "Middle of the road": (
        "They are okay for podcasts. The price feels a bit high for the sound."
    ),
}


def inject_css() -> None:
    st.markdown(
        """
        <style>
        .block-container {max-width: 1080px; padding-top: 1.4rem; padding-bottom: 3rem;}
        h1, h2, h3 {
            font-family: Georgia, "Palatino Linotype", serif;
            font-weight: 500;
            letter-spacing: -0.02em;
        }
        .eyebrow {
            letter-spacing: 0.14em;
            text-transform: uppercase;
            font-size: 0.74rem;
            color: #1F4E46;
            font-weight: 650;
            margin: 0 0 0.35rem 0;
        }
        .lede {color: #4E493F; font-size: 1.05rem; max-width: 44rem; margin-bottom: 0.4rem;}
        .insight {
            background: #fffdf8;
            border: 1px solid #e4ddd0;
            border-left: 4px solid #1F4E46;
            border-radius: 10px;
            padding: 0.7rem 0.9rem;
            margin: 0.35rem 0;
            color: #1C1915;
            line-height: 1.45;
        }
        div[data-testid="stMetric"] {
            background: #fffdf8;
            border: 1px solid #e4ddd0;
            border-radius: 12px;
            padding: 0.7rem 0.85rem;
        }
        div[data-testid="stMetricValue"] {color: #1F4E46;}
        .pred {
            display: inline-block;
            border: 2px solid;
            border-radius: 999px;
            padding: 0.2rem 0.8rem;
            font-weight: 650;
            margin: 0.2rem 0 0.6rem 0;
        }
        .note {color: #5C564C; font-size: 0.92rem;}
        </style>
        """,
        unsafe_allow_html=True,
    )


def style_chart(fig, height: int = 340):
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"family": "Segoe UI, sans-serif", "color": "#1C1915", "size": 13},
        height=height,
        margin={"l": 8, "r": 8, "t": 28, "b": 8},
        legend={"orientation": "h", "y": -0.18, "title": ""},
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor="#E6E0D6")
    return fig


@st.cache_data
def cached_reviews() -> pd.DataFrame:
    return load_reviews()


@st.cache_resource
def cached_model():
    return train_model(load_reviews())


def sidebar(full: pd.DataFrame) -> tuple[list[str], int, int]:
    product = full["product"].iloc[0]
    start = full["date"].min().strftime("%b %Y")
    end = full["date"].max().strftime("%b %Y")
    st.sidebar.markdown(f"**{product}**")
    st.sidebar.caption(
        f"{len(full)} labeled demo reviews, {start} to {end}. "
        "Filters change the stats and the table. The model stays trained on the full set."
    )
    sentiments = st.sidebar.multiselect(
        "Sentiment",
        SENTIMENT_ORDER,
        default=SENTIMENT_ORDER,
    )
    rating_min, rating_max = st.sidebar.slider("Star rating", 1, 5, (1, 5))
    with st.sidebar.expander("What each library does"):
        st.markdown(
            """
            **pandas** loads the CSV and calculates counts, averages, and monthly groups.

            **nltk** splits reviews into words and removes stopwords such as "the" and "is".

            **scikit-learn** turns those words into TF-IDF numbers and trains a Naive Bayes model.

            **plotly** draws the interactive charts.

            **matplotlib** draws the top-words chart.

            **streamlit** is this web page.
            """
        )
    st.sidebar.caption("Start the page with `py -m streamlit run app.py` from the project folder.")
    return sentiments, rating_min, rating_max


def render_header(full: pd.DataFrame) -> None:
    product = full["product"].iloc[0]
    st.markdown("<p class='eyebrow'>Beginner NLP demo</p>", unsafe_allow_html=True)
    st.title("Review analysis")
    st.markdown(
        f"<p class='lede'>Demo reviews for the <b>{product}</b>. "
        "Each one is already labeled positive, neutral, or negative. "
        "The numbers below describe this set: how people scored the product, "
        "how that changed over the months, and which words separate the labels.</p>",
        unsafe_allow_html=True,
    )


def render_insights(lines: list[str]) -> None:
    blocks = "".join(f"<div class='insight'>{line}</div>" for line in lines)
    st.markdown(blocks, unsafe_allow_html=True)


def render_metrics(stats: dict) -> None:
    positive = stats["counts"].set_index("sentiment").loc["Positive", "share"]
    negative = stats["counts"].set_index("sentiment").loc["Negative", "share"]
    columns = st.columns(4)
    columns[0].metric("Reviews", f"{stats['n']}")
    columns[1].metric("Average rating", f"{stats['avg_rating']:.1f} / 5")
    columns[2].metric("Positive", f"{positive * 100:.0f}%")
    columns[3].metric("Negative", f"{negative * 100:.0f}%")
    st.caption(
        f"A typical review is {stats['median_words']:.0f} words. "
        f"The shortest is {stats['min_words']} and the longest is {stats['max_words']}."
    )


def render_overview_charts(frame: pd.DataFrame) -> None:
    left, right = st.columns(2)
    counts = sentiment_counts(frame)
    with left:
        st.subheader("Sentiment mix")
        fig = px.pie(
            counts,
            names="sentiment",
            values="reviews",
            hole=0.58,
            color="sentiment",
            color_discrete_map=COLORS,
            category_orders={"sentiment": SENTIMENT_ORDER},
        )
        fig.update_traces(textposition="inside", textinfo="percent+label", sort=False)
        style_chart(fig, 320)
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
        st.caption("Share of reviews in each label.")

    with right:
        st.subheader("Star ratings")
        stars = (
            frame["rating"]
            .value_counts()
            .reindex([1, 2, 3, 4, 5], fill_value=0)
            .rename_axis("stars")
            .reset_index(name="reviews")
        )
        stars["label"] = stars["stars"].map(
            lambda rating: "Negative" if rating <= 2 else "Neutral" if rating == 3 else "Positive"
        )
        fig = px.bar(
            stars,
            x="stars",
            y="reviews",
            color="label",
            color_discrete_map=COLORS,
            category_orders={"label": SENTIMENT_ORDER},
        )
        style_chart(fig, 320)
        fig.update_layout(xaxis_title="", yaxis_title="Reviews", legend_title="")
        fig.update_xaxes(tickmode="array", tickvals=[1, 2, 3, 4, 5])
        fig.update_yaxes(range=[0, float(stars["reviews"].max()) + 1.4])
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
        st.caption("In this demo, 1–2 stars are negative, 3 is neutral, and 4–5 are positive.")


def render_trend(frame: pd.DataFrame) -> None:
    monthly = monthly_summary(frame)
    st.subheader("How the reviews change by month")
    left, right = st.columns(2)
    with left:
        fig = px.bar(
            monthly,
            x="month",
            y="reviews",
            color_discrete_sequence=["#1F4E46"],
        )
        style_chart(fig, 320)
        fig.update_layout(xaxis_title="", yaxis_title="Reviews", showlegend=False)
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
        st.caption("How many reviews were written each month.")
    with right:
        fig = px.line(monthly, x="month", y="avg_rating", markers=True)
        fig.update_traces(line_color="#1F4E46", marker_color="#1F4E46")
        style_chart(fig, 320)
        fig.update_yaxes(range=[1, 5], title="Average stars")
        fig.update_layout(xaxis_title="", showlegend=False)
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
        st.caption("Read left to right. A rise means later reviews scored the product higher.")

    with st.expander("Month-by-month numbers"):
        pretty = monthly.copy()
        pretty["avg_rating"] = pretty["avg_rating"].map(lambda value: f"{value:.1f}")
        pretty["negative_share"] = pretty["negative_share"].map(lambda value: f"{value:.0%}")
        pretty = pretty.rename(
            columns={
                "month": "Month",
                "reviews": "Reviews",
                "avg_rating": "Average rating",
                "negative_reviews": "Negative reviews",
                "negative_share": "Negative share",
            }
        )
        st.dataframe(pretty, hide_index=True, width="stretch")
        st.caption("Negative share is the portion of that month's reviews labeled negative.")


def draw_word_bars(table: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(8.2, 4.2), dpi=140)
    fig.patch.set_facecolor("#F6F3EC")
    ax.set_facecolor("#F6F3EC")
    ordered = table.iloc[::-1]
    ax.barh(ordered["word"], ordered["count"], color="#1F4E46", height=0.62)
    ax.set_xlabel("Times used")
    ax.tick_params(colors="#1C1915", labelsize=10)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout()
    return fig


def render_words(frame: pd.DataFrame) -> None:
    st.subheader("Words that separate the labels")
    single_sentiment = frame["sentiment"].nunique() == 1
    if single_sentiment:
        st.caption("Only one sentiment is selected, so these lists are plain word counts.")
    else:
        st.caption(
            "A word is listed when it makes up more of that sentiment than of the other reviews, "
            "and it appears at least twice. Stopwords such as “the” and “is” are removed first."
        )

    columns = st.columns(3)
    for column, label in zip(columns, SENTIMENT_ORDER):
        with column:
            st.markdown(f"**{label}**")
            table = distinctive_words(frame, label, n=6)
            if table.empty:
                st.caption("Not enough repeated words in this group.")
                continue
            fig = px.bar(
                table.iloc[::-1],
                x="count",
                y="word",
                orientation="h",
                color_discrete_sequence=[COLORS[label]],
            )
            style_chart(fig, 280)
            fig.update_layout(
                xaxis_title="Uses",
                yaxis_title="",
                showlegend=False,
                margin={"l": 8, "r": 8, "t": 8, "b": 8},
            )
            st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

    st.subheader("Most used words")
    frequent = word_frequencies(frame, n=10)
    if frequent.empty:
        st.caption("No content words left after cleanup.")
        return
    fig = draw_word_bars(frequent)
    st.pyplot(fig, width="stretch")
    plt.close(fig)
    st.caption("Overall word counts for the reviews in the current filter, drawn with Matplotlib.")


def render_table(frame: pd.DataFrame) -> None:
    st.subheader("Demo reviews")
    query = st.text_input("Search the review text", placeholder="Try battery, clear, podcasts")
    view = frame
    if query.strip():
        view = frame[frame["review"].str.contains(query.strip(), case=False, na=False)]
    st.caption(f"{len(view)} review{'s' if len(view) != 1 else ''} in this view.")

    show = view.copy()
    show["date"] = show["date"].dt.strftime("%Y-%m-%d")
    show = show.rename(
        columns={
            "date": "Date",
            "reviewer": "Reviewer",
            "rating": "Stars",
            "sentiment": "Sentiment",
            "review": "Review",
            "word_count": "Words",
        }
    )
    show = show[["Date", "Reviewer", "Stars", "Sentiment", "Words", "Review"]]
    st.dataframe(
        show,
        hide_index=True,
        width="stretch",
        height=440,
        column_config={
            "Review": st.column_config.TextColumn(width="large"),
            "Stars": st.column_config.NumberColumn(format="%d"),
            "Words": st.column_config.NumberColumn(format="%d"),
        },
    )
    st.download_button(
        "Download this view as CSV",
        data=show.to_csv(index=False).encode("utf-8"),
        file_name="aurora_x1_reviews.csv",
        mime="text/csv",
    )


def render_prediction(pipeline, text: str) -> None:
    result = predict_review(pipeline, text)
    label = result["label"]
    st.markdown(
        f"<span class='pred' style='color:{COLORS[label]};border-color:{COLORS[label]}'>{label}</span>",
        unsafe_allow_html=True,
    )
    if result["tokens"]:
        shown = " · ".join(result["tokens"][:24])
        st.markdown(f"<p class='note'>Words kept after cleanup: {shown}</p>", unsafe_allow_html=True)
    else:
        st.caption(
            "No content words were left after cleanup, so this guess mostly reflects "
            "how common each label is in the demo."
        )

    probs = pd.DataFrame(result["probabilities"])
    probs["sentiment"] = pd.Categorical(probs["sentiment"], SENTIMENT_ORDER, ordered=True)
    probs = probs.sort_values("sentiment")
    probs["percent"] = (probs["probability"] * 100).round(1)
    fig = px.bar(
        probs,
        x="percent",
        y="sentiment",
        orientation="h",
        color="sentiment",
        color_discrete_map=COLORS,
    )
    style_chart(fig, 240)
    fig.update_layout(
        xaxis_title="Model confidence (%)",
        yaxis_title="",
        xaxis_range=[0, 100],
        showlegend=False,
        margin={"l": 8, "r": 8, "t": 8, "b": 8},
    )
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})


def render_model(full: pd.DataFrame) -> None:
    st.subheader("A simple model of the same reviews")
    st.markdown(
        "<p class='note'>Naive Bayes learns which words tend to sit with each label, "
        "then picks the label that best fits a new sentence. "
        "The accuracy below is from 5-fold cross-validation: every demo review was scored "
        "by a model that had not trained on it. The try-it box uses a model trained on all "
        f"{len(full)} reviews, so it can judge a sentence that is not in the file.</p>",
        unsafe_allow_html=True,
    )

    model = cached_model()
    accuracy_col, detail_col = st.columns([1, 2])
    with accuracy_col:
        st.metric("Cross-validated accuracy", f"{model['accuracy'] * 100:.0f}%")
        st.caption(f"On {len(full)} demo reviews. Real review text is messier than this set.")
    with detail_col:
        scores = score_table(model["report"]).copy()
        for column in ("Precision", "Recall", "F1"):
            scores[column] = scores[column].map(lambda value: f"{value:.0%}")
        st.dataframe(scores, hide_index=True, width="stretch")

    matrix = pd.DataFrame(
        model["matrix"],
        index=pd.Index(SENTIMENT_ORDER, name="Actual"),
        columns=SENTIMENT_ORDER,
    )
    left, right = st.columns(2)
    with left:
        st.markdown("**Where the guesses landed**")
        fig = px.imshow(
            matrix,
            text_auto=True,
            color_continuous_scale=["#F6F3EC", "#1F4E46"],
            aspect="equal",
        )
        style_chart(fig, 340)
        fig.update_layout(
            xaxis_title="Predicted",
            yaxis_title="Actual label",
            coloraxis_showscale=False,
        )
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
        st.caption("Each cell is a count of reviews. The diagonal is where the guess matched the label.")
    with right:
        st.markdown("**Words the model leans on**")
        words = top_model_words(model["pipeline"], n=6)
        pivot = words.pivot(index="rank", columns="sentiment", values="word")
        pivot = pivot.reindex(columns=[label for label in SENTIMENT_ORDER if label in pivot.columns])
        st.dataframe(pivot, width="stretch")
        st.caption("Top TF-IDF phrases for each label after the model is trained on the full demo.")

    st.subheader("Try a new review")
    if "draft" not in st.session_state:
        st.session_state["draft"] = ""

    clicked = None
    buttons = st.columns(3)
    for column, (name, sample) in zip(buttons, EXAMPLES.items()):
        if column.button(name, width="stretch"):
            clicked = sample
    if clicked is not None:
        st.session_state["draft"] = clicked

    text = st.text_area(
        "Review text",
        key="draft",
        height=110,
        placeholder="Type a short review of the Aurora X1 earbuds.",
    )
    analyze_clicked = st.button("Analyze review", type="primary")
    if clicked is not None or analyze_clicked:
        if text.strip():
            st.session_state["prediction_text"] = text
        else:
            st.session_state["prediction_text"] = ""
            st.info("Type a review first, or use one of the examples.")

    saved = st.session_state.get("prediction_text", "")
    if saved and saved == text:
        render_prediction(model["pipeline"], saved)


def main() -> None:
    st.set_page_config(
        page_title="Aurora X1 review analysis",
        page_icon="🎧",
        layout="centered",
    )
    inject_css()
    full = cached_reviews()
    sentiments, rating_min, rating_max = sidebar(full)
    render_header(full)

    filtered = filter_reviews(full, sentiments, rating_min, rating_max)
    if len(filtered) != len(full):
        st.caption(f"Stats and the table are using {len(filtered)} of {len(full)} reviews.")

    if not sentiments or filtered.empty:
        st.warning("No reviews match the sidebar filters. Pick a sentiment and a star range.")
    else:
        render_insights(insights(filtered))
        render_metrics(summary_stats(filtered))
        render_overview_charts(filtered)
        render_trend(filtered)
        render_words(filtered)
        render_table(filtered)

    render_model(full)


if __name__ == "__main__":
    main()
