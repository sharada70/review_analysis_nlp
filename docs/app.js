/* ReviewLab static frontend — cream UI, scroll reveals, keyword/category detail. */

const COLORS = {
  Negative: "#b42318",
  Neutral: "#a15c07",
  Positive: "#067647",
};

const SENTIMENT_ORDER = ["Negative", "Neutral", "Positive"];

const EXAMPLES = [
  {
    label: "Battery complaint",
    text: "The battery dies in an hour and the audio drops whenever I walk.",
  },
  {
    label: "Happy with it",
    text: "Clear sound, a comfortable fit, and a stable connection all day.",
  },
  {
    label: "Middle of the road",
    text: "They are okay for podcasts. The price feels a bit high for the sound.",
  },
];

const FALLBACK_STOP = new Set([
  "the", "a", "an", "and", "or", "but", "if", "in", "on", "at", "to", "for",
  "of", "is", "it", "this", "that", "with", "my", "i", "me", "they", "them",
  "was", "are", "be", "have", "has", "not", "no", "so", "too", "very", "just",
  "from", "when", "about", "after", "before", "into", "than", "then", "there",
  "their", "what", "which", "would", "could", "should", "you", "your", "we",
  "our", "he", "she", "his", "her", "as", "by", "do", "did", "does", "been",
  "being", "am", "will", "can", "these", "those", "had", "have", "really",
  "much", "one", "also", "even", "still", "get", "got", "made", "make", "like",
]);

const CATEGORY_MAP = {
  battery: "Battery & Power", batteries: "Battery & Power", charge: "Battery & Power",
  charging: "Battery & Power", charged: "Battery & Power", charges: "Battery & Power",
  dies: "Battery & Power", died: "Battery & Power", drains: "Battery & Power",
  drain: "Battery & Power", lasts: "Battery & Power", lasting: "Battery & Power",
  power: "Battery & Power",
  sound: "Sound Quality", audio: "Sound Quality", bass: "Sound Quality",
  vocals: "Sound Quality", voices: "Sound Quality", voice: "Sound Quality",
  tone: "Sound Quality", crackle: "Sound Quality", crackles: "Sound Quality",
  muddy: "Sound Quality", tinny: "Sound Quality", distortion: "Sound Quality",
  clear: "Sound Quality", clarity: "Sound Quality", crisp: "Sound Quality",
  warm: "Sound Quality", flat: "Sound Quality", volume: "Sound Quality",
  loud: "Sound Quality", quiet: "Sound Quality", hiss: "Sound Quality",
  midrange: "Sound Quality", soundstage: "Sound Quality",
  fit: "Fit & Comfort", fits: "Fit & Comfort", comfortable: "Fit & Comfort",
  comfort: "Fit & Comfort", tips: "Fit & Comfort", painful: "Fit & Comfort",
  pads: "Fit & Comfort", cushioning: "Fit & Comfort", clamp: "Fit & Comfort",
  headband: "Fit & Comfort", seal: "Fit & Comfort", hot: "Fit & Comfort",
  sweaty: "Fit & Comfort", irritates: "Fit & Comfort", skin: "Fit & Comfort",
  strap: "Fit & Comfort", clasp: "Fit & Comfort",
  bluetooth: "Connectivity", connection: "Connectivity", connect: "Connectivity",
  connecting: "Connectivity", connected: "Connectivity", pairing: "Connectivity",
  pairs: "Connectivity", drops: "Connectivity", drop: "Connectivity",
  lag: "Connectivity", lags: "Connectivity", latency: "Connectivity",
  sync: "Connectivity", syncs: "Connectivity", multipoint: "Connectivity",
  stable: "Connectivity",
  price: "Value & Price", priced: "Value & Price", expensive: "Value & Price",
  worth: "Value & Price", value: "Value & Price", cheap: "Value & Price",
  cheaper: "Value & Price", cost: "Value & Price",
  calls: "Calls & Mic", call: "Calls & Mic", mic: "Calls & Mic",
  microphone: "Calls & Mic", speakerphone: "Calls & Mic",
  anc: "Noise Control", isolation: "Noise Control", noise: "Noise Control",
  transparency: "Noise Control", rumble: "Noise Control",
  steps: "Fitness & Health", sleep: "Fitness & Health", heart: "Fitness & Health",
  rate: "Fitness & Health", gps: "Fitness & Health", workout: "Fitness & Health",
  workouts: "Fitness & Health", calories: "Fitness & Health", swim: "Fitness & Health",
  tracking: "Fitness & Health", tracks: "Fitness & Health",
};

const REVIEWS_PER_PAGE = 10;

const state = {
  products: [],
  data: null,
  charts: {},
  liveFeedback: [],
  revealObserver: null,
  reviewPage: 1,
  reviewFiltered: [],
};

function $(id) {
  return document.getElementById(id);
}

function destroyChart(key) {
  if (state.charts[key]) {
    state.charts[key].destroy();
    delete state.charts[key];
  }
}

function chartDefaults() {
  Chart.defaults.color = "#6d6458";
  Chart.defaults.borderColor = "rgba(70, 56, 40, 0.1)";
  Chart.defaults.font.family = "Figtree, sans-serif";
}

function selectedSentiments() {
  return [...document.querySelectorAll("#sentimentFilters input:checked")].map(
    (input) => input.value
  );
}

function ratingRange() {
  let min = Number($("ratingMin").value);
  let max = Number($("ratingMax").value);
  if (min > max) [min, max] = [max, min];
  return { min, max };
}

function filteredReviews() {
  if (!state.data) return [];
  const sentiments = new Set(selectedSentiments());
  const { min, max } = ratingRange();
  return state.data.reviews.filter(
    (row) =>
      sentiments.has(row.sentiment) &&
      row.rating >= min &&
      row.rating <= max
  );
}

function recount(rows) {
  const sentiment = Object.fromEntries(SENTIMENT_ORDER.map((label) => [label, 0]));
  const stars = { 1: 0, 2: 0, 3: 0, 4: 0, 5: 0 };
  let ratingSum = 0;
  for (const row of rows) {
    sentiment[row.sentiment] += 1;
    stars[row.rating] += 1;
    ratingSum += row.rating;
  }
  const n = rows.length || 1;
  return {
    n: rows.length,
    avg_rating: rows.length ? ratingSum / rows.length : 0,
    sentiment_counts: SENTIMENT_ORDER.map((label) => ({
      sentiment: label,
      reviews: sentiment[label],
      share: sentiment[label] / n,
    })),
    star_counts: [1, 2, 3, 4, 5].map((value) => ({
      stars: value,
      reviews: stars[value],
    })),
  };
}

function monthlyFrom(rows) {
  const buckets = new Map();
  for (const row of rows) {
    const date = new Date(row.date + "T00:00:00");
    const key = `${date.getFullYear()}-${String(date.getMonth() + 1).padStart(2, "0")}`;
    if (!buckets.has(key)) {
      buckets.set(key, { key, reviews: 0, ratingSum: 0 });
    }
    const bucket = buckets.get(key);
    bucket.reviews += 1;
    bucket.ratingSum += row.rating;
  }
  return [...buckets.values()]
    .sort((a, b) => a.key.localeCompare(b.key))
    .map((bucket) => {
      const [year, month] = bucket.key.split("-");
      const label = new Date(Number(year), Number(month) - 1, 1).toLocaleString(
        "en-US",
        { month: "short", year: "numeric" }
      );
      return {
        month: label,
        reviews: bucket.reviews,
        avg_rating: bucket.ratingSum / bucket.reviews,
      };
    });
}

async function loadProducts() {
  const response = await fetch("./data/products.json");
  const payload = await response.json();
  state.products = payload.products;
  const select = $("productSelect");
  select.innerHTML = state.products
    .map(
      (product) =>
        `<option value="${product.slug}">${product.name} (${product.reviews})</option>`
    )
    .join("");
  await loadProduct(select.value);
}

async function loadProduct(slug) {
  const response = await fetch(`./data/${slug}.json`);
  state.data = await response.json();
  state.liveFeedback = [];
  renderLiveFeedback();
  renderAll();
  refreshReveals();
}

function renderAll() {
  const data = state.data;
  const rows = filteredReviews();
  const live = recount(rows);
  const { min, max } = ratingRange();

  $("pageTitle").textContent = data.product;
  $("pageLede").innerHTML = `Demo reviews for <b>${data.product}</b>. Keywords are grouped into themes, recent critical feedback is listed separately, and you can try classifying a new review.`;
  $("productMeta").innerHTML = `
    <p><strong>${data.stats.n}</strong> labeled reviews</p>
    <p class="muted">${data.date_range.start} to ${data.date_range.end}</p>
    <p class="muted">Avg ${data.stats.avg_rating.toFixed(1)} / 5 across the full set</p>
    <p class="muted">${data.improvements?.opportunity_count || 0} improvement themes</p>
  `;
  $("ratingLabel").textContent = `${min} – ${max} stars`;
  $("filterNote").textContent =
    rows.length === data.reviews.length
      ? ""
      : `Showing ${rows.length} of ${data.reviews.length} reviews after filters.`;

  renderInsights(data.insights);
  renderMetrics(live);
  renderSentimentChart(live.sentiment_counts);
  renderStarsChart(live.star_counts);
  renderTrendCharts(monthlyFrom(rows));
  renderImprove(data);
  renderLanguage(data);
  renderReviewCards(rows);
  renderModel(data);
}

function renderInsights(lines) {
  $("insightList").innerHTML = lines
    .map((line) => `<div class="insight">${line}</div>`)
    .join("");
}

function renderMetrics(live) {
  const positive = live.sentiment_counts.find((row) => row.sentiment === "Positive");
  const negative = live.sentiment_counts.find((row) => row.sentiment === "Negative");
  $("metricGrid").innerHTML = `
    <div class="reveal"><div class="label">Reviews</div><div class="value">${live.n}</div></div>
    <div class="reveal"><div class="label">Average rating</div><div class="value">${live.avg_rating.toFixed(1)} / 5</div></div>
    <div class="reveal"><div class="label">Positive</div><div class="value">${((positive?.share || 0) * 100).toFixed(0)}%</div></div>
    <div class="reveal"><div class="label">Negative</div><div class="value">${((negative?.share || 0) * 100).toFixed(0)}%</div></div>
  `;
}

function barOrDoughnut(key, canvasId, config) {
  destroyChart(key);
  const ctx = $(canvasId);
  if (!ctx) return;
  state.charts[key] = new Chart(ctx, config);
}

function renderSentimentChart(counts) {
  barOrDoughnut("sentiment", "sentimentChart", {
    type: "doughnut",
    data: {
      labels: counts.map((row) => row.sentiment),
      datasets: [
        {
          data: counts.map((row) => row.reviews),
          backgroundColor: counts.map((row) => COLORS[row.sentiment]),
          borderWidth: 0,
          hoverOffset: 6,
        },
      ],
    },
    options: {
      cutout: "62%",
      plugins: { legend: { position: "bottom" } },
    },
  });
}

function renderStarsChart(counts) {
  barOrDoughnut("stars", "starsChart", {
    type: "bar",
    data: {
      labels: counts.map((row) => `${row.stars}★`),
      datasets: [
        {
          data: counts.map((row) => row.reviews),
          backgroundColor: counts.map((row) =>
            row.stars <= 2 ? COLORS.Negative : row.stars === 3 ? COLORS.Neutral : COLORS.Positive
          ),
          borderRadius: 10,
          borderSkipped: false,
        },
      ],
    },
    options: {
      plugins: { legend: { display: false } },
      scales: {
        x: { grid: { display: false } },
        y: { beginAtZero: true, ticks: { precision: 0 } },
      },
    },
  });
}

function renderTrendCharts(monthly) {
  barOrDoughnut("monthlyCount", "monthlyCountChart", {
    type: "bar",
    data: {
      labels: monthly.map((row) => row.month),
      datasets: [
        {
          data: monthly.map((row) => row.reviews),
          backgroundColor: "#0f766e",
          borderRadius: 10,
          borderSkipped: false,
        },
      ],
    },
    options: {
      plugins: { legend: { display: false } },
      scales: {
        x: { grid: { display: false } },
        y: { beginAtZero: true, ticks: { precision: 0 } },
      },
    },
  });

  barOrDoughnut("monthlyRating", "monthlyRatingChart", {
    type: "line",
    data: {
      labels: monthly.map((row) => row.month),
      datasets: [
        {
          data: monthly.map((row) => Number(row.avg_rating.toFixed(2))),
          borderColor: "#1d4e89",
          backgroundColor: "rgba(29, 78, 137, 0.12)",
          fill: true,
          tension: 0.35,
          pointRadius: 4,
          pointBackgroundColor: "#1d4e89",
        },
      ],
    },
    options: {
      plugins: { legend: { display: false } },
      scales: {
        x: { grid: { display: false } },
        y: { min: 1, max: 5 },
      },
    },
  });
}

function horizontalWords(key, canvasId, rows, color) {
  const ordered = [...rows].reverse();
  barOrDoughnut(key, canvasId, {
    type: "bar",
    data: {
      labels: ordered.map((row) => row.word),
      datasets: [
        {
          data: ordered.map((row) => row.count),
          backgroundColor: color,
          borderRadius: 8,
          borderSkipped: false,
        },
      ],
    },
    options: {
      indexAxis: "y",
      plugins: { legend: { display: false } },
      scales: {
        x: { beginAtZero: true, ticks: { precision: 0 } },
        y: { grid: { display: false } },
      },
    },
  });
}

function keywordChips(items, className = "") {
  return (items || [])
    .map((item) => {
      const label = typeof item === "string" ? item : `${item.word}${item.count ? ` · ${item.count}` : ""}`;
      return `<span class="${className}">${escapeHtml(label)}</span>`;
    })
    .join("");
}

function renderImprove(data) {
  const improvements = data.improvements || { recent: [], opportunities: [] };
  $("recentFeedback").innerHTML = improvements.recent.length
    ? improvements.recent
        .map(
          (row) => `<article class="feedback-card reveal">
            <div class="meta-line">
              <span class="badge ${row.sentiment}">${row.sentiment}</span>
              <span>${row.rating}★</span>
              <span>${row.date}</span>
              <span>${escapeHtml(row.reviewer)}</span>
            </div>
            <p class="review-text">${escapeHtml(row.review)}</p>
            <div class="kw-chips">${keywordChips(row.keywords)}</div>
            <div class="kw-chips">${keywordChips(row.categories, "cat")}</div>
          </article>`
        )
        .join("")
    : `<p class="muted">No recent negative or neutral feedback in this product set.</p>`;

  $("opportunityList").innerHTML = improvements.opportunities.length
    ? improvements.opportunities
        .map(
          (item) => `<article class="opportunity-card reveal">
            <h3>${escapeHtml(item.category)}</h3>
            <p class="muted">${item.negative_mentions} negative mentions</p>
            <p class="review-text">${escapeHtml(item.action)}</p>
            <div class="kw-chips">${keywordChips(item.keywords)}</div>
          </article>`
        )
        .join("")
    : `<p class="muted">No negative category patterns found.</p>`;
}

function renderLanguage(data) {
  const categories = data.categories || [];
  barOrDoughnut("categories", "categoryChart", {
    type: "bar",
    data: {
      labels: categories.map((row) => row.category),
      datasets: [
        {
          label: "Reviews mentioning theme",
          data: categories.map((row) => row.mentions),
          backgroundColor: "#1d4e89",
          borderRadius: 10,
          borderSkipped: false,
        },
      ],
    },
    options: {
      plugins: { legend: { display: false } },
      scales: {
        x: { ticks: { maxRotation: 45, minRotation: 0 } },
        y: { beginAtZero: true, ticks: { precision: 0 } },
      },
    },
  });

  $("categoryCards").innerHTML = categories
    .map((row) => {
      const sentimentLine = SENTIMENT_ORDER.map(
        (label) => `${label}: ${row.by_sentiment[label] || 0}`
      ).join(" · ");
      return `<div class="category-card reveal">
        <h3>${escapeHtml(row.category)}</h3>
        <p class="muted">${row.mentions} reviews · ${sentimentLine}</p>
        <div class="kw-chips">${keywordChips(row.keywords)}</div>
      </div>`;
    })
    .join("");

  const select = $("themeSelect");
  const previous = select.value;
  select.innerHTML = categories
    .map((row) => `<option value="${escapeHtml(row.category)}">${escapeHtml(row.category)} (${row.mentions})</option>`)
    .join("");
  if (previous && categories.some((row) => row.category === previous)) {
    select.value = previous;
  }
  renderThemeReviews();

  horizontalWords("negWords", "negWordsChart", data.distinctive_words.Negative || [], COLORS.Negative);
  horizontalWords("neuWords", "neuWordsChart", data.distinctive_words.Neutral || [], COLORS.Neutral);
  horizontalWords("posWords", "posWordsChart", data.distinctive_words.Positive || [], COLORS.Positive);
  horizontalWords("topWords", "topWordsChart", data.top_words || [], "#0f766e");
}

function sortThemeReviews(rows, mode) {
  const sentimentRank = { Negative: 0, Neutral: 1, Positive: 2 };
  const copy = [...rows];
  copy.sort((a, b) => {
    if (mode === "oldest") return a.date.localeCompare(b.date) || a.review_id - b.review_id;
    if (mode === "rating-asc") return a.rating - b.rating || b.date.localeCompare(a.date);
    if (mode === "rating-desc") return b.rating - a.rating || b.date.localeCompare(a.date);
    if (mode === "negative") {
      return (
        (sentimentRank[a.sentiment] ?? 9) - (sentimentRank[b.sentiment] ?? 9) ||
        a.rating - b.rating ||
        b.date.localeCompare(a.date)
      );
    }
    if (mode === "keywords") {
      const ka = (a.keywords || []).length;
      const kb = (b.keywords || []).length;
      return kb - ka || b.date.localeCompare(a.date);
    }
    // newest
    return b.date.localeCompare(a.date) || b.review_id - a.review_id;
  });
  return copy;
}

function renderThemeReviews() {
  const category = $("themeSelect").value;
  const sortMode = $("themeSort")?.value || "newest";
  if (!category || !state.data) {
    $("themeHint").textContent = "No theme selected.";
    $("themeReviews").innerHTML = "";
    return;
  }

  const matched = (state.data.reviews || []).filter((row) =>
    (row.categories || []).includes(category)
  );
  const sorted = sortThemeReviews(matched, sortMode);
  const view = sorted.slice(0, 40);

  $("themeHint").textContent = `${matched.length} reviews mention ${category}. Sorted by ${
    $("themeSort").selectedOptions[0]?.text || sortMode
  }.`;
  $("themeReviews").innerHTML = view.length
    ? view
        .map((row) => {
          const themeKeywords = (row.category_keywords?.[category] || []).map(
            (item) => item.word
          );
          const fallback = (row.keywords || []).map((item) => item.word);
          return `<article class="theme-card reveal">
            <div class="meta-line">
              <span class="badge ${row.sentiment}">${row.sentiment}</span>
              <span>${row.rating}★</span>
              <span>${row.date}</span>
              <span>${escapeHtml(row.reviewer)}</span>
            </div>
            <p class="review-text">${escapeHtml(row.review)}</p>
            <div class="kw-chips">${keywordChips(themeKeywords.length ? themeKeywords : fallback)}</div>
          </article>`;
        })
        .join("")
    : `<p class="muted">No reviews found for this theme.</p>`;

  if (matched.length > 40) {
    $("themeReviews").innerHTML += `<p class="muted">Showing first 40 of ${matched.length}. Change sort or filters to focus.</p>`;
  }
  revealPanel($("panel-themes"));
}

function renderReviewCards(rows, resetPage = true) {
  const query = $("reviewSearch").value.trim().toLowerCase();
  const view = query
    ? rows.filter((row) => {
        const hay = [
          row.review,
          ...(row.keywords || []).map((item) => item.word),
          ...(row.categories || []),
        ]
          .join(" ")
          .toLowerCase();
        return hay.includes(query);
      })
    : rows;

  state.reviewFiltered = view;
  state.visibleRows = view;
  if (resetPage) state.reviewPage = 1;
  renderReviewPage();
}

function renderReviewPage() {
  const view = state.reviewFiltered || [];
  const totalPages = Math.max(1, Math.ceil(view.length / REVIEWS_PER_PAGE));
  if (state.reviewPage > totalPages) state.reviewPage = totalPages;
  if (state.reviewPage < 1) state.reviewPage = 1;

  const start = (state.reviewPage - 1) * REVIEWS_PER_PAGE;
  const pageRows = view.slice(start, start + REVIEWS_PER_PAGE);
  const from = view.length ? start + 1 : 0;
  const to = Math.min(start + REVIEWS_PER_PAGE, view.length);

  $("reviewCount").textContent = view.length
    ? `Showing ${from}–${to} of ${view.length} review${view.length === 1 ? "" : "s"} · page ${state.reviewPage} of ${totalPages}`
    : "0 reviews in this view.";

  $("reviewCards").innerHTML = pageRows.length
    ? pageRows
        .map(
          (row) => `<article class="review-card reveal">
            <div class="meta-line">
              <span class="badge ${row.sentiment}">${row.sentiment}</span>
              <span>${row.rating}★</span>
              <span>${row.date}</span>
              <span>${escapeHtml(row.reviewer)}</span>
              <span>${row.word_count} words</span>
            </div>
            <p class="review-text">${escapeHtml(row.review)}</p>
            <div class="kw-chips">${keywordChips(row.keywords)}</div>
            <div class="kw-chips">${keywordChips(row.categories, "cat")}</div>
          </article>`
        )
        .join("")
    : `<p class="muted">No reviews match the current filters.</p>`;

  const pager = $("reviewPagination");
  const buttons = $("reviewPageButtons");
  if (!view.length) {
    pager.hidden = true;
  } else {
    pager.hidden = false;
    $("reviewPrev").disabled = state.reviewPage <= 1;
    $("reviewNext").disabled = state.reviewPage >= totalPages;
    buttons.innerHTML = Array.from({ length: totalPages }, (_, index) => {
      const page = index + 1;
      const active = page === state.reviewPage ? "active" : "";
      return `<button type="button" class="page-btn ${active}" data-page="${page}">${page}</button>`;
    }).join("");
  }

  if ($("panel-reviews")?.classList.contains("active")) {
    revealPanel($("panel-reviews"));
  }
}

function escapeHtml(text) {
  return String(text)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

function renderModel(data) {
  const model = data.model;
  $("modelMetrics").innerHTML = `
    <div class="reveal"><div class="label">Cross-validated accuracy</div><div class="value">${(
      model.accuracy * 100
    ).toFixed(0)}%</div></div>
    <div class="reveal"><div class="label">Training reviews</div><div class="value">${data.stats.n}</div></div>
  `;

  const labels = model.labels;
  barOrDoughnut("matrix", "matrixChart", {
    type: "bar",
    data: {
      labels,
      datasets: labels.map((predicted, index) => ({
        label: `Predicted ${predicted}`,
        data: model.matrix.map((row) => row[index]),
        backgroundColor: COLORS[predicted],
        borderRadius: 6,
        stack: "matrix",
      })),
    },
    options: {
      plugins: { legend: { position: "bottom" } },
      scales: {
        x: { stacked: true, title: { display: true, text: "Actual" } },
        y: { stacked: true, beginAtZero: true, ticks: { precision: 0 } },
      },
    },
  });

  const grouped = {};
  for (const row of model.top_words) {
    grouped[row.sentiment] ||= [];
    grouped[row.sentiment].push(row.word);
  }
  $("modelWords").innerHTML = SENTIMENT_ORDER.map((label) => {
    const words = (grouped[label] || []).map((word) => `<span>${escapeHtml(word)}</span>`).join("");
    return `<div class="group reveal"><strong class="badge ${label}">${label}</strong><div class="kw-chips" style="margin-top:0.5rem">${words}</div></div>`;
  }).join("");
}

function tokenizeClient(text) {
  return (
    text
      .toLowerCase()
      .match(/[a-z]+/g)
      ?.filter((word) => word.length > 2 && !FALLBACK_STOP.has(word)) || []
  );
}

function extractClient(text) {
  const tokens = tokenizeClient(text);
  const categories = new Set();
  for (const word of tokens) {
    if (CATEGORY_MAP[word]) categories.add(CATEGORY_MAP[word]);
  }
  return {
    tokens,
    keywords: [...new Set(tokens)].slice(0, 12),
    categories: [...categories].sort(),
  };
}

function predictLocal(text) {
  const model = state.data.model.export;
  const extracted = extractClient(text);
  const tokens = extracted.tokens;
  const vocab = model.vocabulary;
  const idf = model.idf;
  const [minN, maxN] = model.ngram_range;
  const features = new Map();

  for (let n = minN; n <= maxN; n += 1) {
    for (let i = 0; i <= tokens.length - n; i += 1) {
      const term = tokens.slice(i, i + n).join(" ");
      if (term in vocab) features.set(term, (features.get(term) || 0) + 1);
    }
  }

  const vector = new Array(idf.length).fill(0);
  let sumSquares = 0;
  for (const [term, count] of features.entries()) {
    const index = vocab[term];
    const value = count * idf[index];
    vector[index] = value;
    sumSquares += value * value;
  }
  const norm = Math.sqrt(sumSquares) || 1;
  for (let i = 0; i < vector.length; i += 1) vector[i] /= norm;

  const scores = model.classes.map((label, classIndex) => {
    let score = model.class_log_prior[classIndex];
    const weights = model.feature_log_prob[classIndex];
    for (let i = 0; i < vector.length; i += 1) {
      if (vector[i]) score += vector[i] * weights[i];
    }
    return { sentiment: label, score };
  });

  const maxScore = Math.max(...scores.map((row) => row.score));
  const exps = scores.map((row) => Math.exp(row.score - maxScore));
  const total = exps.reduce((sum, value) => sum + value, 0);
  const probabilities = scores.map((row, index) => ({
    sentiment: row.sentiment,
    probability: exps[index] / total,
  }));
  probabilities.sort(
    (a, b) => SENTIMENT_ORDER.indexOf(a.sentiment) - SENTIMENT_ORDER.indexOf(b.sentiment)
  );
  const best = probabilities.reduce((a, b) =>
    a.probability >= b.probability ? a : b
  );
  return { label: best.sentiment, probabilities, ...extracted };
}

function renderPrediction(result, originalText) {
  const box = $("predictResult");
  box.hidden = false;
  const bars = result.probabilities
    .map((row) => {
      const pct = (row.probability * 100).toFixed(1);
      return `<div style="margin:0.35rem 0">
        <div style="display:flex;justify-content:space-between;font-size:0.85rem">
          <span>${row.sentiment}</span><span>${pct}%</span>
        </div>
        <div style="height:8px;border-radius:999px;background:rgba(70,56,40,0.08);overflow:hidden">
          <div style="width:${pct}%;height:100%;background:${COLORS[row.sentiment]}"></div>
        </div>
      </div>`;
    })
    .join("");
  box.innerHTML = `
    <span class="pred" style="color:${COLORS[result.label]}">${result.label}</span>
    <p class="muted">Keywords mentioned: ${result.keywords.join(" · ") || "none"}</p>
    <div class="kw-chips">${keywordChips(result.keywords)}</div>
    <div class="kw-chips">${keywordChips(result.categories, "cat")}</div>
    ${bars}
  `;

  state.liveFeedback.unshift({
    text: originalText,
    label: result.label,
    keywords: result.keywords,
    categories: result.categories,
    at: new Date().toLocaleTimeString(),
  });
  state.liveFeedback = state.liveFeedback.slice(0, 8);
  renderLiveFeedback();
}

function renderLiveFeedback() {
  const tray = $("liveFeedbackTray");
  const list = $("liveFeedbackList");
  if (!state.liveFeedback.length) {
    tray.hidden = true;
    list.innerHTML = "";
    return;
  }
  tray.hidden = false;
  list.innerHTML = state.liveFeedback
    .map(
      (item) => `<article class="live-item reveal">
        <div class="meta-line">
          <span class="badge ${item.label}">${item.label}</span>
          <span>${item.at}</span>
        </div>
        <p class="review-text">${escapeHtml(item.text)}</p>
        <div class="kw-chips">${keywordChips(item.keywords)}</div>
        <div class="kw-chips">${keywordChips(item.categories, "cat")}</div>
      </article>`
    )
    .join("");
  refreshReveals();
}

function downloadCsv() {
  const rows = state.visibleRows || [];
  const header = ["Date", "Reviewer", "Stars", "Sentiment", "Words", "Keywords", "Categories", "Review"];
  const lines = [header.join(",")].concat(
    rows.map((row) =>
      [
        row.date,
        row.reviewer,
        row.rating,
        row.sentiment,
        row.word_count,
        `"${(row.keywords || []).map((item) => item.word).join("; ")}"`,
        `"${(row.categories || []).join("; ")}"`,
        `"${row.review.replaceAll('"', '""')}"`,
      ].join(",")
    )
  );
  const blob = new Blob([lines.join("\n")], { type: "text/csv;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = `${state.data.product.replaceAll(" ", "_").toLowerCase()}_reviews.csv`;
  link.click();
  URL.revokeObjectURL(url);
}

function setupTabs() {
  document.querySelectorAll(".tab").forEach((button) => {
    button.addEventListener("click", () => {
      document.querySelectorAll(".tab").forEach((tab) => {
        tab.classList.toggle("active", tab === button);
        tab.setAttribute("aria-selected", tab === button ? "true" : "false");
      });
      document.querySelectorAll(".tab-panel").forEach((panel) => {
        panel.classList.toggle("active", panel.id === `panel-${button.dataset.tab}`);
      });
      const activePanel = document.getElementById(`panel-${button.dataset.tab}`);
      // Hidden tabs keep opacity 0 from .reveal — force-show the active panel.
      requestAnimationFrame(() => revealPanel(activePanel));
    });
  });
}

function setupExamples() {
  $("exampleButtons").innerHTML = EXAMPLES.map(
    (item) => `<button type="button" data-text="${escapeHtml(item.text)}">${item.label}</button>`
  ).join("");
  $("exampleButtons").addEventListener("click", (event) => {
    const button = event.target.closest("button");
    if (!button) return;
    $("predictInput").value = button.dataset.text;
  });
}

function revealPanel(panel) {
  if (!panel) return;
  const nodes = panel.querySelectorAll(".reveal");
  if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) {
    nodes.forEach((node) => node.classList.add("is-visible"));
    return;
  }
  nodes.forEach((node, index) => {
    node.classList.remove("is-visible");
    window.setTimeout(() => node.classList.add("is-visible"), 40 + index * 35);
  });
}

function refreshReveals() {
  const reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const activePanel = document.querySelector(".tab-panel.active");
  const alwaysVisible = [document.getElementById("sidebar"), document.querySelector(".hero")].filter(
    Boolean
  );

  if (state.revealObserver) state.revealObserver.disconnect();

  if (reduce) {
    document.querySelectorAll(".reveal").forEach((node) => node.classList.add("is-visible"));
    return;
  }

  alwaysVisible.forEach((node) => {
    node.classList.add("is-visible");
    node.querySelectorAll(".reveal").forEach((child) => child.classList.add("is-visible"));
  });

  // Content inside hidden tabs stays invisible until the tab is opened.
  document.querySelectorAll(".tab-panel:not(.active) .reveal").forEach((node) => {
    node.classList.remove("is-visible");
  });

  revealPanel(activePanel);

  state.revealObserver = new IntersectionObserver(
    (entries) => {
      for (const entry of entries) {
        if (entry.isIntersecting) {
          entry.target.classList.add("is-visible");
          state.revealObserver.unobserve(entry.target);
        }
      }
    },
    { threshold: 0.08, rootMargin: "0px 0px -4% 0px" }
  );

  if (activePanel) {
    activePanel.querySelectorAll(".reveal:not(.is-visible)").forEach((node) => {
      state.revealObserver.observe(node);
    });
  }
}

function bindControls() {
  $("productSelect").addEventListener("change", (event) => loadProduct(event.target.value));
  $("sentimentFilters").addEventListener("change", () => {
    renderAll();
    refreshReveals();
  });
  $("ratingMin").addEventListener("input", () => {
    renderAll();
    refreshReveals();
  });
  $("ratingMax").addEventListener("input", () => {
    renderAll();
    refreshReveals();
  });
  $("reviewSearch").addEventListener("input", () => renderReviewCards(filteredReviews(), true));
  $("themeSelect").addEventListener("change", renderThemeReviews);
  $("themeSort").addEventListener("change", renderThemeReviews);
  $("downloadCsv").addEventListener("click", downloadCsv);
  $("reviewPrev").addEventListener("click", () => {
    state.reviewPage -= 1;
    renderReviewPage();
  });
  $("reviewNext").addEventListener("click", () => {
    state.reviewPage += 1;
    renderReviewPage();
  });
  $("reviewPageButtons").addEventListener("click", (event) => {
    const button = event.target.closest("[data-page]");
    if (!button) return;
    state.reviewPage = Number(button.dataset.page);
    renderReviewPage();
  });
  $("predictBtn").addEventListener("click", () => {
    const text = $("predictInput").value.trim();
    if (!text) {
      $("predictResult").hidden = false;
      $("predictResult").innerHTML = `<p class="muted">Type a review first, or use one of the examples.</p>`;
      return;
    }
    renderPrediction(predictLocal(text), text);
  });
}

async function init() {
  chartDefaults();
  setupTabs();
  setupExamples();
  bindControls();
  try {
    await loadProducts();
  } catch (error) {
    $("pageLede").textContent =
      "Could not load demo data. Run `py build_site.py` then serve the docs/ folder.";
    console.error(error);
  }
}

document.addEventListener("DOMContentLoaded", init);
