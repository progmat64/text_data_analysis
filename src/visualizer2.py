import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
from sklearn.feature_extraction.text import CountVectorizer
from gensim import corpora, models
from sentence_transformers import SentenceTransformer
import umap
import hdbscan
from bertopic import BERTopic

df = pd.read_excel(
    "data/wb_reviews_11524.xlsx",
    engine="openpyxl",
    parse_dates=["review_date"]
)

df["text"] = df["comment"].fillna("").astype(str)

rus_stop = set(STOPWORDS)
eng_stop = set(WordCloud().stopwords)

vectorizer = CountVectorizer(
    token_pattern=r"\b[А-Яа-яЁё]{3,}\b",
    stop_words=list(rus_stop | eng_stop),
    max_df=0.9,
    min_df=5
)
X = vectorizer.fit_transform(df["text"])
words = vectorizer.get_feature_names_out()
counts = X.sum(axis=0).A1
freq = dict(zip(words, counts))
top20 = dict(sorted(freq.items(), key=lambda x: x[1], reverse=True)[:20])

plt.figure(figsize=(10, 5))
plt.bar(range(len(top20)), list(top20.values()), align="center")
plt.xticks(range(len(top20)), list(top20.keys()), rotation=45, ha="right")
plt.title("Топ-20 слов")
plt.xlabel("Слово")
plt.ylabel("Частота")
plt.tight_layout()
plt.show()

wc = WordCloud(
    width=800, height=400,
    background_color="white",
    stopwords=rus_stop | eng_stop
).generate(" ".join(df["text"]))
plt.figure(figsize=(12, 6))
plt.imshow(wc, interpolation="bilinear")
plt.axis("off")
plt.show()



tokenized = [doc.split() for doc in df["text"]]
dictionary = corpora.Dictionary(tokenized)
corpus = [dictionary.doc2bow(text) for text in tokenized]

lda = models.LdaModel(
    corpus=corpus,
    id2word=dictionary,
    num_topics=5,
    random_state=42,
    passes=10
)

lda_topics = [max(lda[doc], key=lambda x: x[1])[0] for doc in corpus]
df["topic_lda"] = lda_topics
dist_lda = df["topic_lda"].value_counts().sort_index()

plt.figure(figsize=(8, 4))
plt.bar(dist_lda.index, dist_lda.values)
plt.title("Распределение тем LDA")
plt.xlabel("Номер темы")
plt.ylabel("Кол-во документов")
plt.tight_layout()
plt.show()


embedder = SentenceTransformer("distiluse-base-multilingual-cased-v1")
embeddings = embedder.encode(df["text"].tolist(), show_progress_bar=True)

umap_model = umap.UMAP(random_state=42)
umap_emb = umap_model.fit_transform(embeddings)

clusterer = hdbscan.HDBSCAN(min_cluster_size=10, prediction_data=True, core_dist_n_jobs=4)
clusters = clusterer.fit_predict(umap_emb)

plt.figure(figsize=(8, 6))
sc = plt.scatter(umap_emb[:, 0], umap_emb[:, 1], c=clusters, cmap="tab10", s=10)
plt.colorbar(sc, label="Кластер HDBSCAN")
plt.title("UMAP + HDBSCAN кластеризация эмбеддингов")
plt.xlabel("UMAP 1")
plt.ylabel("UMAP 2")
plt.tight_layout()
plt.show()

topic_model = BERTopic(
    umap_model=umap_model,
    hdbscan_model=clusterer,
    calculate_probabilities=True,
    random_state=42,
    verbose=False
)
bt_topics, bt_probs = topic_model.fit_transform(df["text"].tolist())
df["topic_bt"] = bt_topics

fig1 = topic_model.visualize_barchart(top_n_topics=5)
fig1.show()

fig2 = topic_model.visualize_heatmap()
fig2.show()

df_time = df.dropna(subset=["review_date"]).set_index("review_date")
monthly = (
    df_time
    .groupby([pd.Grouper(freq="M"), "topic_bt"])
    .size()
    .unstack(fill_value=0)
)
monthly_prop = monthly.div(monthly.sum(axis=1), axis=0)

plt.figure(figsize=(10, 6))
for topic in monthly_prop.columns[:5]:
    plt.plot(monthly_prop.index, monthly_prop[topic], linewidth=2, label=f"Topic {topic}")
plt.legend(title="Тема")
plt.title("Доля топ-5 тем BERTopic во времени")
plt.xlabel("Дата")
plt.ylabel("Доля")
plt.tight_layout()
plt.show()
