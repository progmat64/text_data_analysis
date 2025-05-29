import os
import random
import numpy as np
import torch
import warnings

import streamlit as st
import pandas as pd
from data_loader import load_and_preprocess
from review_classifier import ReviewClassifier
import plotly.express as px
import plotly.graph_objects as go
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.exceptions import InconsistentVersionWarning

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

st.set_page_config(
    page_title="Анализ отзывов: темы и тональности",
    layout="wide"
)

def main():
    st.title("Интерактивный анализ отзывов: ключевые темы и тональность")

    try:
        df = load_and_preprocess(
            path_pattern="data/wb_reviews_*.xlsx",
            text_cols=['advantages', 'disadvantages', 'comment'],
            date_col=None
        )
    except FileNotFoundError as e:
        st.error(f"Не удалось загрузить файлы: {e}")
        return

    classifier = ReviewClassifier(
        sentiment_model_name="cointegrated/rubert-tiny2",
        topic_model_path=os.path.join("models", "topic_multilabel_model.joblib"),
        topic_encoder_path=os.path.join("models", "topic_mlb.joblib"),
        topic_embedder_path=os.path.join("models", "sbert_model.joblib"),
    )
    df = classifier.run(df)

    if 'review_date' in df.columns:
        df['review_date'] = pd.to_datetime(df['review_date'], errors='coerce')

    tab_overview, tab_trends = st.tabs(["Обзор", "Тренды"])

    with tab_overview:
        st.subheader("1. Тональность и темы (сводка)")
        col1, col2 = st.columns(2, gap="medium")

        with col1:
            counts = (
                df['sentiment']
                .value_counts()
                .reindex(['negative','neutral','positive'], fill_value=0)
            )
            df_counts = counts.rename_axis('sentiment').reset_index(name='count')
            df_counts['sentiment'] = df_counts['sentiment'].str.capitalize()

            fig1 = px.bar(df_counts,
                          x='sentiment', y='count', text='count',
                          labels={'sentiment':'Тональность','count':'Количество'},
                          title="Распределение тональности")
            fig1.update_layout(yaxis=dict(showgrid=True), uniformtext_minsize=12)
            st.plotly_chart(fig1, use_container_width=True)

        prim = (
            df['topics']
            .apply(lambda lst: lst[0] if isinstance(lst, list) and lst else None)
            .value_counts()
            .nlargest(8)
        )
        if not prim.empty:
            with col2:
                df_prim = prim.rename_axis('topic').reset_index(name='count')
                fig2 = px.bar(df_prim,
                              x='topic', y='count', text='count',
                              labels={'topic':'Тема','count':'Количество'},
                              title="Топ-8 тем (первичные)")
                fig2.update_layout(yaxis=dict(showgrid=True), xaxis_tickangle=-45)
                st.plotly_chart(fig2, use_container_width=True)

        st.markdown("---")
        st.subheader("2. Длина отзывов и комментариев")
        col3, col4 = st.columns(2, gap="medium")

        with col3:
            comment_lengths = df['comment'].astype(str).str.split().str.len()
            fig3 = px.histogram(pd.DataFrame({'length':comment_lengths}),
                                x='length', nbins=20,
                                labels={'length':'Слов','count':'Частота'},
                                title="Длина комментариев")
            fig3.update_traces(marker_line_width=1, marker_line_color='black', opacity=0.7)
            st.plotly_chart(fig3, use_container_width=True)

        with col4:
            text_lengths = df['text'].astype(str).str.split().str.len()
            fig4 = px.histogram(pd.DataFrame({'length':text_lengths}),
                                x='length', nbins=30,
                                labels={'length':'Слов','count':'Частота'},
                                title="Длина полного текста")
            fig4.update_traces(marker_line_width=1, marker_line_color='black', opacity=0.7)
            st.plotly_chart(fig4, use_container_width=True)

        st.markdown("---")
        st.subheader("3. Топ-20 слов корпуса")
        corpus = df['text'].astype(str).tolist()
        vect = CountVectorizer(
            max_df=0.9, min_df=5,
            stop_words='english',
            token_pattern=r'\b[А-Яа-яЁё]{3,}\b'
        )
        X = vect.fit_transform(corpus)
        freqs = sorted(zip(vect.get_feature_names_out(), X.sum(axis=0).A1),
                       key=lambda x: -x[1])[:20]
        if freqs:
            words, counts_w = zip(*freqs)
            df_words = pd.DataFrame({'word':words, 'count':counts_w})
            fig5 = px.bar(df_words,
                          x='word', y='count', text='count',
                          labels={'word':'Слово','count':'Частота'},
                          title="Топ-20 слов корпуса")
            fig5.update_layout(xaxis_tickangle=-45, yaxis=dict(showgrid=True))
            st.plotly_chart(fig5, use_container_width=True)

    with tab_trends:
        st.subheader("4. Динамика тональности во времени")
        if 'review_date' in df.columns and df['review_date'].notna().any():
            df_time = df.dropna(subset=['review_date']).set_index('review_date')
            sent_trends = (
                df_time
                .groupby([pd.Grouper(freq='M'),'sentiment'])
                .size()
                .unstack(fill_value=0)
                .pipe(lambda x: x.div(x.sum(axis=1), axis=0))
            )
            fig6 = px.line(sent_trends,
                           labels={'value':'Доля','review_date':'Дата','variable':'Тональность'},
                           title="Доля тональностей во времени")
            fig6.update_traces(line=dict(width=3))
            st.plotly_chart(fig6, use_container_width=True)
        else:
            st.info("Нет данных с датами для тренда тональности.")

        st.markdown("---")
        st.subheader("5. Динамика упоминаний тем")
        if 'review_date' in df.columns and df['review_date'].notna().any():
            df_time['primary_topic'] = df_time['topics'].apply(
                lambda lst: lst[0] if isinstance(lst, list) and lst else None
            )
            topic_trends = (
                df_time
                .groupby([pd.Grouper(freq='M'),'primary_topic'])
                .size()
                .unstack(fill_value=0)
            )
            top_ts = topic_trends.sum().nlargest(5).index
            fig7 = px.line(topic_trends[top_ts],
                           labels={'value':'Кол-во','review_date':'Дата','variable':'Тема'},
                           title="Тренды топ-5 тем")
            fig7.update_traces(line=dict(width=3))
            st.plotly_chart(fig7, use_container_width=True)
        else:
            st.info("Нет данных с датами для тренда тем.")

        st.markdown("---")
        st.subheader("6. Матрица: тема vs тональность")
        exploded = df.explode('topics').dropna(subset=['topics'])
        heat = exploded.pivot_table(
            index='topics', columns='sentiment',
            values='text', aggfunc='count', fill_value=0
        )
        if not heat.empty:
            fig8 = go.Figure(go.Heatmap(
                z=heat.values,
                x=heat.columns,
                y=heat.index,
                colorscale="Blues"
            ))
            fig8.update_layout(
                title="Тема vs Тональность",
                xaxis_title="Тональность",
                yaxis_title="Тема",
                height=600
            )
            st.plotly_chart(fig8, use_container_width=True)
        else:
            st.warning("Недостаточно данных для heatmap тема×тональность.")

    topic_lists = df['topics'].dropna().tolist()
    all_topics = sorted({t for lst in topic_lists if isinstance(lst, list) for t in lst})
    if all_topics:
        st.markdown("---")
        st.subheader("7. Тональность по теме")
        sel = st.selectbox("Выберите тему", all_topics)
        subset = df[df['topics'].apply(lambda lst: sel in lst if isinstance(lst, list) else False)]
        ct = subset['sentiment'].value_counts().reindex(
            ['negative','neutral','positive'], fill_value=0
        )
        df_ct = ct.rename_axis('sentiment').reset_index(name='count')
        df_ct['sentiment'] = df_ct['sentiment'].str.capitalize()
        fig9 = px.bar(df_ct,
                      x='sentiment', y='count', text='count',
                      labels={'sentiment':'Тональность','count':'Количество'},
                      title=f"Тональность по теме «{sel}»")
        fig9.update_layout(yaxis=dict(showgrid=True))
        st.plotly_chart(fig9, use_container_width=True)


if __name__ == "__main__":
    main()
