import os
from data_loader import load_and_preprocess
from review_classifier import ReviewClassifier
from visualizer import (
    plot_topic_distribution,
    plot_sentiment_distribution,
    plot_sentiment_by_topic,
    plot_review_length_distribution,
    plot_top_words,
    plot_sentiment_trend,
    plot_topic_trends,
    plot_topic_sentiment_heatmap,
)

def main():
    df = load_and_preprocess(
        path_pattern='data/wb_reviews_*.xlsx',
        text_cols=['advantages', 'disadvantages', 'comment'],
        date_col='review_date'
    )

    classifier = ReviewClassifier(
        topic_model_path=os.path.join('models', 'topic_multilabel_model.joblib'),
        topic_encoder_path=os.path.join('models', 'topic_mlb.joblib'),
        topic_embedder_path=os.path.join('models', 'sbert_model.joblib'),
        sentiment_model_name='cointegrated/rubert-tiny2'
    )

    df = classifier.run(df)

    plot_sentiment_distribution(df)

    plot_topic_distribution(df, top_n=8, primary_only=True)
    plot_topic_distribution(df, top_n=8, primary_only=False)

    plot_review_length_distribution(df)

    plot_top_words(df, top_n=20, stop_words='russian')

    plot_topic_sentiment_heatmap(df)

    plot_sentiment_trend(df, date_col='review_date', freq='M')
    plot_topic_trends(df, date_col='review_date', top_n=5, freq='M', primary_only=True)

    plot_sentiment_by_topic(df, topic='доставка')


if __name__ == "__main__":
    main()
