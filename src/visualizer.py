import os
import random
import numpy as np
import torch
import joblib
from transformers import pipeline
from umap import UMAP
from hdbscan import HDBSCAN
from bertopic import BERTopic

class ReviewClassifier:
    def __init__(
        self,
        sentiment_model_name: str,
        topic_model_path: str,
        topic_encoder_path: str,
        topic_embedder_path: str,
    ):
        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)

        self.sentiment = pipeline(
            "sentiment-analysis",
            model=sentiment_model_name,
            device=0 if torch.cuda.is_available() else -1,
            framework="pt",
        )

        umap_model = UMAP(
            n_neighbors=15,
            min_dist=0.1,
            random_state=42,
            metric='cosine'
        )
        hdbscan_model = HDBSCAN(
            min_cluster_size=10,
            prediction_data=True,
            random_state=42
        )
        self.topic_model = BERTopic(
            umap_model=umap_model,
            hdbscan_model=hdbscan_model,
            calculate_probabilities=True,
            verbose=False,
            random_state=42
        )

        self.topic_encoder = joblib.load(topic_encoder_path)
        self.topic_embedder = joblib.load(topic_embedder_path)

        if os.path.exists(topic_model_path):
            self.topic_model = joblib.load(topic_model_path)

    def run(self, df):
        df['sentiment'] = df['comment'].fillna("").apply(
            lambda txt: self.sentiment(txt)[0]['label'].lower()
        )

        docs = df['text'].astype(str).tolist()
        topics, _ = self.topic_model.fit_transform(docs)

        df['topics'] = self.topic_encoder.inverse_transform(self.topic_encoder.transform(topics))
        return df
