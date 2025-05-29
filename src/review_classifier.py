import pandas as pd
import unicodedata
import torch
import numpy as np
import joblib
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer
from typing import List


class ReviewClassifier:
    def __init__(
        self,
        sentiment_model_name: str = "cointegrated/rubert-tiny2",
        topic_model_path: str = "models/topic_multilabel_model.joblib",
        topic_encoder_path: str = "models/topic_mlb.joblib",
        topic_embedder_path: str = "models/sbert_model.joblib",
    ):
        self.sentiment_tokenizer = AutoTokenizer.from_pretrained(sentiment_model_name)
        self.sentiment_model = AutoModelForSequenceClassification.from_pretrained(
            sentiment_model_name, num_labels=3
        )
        self.sentiment_model.eval()

        self.topic_clf = joblib.load(topic_model_path)
        self.topic_encoder = joblib.load(topic_encoder_path)
        self.topic_embedder = joblib.load(topic_embedder_path)

    @staticmethod
    def _remove_emojis(text: str) -> str:
        return ''.join(ch for ch in text if unicodedata.category(ch) not in {'So', 'Sk'})

    @staticmethod
    def _is_uninformative(row) -> bool:
        for col in ['advantages', 'disadvantages', 'comment']:
            if pd.notna(row.get(col)) and len(str(row[col]).split()) > 1:
                return False
        return True

    @staticmethod
    def _combine_text(row) -> str:
        return (
            f"Достоинства: {row.get('advantages', '')}; "
            f"Недостатки: {row.get('disadvantages', '')}; "
            f"Комментарий: {row.get('comment', '')}"
        )

    def filter_informative(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df['drop'] = df.apply(self._is_uninformative, axis=1)
        df = df[~df['drop']].copy()
        df['text'] = df.apply(self._combine_text, axis=1)
        return df

    def classify_sentiment(self, text: str) -> str:
        tokens = self.sentiment_tokenizer(
            text, return_tensors="pt", padding=True,
            truncation=True, max_length=256
        )
        with torch.no_grad():
            logits = self.sentiment_model(**tokens).logits
            probs = torch.softmax(logits, dim=1).squeeze().numpy()
        return ["negative", "neutral", "positive"][np.argmax(probs)]

    def classify_topics(self, text: str) -> List[str]:
        emb = self.topic_embedder.encode([text])
        pred = self.topic_clf.predict(emb)
        labels = self.topic_encoder.inverse_transform(pred)
        return labels[0] if labels else []

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        for col in ['advantages', 'disadvantages', 'comment']:
            if col in df.columns:
                df[col] = df[col].astype(str).apply(self._remove_emojis)

        df = self.filter_informative(df)
        df['sentiment'] = df['text'].apply(self.classify_sentiment)
        df['topics'] = df['text'].apply(self.classify_topics)
        return df