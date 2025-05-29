import pandas as pd
import joblib
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import LogisticRegression


def combine_text_fields(row):
    parts = []
    for field in ['advantages', 'disadvantages', 'comment']:
        val = row.get(field)
        if pd.isna(val):
            continue
        parts.append(str(val).strip())
    return '. '.join(parts)


def extract_topics(text):
    text = text.lower()
    topics = []
    if any(w in text for w in ['доставк', 'привез', 'курьер']):
        topics.append('доставка')
    if any(w in text for w in ['упаковк', 'коробк', 'тара']):
        topics.append('упаковка')
    if any(w in text for w in ['качество', 'брак', 'слом', 'порвал']):
        topics.append('качество')
    if any(w in text for w in ['цена', 'стоимость', 'дешево']):
        topics.append('цена')
    if any(w in text for w in ['запах', 'аромат', 'воня']):
        topics.append('запах')
    return topics or ['прочее']


def train_model(data_path: str):
    df = pd.read_excel(data_path)
    df['text'] = df.apply(combine_text_fields, axis=1)
    df['topics'] = df['text'].apply(extract_topics)

    X_texts = df['text'].tolist()
    y_topics = df['topics'].tolist()

    sbert = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    X = sbert.encode(X_texts, convert_to_numpy=True)

    mlb = MultiLabelBinarizer()
    Y = mlb.fit_transform(y_topics)

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    clf = MultiOutputClassifier(LogisticRegression(max_iter=1000, class_weight="balanced"))
    clf.fit(X_train, y_train)

    joblib.dump(clf, "topic_multilabel_model.joblib")
    joblib.dump(mlb, "topic_mlb.joblib")
    joblib.dump(sbert, "sbert_model.joblib")

    print("Модель тем успешно обучена и сохранена.")
    print("Темы:", mlb.classes_)


if __name__ == "__main__":
    train_model("wb_reviews_153403514_144202977_20250214_0327.xlsx")
