import os
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from detection.url.features import extract_url_features

DATA_PATH = os.path.join(os.path.dirname(__file__), 'data', 'url_dataset.csv')
MODEL_OUT = os.path.join(os.path.dirname(__file__), '..', '..', 'models', 'url_model.pkl')


def train():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"URL dataset not found: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH, encoding='utf-8')

    features, labels = [], []
    for _, row in df.iterrows():
        try:
            features.append(extract_url_features(row['url']))
            labels.append(row['label'])
        except Exception as e:
            print(f"[url] skipping {row['url']}: {e}")

    X = np.array(features)
    y = np.array(labels)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    clf = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
    clf.fit(X_train, y_train)
    acc = accuracy_score(y_test, clf.predict(X_test))
    print(f"[url] accuracy: {acc:.2%}  ({len(df)} samples)")

    os.makedirs(os.path.dirname(MODEL_OUT), exist_ok=True)
    joblib.dump(clf, MODEL_OUT)
    print(f"[url] saved -> {MODEL_OUT}")
    return clf


if __name__ == '__main__':
    train()
