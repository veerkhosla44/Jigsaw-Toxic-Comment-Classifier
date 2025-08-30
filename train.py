from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import joblib

ROOT = Path(__file__).resolve().parents[1]
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

def plot_confusion_matrix(cm, classes, title, out_path):
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, interpolation='nearest')
    ax.figure.colorbar(im, ax=ax)
    ax.set_xticks(np.arange(len(classes)))
    ax.set_yticks(np.arange(len(classes)))
    ax.set_xticklabels(classes, rotation=45, ha="right")
    ax.set_yticklabels(classes)
    ax.set_ylabel('True label')
    ax.set_xlabel('Predicted label')
    ax.set_title(title)
    fig.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)

def main():
    df = pd.read_csv(DATA_DIR / "train.csv")
    X = df["text"].astype(str).values
    y = df["label"].values

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1,2), min_df=2)),
        ("clf", LogisticRegression(max_iter=2000, class_weight="balanced"))
    ])

    pipe.fit(X_train, y_train)
    joblib.dump(pipe, MODELS_DIR / "model.joblib")

    # Validation
    y_pred = pipe.predict(X_val)
    y_proba = pipe.predict_proba(X_val)[:,1]
    report = classification_report(y_val, y_pred, digits=4)
    auc = roc_auc_score(y_val, y_proba)
    (REPORTS_DIR / "classification_report_val.txt").write_text(f"ROC-AUC: {auc:.4f}\n\n{report}")

    cm = confusion_matrix(y_val, y_pred)
    plot_confusion_matrix(cm, ["non-toxic","toxic"], "Confusion Matrix (Validation)", REPORTS_DIR / "confusion_matrix_val.png")

    print("Training complete.")
    print(f"Validation ROC-AUC: {auc:.4f}")
    print(report)

if __name__ == "__main__":
    main()
