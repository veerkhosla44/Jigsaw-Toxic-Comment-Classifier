from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import joblib
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"
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
    model_path = MODELS_DIR / "model.joblib"
    if not model_path.exists():
        raise FileNotFoundError("Model not found. Run `python src/train.py` first.")
    model = joblib.load(model_path)

    df_test = pd.read_csv(DATA_DIR / "test.csv")
    X = df_test["text"].astype(str).values
    y = df_test["label"].values

    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)[:,1]

    report = classification_report(y, y_pred, digits=4)
    auc = roc_auc_score(y, y_proba)
    (REPORTS_DIR / "classification_report_test.txt").write_text(f"ROC-AUC: {auc:.4f}\n\n{report}")

    cm = confusion_matrix(y, y_pred)
    plot_confusion_matrix(cm, ["non-toxic","toxic"], "Confusion Matrix (Test)", REPORTS_DIR / "confusion_matrix_test.png")

    print("Test Evaluation")
    print(f"ROC-AUC: {auc:.4f}")
    print(report)

if __name__ == "__main__":
    main()
