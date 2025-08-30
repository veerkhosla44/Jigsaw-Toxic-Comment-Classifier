import sys
from pathlib import Path
import joblib

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"

def main():
    model_path = MODELS_DIR / "model.joblib"
    if not model_path.exists():
        print("Model not found. Run `python src/train.py` first.")
        return
    model = joblib.load(model_path)

    if len(sys.argv) < 2:
        print('Usage: python src/predict.py "your text" ["more text" ...]')
        return

    texts = sys.argv[1:]
    preds = model.predict(texts)
    probs = model.predict_proba(texts)[:,1]

    for t, y, p in zip(texts, preds, probs):
        label = "toxic" if y == 1 else "non-toxic"
        print(f"[{label:9s}] {p:.3f} :: {t}")

if __name__ == "__main__":
    main()
