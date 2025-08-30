# Toxic Comment Classification (YouTube Trust & Safety Demo)

This repo trains a **binary toxic vs non-toxic** classifier using the **Jigsaw Toxic Comment** data you downloaded.
It accepts either raw Jigsaw CSVs (multi-label) or already-processed `text,label` CSVs.

## Directory
```
yt_toxic_project/
├── raw/                       # put your original CSVs here (e.g., Jigsaw train.csv)
│   ├── train.csv              # REQUIRED (raw Jigsaw or already processed text,label)
│   └── test.csv               # OPTIONAL (raw Jigsaw with labels). If absent, we split from train.
├── data/                      # auto-generated processed files (text,label)
│   ├── train.csv
│   └── test.csv
├── src/
│   ├── prepare.py             # converts raw Jigsaw -> data/ schema
│   ├── train.py               # trains TF-IDF + LogisticRegression
│   ├── evaluate.py            # test-set evaluation + plots
│   └── predict.py             # quick CLI predictions
├── models/
│   └── model.joblib           # saved model after training
├── reports/
│   ├── classification_report_val.txt
│   ├── classification_report_test.txt
│   ├── confusion_matrix_val.png
│   └── confusion_matrix_test.png
└── requirements.txt
```

## Quickstart

1) **Place your downloaded CSVs** into `raw/`:
   - At minimum: `raw/train.csv`
   - Optional: `raw/test.csv` (must include the same toxicity label columns as train).

2) **Create processed train/test** (binary label; stratified split if needed):
```bash
python src/prepare.py --train_path raw/train.csv --test_path raw/test.csv
# If you don't have a labeled test.csv, omit --test_path and the script will split from train.
```

3) **Install deps & train**:
```bash
pip install -r requirements.txt
python src/train.py
```

4) **Evaluate & predict**:
```bash
python src/evaluate.py
python src/predict.py "I love this video" "this is stupid, seriously"
```

## Notes
- Binary label: `label = 1` if any of {toxic, severe_toxic, obscene, threat, insult, identity_hate} = 1, else 0.
- If your `raw/train.csv` already has two columns `text,label`, the prepare script will pass it through as-is.
