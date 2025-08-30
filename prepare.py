#!/usr/bin/env python3
import argparse
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

TOX_COLS = ["toxic","severe_toxic","obscene","threat","insult","identity_hate"]

def to_binary(df: pd.DataFrame) -> pd.DataFrame:
    """Return df with columns: text,label (1 if any toxic label == 1)."""
    out = pd.DataFrame()
    if "text" in df.columns and "label" in df.columns:
        out["text"]  = df["text"].astype(str)
        out["label"] = df["label"].astype(int)
        return out

    if "comment_text" in df.columns:
        out["text"] = df["comment_text"].astype(str)
        # if toxicity columns exist, compute binary label
        if all(c in df.columns for c in TOX_COLS):
            bin_label = (df[TOX_COLS] == 1).any(axis=1).astype(int)
            out["label"] = bin_label
            return out
        else:
            # no labels present (e.g., Jigsaw test.csv) -> return text only
            return out

    raise ValueError(
        "Unsupported CSV schema. Expected either tidy ['text','label'] "
        "or Jigsaw with 'comment_text' (+ toxicity label columns for train)."
    )

def maybe_merge_test_labels(test_df: pd.DataFrame, labels_path: Path | None) -> pd.DataFrame:
    """Merge Jigsaw test_labels.csv if provided; compute binary label; drop -1 rows."""
    if labels_path is None or not labels_path.exists():
        return test_df  # unlabeled test set is fine for inference-time evaluation elsewhere
    if "id" not in test_df.columns:
        return test_df

    lab = pd.read_csv(labels_path)
    # Keep only required columns
    needed = ["id"] + TOX_COLS
    lab = lab[[c for c in needed if c in lab.columns]].copy()

    merged = test_df.merge(lab, on="id", how="left")
    # Drop rows where labels are -1 across the board (not used in Kaggle scoring)
    if all(c in merged.columns for c in TOX_COLS):
        mask_all_neg1 = (merged[TOX_COLS] == -1).all(axis=1)
        merged = merged.loc[~mask_all_neg1].copy()
        merged["label"] = (merged[TOX_COLS] == 1).any(axis=1).astype(int)
        merged = merged.rename(columns={"comment_text":"text"})[["text","label"]]
    else:
        merged = merged.rename(columns={"comment_text":"text"})[["text"]]
    return merged

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_path", required=True, type=str)
    ap.add_argument("--test_path", required=True, type=str)
    ap.add_argument("--test_labels_path", type=str, default=None,
                    help="Optional path to Jigsaw test_labels.csv to attach labels.")
    ap.add_argument("--test_size", type=float, default=0.2)
    ap.add_argument("--random_state", type=int, default=42)
    ap.add_argument("--out_dir", type=str, default="data")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_df_raw = pd.read_csv(args.train_path)
    test_df_raw  = pd.read_csv(args.test_path)

    # Convert train: must end up with text,label
    train_df = to_binary(train_df_raw)
    if "label" not in train_df.columns:
        raise ValueError("Training data ended up without labels. Make sure train.csv has toxicity columns or tidy labels.")
    train_df = train_df[["text","label"]].dropna(subset=["text"]).reset_index(drop=True)

    # Split off validation from train
    tr, val = train_df.sample(frac=1, random_state=args.random_state), None
    tr, val = train_test_split(tr, test_size=args.test_size, random_state=args.random_state, stratify=tr["label"])

    # Convert test: may be text-only or labeled if you pass test_labels_path
    test_df = to_binary(test_df_raw)
    if "label" not in test_df.columns:
        lbl_path = Path(args.test_labels_path) if args.test_labels_path else None
        test_df = maybe_merge_test_labels(test_df_raw, lbl_path)
    # Ensure only the expected columns for outputs
    if "label" in test_df.columns:
        test_df = test_df[["text","label"]].dropna(subset=["text"]).reset_index(drop=True)
    else:
        test_df = test_df[["text"]].dropna(subset=["text"]).reset_index(drop=True)

    # Write out
    (out_dir / "train.csv").write_text(train_df_to_csv(tr))
    (out_dir / "val.csv").write_text(train_df_to_csv(val))
    if "label" in test_df.columns:
        (out_dir / "test.csv").write_text(train_df_to_csv(test_df))
    else:
        test_df.to_csv(out_dir / "test_unlabeled.csv", index=False)

    print(f"Wrote:\n- {out_dir/'train.csv'}\n- {out_dir/'val.csv'}")
    if "label" in test_df.columns:
        print(f"- {out_dir/'test.csv'}")
    else:
        print(f"- {out_dir/'test_unlabeled.csv'} (no labels found)")

def train_df_to_csv(df: pd.DataFrame) -> str:
    return df.to_csv(index=False)

if __name__ == "__main__":
    main()
