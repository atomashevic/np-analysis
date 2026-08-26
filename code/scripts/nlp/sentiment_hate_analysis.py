#!/usr/bin/env python3
"""
Sentiment and Hate Speech Analysis for #NisamPrijavila tweets.

Models used:
1. cardiffnlp/twitter-xlm-roberta-base-sentiment (multilingual sentiment)
2. classla/bcms-bertic-frenk-hate (Serbian hate speech)
3. Hate-speech-CNERG/bert-base-uncased-hatexplain (English hate speech)
4. cardiffnlp/twitter-roberta-base-offensive (English offensive)

Usage: uv run python scripts/sentiment_hate_analysis.py
"""

import re
import pandas as pd
import torch
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
from cyrtranslit import to_latin
from deep_translator import GoogleTranslator
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")

# Configuration
DATA_PATH = "data/np.csv"
OUTPUT_PATH = "data/np_analyzed.csv"
BATCH_SIZE = 32
TRANSLATION_BATCH_SIZE = 50  # Google Translate has limits


def preprocess_serbian_tweet(text: str) -> str:
    """Preprocess Serbian tweet text."""
    text = to_latin(str(text), "sr")  # Cyrillic → Latin
    text = re.sub(r"http\S+|www\.\S+", "", text)  # Remove URLs
    text = re.sub(r"@\w+", "@user", text)  # Normalize mentions
    text = re.sub(r"#(\w+)", r"\1", text)  # Remove # from hashtags
    text = re.sub(r"^RT\s*:?\s*", "", text)  # Remove RT markers
    return " ".join(text.split()).strip()


def batch_translate(texts: list[str], batch_size: int = 50) -> list[str]:
    """Translate Serbian texts to English in batches."""
    translator = GoogleTranslator(source="sr", target="en")
    translations = []

    for i in tqdm(range(0, len(texts), batch_size), desc="Translating"):
        batch = texts[i : i + batch_size]
        # Filter empty strings and translate
        batch_translations = []
        for text in batch:
            if text.strip():
                try:
                    translated = translator.translate(text)
                    batch_translations.append(translated if translated else text)
                except Exception:
                    batch_translations.append(text)  # Keep original on error
            else:
                batch_translations.append("")
        translations.extend(batch_translations)

    return translations


def run_pipeline_batched(pipe, texts: list[str], desc: str) -> list[dict]:
    """Run a HuggingFace pipeline in batches with progress bar."""
    results = []
    for i in tqdm(range(0, len(texts), BATCH_SIZE), desc=desc):
        batch = texts[i : i + BATCH_SIZE]
        # Handle empty texts
        batch_results = pipe(batch, truncation=True, max_length=512)
        results.extend(batch_results)
    return results


def print_summary(df: pd.DataFrame):
    """Print summary statistics."""
    print("\n" + "=" * 60)
    print("ANALYSIS SUMMARY - #NisamPrijavila Dataset")
    print("=" * 60)

    print(f"\nTotal tweets analyzed: {len(df):,}")

    # Sentiment distribution
    print("\n" + "-" * 40)
    print("SENTIMENT ANALYSIS (XLM-RoBERTa)")
    print("-" * 40)
    sentiment_counts = df["sentiment_label"].value_counts()
    for label, count in sentiment_counts.items():
        pct = count / len(df) * 100
        print(f"  {label:12}: {count:6,} ({pct:5.1f}%)")

    # Serbian hate speech
    print("\n" + "-" * 40)
    print("HATE SPEECH - Serbian (BERTić)")
    print("-" * 40)
    hate_sr_counts = df["hate_sr_label"].value_counts()
    for label, count in hate_sr_counts.items():
        pct = count / len(df) * 100
        print(f"  {label:12}: {count:6,} ({pct:5.1f}%)")

    # English hate speech (HateXplain)
    print("\n" + "-" * 40)
    print("HATE SPEECH - English (HateXplain)")
    print("-" * 40)
    hate_en_counts = df["hate_en_label"].value_counts()
    for label, count in hate_en_counts.items():
        pct = count / len(df) * 100
        print(f"  {label:12}: {count:6,} ({pct:5.1f}%)")

    # English offensive
    print("\n" + "-" * 40)
    print("OFFENSIVE LANGUAGE (RoBERTa)")
    print("-" * 40)
    off_counts = df["offensive_label"].value_counts()
    for label, count in off_counts.items():
        pct = count / len(df) * 100
        print(f"  {label:12}: {count:6,} ({pct:5.1f}%)")

    # Cross-tabulation: Sentiment × Serbian hate
    print("\n" + "-" * 40)
    print("CROSS-TAB: Sentiment × Serbian Hate Speech")
    print("-" * 40)
    crosstab = pd.crosstab(df["sentiment_label"], df["hate_sr_label"])
    print(crosstab.to_string())

    # Model agreement
    print("\n" + "-" * 40)
    print("MODEL AGREEMENT: Serbian vs English Hate Detection")
    print("-" * 40)
    # Map labels to binary
    df["hate_sr_binary"] = df["hate_sr_label"].map(
        lambda x: 1 if "Offensive" in str(x) else 0
    )
    df["hate_en_binary"] = df["hate_en_label"].map(
        lambda x: 1 if x in ["hatespeech", "offensive"] else 0
    )
    agreement = (df["hate_sr_binary"] == df["hate_en_binary"]).sum()
    print(f"  Agreement: {agreement:,} / {len(df):,} ({agreement/len(df)*100:.1f}%)")

    # High confidence hate speech examples
    print("\n" + "-" * 40)
    print("TOP 10 HATE SPEECH (Serbian model, highest confidence)")
    print("-" * 40)
    hate_tweets = df[df["hate_sr_label"].str.contains("Offensive", na=False)].nlargest(
        10, "hate_sr_score"
    )
    for i, row in enumerate(hate_tweets.itertuples(), 1):
        text = row.text_clean[:80] + "..." if len(row.text_clean) > 80 else row.text_clean
        print(f"  {i}. [{row.hate_sr_score:.3f}] {text}")

    print("\n" + "=" * 60)


def main():
    # Check GPU
    device = 0 if torch.cuda.is_available() else -1
    device_name = "GPU" if device == 0 else "CPU"
    print(f"Using device: {device_name}")
    if device == 0:
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Load data
    print(f"\nLoading data from {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded {len(df):,} tweets")

    # Preprocess Serbian text
    print("\nPreprocessing tweets...")
    df["text_clean"] = df["text"].apply(preprocess_serbian_tweet)

    # Translate to English for English models
    print("\nTranslating to English (this may take a while)...")
    df["text_en"] = batch_translate(df["text_clean"].tolist(), TRANSLATION_BATCH_SIZE)

    # Initialize pipelines
    print("\nLoading models...")

    # 1. Multilingual sentiment
    print("  Loading sentiment model...")
    sentiment_pipe = pipeline(
        "sentiment-analysis",
        model="cardiffnlp/twitter-xlm-roberta-base-sentiment",
        device=device,
    )

    # 2. Serbian hate speech
    print("  Loading Serbian hate speech model...")
    hate_sr_pipe = pipeline(
        "text-classification",
        model="classla/bcms-bertic-frenk-hate",
        device=device,
    )

    # 3. English hate speech (HateXplain)
    print("  Loading HateXplain model...")
    hate_en_pipe = pipeline(
        "text-classification",
        model="Hate-speech-CNERG/bert-base-uncased-hatexplain",
        device=device,
    )

    # 4. English offensive
    print("  Loading offensive language model...")
    offensive_pipe = pipeline(
        "text-classification",
        model="cardiffnlp/twitter-roberta-base-offensive",
        device=device,
    )

    # Run inference
    print("\nRunning inference...")
    texts_sr = df["text_clean"].tolist()
    texts_en = df["text_en"].tolist()

    # Sentiment (on Serbian text - model is multilingual)
    sentiment_results = run_pipeline_batched(sentiment_pipe, texts_sr, "Sentiment")
    df["sentiment_label"] = [r["label"] for r in sentiment_results]
    df["sentiment_score"] = [r["score"] for r in sentiment_results]

    # Serbian hate speech
    hate_sr_results = run_pipeline_batched(hate_sr_pipe, texts_sr, "Hate (Serbian)")
    df["hate_sr_label"] = [r["label"] for r in hate_sr_results]
    df["hate_sr_score"] = [r["score"] for r in hate_sr_results]

    # English hate speech
    hate_en_results = run_pipeline_batched(hate_en_pipe, texts_en, "Hate (English)")
    df["hate_en_label"] = [r["label"] for r in hate_en_results]
    df["hate_en_score"] = [r["score"] for r in hate_en_results]

    # English offensive
    off_results = run_pipeline_batched(offensive_pipe, texts_en, "Offensive")
    df["offensive_label"] = [r["label"] for r in off_results]
    df["offensive_score"] = [r["score"] for r in off_results]

    # Save results
    print(f"\nSaving results to {OUTPUT_PATH}...")
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved {len(df):,} rows")

    # Print summary
    print_summary(df)


if __name__ == "__main__":
    main()
