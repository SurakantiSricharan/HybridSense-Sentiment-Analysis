"""
Dataset Builder Module for HybridSense Sentiment Analysis
Constructs the 4-Class Sentiment Dataset with expanded Ambivalent class,
applies quality validation, and creates stratified train/val/test splits.
"""
import re
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from src import config
from src.preprocessing import clean_tweet_text

analyzer = SentimentIntensityAnalyzer()

CONTRAST_CONJUNCTIONS = r"\b(?:but|however|although|though|yet|while|despite)\b"

def extract_clause_ambivalence(text: str):
    """
    Evaluates whether a text has opposing clause sentiments across contrastive conjunctions.
    Returns True if clause 1 is positive and clause 2 is negative (or vice versa).
    """
    if not isinstance(text, str) or len(text.strip()) < 20:
        return False
        
    parts = re.split(CONTRAST_CONJUNCTIONS, text, flags=re.IGNORECASE)
    if len(parts) >= 2:
        s1 = analyzer.polarity_scores(parts[0])
        s2 = analyzer.polarity_scores(parts[1])
        # Check opposing polarities
        if (s1["compound"] >= 0.22 and s2["compound"] <= -0.22) or (s1["compound"] <= -0.22 and s2["compound"] >= 0.22):
            return True
            
    # Also check prominent dual polarity with contrastive marker
    scores = analyzer.polarity_scores(text)
    has_contrast = bool(re.search(CONTRAST_CONJUNCTIONS, text, re.IGNORECASE))
    if has_contrast and scores["pos"] >= 0.14 and scores["neg"] >= 0.14:
        return True
        
    return False

def build_datasets():
    print("=" * 60)
    print("HYBRIDSENSE: 4-CLASS DATASET BUILDER")
    print("=" * 60)

    # 1. Load Raw Training Tweets
    if not config.RAW_TRAIN_CSV.exists():
        raise FileNotFoundError(f"Missing {config.RAW_TRAIN_CSV}")
        
    raw_train_df = pd.read_csv(config.RAW_TRAIN_CSV)
    print(f"Loaded raw TweetEval data: {raw_train_df.shape}")

    # Standardize columns
    if "sentiment" not in raw_train_df.columns:
        if "label" in raw_train_df.columns:
            raw_train_df["sentiment"] = raw_train_df["label"].map(config.LABEL_TO_NAME)
        else:
            raise ValueError("tweet_eval_sentiment_train.csv missing 'sentiment' or 'label'")
            
    base_df = raw_train_df[["text", "sentiment"]].copy()

    # 2. Load Verified Ambivalent Candidates
    verified_candidates = []
    if config.RAW_AMBIVALENT_CSV.exists():
        cand_df = pd.read_csv(config.RAW_AMBIVALENT_CSV)
        if "ambivalent_label" in cand_df.columns:
            verified_cand_df = cand_df[cand_df["ambivalent_label"] == 1].copy()
            verified_candidates = verified_cand_df["text"].dropna().tolist()
            print(f"Loaded {len(verified_candidates)} verified Ambivalent samples from candidate file.")

    # 3. Linguistic Contrast Mining for Ambivalent Expansion
    print("Mining contrastive Ambivalent samples from dataset using clause polarity analysis...")
    mined_ambivalent = []
    
    for text in base_df["text"]:
        if extract_clause_ambivalence(text):
            mined_ambivalent.append(text)

    print(f"Mined {len(mined_ambivalent)} candidate contrastive tweets.")

    # Combine all verified ambivalent texts
    all_ambivalent_texts = list(set(verified_candidates + mined_ambivalent))
    print(f"Total unique Ambivalent samples identified: {len(all_ambivalent_texts)}")

    # 4. Construct Ambivalent DataFrame
    ambivalent_df = pd.DataFrame({
        "text": all_ambivalent_texts,
        "sentiment": "Ambivalent"
    })

    # 5. Remove Ambivalent texts from the base dataset to avoid multi-label collision
    ambivalent_set = set(all_ambivalent_texts)
    base_filtered = base_df[~base_df["text"].isin(ambivalent_set)].copy()

    # 6. Combine Datasets
    combined_df = pd.concat([base_filtered, ambivalent_df], ignore_index=True)

    # 7. Quality Checks & Deduplication
    print("\nApplying Data Quality Checks:")
    # Drop NaNs
    initial_count = len(combined_df)
    combined_df = combined_df.dropna(subset=["text", "sentiment"]).copy()
    
    # Strip text and remove empty strings
    combined_df["text"] = combined_df["text"].astype(str).str.strip()
    combined_df = combined_df[combined_df["text"].str.len() > 3].copy()
    
    # Remove duplicates
    combined_df = combined_df.drop_duplicates(subset=["text"]).reset_index(drop=True)
    print(f"Removed {initial_count - len(combined_df)} duplicates/invalid rows. Final count: {len(combined_df)}")

    # Map sentiment string to numeric label
    combined_df["label"] = combined_df["sentiment"].map(config.NAME_TO_LABEL)

    # Validate labels
    assert combined_df["label"].isnull().sum() == 0, "Null labels detected!"
    assert set(combined_df["label"].unique()) == {0, 1, 2, 3}, f"Unexpected labels: {combined_df['label'].unique()}"

    print("\nClass Distribution in Final Dataset:")
    print(combined_df["sentiment"].value_counts())
    print("\nLabel Distribution:")
    print(combined_df["label"].value_counts().sort_index())

    # Save Final Dataset
    config.PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    combined_df.to_csv(config.FINAL_DATASET_CSV, index=False)
    print(f"\nSaved final 4-class dataset to: {config.FINAL_DATASET_CSV}")

    # 8. Create Stratified Train / Validation / Test Splits (70% / 10% / 20%)
    print("\nCreating Stratified Train / Validation / Test Splits (70% / 10% / 20%):")
    
    train_df, temp_df = train_test_split(
        combined_df,
        test_size=0.30,
        random_state=config.RANDOM_SEED,
        stratify=combined_df["label"]
    )

    val_df, test_df = train_test_split(
        temp_df,
        test_size=(2.0 / 3.0), # 1/3 of 30% is 10% val, 2/3 is 20% test
        random_state=config.RANDOM_SEED,
        stratify=temp_df["label"]
    )

    train_df.to_csv(config.TRAIN_CSV, index=False)
    val_df.to_csv(config.VAL_CSV, index=False)
    test_df.to_csv(config.TEST_CSV, index=False)

    print(f"Train split saved ({len(train_df)} rows) -> {config.TRAIN_CSV}")
    print(f"Validation split saved ({len(val_df)} rows) -> {config.VAL_CSV}")
    print(f"Test split saved ({len(test_df)} rows) -> {config.TEST_CSV}")

    print("\nTrain Split Class Distribution:")
    print(train_df["sentiment"].value_counts())
    print("\nTest Split Class Distribution:")
    print(test_df["sentiment"].value_counts())

    print("=" * 60)
    print("DATASET PREPARATION COMPLETED SUCCESSFULLY")
    print("=" * 60)

if __name__ == "__main__":
    build_datasets()
