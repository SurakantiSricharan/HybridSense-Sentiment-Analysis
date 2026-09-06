"""
Baseline Model: TF-IDF + Logistic Regression
Trains and evaluates a 4-class sentiment classifier using scikit-learn.
"""
import pickle
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from src import config
from src.preprocessing import clean_tweet_text

def train_baseline():
    print("=" * 60)
    print("HYBRIDSENSE: BASELINE MODEL (TF-IDF + LOGISTIC REGRESSION)")
    print("=" * 60)

    # 1. Load Data
    train_df = pd.read_csv(config.TRAIN_CSV)
    test_df = pd.read_csv(config.TEST_CSV)
    print(f"Train samples: {len(train_df)} | Test samples: {len(test_df)}")

    # 2. Clean Text
    print("Preprocessing text...")
    X_train_raw = train_df["text"].apply(clean_tweet_text)
    y_train = train_df["label"].values

    X_test_raw = test_df["text"].apply(clean_tweet_text)
    y_test = test_df["label"].values

    # 3. Fit TF-IDF Vectorizer
    print("Extracting TF-IDF features (unigrams & bigrams, max 15,000 features)...")
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=15000,
        sublinear_tf=True,
        min_df=2
    )
    X_train_vec = vectorizer.fit_transform(X_train_raw)
    X_test_vec = vectorizer.transform(X_test_raw)

    # 4. Train Logistic Regression
    print("Training Logistic Regression with balanced class weights...")
    clf = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=config.RANDOM_SEED,
        C=1.0,
        solver="lbfgs"
    )
    clf.fit(X_train_vec, y_train)

    # 5. Evaluate on Test Set
    y_pred = clf.predict(X_test_vec)

    acc = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro")
    weighted_f1 = f1_score(y_test, y_pred, average="weighted")
    prec = precision_score(y_test, y_pred, average="weighted")
    rec = recall_score(y_test, y_pred, average="weighted")

    print("\n--- Test Set Evaluation Results ---")
    print(f"Accuracy:    {acc:.4f}")
    print(f"Precision:   {prec:.4f}")
    print(f"Recall:      {rec:.4f}")
    print(f"Macro F1:    {macro_f1:.4f}")
    print(f"Weighted F1: {weighted_f1:.4f}")

    target_names = [config.LABEL_TO_NAME[i] for i in range(config.NUM_CLASSES)]
    report = classification_report(y_test, y_pred, target_names=target_names, digits=4)
    print("\nClassification Report:\n", report)

    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:\n", cm)

    # 6. Save Model and Vectorizer Artifacts
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    with open(config.TFIDF_MODEL_PATH, "wb") as f:
        pickle.dump(clf, f)
    with open(config.TFIDF_VECTORIZER_PATH, "wb") as f:
        pickle.dump(vectorizer, f)

    report_path = config.RESULTS_DIR / "classification_report_baseline.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("BASELINE MODEL (TF-IDF + Logistic Regression)\n")
        f.write("=" * 60 + "\n")
        f.write(f"Accuracy:    {acc:.4f}\n")
        f.write(f"Precision:   {prec:.4f}\n")
        f.write(f"Recall:      {rec:.4f}\n")
        f.write(f"Macro F1:    {macro_f1:.4f}\n")
        f.write(f"Weighted F1: {weighted_f1:.4f}\n\n")
        f.write(report)
        f.write("\nConfusion Matrix:\n")
        f.write(np.array2string(cm))

    print(f"\nModel saved to: {config.TFIDF_MODEL_PATH}")
    print(f"Vectorizer saved to: {config.TFIDF_VECTORIZER_PATH}")
    print(f"Report saved to: {report_path}")

    return {
        "model": "TF-IDF + Logistic Regression",
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1
    }

if __name__ == "__main__":
    train_baseline()
