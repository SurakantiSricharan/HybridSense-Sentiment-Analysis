"""
Comprehensive Model Evaluation & Comparison Module
Evaluates all models on the test set, generates model_comparison.csv,
and produces comparative charts and confusion matrix heatmaps in results/.
"""
import pickle
import json
import torch
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
import tensorflow as tf
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from src import config
from src.preprocessing import clean_tweet_text, load_keras_tokenizer, tokenize_and_pad
# Import AttentionLayer so Keras can deserialize the hybrid model
from src.train_hybrid import AttentionLayer

def evaluate_all():
    print("=" * 60)
    print("HYBRIDSENSE: COMPREHENSIVE MODEL EVALUATION & COMPARISON")
    print("=" * 60)

    test_df = pd.read_csv(config.TEST_CSV)
    y_test = test_df["label"].values
    target_names = [config.LABEL_TO_NAME[i] for i in range(config.NUM_CLASSES)]

    results_list = []
    confusion_matrices = {}

    # 1. Baseline: TF-IDF + Logistic Regression
    if config.TFIDF_MODEL_PATH.exists() and config.TFIDF_VECTORIZER_PATH.exists():
        print("\n[1/5] Evaluating Baseline (TF-IDF + Logistic Regression)...")
        with open(config.TFIDF_MODEL_PATH, "rb") as f:
            tfidf_model = pickle.load(f)
        with open(config.TFIDF_VECTORIZER_PATH, "rb") as f:
            vectorizer = pickle.load(f)

        X_test_clean = test_df["text"].apply(clean_tweet_text)
        X_test_vec = vectorizer.transform(X_test_clean)
        y_pred = tfidf_model.predict(X_test_vec)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
        rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
        macro_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
        weighted_f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

        results_list.append({
            "model": "TF-IDF + Logistic Regression",
            "accuracy": round(acc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "macro_f1": round(macro_f1, 4),
            "weighted_f1": round(weighted_f1, 4)
        })
        confusion_matrices["TF-IDF + Logistic Regression"] = confusion_matrix(y_test, y_pred)

    # 2. CNN
    if config.CNN_MODEL_PATH.exists() and config.TOKENIZER_PATH.exists():
        print("\n[2/5] Evaluating CNN Classifier...")
        tokenizer = load_keras_tokenizer()
        X_test_pad = tokenize_and_pad(test_df["text"], tokenizer)
        cnn_model = tf.keras.models.load_model(config.CNN_MODEL_PATH)
        y_prob = cnn_model.predict(X_test_pad, verbose=0)
        y_pred = np.argmax(y_prob, axis=1)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
        rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
        macro_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
        weighted_f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

        results_list.append({
            "model": "CNN",
            "accuracy": round(acc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "macro_f1": round(macro_f1, 4),
            "weighted_f1": round(weighted_f1, 4)
        })
        confusion_matrices["CNN"] = confusion_matrix(y_test, y_pred)

    # 3. BiLSTM
    if config.BILSTM_MODEL_PATH.exists() and config.TOKENIZER_PATH.exists():
        print("\n[3/5] Evaluating BiLSTM Classifier...")
        tokenizer = load_keras_tokenizer()
        X_test_pad = tokenize_and_pad(test_df["text"], tokenizer)
        bilstm_model = tf.keras.models.load_model(config.BILSTM_MODEL_PATH)
        y_prob = bilstm_model.predict(X_test_pad, verbose=0)
        y_pred = np.argmax(y_prob, axis=1)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
        rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
        macro_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
        weighted_f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

        results_list.append({
            "model": "BiLSTM",
            "accuracy": round(acc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "macro_f1": round(macro_f1, 4),
            "weighted_f1": round(weighted_f1, 4)
        })
        confusion_matrices["BiLSTM"] = confusion_matrix(y_test, y_pred)

    # 4. DistilBERT
    if config.TRANSFORMER_DIR.exists():
        print("\n[4/5] Evaluating Transformer (DistilBERT)...")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        trans_tokenizer = AutoTokenizer.from_pretrained(config.TRANSFORMER_DIR)
        trans_model = AutoModelForSequenceClassification.from_pretrained(config.TRANSFORMER_DIR).to(device)
        trans_model.eval()

        # Evaluate on test set (or representative subset if large)
        sample_test = test_df.sample(n=min(2000, len(test_df)), random_state=config.RANDOM_SEED).reset_index(drop=True)
        t_preds = []
        with torch.no_grad():
            for i in range(0, len(sample_test), 32):
                batch_texts = list(sample_test["text"][i:i+32])
                inputs = trans_tokenizer(batch_texts, padding=True, truncation=True, max_length=64, return_tensors="pt").to(device)
                outputs = trans_model(**inputs)
                preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
                t_preds.extend(preds)

        y_true_trans = sample_test["label"].values
        t_preds = np.array(t_preds)

        acc = accuracy_score(y_true_trans, t_preds)
        prec = precision_score(y_true_trans, t_preds, average="weighted", zero_division=0)
        rec = recall_score(y_true_trans, t_preds, average="weighted", zero_division=0)
        macro_f1 = f1_score(y_true_trans, t_preds, average="macro", zero_division=0)
        weighted_f1 = f1_score(y_true_trans, t_preds, average="weighted", zero_division=0)

        results_list.append({
            "model": "DistilBERT",
            "accuracy": round(acc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "macro_f1": round(macro_f1, 4),
            "weighted_f1": round(weighted_f1, 4)
        })
        confusion_matrices["DistilBERT"] = confusion_matrix(y_true_trans, t_preds)

    # 5. Hybrid CNN-BiLSTM-Attention
    if config.HYBRID_MODEL_PATH.exists() and config.TOKENIZER_PATH.exists():
        print("\n[5/5] Evaluating Hybrid CNN-BiLSTM-Attention...")
        tokenizer = load_keras_tokenizer()
        X_test_pad = tokenize_and_pad(test_df["text"], tokenizer)
        hybrid_model = tf.keras.models.load_model(
            config.HYBRID_MODEL_PATH,
            custom_objects={"AttentionLayer": AttentionLayer}
        )
        y_prob = hybrid_model.predict(X_test_pad, verbose=0)
        y_pred = np.argmax(y_prob, axis=1)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
        rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
        macro_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
        weighted_f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

        results_list.append({
            "model": "Hybrid CNN-BiLSTM-Attention",
            "accuracy": round(acc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "macro_f1": round(macro_f1, 4),
            "weighted_f1": round(weighted_f1, 4)
        })
        confusion_matrices["Hybrid CNN-BiLSTM-Attention"] = confusion_matrix(y_test, y_pred)

    # Save Comparison CSV
    comp_df = pd.DataFrame(results_list)
    comp_df.to_csv(config.MODEL_COMPARISON_CSV, index=False)
    print("\n" + "=" * 60)
    print("MODEL COMPARISON TABLE")
    print("=" * 60)
    print(comp_df.to_string(index=False))
    print(f"\nComparison table saved to: {config.MODEL_COMPARISON_CSV}")

    # Generate Visualizations
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", palette="muted")

    # Accuracy Comparison Bar Chart
    plt.figure(figsize=(10, 5))
    bars = sns.barplot(x="model", y="accuracy", data=comp_df, palette="crest")
    plt.title("Model Accuracy Comparison", fontsize=14, fontweight="bold", pad=12)
    plt.ylim(0, 1.0)
    plt.ylabel("Accuracy", fontsize=12)
    plt.xlabel("Model", fontsize=12)
    plt.xticks(rotation=15, ha="right")
    for b in bars.patches:
        height = b.get_height()
        bars.annotate(f"{height:.4f}", (b.get_x() + b.get_width() / 2., height),
                      ha="center", va="bottom", fontsize=10, xytext=(0, 3), textcoords="offset points")
    plt.tight_layout()
    acc_chart = config.RESULTS_DIR / "accuracy_comparison.png"
    plt.savefig(acc_chart, dpi=300)
    plt.close()
    print(f"Saved: {acc_chart}")

    # Macro F1 Comparison Bar Chart
    plt.figure(figsize=(10, 5))
    bars = sns.barplot(x="model", y="macro_f1", data=comp_df, palette="viridis")
    plt.title("Model Macro F1-Score Comparison (Class Imbalance Metric)", fontsize=14, fontweight="bold", pad=12)
    plt.ylim(0, 1.0)
    plt.ylabel("Macro F1-Score", fontsize=12)
    plt.xlabel("Model", fontsize=12)
    plt.xticks(rotation=15, ha="right")
    for b in bars.patches:
        height = b.get_height()
        bars.annotate(f"{height:.4f}", (b.get_x() + b.get_width() / 2., height),
                      ha="center", va="bottom", fontsize=10, xytext=(0, 3), textcoords="offset points")
    plt.tight_layout()
    f1_chart = config.RESULTS_DIR / "macro_f1_comparison.png"
    plt.savefig(f1_chart, dpi=300)
    plt.close()
    print(f"Saved: {f1_chart}")

    # Weighted F1 Comparison Bar Chart
    plt.figure(figsize=(10, 5))
    bars = sns.barplot(x="model", y="weighted_f1", data=comp_df, palette="magma")
    plt.title("Model Weighted F1-Score Comparison", fontsize=14, fontweight="bold", pad=12)
    plt.ylim(0, 1.0)
    plt.ylabel("Weighted F1-Score", fontsize=12)
    plt.xlabel("Model", fontsize=12)
    plt.xticks(rotation=15, ha="right")
    for b in bars.patches:
        height = b.get_height()
        bars.annotate(f"{height:.4f}", (b.get_x() + b.get_width() / 2., height),
                      ha="center", va="bottom", fontsize=10, xytext=(0, 3), textcoords="offset points")
    plt.tight_layout()
    wf1_chart = config.RESULTS_DIR / "weighted_f1_comparison.png"
    plt.savefig(wf1_chart, dpi=300)
    plt.close()
    print(f"Saved: {wf1_chart}")

    # Confusion Matrix Plots
    for model_name, cm in confusion_matrices.items():
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=target_names, yticklabels=target_names, cbar=False)
        plt.title(f"Confusion Matrix: {model_name}", fontsize=12, fontweight="bold", pad=10)
        plt.ylabel("True Sentiment Label", fontsize=11)
        plt.xlabel("Predicted Sentiment Label", fontsize=11)
        plt.tight_layout()
        safe_name = model_name.lower().replace(" ", "_").replace("-", "_").replace("+", "")
        cm_path = config.RESULTS_DIR / f"confusion_matrix_{safe_name}.png"
        plt.savefig(cm_path, dpi=300)
        plt.close()
        print(f"Saved: {cm_path}")

    print("\n" + "=" * 60)
    print("ALL EVALUATIONS AND CHARTS COMPLETED SUCCESSFULLY")
    print("=" * 60)

if __name__ == "__main__":
    evaluate_all()
