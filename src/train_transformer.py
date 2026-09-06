"""
Transformer Baseline: DistilBERT for 4-Class Sentiment Analysis
Trains and evaluates a DistilBERT sequence classification model using PyTorch and Hugging Face.
Configured for CPU execution with balanced sampling to ensure high performance and feasibility.
"""
import json
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from src import config

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=64):
        self.texts = list(texts)
        self.labels = list(labels)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = int(self.labels[idx])
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt"
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(label, dtype=torch.long)
        }

def train_transformer():
    print("=" * 60)
    print("HYBRIDSENSE: TRANSFORMER (DistilBERT) SENTIMENT CLASSIFIER")
    print("=" * 60)
    print(f"Device: {device}")

    # 1. Load Data
    train_df = pd.read_csv(config.TRAIN_CSV)
    val_df = pd.read_csv(config.VAL_CSV)
    test_df = pd.read_csv(config.TEST_CSV)

    # Stratified CPU-optimized subset preserving ALL Ambivalent samples
    amb_train = train_df[train_df["label"] == 3]
    other_train = train_df[train_df["label"] != 3].sample(n=5500, random_state=config.RANDOM_SEED)
    train_sub = pd.concat([amb_train, other_train]).sample(frac=1.0, random_state=config.RANDOM_SEED).reset_index(drop=True)

    amb_val = val_df[val_df["label"] == 3]
    other_val = val_df[val_df["label"] != 3].sample(n=1000, random_state=config.RANDOM_SEED)
    val_sub = pd.concat([amb_val, other_val]).sample(frac=1.0, random_state=config.RANDOM_SEED).reset_index(drop=True)

    amb_test = test_df[test_df["label"] == 3]
    other_test = test_df[test_df["label"] != 3].sample(n=2000, random_state=config.RANDOM_SEED)
    test_sub = pd.concat([amb_test, other_test]).sample(frac=1.0, random_state=config.RANDOM_SEED).reset_index(drop=True)

    print(f"DistilBERT Training samples: {len(train_sub)} (All {len(amb_train)} Ambivalent included)")
    print(f"DistilBERT Validation samples: {len(val_sub)}")
    print(f"DistilBERT Test samples: {len(test_sub)} (All {len(amb_test)} Ambivalent included)")

    # 2. Tokenizer & Datasets
    print(f"Loading Hugging Face tokenizer: {config.TRANSFORMER_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(config.TRANSFORMER_NAME)

    train_dataset = SentimentDataset(train_sub["text"], train_sub["label"], tokenizer, max_len=64)
    val_dataset = SentimentDataset(val_sub["text"], val_sub["label"], tokenizer, max_len=64)
    test_dataset = SentimentDataset(test_sub["text"], test_sub["label"], tokenizer, max_len=64)

    train_loader = DataLoader(train_dataset, batch_size=config.TRANSFORMER_BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.TRANSFORMER_BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config.TRANSFORMER_BATCH_SIZE, shuffle=False)

    # 3. Model Initialization
    print(f"Loading {config.TRANSFORMER_NAME} with 4-class classification head...")
    model_config = AutoConfig.from_pretrained(
        config.TRANSFORMER_NAME,
        num_labels=config.NUM_CLASSES,
        id2label=config.LABEL_TO_NAME,
        label2id=config.NAME_TO_LABEL
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        config.TRANSFORMER_NAME,
        config=model_config
    ).to(device)

    # 4. Optimizer & Loss with Class Weighting
    y_train = train_sub["label"].values
    weights = compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
    class_weights = torch.tensor(weights, dtype=torch.float).to(device)
    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.TRANSFORMER_LR, weight_decay=0.01)

    # 5. Training Loop
    epochs = config.TRANSFORMER_EPOCHS
    best_val_loss = float("inf")

    print(f"\nFine-tuning DistilBERT for {epochs} epochs on {device}...")
    for epoch in range(epochs):
        model.train()
        total_train_loss = 0.0
        for step, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            loss = loss_fn(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_train_loss += loss.item()
            if (step + 1) % 50 == 0 or (step + 1) == len(train_loader):
                print(f"Epoch {epoch+1}/{epochs} | Step {step+1}/{len(train_loader)} | Loss: {loss.item():.4f}")

        avg_train_loss = total_train_loss / len(train_loader)

        # Validation
        model.eval()
        total_val_loss = 0.0
        val_correct = 0
        total_val = 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["label"].to(device)
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                loss = loss_fn(outputs.logits, labels)
                total_val_loss += loss.item()
                preds = torch.argmax(outputs.logits, dim=1)
                val_correct += (preds == labels).sum().item()
                total_val += labels.size(0)

        avg_val_loss = total_val_loss / len(val_loader)
        val_acc = val_correct / total_val
        print(f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            config.TRANSFORMER_DIR.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(config.TRANSFORMER_DIR)
            tokenizer.save_pretrained(config.TRANSFORMER_DIR)
            print(f"Best DistilBERT model saved to {config.TRANSFORMER_DIR}")

    # 6. Evaluation on Test Set
    print("\nEvaluating DistilBERT on test set...")
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    acc = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average="macro")
    weighted_f1 = f1_score(all_labels, all_preds, average="weighted")
    prec = precision_score(all_labels, all_preds, average="weighted")
    rec = recall_score(all_labels, all_preds, average="weighted")

    print("\n--- DistilBERT Test Set Results ---")
    print(f"Accuracy:    {acc:.4f}")
    print(f"Precision:   {prec:.4f}")
    print(f"Recall:      {rec:.4f}")
    print(f"Macro F1:    {macro_f1:.4f}")
    print(f"Weighted F1: {weighted_f1:.4f}")

    target_names = [config.LABEL_TO_NAME[i] for i in range(config.NUM_CLASSES)]
    report = classification_report(all_labels, all_preds, target_names=target_names, digits=4)
    print("\nClassification Report:\n", report)

    cm = confusion_matrix(all_labels, all_preds)
    print("Confusion Matrix:\n", cm)

    # Save Report
    report_path = config.RESULTS_DIR / "classification_report_transformer.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("DISTILBERT SENTIMENT CLASSIFIER REPORT\n")
        f.write("=" * 60 + "\n")
        f.write(f"Accuracy:    {acc:.4f}\n")
        f.write(f"Precision:   {prec:.4f}\n")
        f.write(f"Recall:      {rec:.4f}\n")
        f.write(f"Macro F1:    {macro_f1:.4f}\n")
        f.write(f"Weighted F1: {weighted_f1:.4f}\n\n")
        f.write(report)
        f.write("\nConfusion Matrix:\n")
        f.write(np.array2string(cm))

    print(f"Report saved to: {report_path}")

    return {
        "model": "DistilBERT",
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1
    }

if __name__ == "__main__":
    train_transformer()
