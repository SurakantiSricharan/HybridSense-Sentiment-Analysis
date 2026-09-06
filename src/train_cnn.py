"""
CNN Model for HybridSense Sentiment Analysis
1D Convolutional Neural Network capturing local n-gram sentiment patterns.
"""
import json
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from src import config
from src.preprocessing import fit_keras_tokenizer, tokenize_and_pad

def build_cnn_model(vocab_size: int, embedding_dim: int, max_seq_len: int, num_classes: int):
    model = Sequential([
        Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_seq_len),
        Conv1D(filters=128, kernel_size=5, activation="relu", padding="same"),
        GlobalMaxPooling1D(),
        Dense(64, activation="relu"),
        Dropout(0.4),
        Dense(num_classes, activation="softmax")
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config.LEARNING_RATE),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

def train_cnn():
    print("=" * 60)
    print("HYBRIDSENSE: CNN SENTIMENT CLASSIFIER")
    print("=" * 60)

    # 1. Load Data
    train_df = pd.read_csv(config.TRAIN_CSV)
    val_df = pd.read_csv(config.VAL_CSV)
    test_df = pd.read_csv(config.TEST_CSV)
    print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

    # 2. Tokenizer
    print("Fitting Keras Tokenizer on train split...")
    tokenizer = fit_keras_tokenizer(train_df["text"])

    # 3. Padded Sequences
    X_train = tokenize_and_pad(train_df["text"], tokenizer)
    y_train = train_df["label"].values

    X_val = tokenize_and_pad(val_df["text"], tokenizer)
    y_val = val_df["label"].values

    X_test = tokenize_and_pad(test_df["text"], tokenizer)
    y_test = test_df["label"].values

    # 4. Balanced Class Weights
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(y_train),
        y=y_train
    )
    class_weight_dict = {i: float(w) for i, w in enumerate(class_weights)}
    print("Computed Class Weights:", class_weight_dict)

    # 5. Build Model
    model = build_cnn_model(
        vocab_size=config.MAX_VOCAB_SIZE,
        embedding_dim=config.EMBEDDING_DIM,
        max_seq_len=config.MAX_SEQ_LEN,
        num_classes=config.NUM_CLASSES
    )
    model.summary()

    # 6. Callbacks
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=config.PATIENCE, restore_best_weights=True, verbose=1),
        ModelCheckpoint(filepath=str(config.CNN_MODEL_PATH), monitor="val_loss", save_best_only=True, verbose=1)
    ]

    # 7. Train Model
    print(f"\nTraining CNN model for up to {config.EPOCHS} epochs...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=config.EPOCHS,
        batch_size=config.BATCH_SIZE,
        class_weight=class_weight_dict,
        callbacks=callbacks,
        verbose=1
    )

    # 8. Evaluate on Test Set
    print("\nEvaluating CNN on test set...")
    y_prob = model.predict(X_test)
    y_pred = np.argmax(y_prob, axis=1)

    acc = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro")
    weighted_f1 = f1_score(y_test, y_pred, average="weighted")
    prec = precision_score(y_test, y_pred, average="weighted")
    rec = recall_score(y_test, y_pred, average="weighted")

    print("\n--- CNN Test Set Results ---")
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

    # Save History and Report
    report_path = config.RESULTS_DIR / "classification_report_cnn.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("CNN SENTIMENT CLASSIFIER REPORT\n")
        f.write("=" * 60 + "\n")
        f.write(f"Accuracy:    {acc:.4f}\n")
        f.write(f"Precision:   {prec:.4f}\n")
        f.write(f"Recall:      {rec:.4f}\n")
        f.write(f"Macro F1:    {macro_f1:.4f}\n")
        f.write(f"Weighted F1: {weighted_f1:.4f}\n\n")
        f.write(report)
        f.write("\nConfusion Matrix:\n")
        f.write(np.array2string(cm))

    history_path = config.RESULTS_DIR / "history_cnn.json"
    with open(history_path, "w") as f:
        json.dump({k: [float(x) for x in v] for k, v in history.history.items()}, f)

    print(f"Report saved to: {report_path}")
    print(f"History saved to: {history_path}")

    return {
        "model": "CNN",
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1
    }

if __name__ == "__main__":
    train_cnn()
