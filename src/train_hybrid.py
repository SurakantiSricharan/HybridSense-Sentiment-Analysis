"""
Hybrid CNN-BiLSTM-Attention Sentiment Classifier
Primary Proposed Architecture:
Input -> Embedding -> Conv1D -> BiLSTM -> Self-Attention -> Dense -> Dropout -> Softmax
Captures local n-gram patterns (CNN), sequential bidirectional context (BiLSTM),
and dynamically attends to salient sentiment-bearing tokens (Attention).
"""
import json
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Embedding, Conv1D, Bidirectional, LSTM, Dense, Dropout, Layer
)
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from src import config
from src.preprocessing import load_keras_tokenizer, tokenize_and_pad

@tf.keras.utils.register_keras_serializable(package="HybridSense")
class AttentionLayer(Layer):
    """
    Bahdanau-style self-attention mechanism over sequence representations.
    Learns attention weights to produce a context vector focusing on sentiment-bearing tokens.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        hidden_dim = int(input_shape[-1])
        self.W = self.add_weight(
            name="attention_weight",
            shape=(hidden_dim, hidden_dim),
            initializer="glorot_uniform",
            trainable=True
        )
        self.b = self.add_weight(
            name="attention_bias",
            shape=(hidden_dim,),
            initializer="zeros",
            trainable=True
        )
        self.u = self.add_weight(
            name="attention_context",
            shape=(hidden_dim, 1),
            initializer="glorot_uniform",
            trainable=True
        )
        super().build(input_shape)

    def call(self, inputs, mask=None):
        # inputs shape: (batch_size, time_steps, hidden_dim)
        uit = tf.tanh(tf.tensordot(inputs, self.W, axes=1) + self.b)
        ait = tf.tensordot(uit, self.u, axes=1) # (batch_size, time_steps, 1)
        ait = tf.squeeze(ait, -1)              # (batch_size, time_steps)
        if mask is not None:
            ait += (1.0 - tf.cast(mask, tf.float32)) * -1e9
        alpha = tf.nn.softmax(ait, axis=1)     # (batch_size, time_steps)
        alpha_expanded = tf.expand_dims(alpha, -1) # (batch_size, time_steps, 1)
        context = tf.reduce_sum(inputs * alpha_expanded, axis=1) # (batch_size, hidden_dim)
        return context

    def get_config(self):
        return super().get_config()

def build_hybrid_model(vocab_size: int, embedding_dim: int, max_seq_len: int, num_classes: int):
    """
    Constructs the full CNN-BiLSTM-Attention hybrid network.
    """
    inputs = Input(shape=(max_seq_len,), name="input_ids")

    # 1. Embedding Layer
    embed = Embedding(
        input_dim=vocab_size,
        output_dim=embedding_dim,
        name="embedding"
    )(inputs)

    # 2. CNN: Local Phrase-level Feature Extraction
    conv = Conv1D(
        filters=128,
        kernel_size=3,
        padding="same",
        activation="relu",
        name="conv1d_local_features"
    )(embed)

    # 3. BiLSTM: Bidirectional Sequential and Contextual Feature Extraction
    bilstm = Bidirectional(
        LSTM(64, return_sequences=True, dropout=0.2, recurrent_dropout=0.0),
        name="bilstm_context"
    )(conv)

    # 4. Attention Mechanism: Dynamically weights salient tokens
    attn_vector = AttentionLayer(name="attention_layer")(bilstm)

    # 5. Fully Connected Classification Head
    dense = Dense(64, activation="relu", name="dense_features")(attn_vector)
    drop = Dropout(0.4, name="dropout")(dense)
    outputs = Dense(num_classes, activation="softmax", name="output_probabilities")(drop)

    model = Model(inputs=inputs, outputs=outputs, name="Hybrid_CNN_BiLSTM_Attention")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config.LEARNING_RATE),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

def train_hybrid():
    print("=" * 60)
    print("HYBRIDSENSE: PRIMARY PROPOSED MODEL")
    print("HYBRID CNN + BiLSTM + ATTENTION CLASSIFIER")
    print("=" * 60)

    # 1. Load Data
    train_df = pd.read_csv(config.TRAIN_CSV)
    val_df = pd.read_csv(config.VAL_CSV)
    test_df = pd.read_csv(config.TEST_CSV)
    print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

    # 2. Load Tokenizer & Create Sequences
    tokenizer = load_keras_tokenizer()

    X_train = tokenize_and_pad(train_df["text"], tokenizer)
    y_train = train_df["label"].values

    X_val = tokenize_and_pad(val_df["text"], tokenizer)
    y_val = val_df["label"].values

    X_test = tokenize_and_pad(test_df["text"], tokenizer)
    y_test = test_df["label"].values

    # 3. Balanced Class Weights
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(y_train),
        y=y_train
    )
    class_weight_dict = {i: float(w) for i, w in enumerate(class_weights)}
    print("Computed Class Weights:", class_weight_dict)

    # 4. Build Model
    model = build_hybrid_model(
        vocab_size=config.MAX_VOCAB_SIZE,
        embedding_dim=config.EMBEDDING_DIM,
        max_seq_len=config.MAX_SEQ_LEN,
        num_classes=config.NUM_CLASSES
    )
    model.summary()

    # 5. Callbacks
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=config.PATIENCE, restore_best_weights=True, verbose=1),
        ModelCheckpoint(filepath=str(config.HYBRID_MODEL_PATH), monitor="val_loss", save_best_only=True, verbose=1)
    ]

    # 6. Train Model
    print(f"\nTraining Hybrid model for up to {config.EPOCHS} epochs...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=config.EPOCHS,
        batch_size=config.BATCH_SIZE,
        class_weight=class_weight_dict,
        callbacks=callbacks,
        verbose=1
    )

    # 7. Evaluate on Test Set
    print("\nEvaluating Hybrid CNN-BiLSTM-Attention on test set...")
    y_prob = model.predict(X_test)
    y_pred = np.argmax(y_prob, axis=1)

    acc = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro")
    weighted_f1 = f1_score(y_test, y_pred, average="weighted")
    prec = precision_score(y_test, y_pred, average="weighted")
    rec = recall_score(y_test, y_pred, average="weighted")

    print("\n--- Hybrid Model Test Set Results ---")
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
    report_path = config.RESULTS_DIR / "classification_report_hybrid.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("HYBRID CNN-BiLSTM-ATTENTION MODEL REPORT\n")
        f.write("=" * 60 + "\n")
        f.write(f"Accuracy:    {acc:.4f}\n")
        f.write(f"Precision:   {prec:.4f}\n")
        f.write(f"Recall:      {rec:.4f}\n")
        f.write(f"Macro F1:    {macro_f1:.4f}\n")
        f.write(f"Weighted F1: {weighted_f1:.4f}\n\n")
        f.write(report)
        f.write("\nConfusion Matrix:\n")
        f.write(np.array2string(cm))

    history_path = config.RESULTS_DIR / "history_hybrid.json"
    with open(history_path, "w") as f:
        json.dump({k: [float(x) for x in v] for k, v in history.history.items()}, f)

    print(f"Report saved to: {report_path}")
    print(f"History saved to: {history_path}")

    return {
        "model": "Hybrid CNN-BiLSTM-Attention",
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1
    }

if __name__ == "__main__":
    train_hybrid()
