"""
HybridSense-X Inference Engine
Provides high-performance, unified on-demand inference across all trained models:
- Hybrid CNN-BiLSTM-Attention (Proposed architecture)
- DistilBERT (Fine-tuned transformer)
- CNN (1D Convolutional Network)
- BiLSTM (Bidirectional LSTM)
- TF-IDF + Logistic Regression (Linguistic Baseline)
"""

import os
import sys
import time
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

# Ensure project root is in sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src import config
from src.preprocessing import clean_tweet_text, tokenize_and_pad

LABEL_MAP = {0: "Negative", 1: "Neutral", 2: "Positive", 3: "Ambivalent"}
INDEX_MAP = {v: k for k, v in LABEL_MAP.items()}

# Cached models & tokenizer
_LOADED_MODELS: Dict[str, Any] = {}
_KERAS_TOKENIZER = None


def get_keras_tokenizer():
    global _KERAS_TOKENIZER
    if _KERAS_TOKENIZER is None:
        tok_path = getattr(config, "TOKENIZER_PATH", config.MODELS_DIR / "keras_tokenizer.pkl")
        if tok_path.exists():
            with open(tok_path, "rb") as f:
                _KERAS_TOKENIZER = pickle.load(f)
    return _KERAS_TOKENIZER


def get_model(model_name: str):
    """Lazy-load the requested model on demand."""
    global _LOADED_MODELS
    if model_name in _LOADED_MODELS:
        return _LOADED_MODELS[model_name]

    if model_name == "TF-IDF + Logistic Regression":
        tfidf_path = getattr(config, "TFIDF_MODEL_PATH", config.MODELS_DIR / "tfidf_logistic_regression.pkl")
        vec_path = getattr(config, "TFIDF_VECTORIZER_PATH", config.MODELS_DIR / "tfidf_vectorizer.pkl")
        if tfidf_path.exists() and vec_path.exists():
            with open(tfidf_path, "rb") as f:
                clf = pickle.load(f)
            with open(vec_path, "rb") as f:
                vec = pickle.load(f)
            _LOADED_MODELS[model_name] = {"model": clf, "vectorizer": vec}
            return _LOADED_MODELS[model_name]

    elif model_name == "DistilBERT":
        try:
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            distil_path = getattr(config, "TRANSFORMER_DIR", config.MODELS_DIR / "distilbert_sentiment")
            if distil_path.exists():
                _LOADED_MODELS[model_name] = {
                    "tokenizer": AutoTokenizer.from_pretrained(str(distil_path)),
                    "model": AutoModelForSequenceClassification.from_pretrained(str(distil_path))
                }
                return _LOADED_MODELS[model_name]
        except Exception as e:
            print(f"[Warning] DistilBERT load failed: {e}")

    # Keras models require tensorflow and custom layers
    import tensorflow as tf
    from src.train_hybrid import AttentionLayer

    if model_name == "CNN":
        cnn_path = getattr(config, "CNN_MODEL_PATH", config.MODELS_DIR / "cnn_model.keras")
        if cnn_path.exists():
            _LOADED_MODELS[model_name] = tf.keras.models.load_model(cnn_path)
            return _LOADED_MODELS[model_name]

    elif model_name == "BiLSTM":
        bilstm_path = getattr(config, "BILSTM_MODEL_PATH", config.MODELS_DIR / "bilstm_model.keras")
        if bilstm_path.exists():
            _LOADED_MODELS[model_name] = tf.keras.models.load_model(bilstm_path)
            return _LOADED_MODELS[model_name]

    # Default to Hybrid CNN-BiLSTM-Attention
    hybrid_path = getattr(config, "HYBRID_MODEL_PATH", config.MODELS_DIR / "hybrid_model.keras")
    if hybrid_path.exists():
        _LOADED_MODELS["Hybrid CNN-BiLSTM-Attention"] = tf.keras.models.load_model(
            hybrid_path,
            custom_objects={"AttentionLayer": AttentionLayer}
        )
        return _LOADED_MODELS["Hybrid CNN-BiLSTM-Attention"]

    return None


def predict_single(
    text: str,
    model_name: str = "Hybrid CNN-BiLSTM-Attention"
) -> Tuple[str, float, List[float], Optional[List[float]]]:
    """
    Perform 4-class sentiment inference on a single text string.

    Returns:
        pred_label (str): One of ["Negative", "Neutral", "Positive", "Ambivalent"]
        confidence (float): Confidence score (0.0 to 1.0)
        probabilities (List[float]): 4-class probability distribution [Neg, Neu, Pos, Amb]
        attention_weights (Optional[List[float]]): Saliency attention weights if available
    """
    text = text.strip() if text else ""
    if not text:
        return "Neutral", 0.25, [0.25, 0.25, 0.25, 0.25], None

    probs = [0.25, 0.25, 0.25, 0.25]
    att_weights = None

    if model_name == "TF-IDF + Logistic Regression":
        entry = get_model(model_name)
        if entry:
            vec = entry["vectorizer"]
            clf = entry["model"]
            cleaned = clean_tweet_text(text)
            feat = vec.transform([cleaned])
            probs = clf.predict_proba(feat)[0].tolist()
    elif model_name == "DistilBERT":
        entry = get_model(model_name)
        if entry:
            import torch
            tok = entry["tokenizer"]
            mdl = entry["model"]
            mdl.eval()
            inputs = tok(
                text,
                return_tensors="pt",
                max_length=config.MAX_SEQ_LEN,
                truncation=True,
                padding=True
            )
            with torch.no_grad():
                out = mdl(**inputs)
                probs = torch.softmax(out.logits, dim=-1).cpu().numpy()[0].tolist()
    else:
        # Keras Neural Models (Hybrid, CNN, BiLSTM)
        mdl = get_model(model_name)
        tokenizer = get_keras_tokenizer()
        if mdl and tokenizer:
            seq = tokenize_and_pad([text], tokenizer, max_seq_len=config.MAX_SEQ_LEN)
            raw = mdl.predict(seq, verbose=0)
            if isinstance(raw, (list, tuple)):
                probs = raw[0][0].tolist()
                att_weights = raw[1][0].flatten().tolist()
            else:
                probs = raw[0].tolist()

    pred_idx = int(np.argmax(probs))
    pred_label = LABEL_MAP[pred_idx]
    confidence = float(probs[pred_idx])

    return pred_label, confidence, probs, att_weights
