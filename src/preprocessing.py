"""
Text Preprocessing Module for HybridSense Sentiment Analysis
Preserves crucial sentiment signals (negations, contrastive markers, emojis)
while removing noise (redundant whitespace, raw URLs).
"""
import re
import html
import pickle
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from src import config

def clean_tweet_text(text: str) -> str:
    """
    Cleans raw tweet text while preserving negation words and contrastive tokens.
    """
    if not isinstance(text, str):
        return ""

    # Unescape HTML entities (e.g., &amp; -> &, &quot; -> ")
    text = html.unescape(text)

    # Normalize URLs
    text = re.sub(r"https?://\S+|www\.\S+", "", text)

    # Standardize user mentions
    text = re.sub(r"@\w+", "@user", text)

    # Replace newlines and tabs with space
    text = re.sub(r"[\r\n\t]+", " ", text)

    # Standardize repeated punctuation (e.g. !!!! -> !)
    text = re.sub(r"([!?.,]){2,}", r"\1", text)

    # Normalize excessive whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text

def fit_keras_tokenizer(texts, max_vocab: int = config.MAX_VOCAB_SIZE, save_path=config.TOKENIZER_PATH):
    """
    Fits and saves a Keras Tokenizer on a corpus of texts.
    """
    tokenizer = Tokenizer(num_words=max_vocab, oov_token="<OOV>", lower=True)
    tokenizer.fit_on_texts(texts)
    
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "wb") as f:
            pickle.dump(tokenizer, f)
        print(f"Tokenizer saved to {save_path}")

    return tokenizer

def load_keras_tokenizer(load_path=config.TOKENIZER_PATH):
    """
    Loads a saved Keras Tokenizer.
    """
    with open(load_path, "rb") as f:
        return pickle.load(f)

def tokenize_and_pad(texts, tokenizer, max_seq_len: int = config.MAX_SEQ_LEN, maxlen: int = None):
    """
    Converts list of texts to padded integer sequences.
    Supports both max_seq_len and maxlen keyword arguments.
    """
    target_len = maxlen if maxlen is not None else max_seq_len
    cleaned = [clean_tweet_text(t) for t in texts]
    sequences = tokenizer.texts_to_sequences(cleaned)
    padded = pad_sequences(sequences, maxlen=target_len, padding="post", truncating="post")
    return padded

