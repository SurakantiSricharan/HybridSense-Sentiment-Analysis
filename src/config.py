"""
Centralized Configuration for HybridSense-Sentiment-Analysis
"""
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"
APP_DIR = BASE_DIR / "app"

# Data Files
RAW_TRAIN_CSV = RAW_DATA_DIR / "tweet_eval_sentiment_train.csv"
RAW_AMBIVALENT_CSV = RAW_DATA_DIR / "ambivalent_candidates.csv"
FINAL_DATASET_CSV = PROCESSED_DATA_DIR / "final_sentiment_dataset.csv"
TRAIN_CSV = PROCESSED_DATA_DIR / "train.csv"
VAL_CSV = PROCESSED_DATA_DIR / "val.csv"
TEST_CSV = PROCESSED_DATA_DIR / "test.csv"

# Model File Paths
TFIDF_MODEL_PATH = MODELS_DIR / "tfidf_logistic_regression.pkl"
TFIDF_VECTORIZER_PATH = MODELS_DIR / "tfidf_vectorizer.pkl"
TOKENIZER_PATH = MODELS_DIR / "keras_tokenizer.pkl"
CNN_MODEL_PATH = MODELS_DIR / "cnn_model.keras"
BILSTM_MODEL_PATH = MODELS_DIR / "bilstm_model.keras"
HYBRID_MODEL_PATH = MODELS_DIR / "hybrid_model.keras"
TRANSFORMER_DIR = MODELS_DIR / "distilbert_sentiment"

# Results
MODEL_COMPARISON_CSV = RESULTS_DIR / "model_comparison.csv"

# Class Mapping
LABEL_TO_NAME = {
    0: "Negative",
    1: "Neutral",
    2: "Positive",
    3: "Ambivalent"
}

NAME_TO_LABEL = {v: k for k, v in LABEL_TO_NAME.items()}

NUM_CLASSES = 4
RANDOM_SEED = 42

# Preprocessing & Sequence Settings
MAX_VOCAB_SIZE = 20000
MAX_SEQ_LEN = 80
EMBEDDING_DIM = 128

# Training Settings (CPU-optimized for reliable E-batch-5 execution)
BATCH_SIZE = 64
EPOCHS = 6
PATIENCE = 2
LEARNING_RATE = 1e-3

# Transformer Settings
TRANSFORMER_NAME = "distilbert-base-uncased"
TRANSFORMER_BATCH_SIZE = 32
TRANSFORMER_EPOCHS = 2
TRANSFORMER_LR = 3e-5
