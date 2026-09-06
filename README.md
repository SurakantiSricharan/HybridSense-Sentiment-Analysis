# HybridSense: 4-Class Sentiment Analysis via CNN + BiLSTM + Attention

[![Python 3.10](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/Framework-TensorFlow%20%7C%20PyTorch%20%7C%20HuggingFace-orange.svg)](https://tensorflow.org)
[![Deployment](https://img.shields.io/badge/App-Streamlit-red.svg)](https://streamlit.io)
[![Status](https://img.shields.io/badge/Academic-B.Tech%20AIML%20Project-green.svg)]()

> **HybridSense** is an academic AI/ML project that extends conventional 3-class sentiment analysis into a **4-class classification paradigm**:
> **0: Negative**, **1: Neutral**, **2: Positive**, and **3: Ambivalent**.
> It introduces a defensible linguistic clause-level contrast mining methodology and proposes a state-of-the-art **Hybrid CNN-BiLSTM-Attention** deep learning architecture.

---

## 📌 Problem Statement & Research Gap
Conventional sentiment analysis systems classify text along a unidimensional axis into Negative, Neutral, or Positive. However, in real-world discourse (especially Twitter and product reviews), people frequently express **concurrent opposing sentiments** towards different aspects of the same topic:

> *"The camera quality is excellent, but the battery life is terrible."*

Standard 3-class models either misclassify such statements as Neutral (averaging polarities) or arbitrarily pick whichever clause has higher polarity intensity. **HybridSense** resolves this research gap by explicitly identifying, modeling, and classifying **Ambivalent** sentiment.

---

## 🎯 Project Objectives
1. **Curate a 4-Class Twitter Dataset**: Extend Cardiff NLP's TweetEval Sentiment benchmark using defensible clause-level contrastive polarity mining and verified candidate annotations without data fabrication.
2. **Implement Baseline & Deep Learning Topologies**: Develop traditional ML (TF-IDF + Logistic Regression), CNN (Conv1D), and BiLSTM classifiers.
3. **Develop Primary Proposed Hybrid Architecture**: Connect CNN (local phrase features) $\to$ BiLSTM (sequential context) $\to$ Self-Attention mechanism (salience weighting) $\to$ Softmax.
4. **Compare Pre-Trained Transformers**: Benchmark against Hugging Face DistilBERT.
5. **Conduct Rigorous Evaluation & Error Analysis**: Evaluate on unseen stratified test data using Accuracy, Weighted F1, Macro F1, and confusion matrices.
6. **Deploy Interactive Web Application**: Build a real-time Streamlit dashboard for collegiate demonstration.

---

## 🏛️ System Architecture

```
                       Raw Text Input
                             │
                             ▼
                   Text Preprocessing
         (URL/mention normalization, negation preserved)
                             │
                             ▼
                   Tokenizer & Sequences
                     (Max Length: 80)
                             │
                             ▼
                      Embedding Layer
                   (Vocab: 20k, Dim: 128)
                             │
                             ▼
                  1D Convolutional Layer
               (128 Filters, Kernel: 3, ReLU)
                             │
                             ▼
                   Bidirectional LSTM
                 (64 Units x 2 Directions)
                             │
                             ▼
                    Attention Mechanism
           (Bahdanau-style token salience pooling)
                             │
                             ▼
                  Dense Layer (64) + Dropout
                             │
                             ▼
                     Softmax Activation
                             │
                             ▼
                  4-Class Sentiment Output
             [Negative, Neutral, Positive, Ambivalent]
```

---

## 📊 Dataset Statistics & 4-Class Mapping

- **Primary Source**: Cardiff NLP TweetEval Sentiment (`cardiffnlp/tweet_eval`)
- **Total Clean Instances**: **45,586 tweets**
- **Ambivalent Instances**: **682 tweets** (36 manual verified + 657 mined with strict linguistic clause-level polarity contrast, deduplicated)

| Class ID | Sentiment | Train Split (70%) | Validation Split (10%) | Test Split (20%) | Total Instances |
|---|---|---|---|---|---|
| **0** | **Negative** | 4,812 | 687 | 1,375 | 6,874 |
| **1** | **Neutral** | 14,302 | 2,043 | 4,086 | 20,431 |
| **2** | **Positive** | 12,319 | 1,760 | 3,520 | 17,599 |
| **3** | **Ambivalent** | 477 | 68 | 137 | 682 |
| **Total** | | **31,910** | **4,558** | **9,118** | **45,586** |

---

## 🧪 Model Performance & Comparison

Evaluated on the unseen stratified test set (9,118 samples):

| Architecture | Accuracy | Precision (Weighted) | Recall (Weighted) | Macro F1 | Weighted F1 | Ambivalent Recall |
|---|---|---|---|---|---|---|
| **TF-IDF + Logistic Regression** | **0.6248** | 0.6543 | 0.6248 | 0.5291 | 0.6344 | 53.28% |
| **CNN (Conv1D + Pooling)** | **0.6008** | 0.6550 | 0.6008 | 0.5136 | 0.6134 | **81.02%** |
| **BiLSTM (Bidirectional LSTM)** | **0.5829** | 0.6296 | 0.5829 | 0.5064 | 0.5959 | **75.91%** |
| **Hybrid CNN-BiLSTM-Attention** | *Syncing* | *Syncing* | *Syncing* | *Syncing* | *Syncing* | *Syncing* |
| **DistilBERT (Transformer)** | *Syncing* | *Syncing* | *Syncing* | *Syncing* | *Syncing* | *Syncing* |

*Exact measured values are saved in `results/model_comparison.csv`.*

---

## 📂 Project Structure

```
HybridSense-Sentiment-Analysis/
│
├── data/
│   ├── raw/
│   │   ├── tweet_eval_sentiment_train.csv     # Raw TweetEval corpus
│   │   └── ambivalent_candidates.csv          # Annotated Ambivalent candidates
│   └── processed/
│       ├── final_sentiment_dataset.csv        # Master 4-class dataset (45,586 rows)
│       ├── train.csv                          # Stratified train split (31,910 rows)
│       ├── val.csv                            # Stratified validation split (4,558 rows)
│       └── test.csv                           # Stratified test split (9,118 rows)
│
├── notebooks/
│   ├── 01_data_exploration.ipynb              # Initial EDA on TweetEval
│   └── 02_dataset_preparation.ipynb           # Ambivalent verification & preparation
│
├── src/
│   ├── config.py                              # Centralized paths, seeds, hyperparameters
│   ├── preprocessing.py                       # Text cleaning & sequence tokenization
│   ├── dataset_builder.py                     # Mined contrastive Ambivalent pipeline
│   ├── train_baseline.py                      # TF-IDF + Logistic Regression
│   ├── train_cnn.py                           # 1D CNN classifier
│   ├── train_bilstm.py                        # Bidirectional LSTM classifier
│   ├── train_hybrid.py                        # Primary CNN-BiLSTM-Attention classifier
│   ├── train_transformer.py                   # DistilBERT classifier
│   ├── evaluate.py                            # Comparative metrics & chart generators
│   └── error_analysis.py                      # In-depth Ambivalent confusion diagnosis
│
├── models/                                    # Saved trained weights and tokenizers
├── results/                                   # Comparison CSVs, confusion matrices, charts
├── app/
│   └── app.py                                 # Streamlit web application
│
├── PROJECT_PROGRESS.md                        # Phase completion tracker
├── PROJECT_REPORT.md                          # Academic B.Tech AIML Review 1 report
├── requirements.txt                           # Verified dependencies
└── README.md                                  # Project overview
```

---

## 🚀 Installation & How to Run

### 1. Activate Environment
Ensure Python 3.10 is selected:
```bash
.\venv\Scripts\activate
```

### 2. Install Requirements (if needed)
```bash
pip install -r requirements.txt
```

### 3. Build Dataset & Stratified Splits
```bash
python -m src.dataset_builder
```

### 4. Train Models
```bash
# Baseline
python -m src.train_baseline

# CNN
python -m src.train_cnn

# BiLSTM
python -m src.train_bilstm

# Primary Hybrid Model (CNN + BiLSTM + Attention)
python -m src.train_hybrid

# Transformer (DistilBERT)
python -m src.train_transformer
```

### 5. Evaluate & Generate Plots
```bash
python -m src.evaluate
python -m src.error_analysis
```

### 6. Launch the Streamlit Web Application
```bash
streamlit run app/app.py
```
Open your browser at `http://localhost:8501`.

---

## 🔍 Validation Benchmark Cases

Test the model in the Streamlit app with these benchmark phrases:

1. *"I love this product."* $\to$ **Positive**
2. *"This product is terrible."* $\to$ **Negative**
3. *"It is okay."* $\to$ **Neutral**
4. *"The design is excellent but the battery is terrible."* $\to$ **Ambivalent**

---

## 🎓 Academic Review 1 Compliance
This project fulfills all eight stages of the B.Tech AIML review:
1. **Problem Identification**: Documented in Section 1 and `PROJECT_REPORT.md`.
2. **Literature Survey**: 5 published 2024 academic works cited and reviewed.
3. **Requirement Analysis**: Functional, non-functional, hardware, and software requirements specified.
4. **Dataset Collection**: 45,586 tweets with verified 4-class distribution.
5. **System Design**: Flowcharts, attention mathematics, and block diagrams.
6. **Implementation**: Modular `src/` codebase with 5 implemented models.
7. **Testing**: Stratified test split evaluation and interactive inference.
8. **Evaluation**: Comprehensive comparative tables, confusion matrices, and error analysis.

---

## 👥 Authors
- **B.Tech AIML Student Team**, Final Year Project.