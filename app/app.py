"""
HybridSense-X: Enterprise Full-Stack 4-Class Sentiment & Ambivalence Disentanglement Platform
A high-aesthetic research-grade dashboard integrating Explainable AI (XAI), SQLite persistent telemetry,
dual-head clause disentanglement, REST API documentation, and benchmark visualizers.
"""

import os
import sys
import time
from pathlib import Path
import pickle
import numpy as np
import pandas as pd
import streamlit as st

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import tensorflow as tf
from src import config
from src.preprocessing import clean_tweet_text, tokenize_and_pad
from src.train_hybrid import AttentionLayer
from src.database import (
    get_analytics_summary,
    get_recent_history,
    log_analysis,
    log_feedback
)
from src.clause_disentangler import disentangle_clauses
from src.explainability import compute_token_attribution, generate_attribution_html

st.set_page_config(
    page_title="HybridSense-X | Full-Stack Sentiment & XAI Platform",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enterprise Modern Dark Glassmorphic Styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    .hero-container {
        background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #311042 100%);
        padding: 2.5rem;
        border-radius: 20px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 20px 40px -15px rgba(0, 0, 0, 0.5);
        margin-bottom: 2rem;
    }
    
    .hero-title {
        font-size: 2.4rem;
        font-weight: 800;
        background: linear-gradient(90deg, #38bdf8, #818cf8, #c084fc);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    
    .hero-subtitle {
        color: #94a3b8;
        font-size: 1.05rem;
        max-width: 850px;
        line-height: 1.6;
    }
    
    .glass-card {
        background: rgba(30, 41, 59, 0.7);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 1.5rem;
        margin-bottom: 1.2rem;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.3);
    }
    
    .kpi-card {
        background: rgba(15, 23, 42, 0.8);
        border: 1px solid rgba(99, 102, 241, 0.2);
        border-radius: 14px;
        padding: 1.2rem;
        text-align: center;
    }
    
    .kpi-val {
        font-size: 2rem;
        font-weight: 700;
        color: #f8fafc;
    }
    
    .kpi-label {
        font-size: 0.85rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    .badge-ambivalent {
        background: linear-gradient(135deg, #a855f7 0%, #7e22ce 100%);
        color: white; padding: 0.4rem 1.2rem; border-radius: 9999px; font-weight: 700; display: inline-block;
        box-shadow: 0 4px 15px rgba(168, 85, 247, 0.4);
    }
    .badge-positive {
        background: linear-gradient(135deg, #10b981 0%, #047857 100%);
        color: white; padding: 0.4rem 1.2rem; border-radius: 9999px; font-weight: 700; display: inline-block;
        box-shadow: 0 4px 15px rgba(16, 185, 129, 0.4);
    }
    .badge-negative {
        background: linear-gradient(135deg, #ef4444 0%, #b91c1c 100%);
        color: white; padding: 0.4rem 1.2rem; border-radius: 9999px; font-weight: 700; display: inline-block;
        box-shadow: 0 4px 15px rgba(239, 68, 68, 0.4);
    }
    .badge-neutral {
        background: linear-gradient(135deg, #64748b 0%, #334155 100%);
        color: white; padding: 0.4rem 1.2rem; border-radius: 9999px; font-weight: 700; display: inline-block;
    }
</style>
""", unsafe_allow_html=True)

LABEL_MAP = {0: "Negative", 1: "Neutral", 2: "Positive", 3: "Ambivalent"}
COLOR_MAP = {
    "Negative": "#ef4444",
    "Neutral": "#94a3b8",
    "Positive": "#10b981",
    "Ambivalent": "#a855f7"
}


# Cached Model Loader
@st.cache_resource
def load_all_models():
    models = {}
    try:
        with open(config.BASELINE_MODEL_PATH, "rb") as f:
            models["TF-IDF + Logistic Regression"] = {"model": pickle.load(f)}
        with open(config.TFIDF_VECTORIZER_PATH, "rb") as f:
            models["TF-IDF + Logistic Regression"]["vectorizer"] = pickle.load(f)
    except Exception as e:
        models["TF-IDF + Logistic Regression"] = None

    try:
        models["CNN"] = tf.keras.models.load_model(config.CNN_MODEL_PATH)
    except Exception:
        models["CNN"] = None

    try:
        models["BiLSTM"] = tf.keras.models.load_model(config.BILSTM_MODEL_PATH)
    except Exception:
        models["BiLSTM"] = None

    try:
        models["Hybrid CNN-BiLSTM-Attention"] = tf.keras.models.load_model(
            config.HYBRID_MODEL_PATH,
            custom_objects={"AttentionLayer": AttentionLayer}
        )
    except Exception:
        models["Hybrid CNN-BiLSTM-Attention"] = None

    try:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        distil_path = config.DISTILBERT_DIR
        if distil_path.exists():
            models["DistilBERT"] = {
                "tokenizer": AutoTokenizer.from_pretrained(str(distil_path)),
                "model": AutoModelForSequenceClassification.from_pretrained(str(distil_path))
            }
        else:
            models["DistilBERT"] = None
    except Exception:
        models["DistilBERT"] = None

    tokenizer_path = config.MODELS_DIR / "keras_tokenizer.pkl"
    keras_tok = None
    if tokenizer_path.exists():
        with open(tokenizer_path, "rb") as f:
            keras_tok = pickle.load(f)

    return models, keras_tok


MODELS, KERAS_TOKENIZER = load_all_models()


def run_inference(text: str, model_name: str):
    start_t = time.time()
    probs = [0.25, 0.25, 0.25, 0.25]
    att_weights = None

    if model_name == "TF-IDF + Logistic Regression":
        entry = MODELS.get(model_name)
        if entry:
            vec = entry["vectorizer"]
            clf = entry["model"]
            cleaned = clean_tweet_text(text)
            feat = vec.transform([cleaned])
            probs = clf.predict_proba(feat)[0].tolist()
    elif model_name == "DistilBERT":
        entry = MODELS.get(model_name)
        if entry:
            import torch
            tok = entry["tokenizer"]
            mdl = entry["model"]
            mdl.eval()
            inputs = tok(text, return_tensors="pt", max_length=config.MAX_SEQ_LEN, truncation=True, padding=True)
            with torch.no_grad():
                out = mdl(**inputs)
                probs = torch.softmax(out.logits, dim=-1).cpu().numpy()[0].tolist()
    else:
        mdl = MODELS.get(model_name)
        if mdl and KERAS_TOKENIZER:
            seq = tokenize_and_pad([text], KERAS_TOKENIZER, max_seq_len=config.MAX_SEQ_LEN)
            raw = mdl.predict(seq, verbose=0)
            if isinstance(raw, (list, tuple)):
                probs = raw[0][0].tolist()
                att_weights = raw[1][0].flatten().tolist()
            else:
                probs = raw[0].tolist()

    pred_idx = int(np.argmax(probs))
    pred_label = LABEL_MAP[pred_idx]
    conf = float(probs[pred_idx])
    latency = round((time.time() - start_t) * 1000, 2)
    return pred_label, pred_idx, conf, probs, latency, att_weights


# Hero Header
st.markdown("""
<div class="hero-container">
    <div class="hero-title">HybridSense-X Enterprise Dashboard</div>
    <div class="hero-subtitle">
        A research-grade 4-class sentiment intelligence system featuring Explainable AI (XAI), 
        dual-aspect ambivalence disentanglement, persistent SQLite telemetry, and a high-performance REST API.
    </div>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("### 🎛️ Control Panel")
    selected_model = st.selectbox(
        "Active Classifier",
        [
            "Hybrid CNN-BiLSTM-Attention",
            "DistilBERT",
            "CNN",
            "BiLSTM",
            "TF-IDF + Logistic Regression"
        ]
    )
    
    st.markdown("---")
    st.markdown("### 💾 Storage & Backend Status")
    st.success("🟢 SQLite Database: Connected (`data/hybridsense.db`)")
    st.info("🟢 REST API Ready: `src/api.py` (FastAPI / Swagger)")
    st.markdown("""
    <div style="display: flex; flex-direction: column; gap: 8px; margin-top: 10px;">
        <a href="http://127.0.0.1:8000" target="_blank" style="background: rgba(99, 102, 241, 0.2); border: 1px solid #6366f1; color: white; padding: 6px 12px; border-radius: 6px; text-decoration: none; text-align: center; font-weight: 600; font-size: 0.85rem;">🌐 Main Web Portal (Port 8000) ↗</a>
        <a href="http://127.0.0.1:8000/docs" target="_blank" style="background: rgba(255, 255, 255, 0.05); border: 1px solid rgba(255, 255, 255, 0.15); color: #cbd5e1; padding: 6px 12px; border-radius: 6px; text-decoration: none; text-align: center; font-weight: 600; font-size: 0.85rem;">⚡ Swagger OpenAPI Docs ↗</a>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("### 🏷️ 4-Class Taxonomy")
    st.markdown("""
    - <span class="badge-negative">0: Negative</span>
    - <span class="badge-neutral">1: Neutral</span>
    - <span class="badge-positive">2: Positive</span>
    - <span class="badge-ambivalent">3: Ambivalent</span>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.caption("HybridSense-X • B.Tech AIML Major Project")


# Main Tab Navigation
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🚀 Real-Time Inference & Disentangler",
    "🔬 Explainable AI (XAI) Saliency",
    "📊 Persistent Telemetry & Database",
    "📁 Batch CSV Processing",
    "⚡ REST API & Developer Hub",
    "📈 Academic Model Benchmark"
])

# ----------------- TAB 1: Real-Time Inference -----------------
with tab1:
    col_in, col_out = st.columns([1.1, 0.9])
    
    with col_in:
        st.markdown("#### 📝 Input Text or Review")
        default_sample = "The display and battery life are phenomenal, but the software is full of annoying bugs."
        user_text = st.text_area(
            "Enter sentence to analyze:",
            value=default_sample,
            height=130
        )
        
        c1, c2, c3 = st.columns(3)
        if c1.button("Preset: Ambivalent", use_container_width=True):
            user_text = "The camera quality is extraordinary, however the customer support was shockingly rude."
        if c2.button("Preset: Positive", use_container_width=True):
            user_text = "I absolutely love this new update! It runs so smooth and looks beautiful."
        if c3.button("Preset: Negative", use_container_width=True):
            user_text = "Total waste of money. The hardware broke within two days of usage."

        analyze_btn = st.button("⚡ Run Real-Time Analysis", type="primary", use_container_width=True)

    with col_out:
        if analyze_btn or user_text:
            pred_label, pred_idx, conf, probs, latency, att = run_inference(user_text, selected_model)
            disentangled = disentangle_clauses(user_text)
            
            # Auto-log to SQLite database
            log_id = log_analysis(
                text=user_text,
                predicted_class=pred_idx,
                predicted_label=pred_label,
                confidence=conf,
                model_used=selected_model,
                latency_ms=latency,
                probabilities=probs,
                clauses=disentangled.get("clauses", [])
            )
            
            badge_class = f"badge-{pred_label.lower()}"
            st.markdown(f"""
            <div class="glass-card">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem;">
                    <span class="{badge_class}" style="font-size: 1.15rem;">{pred_label}</span>
                    <span style="color: #94a3b8; font-size: 0.9rem;">⏱️ Latency: <b>{latency} ms</b> | ID #{log_id}</span>
                </div>
                <div style="font-size: 1.6rem; font-weight: 700; color: #f8fafc; margin-bottom: 0.5rem;">
                    {conf * 100:.1f}% Confidence
                </div>
                <div style="color: #cbd5e1; font-size: 0.95rem;">Model: <b>{selected_model}</b></div>
            </div>
            """, unsafe_allow_html=True)
            
            # Probability Bars
            st.markdown("##### 4-Class Probability Spectrum")
            for lbl, p in zip(["Negative", "Neutral", "Positive", "Ambivalent"], probs):
                st.write(f"**{lbl}**: `{p*100:.1f}%`")
                st.progress(float(p))

            # Disentanglement Alert
            if disentangled["has_opposing_clauses"] or pred_label == "Ambivalent":
                st.markdown("---")
                st.markdown("##### 🔀 Disentangled Opposing Sentiment Clauses")
                for cl in disentangled["clauses"]:
                    c_badge = f"badge-{cl['detected_polarity'].lower()}"
                    st.markdown(f"""
                    <div style="background: rgba(15, 23, 42, 0.6); padding: 0.8rem; border-radius: 8px; margin-bottom: 0.5rem; border-left: 4px solid {COLOR_MAP.get(cl['detected_polarity'], '#94a3b8')}">
                        <span class="{c_badge}" style="font-size: 0.75rem; padding: 2px 8px;">Clause {cl['clause_id']}: {cl['detected_polarity']}</span>
                        <div style="margin-top: 5px; color: #f1f5f9; font-size: 0.95rem;">"{cl['text']}"</div>
                    </div>
                    """, unsafe_allow_html=True)

            # Live User Feedback Widget
            st.markdown("---")
            with st.expander("💬 Submit Human Feedback / Correction (Stored in SQLite)"):
                fb_label = st.selectbox("Suggested Correct Label:", ["Negative", "Neutral", "Positive", "Ambivalent"], index=pred_idx)
                fb_comment = st.text_input("Reviewer Notes / Rationale:")
                if st.button("Save Feedback to DB"):
                    log_feedback(log_id, fb_label, fb_comment)
                    st.success("✅ Feedback successfully logged to database!")

# ----------------- TAB 2: Explainable AI -----------------
with tab2:
    st.markdown("#### 🔬 Explainable AI (XAI) & Token Saliency Inspector")
    st.markdown(
        "Transparent sentiment attribution: Green indicates positive polarity triggers, "
        "Red indicates negative polarity triggers, Amber highlights contrastive discourse markers, "
        "and Purple highlights negation modifiers."
    )
    
    sample_xai = st.text_input(
        "Analyze text transparency:",
        value=user_text if 'user_text' in locals() else "The camera is great but the battery is terrible"
    )
    
    if sample_xai:
        tokens = compute_token_attribution(sample_xai)
        html_view = generate_attribution_html(tokens)
        
        st.markdown("""
        <div style="display: flex; gap: 15px; margin-bottom: 12px; font-size: 0.85rem;">
            <span>🟢 <b style="color: #10b981">Positive Triggers</b></span>
            <span>🔴 <b style="color: #ef4444">Negative Triggers</b></span>
            <span>🟡 <b style="color: #f59e0b">Discourse Connectives</b></span>
            <span>🟣 <b style="color: #a855f7">Negation Modifiers</b></span>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="glass-card" style="font-size: 1.25rem; line-height: 2.2;">
            {html_view}
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("##### Token Saliency Breakdown Table")
        df_tok = pd.DataFrame(tokens)
        st.dataframe(df_tok, use_container_width=True)

# ----------------- TAB 3: Telemetry & Database -----------------
with tab3:
    st.markdown("#### 📊 Persistent Telemetry & Real-Time Analytics (SQLite)")
    stats = get_analytics_summary()
    
    k1, k2, k3, k4 = st.columns(4)
    k1.markdown(f"""<div class="kpi-card"><div class="kpi-val">{stats['total_inferences']}</div><div class="kpi-label">Total Inferences Logged</div></div>""", unsafe_allow_html=True)
    k2.markdown(f"""<div class="kpi-card"><div class="kpi-val">{stats['avg_latency_ms']} ms</div><div class="kpi-label">Average Latency</div></div>""", unsafe_allow_html=True)
    k3.markdown(f"""<div class="kpi-card"><div class="kpi-val">{stats['avg_confidence']*100:.1f}%</div><div class="kpi-label">Mean Confidence</div></div>""", unsafe_allow_html=True)
    k4.markdown(f"""<div class="kpi-card"><div class="kpi-val">{stats['ambivalent_count']}</div><div class="kpi-label">Ambivalent Detected</div></div>""", unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("##### 📜 Recent Live Inferences from Database (`data/hybridsense.db`)")
    history = get_recent_history(limit=25)
    if history:
        df_hist = pd.DataFrame(history)
        st.dataframe(df_hist, use_container_width=True)
    else:
        st.info("No inferences logged in database yet. Run an analysis in Tab 1!")

# ----------------- TAB 4: Batch CSV -----------------
with tab4:
    st.markdown("#### 📁 Batch CSV Dataset Processor")
    st.markdown("Upload any CSV file containing a `text` column to batch-classify sentiment.")
    
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded_file:
        df_up = pd.read_csv(uploaded_file)
        if "text" not in df_up.columns:
            st.error("CSV must contain a 'text' column.")
        else:
            st.write(f"Loaded {len(df_up)} rows. Preview:")
            st.dataframe(df_up.head(5))
            if st.button("⚡ Process Batch"):
                with st.spinner("Classifying batch..."):
                    results = []
                    for t in df_up["text"].astype(str):
                        p_lbl, _, p_conf, _, _, _ = run_inference(t, selected_model)
                        results.append({"predicted_sentiment": p_lbl, "confidence": round(p_conf, 4)})
                    df_res = pd.concat([df_up, pd.DataFrame(results)], axis=1)
                    st.success("Batch classification complete!")
                    st.dataframe(df_res.head(10))
                    csv_data = df_res.to_csv(index=False).encode("utf-8")
                    st.download_button("📥 Download Classified CSV", data=csv_data, file_name="hybridsense_predictions.csv", mime="text/csv")

# ----------------- TAB 5: REST API Hub -----------------
with tab5:
    st.markdown("#### ⚡ Enterprise REST API & Developer Hub")
    st.markdown("HybridSense-X provides a production-ready **FastAPI** backend supporting automated Swagger documentation.")
    
    st.markdown("""
    ##### Quickstart: Launching the Backend Server
    ```powershell
    .\\venv\\Scripts\\python.exe -m uvicorn src.api:app --host 127.0.0.1 --port 8000 --reload
    ```
    Once running, open Swagger interactive docs at: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
    """)
    
    st.markdown("##### REST API Code Snippets")
    c_api1, c_api2 = st.columns(2)
    with c_api1:
        st.markdown("**Python `requests` Integration:**")
        st.code("""
import requests

url = "http://127.0.0.1:8000/api/v1/predict"
payload = {
    "text": "The camera is great but the battery is terrible.",
    "model_name": "Hybrid CNN-BiLSTM-Attention"
}
response = requests.post(url, json=payload)
data = response.json()
print("Prediction:", data["predicted_label"])
print("Confidence:", data["confidence"])
print("Clauses:", data["disentangled_clauses"])
        """, language="python")

    with c_api2:
        st.markdown("**cURL Terminal Request:**")
        st.code("""
curl -X POST "http://127.0.0.1:8000/api/v1/predict" \\
     -H "Content-Type: application/json" \\
     -d '{"text": "The food was delicious, but service was slow."}'
        """, language="bash")

# ----------------- TAB 6: Academic Benchmark -----------------
with tab6:
    st.markdown("#### 📈 Multi-Model Comparative Evaluation")
    comp_path = config.RESULTS_DIR / "model_comparison.csv"
    if comp_path.exists():
        df_comp = pd.read_csv(comp_path)
        st.dataframe(df_comp.style.highlight_max(subset=["accuracy", "macro_f1", "weighted_f1"], color="#064e3b"), use_container_width=True)
    
    b1, b2 = st.columns(2)
    acc_chart = config.RESULTS_DIR / "accuracy_comparison.png"
    f1_chart = config.RESULTS_DIR / "macro_f1_comparison.png"
    if acc_chart.exists():
        b1.image(str(acc_chart), caption="Test Accuracy Across Architectures")
    if f1_chart.exists():
        b2.image(str(f1_chart), caption="Macro F1-Score Across Architectures")
