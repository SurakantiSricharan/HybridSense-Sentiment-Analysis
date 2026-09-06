"""
HybridSense: 4-Class Sentiment Analysis Web Application
Streamlit interactive deployment allowing real-time inference with trained models.
Features rich modern styling, multi-model selection, confidence metrics,
and full 4-class probability distributions.
"""
import sys
from pathlib import Path
import pickle
import numpy as np
import streamlit as st

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import tensorflow as tf
from src import config
from src.preprocessing import clean_tweet_text, tokenize_and_pad
# Register AttentionLayer for Hybrid model deserialization
from src.train_hybrid import AttentionLayer

st.set_page_config(
    page_title="HybridSense | 4-Class Sentiment Analysis",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for rich aesthetics
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    .main-header {
        background: linear-gradient(135deg, #1e1b4b 0%, #312e81 50%, #4338ca 100%);
        padding: 2.2rem 2.5rem;
        border-radius: 16px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 10px 25px -5px rgba(67, 56, 202, 0.3);
    }
    
    .badge-ambivalent {
        background: linear-gradient(135deg, #8b5cf6 0%, #6d28d9 100%);
        color: white;
        padding: 0.4rem 1.2rem;
        border-radius: 9999px;
        font-weight: 600;
        display: inline-block;
        box-shadow: 0 4px 12px rgba(139, 92, 246, 0.4);
    }
    
    .badge-positive {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: white;
        padding: 0.4rem 1.2rem;
        border-radius: 9999px;
        font-weight: 600;
        display: inline-block;
        box-shadow: 0 4px 12px rgba(16, 185, 129, 0.4);
    }
    
    .badge-negative {
        background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
        color: white;
        padding: 0.4rem 1.2rem;
        border-radius: 9999px;
        font-weight: 600;
        display: inline-block;
        box-shadow: 0 4px 12px rgba(239, 68, 68, 0.4);
    }
    
    .badge-neutral {
        background: linear-gradient(135deg, #64748b 0%, #475569 100%);
        color: white;
        padding: 0.4rem 1.2rem;
        border-radius: 9999px;
        font-weight: 600;
        display: inline-block;
        box-shadow: 0 4px 12px rgba(100, 116, 139, 0.4);
    }
    
    .result-card {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 14px;
        padding: 1.8rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        margin-top: 1.5rem;
    }
    
    .metric-value {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1e293b;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_resources():
    models = {}
    tokenizer = None

    # 1. Keras Tokenizer
    if config.TOKENIZER_PATH.exists():
        with open(config.TOKENIZER_PATH, "rb") as f:
            tokenizer = pickle.load(f)

    # 2. Hybrid Model (Primary)
    if config.HYBRID_MODEL_PATH.exists():
        try:
            models["Hybrid CNN-BiLSTM-Attention"] = tf.keras.models.load_model(
                config.HYBRID_MODEL_PATH,
                custom_objects={"AttentionLayer": AttentionLayer}
            )
        except Exception as e:
            st.warning(f"Note loading Hybrid: {e}")

    # 3. CNN Model
    if config.CNN_MODEL_PATH.exists():
        try:
            models["CNN"] = tf.keras.models.load_model(config.CNN_MODEL_PATH)
        except Exception as e:
            st.warning(f"Note loading CNN: {e}")

    # 4. BiLSTM Model
    if config.BILSTM_MODEL_PATH.exists():
        try:
            models["BiLSTM"] = tf.keras.models.load_model(config.BILSTM_MODEL_PATH)
        except Exception as e:
            st.warning(f"Note loading BiLSTM: {e}")

    # 5. Baseline Model
    if config.TFIDF_MODEL_PATH.exists() and config.TFIDF_VECTORIZER_PATH.exists():
        try:
            with open(config.TFIDF_MODEL_PATH, "rb") as f:
                clf = pickle.load(f)
            with open(config.TFIDF_VECTORIZER_PATH, "rb") as f:
                vec = pickle.load(f)
            models["TF-IDF + Logistic Regression"] = (clf, vec)
        except Exception as e:
            st.warning(f"Note loading Baseline: {e}")

    return models, tokenizer

models_dict, keras_tokenizer = load_resources()

# Sidebar Setup
with st.sidebar:
    st.image("https://img.icons8.com/clouds/200/brain.png", width=110)
    st.markdown("### **HybridSense System**")
    st.markdown("**B.Tech AIML Project**")
    st.markdown("---")

    available_models = list(models_dict.keys())
    if not available_models:
        st.error("No models detected! Please train models first.")
        selected_model_name = None
    else:
        # Default to Hybrid if present
        default_idx = available_models.index("Hybrid CNN-BiLSTM-Attention") if "Hybrid CNN-BiLSTM-Attention" in available_models else 0
        selected_model_name = st.selectbox(
            "Select Inference Model:",
            available_models,
            index=default_idx
        )

    st.markdown("---")
    st.markdown("#### **Target Classes**")
    st.markdown("• **0 - Negative**: Critical, dissatisfied")
    st.markdown("• **1 - Neutral**: Objective, factual")
    st.markdown("• **2 - Positive**: Praising, satisfied")
    st.markdown("• **3 - Ambivalent**: Dual positive + negative sentiments")
    st.markdown("---")
    st.caption("College Review 1 Demonstrator")

# Main Header
st.markdown("""
<div class="main-header">
    <h1 style="margin:0; font-size:2.4rem; font-weight:700;">HybridSense</h1>
    <h3 style="margin:0.4rem 0 0 0; font-size:1.2rem; font-weight:400; opacity:0.9;">
        4-Class Sentiment Analysis: Capturing Ambivalence via CNN + BiLSTM + Attention
    </h3>
</div>
""", unsafe_allow_html=True)

# Preset Example Buttons
st.markdown("##### 💡 Try Benchmark Example Presets:")
preset_cols = st.columns(4)

if "input_text_val" not in st.session_state:
    st.session_state["input_text_val"] = "The design is excellent but the battery is terrible."

with preset_cols[0]:
    if st.button("🌟 Positive Example", use_container_width=True):
        st.session_state["input_text_val"] = "I absolutely love this product, it works wonderfully!"
with preset_cols[1]:
    if st.button("⚠️ Negative Example", use_container_width=True):
        st.session_state["input_text_val"] = "This product is terrible and completely broke down."
with preset_cols[2]:
    if st.button("⚖️ Neutral Example", use_container_width=True):
        st.session_state["input_text_val"] = "The item arrived on Tuesday according to the tracking schedule."
with preset_cols[3]:
    if st.button("⚡ Ambivalent Example", use_container_width=True):
        st.session_state["input_text_val"] = "The camera quality is excellent, but the battery life is terrible."

# User Text Input
user_text = st.text_area(
    "Enter your text...",
    value=st.session_state["input_text_val"],
    height=120,
    placeholder="Type or paste a tweet or review here..."
)

col_btn, col_info = st.columns([1, 4])
with col_btn:
    analyze_btn = st.button("🚀 Analyze Sentiment", type="primary", use_container_width=True)

def predict_text(text: str, model_name: str):
    if not text.strip():
        return None, 0.0, None

    clean_text = clean_tweet_text(text)

    if model_name == "TF-IDF + Logistic Regression":
        clf, vec = models_dict[model_name]
        vec_text = vec.transform([clean_text])
        probs = clf.predict_proba(vec_text)[0]
    else:
        model = models_dict[model_name]
        pad_seq = tokenize_and_pad([clean_text], keras_tokenizer)
        probs = model.predict(pad_seq, verbose=0)[0]

    pred_class = int(np.argmax(probs))
    confidence = float(probs[pred_class])
    return pred_class, confidence, probs

if analyze_btn:
    if not selected_model_name:
        st.error("Please ensure a model is loaded.")
    elif not user_text.strip():
        st.warning("Please enter some text to analyze.")
    else:
        with st.spinner(f"Running inference with {selected_model_name}..."):
            pred_class, confidence, probs = predict_text(user_text, selected_model_name)

        if pred_class is not None:
            sentiment_name = config.LABEL_TO_NAME[pred_class]

            badge_class = f"badge-{sentiment_name.lower()}"
            badge_html = f"<span class='{badge_class}' style='font-size:1.25rem;'>{sentiment_name}</span>"

            st.markdown(f"""
            <div class="result-card">
                <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap;">
                    <div>
                        <div style="color: #64748b; font-size: 0.95rem; font-weight: 500; margin-bottom: 0.4rem;">PREDICTED SENTIMENT</div>
                        {badge_html}
                    </div>
                    <div>
                        <div style="color: #64748b; font-size: 0.95rem; font-weight: 500; margin-bottom: 0.4rem;">MODEL CONFIDENCE</div>
                        <div class="metric-value">{confidence * 100:.2f}%</div>
                    </div>
                    <div>
                        <div style="color: #64748b; font-size: 0.95rem; font-weight: 500; margin-bottom: 0.4rem;">ACTIVE ARCHITECTURE</div>
                        <div style="font-weight: 600; color: #334155; font-size: 1.1rem;">{selected_model_name}</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("### **Class Probability Distribution**")
            chart_cols = st.columns(4)

            palette = {
                "Negative": "#ef4444",
                "Neutral": "#64748b",
                "Positive": "#10b981",
                "Ambivalent": "#8b5cf6"
            }

            for idx, col in enumerate(chart_cols):
                c_name = config.LABEL_TO_NAME[idx]
                p_val = float(probs[idx])
                with col:
                    st.markdown(f"**{c_name}**")
                    st.progress(p_val)
                    st.markdown(f"<span style='color:{palette[c_name]}; font-weight:700;'>{p_val * 100:.2f}%</span>", unsafe_allow_html=True)

            if pred_class == 3:
                st.info("💡 **Ambivalent Sentiment Detected**: This text exhibits concurrent positive and negative polarities, contrasting different aspects of the same topic.")
