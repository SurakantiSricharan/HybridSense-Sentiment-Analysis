"""
Explainable AI (XAI) Module for HybridSense-X
Computes token-level importance, attention saliency, and polarity attribution
for transparent, interpretable 4-class sentiment predictions.
"""

import re
from typing import Any, Dict, List
import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

analyzer = SentimentIntensityAnalyzer()


def compute_token_attribution(text: str, attention_weights: List[float] = None) -> List[Dict[str, Any]]:
    """
    Computes token-level polarity attribution for color-coded visual heatmaps.
    Returns list of tokens with their polarity score (-1.0 to +1.0),
    weight (intensity), and highlight category (positive, negative, neutral, discourse).
    """
    tokens = re.findall(r"\w+|[^\w\s]", text)
    if not tokens:
        return []

    contrast_markers = {"but", "however", "although", "though", "yet", "while", "despite"}
    negations = {"not", "never", "no", "neither", "nor", "hardly", "barely", "scarcely"}

    token_data = []
    for i, token in enumerate(tokens):
        lower_token = token.lower()

        # Check if contrast marker or negation
        if lower_token in contrast_markers:
            score = 0.0
            category = "discourse"
            weight = 0.8
        elif lower_token in negations:
            score = -0.4
            category = "negation"
            weight = 0.7
        else:
            # Word-level sentiment from VADER lexicon
            word_score = analyzer.polarity_scores(token)["compound"]
            score = word_score
            if score >= 0.1:
                category = "positive"
                weight = min(1.0, abs(score) * 1.5)
            elif score <= -0.1:
                category = "negative"
                weight = min(1.0, abs(score) * 1.5)
            else:
                category = "neutral"
                weight = 0.1

        token_data.append({
            "token": token,
            "score": round(score, 3),
            "weight": round(weight, 3),
            "category": category
        })

    # If neural attention weights are available, modulate weights
    if attention_weights and len(attention_weights) >= len(tokens):
        norm_att = np.array(attention_weights[:len(tokens)])
        if norm_att.max() > 0:
            norm_att = norm_att / norm_att.max()
            for i, td in enumerate(token_data):
                td["attention_weight"] = round(float(norm_att[i]), 3)

    return token_data


def generate_attribution_html(token_data: List[Dict[str, Any]]) -> str:
    """
    Generates rich HTML snippet with CSS background highlights for Streamlit.
    """
    html_spans = []
    for td in token_data:
        tok = td["token"]
        cat = td["category"]
        w = td["weight"]

        if cat == "positive":
            bg = f"rgba(16, 185, 129, {max(0.2, w * 0.7)})"  # Emerald green
            border = "1px solid #10b981"
            color = "#ffffff"
        elif cat == "negative":
            bg = f"rgba(239, 68, 68, {max(0.2, w * 0.7)})"   # Crimson red
            border = "1px solid #ef4444"
            color = "#ffffff"
        elif cat == "discourse":
            bg = "rgba(245, 158, 11, 0.4)"                    # Amber
            border = "1px dashed #f59e0b"
            color = "#ffffff"
        elif cat == "negation":
            bg = "rgba(168, 85, 247, 0.4)"                    # Purple
            border = "1px solid #a855f7"
            color = "#ffffff"
        else:
            bg = "rgba(255, 255, 255, 0.05)"
            border = "none"
            color = "#d1d5db"

        span = (
            f"<span style='background-color: {bg}; border: {border}; color: {color}; "
            f"padding: 2px 6px; margin: 2px 1px; border-radius: 4px; font-weight: 500; display: inline-block;'>"
            f"{tok}</span>"
        )
        html_spans.append(span)

    return " ".join(html_spans)
