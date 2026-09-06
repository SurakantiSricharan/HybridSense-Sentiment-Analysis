"""
Clause Disentangler Module for HybridSense-X
Splits complex mixed-sentiment and Ambivalent sentences across contrastive discourse
markers, computes sub-clause polarities, and attributes opposing aspect sentiments.
"""

import re
from typing import Any, Dict, List, Optional
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

analyzer = SentimentIntensityAnalyzer()

CONTRASTIVE_MARKERS = [
    "but", "however", "although", "though", "yet", "while",
    "despite", "on the other hand", "nevertheless", "whereas"
]

# Regex pattern matching whole-word contrastive connectives
MARKER_REGEX = re.compile(
    r"\b(" + "|".join(re.escape(m) for m in CONTRASTIVE_MARKERS) + r")\b",
    re.IGNORECASE
)


def disentangle_clauses(text: str) -> Dict[str, Any]:
    """
    Disentangle a sentence into distinct sub-clauses across discourse connectives
    and compute their individual polarities.
    """
    cleaned = text.strip()
    matches = list(MARKER_REGEX.finditer(cleaned))

    if not matches:
        # Fallback to comma/semicolon split if no contrastive marker
        sub_texts = [p.strip() for p in re.split(r"[,;]", cleaned) if len(p.strip()) > 3]
        if len(sub_texts) < 2:
            scores = analyzer.polarity_scores(cleaned)
            return {
                "has_opposing_clauses": False,
                "discourse_marker": None,
                "clauses": [
                    {
                        "clause_id": 1,
                        "text": cleaned,
                        "polarity_score": scores["compound"],
                        "detected_polarity": _label_from_compound(scores["compound"]),
                        "pos_score": scores["pos"],
                        "neg_score": scores["neg"]
                    }
                ],
                "explanation": "Single-clause structure without contrastive conjunctions."
            }
        else:
            clauses_data = []
            for i, st in enumerate(sub_texts[:3]):
                scores = analyzer.polarity_scores(st)
                clauses_data.append({
                    "clause_id": i + 1,
                    "text": st,
                    "polarity_score": scores["compound"],
                    "detected_polarity": _label_from_compound(scores["compound"]),
                    "pos_score": scores["pos"],
                    "neg_score": scores["neg"]
                })
            has_opposing = _check_opposing(clauses_data)
            return {
                "has_opposing_clauses": has_opposing,
                "discourse_marker": "punctuation",
                "clauses": clauses_data,
                "explanation": "Split across punctuation boundaries."
            }

    # Split using the first prominent contrastive marker
    first_match = matches[0]
    marker = first_match.group(0).lower()
    split_idx = first_match.start()
    split_end = first_match.end()

    clause1_text = cleaned[:split_idx].strip().rstrip(",;")
    clause2_text = cleaned[split_end:].strip().lstrip(",;")

    # Compute scores for both clauses
    scores1 = analyzer.polarity_scores(clause1_text) if clause1_text else {"compound": 0, "pos": 0, "neg": 0}
    scores2 = analyzer.polarity_scores(clause2_text) if clause2_text else {"compound": 0, "pos": 0, "neg": 0}

    c1_pol = _label_from_compound(scores1["compound"])
    c2_pol = _label_from_compound(scores2["compound"])

    clauses_data = [
        {
            "clause_id": 1,
            "text": clause1_text,
            "polarity_score": round(scores1["compound"], 3),
            "detected_polarity": c1_pol,
            "pos_score": round(scores1["pos"], 3),
            "neg_score": round(scores1["neg"], 3)
        },
        {
            "clause_id": 2,
            "text": clause2_text,
            "polarity_score": round(scores2["compound"], 3),
            "detected_polarity": c2_pol,
            "pos_score": round(scores2["pos"], 3),
            "neg_score": round(scores2["neg"], 3)
        }
    ]

    has_opposing = _check_opposing(clauses_data)

    if has_opposing:
        explanation = (
            f"Dual-aspect ambivalence detected across discourse marker '{marker}'. "
            f"Clause 1 expresses {c1_pol.upper()} sentiment while Clause 2 expresses {c2_pol.upper()} sentiment."
        )
    else:
        explanation = f"Conjunction '{marker}' links clauses without polarity reversal."

    return {
        "has_opposing_clauses": has_opposing,
        "discourse_marker": marker,
        "clauses": clauses_data,
        "explanation": explanation
    }


def _label_from_compound(compound: float) -> str:
    if compound >= 0.15:
        return "Positive"
    elif compound <= -0.15:
        return "Negative"
    else:
        return "Neutral"


def _check_opposing(clauses: List[Dict[str, Any]]) -> bool:
    polarities = {c["detected_polarity"] for c in clauses}
    return ("Positive" in polarities and "Negative" in polarities)
