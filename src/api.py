"""
HybridSense-X Enterprise REST API
High-performance FastAPI service providing 4-class sentiment inference,
Explainable AI (XAI) token saliency, clause disentanglement, and telemetry.
"""

import time
from typing import Any, Dict, List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.database import (
    get_analytics_summary,
    get_recent_history,
    log_analysis,
    log_feedback
)
from src.clause_disentangler import disentangle_clauses
from src.explainability import compute_token_attribution

app = FastAPI(
    title="HybridSense-X API",
    description="Enterprise 4-Class Sentiment & Ambivalence Disentanglement Engine",
    version="2.0.0"
)

# Enable CORS for full-stack integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

LABEL_MAP = {0: "Negative", 1: "Neutral", 2: "Positive", 3: "Ambivalent"}


class PredictRequest(BaseModel):
    text: str = Field(..., example="The camera is phenomenal but the battery life is terrible.")
    model_name: Optional[str] = Field("Hybrid CNN-BiLSTM-Attention", example="Hybrid CNN-BiLSTM-Attention")


class FeedbackRequest(BaseModel):
    log_id: int
    suggested_label: str
    user_comment: Optional[str] = ""


class DisentangleRequest(BaseModel):
    text: str = Field(..., example="The food was delicious, however the customer service was awful.")


from pathlib import Path
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

@app.get("/")
def root():
    index_file = FRONTEND_DIR / "index.html"
    if index_file.exists():
        return FileResponse(str(index_file))
    return {
        "system": "HybridSense-X Research Platform",
        "version": "2.0.0",
        "docs_url": "/docs",
        "status": "online"
    }

@app.get("/style.css")
def get_css():
    return FileResponse(str(FRONTEND_DIR / "style.css"))

@app.get("/app.js")
def get_js():
    return FileResponse(str(FRONTEND_DIR / "app.js"))


@app.get("/api/v1/health")
def health_check():
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "database": "connected (SQLite)",
        "models_available": [
            "TF-IDF + Logistic Regression",
            "CNN",
            "BiLSTM",
            "DistilBERT",
            "Hybrid CNN-BiLSTM-Attention"
        ]
    }


@app.post("/api/v1/disentangle")
def disentangle_text(req: DisentangleRequest):
    """Disentangle a sentence into opposing polarity clauses across discourse markers."""
    result = disentangle_clauses(req.text)
    return result


@app.post("/api/v1/explain")
def explain_text(req: PredictRequest):
    """Compute token-level polarity attribution and highlight categories."""
    tokens = compute_token_attribution(req.text)
    return {"text": req.text, "tokens": tokens}


@app.post("/api/v1/predict")
def predict_sentiment(req: PredictRequest):
    """
    Perform 4-class sentiment analysis, extract clauses if ambivalent,
    and persist inference telemetry in the SQLite database.
    """
    start_time = time.time()
    text = req.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text must not be empty.")

    # Lazy-load inference to maintain fast startup
    from src.inference import predict_single

    try:
        pred_label, conf, probs, att_weights = predict_single(text, model_name=req.model_name)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")

    latency_ms = round((time.time() - start_time) * 1000, 2)
    pred_class = {v: k for k, v in LABEL_MAP.items()}.get(pred_label, 1)

    # Disentangle sub-clauses
    disentangled = disentangle_clauses(text)
    clauses = disentangled.get("clauses", [])

    # Compute XAI token attribution
    token_attribution = compute_token_attribution(text, att_weights)

    # Persist in SQLite database
    log_id = log_analysis(
        text=text,
        predicted_class=pred_class,
        predicted_label=pred_label,
        confidence=conf,
        model_used=req.model_name,
        latency_ms=latency_ms,
        probabilities=probs,
        clauses=clauses
    )

    return {
        "log_id": log_id,
        "text": text,
        "predicted_label": pred_label,
        "predicted_class": pred_class,
        "confidence": conf,
        "probabilities": {
            "Negative": probs[0] if len(probs) > 0 else 0.0,
            "Neutral": probs[1] if len(probs) > 1 else 0.0,
            "Positive": probs[2] if len(probs) > 2 else 0.0,
            "Ambivalent": probs[3] if len(probs) > 3 else 0.0,
        },
        "model_used": req.model_name,
        "latency_ms": latency_ms,
        "disentangled_clauses": disentangled,
        "token_attribution": token_attribution
    }


@app.get("/api/v1/history")
def get_history(limit: int = 50):
    """Retrieve recent inference events logged in SQLite database."""
    history = get_recent_history(limit=limit)
    return {"count": len(history), "history": history}


@app.get("/api/v1/analytics")
def get_analytics():
    """Retrieve aggregate sentiment analytics and telemetry from database."""
    stats = get_analytics_summary()
    return stats


@app.post("/api/v1/feedback")
def submit_feedback(req: FeedbackRequest):
    """Submit user rating or classification correction to database."""
    log_feedback(req.log_id, req.suggested_label, req.user_comment)
    return {"status": "success", "message": "Feedback recorded in database"}


class StreamStartRequest(BaseModel):
    source: Optional[str] = Field("simulation", example="simulation")
    query: Optional[str] = Field("technology", example="technology")
    interval: Optional[float] = Field(2.0, example=2.0)
    model_name: Optional[str] = Field("Hybrid CNN-BiLSTM-Attention", example="Hybrid CNN-BiLSTM-Attention")
    twitter_bearer: Optional[str] = None


from src.social_stream import STREAM_ENGINE


@app.post("/api/v1/stream/start")
def start_live_stream(req: Optional[StreamStartRequest] = None):
    """Start autonomous background streaming from social media sources."""
    if req is None:
        req = StreamStartRequest()
    status = STREAM_ENGINE.start(
        source=req.source,
        query=req.query,
        interval=req.interval,
        model_name=req.model_name,
        twitter_bearer=req.twitter_bearer
    )
    return {"status": "started", "stream": status}


@app.post("/api/v1/stream/stop")
def stop_live_stream():
    """Halt background social streaming."""
    status = STREAM_ENGINE.stop()
    return {"status": "stopped", "stream": status}


@app.get("/api/v1/stream/status")
def get_stream_status():
    """Check live status, throughput, and counts of the automated social stream."""
    return STREAM_ENGINE.get_status()


@app.get("/api/v1/stream/recent")
def get_stream_recent(limit: int = 25):
    """Get latest analyzed posts from the in-memory stream buffer."""
    posts = STREAM_ENGINE.get_recent_posts(limit=limit)
    return {"count": len(posts), "posts": posts}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)

