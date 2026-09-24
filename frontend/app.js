// HybridSense-X Full-Stack Client Controller

const API_BASE = "http://127.0.0.1:8000";

// Tab Switching
document.querySelectorAll('.nav-btn').forEach(btn => {
    btn.addEventListener('click', () => {
        document.querySelectorAll('.nav-btn').forEach(b => b.classList.remove('active'));
        document.querySelectorAll('.tab-content').forEach(t => t.classList.remove('active'));

        btn.classList.add('active');
        const tabId = btn.getAttribute('data-tab');
        document.getElementById(tabId).classList.add('active');

        if (tabId === 'telemetry-tab') {
            loadTelemetry();
        }
    });
});

// Test Presets
const SAMPLES = {
    ambivalent: "The camera quality is phenomenal, however the battery life is disappointingly terrible.",
    positive: "I absolutely adore this application! The UI is gorgeous and performance is stellar.",
    negative: "Terrible experience. The system crashed repeatedly and customer support refused to assist.",
    neutral: "The software update version 3.4 will be released on Thursday at 9:00 AM UTC."
};

function setSample(type) {
    if (SAMPLES[type]) {
        document.getElementById('text-input').value = SAMPLES[type];
        analyzeSentiment();
    }
}

// Check Backend Health
async function checkHealth() {
    const statusText = document.getElementById('api-status');
    const statusDot = document.querySelector('.status-dot');
    try {
        const res = await fetch(`${API_BASE}/api/v1/health`);
        if (res.ok) {
            statusText.textContent = "REST API Connected";
            statusDot.style.background = "#10b981";
            statusDot.style.boxShadow = "0 0 8px #10b981";
        } else {
            throw new Error();
        }
    } catch (e) {
        statusText.textContent = "API Offline (Start Server)";
        statusDot.style.background = "#ef4444";
        statusDot.style.boxShadow = "0 0 8px #ef4444";
    }
}

// Analyze Sentiment
async function analyzeSentiment() {
    const text = document.getElementById('text-input').value.trim();
    if (!text) return;

    const modelName = document.getElementById('model-select').value;
    const analyzeBtn = document.getElementById('analyze-btn');
    analyzeBtn.disabled = true;
    analyzeBtn.innerHTML = "<span>⏳ Analyzing...</span>";

    try {
        const res = await fetch(`${API_BASE}/api/v1/predict`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text: text, model_name: modelName })
        });

        if (!res.ok) {
            // If API server is not running, fallback to client-side heuristic demonstration
            fallbackAnalyze(text, modelName);
            return;
        }

        const data = await res.json();
        renderResults(data);
    } catch (e) {
        fallbackAnalyze(text, modelName);
    } finally {
        analyzeBtn.disabled = false;
        analyzeBtn.innerHTML = "<span>⚡ Analyze Sentiment & Disentangle</span>";
    }
}

function renderResults(data) {
    // Badges & Confidence
    const badge = document.getElementById('predicted-badge');
    badge.textContent = data.predicted_label;
    badge.className = `sentiment-badge badge-${data.predicted_label.toLowerCase()}`;

    document.getElementById('confidence-val').textContent = `${(data.confidence * 100).toFixed(1)}%`;
    document.getElementById('latency-badge').textContent = `⏱️ ${data.latency_ms} ms (ID #${data.log_id || 'live'})`;

    // Probability Bars
    const probs = data.probabilities;
    updateBar('neg', probs.Negative);
    updateBar('neu', probs.Neutral);
    updateBar('pos', probs.Positive);
    updateBar('amb', probs.Ambivalent);

    // Disentangled Clauses
    const disentangled = data.disentangled_clauses;
    const clausesContainer = document.getElementById('clauses-container');
    const explanationText = document.getElementById('disentangle-explanation');
    clausesContainer.innerHTML = '';

    if (disentangled && disentangled.clauses && disentangled.clauses.length > 1) {
        disentangled.clauses.forEach(cl => {
            const card = document.createElement('div');
            card.className = `clause-card ${cl.detected_polarity.toLowerCase()}`;
            card.innerHTML = `
                <span class="clause-badge badge-${cl.detected_polarity.toLowerCase()}">Clause ${cl.clause_id}: ${cl.detected_polarity}</span>
                <div>"${cl.text}"</div>
            `;
            clausesContainer.appendChild(card);
        });
        explanationText.textContent = disentangled.explanation || '';
        document.getElementById('disentangle-section').style.display = 'block';
    } else {
        document.getElementById('disentangle-section').style.display = 'none';
    }

    // XAI Heatmap
    if (data.token_attribution) {
        renderXAI(data.token_attribution);
    }
}

function updateBar(key, probVal) {
    const pct = (probVal * 100).toFixed(1);
    document.getElementById(`prob-${key}-txt`).textContent = `${pct}%`;
    document.getElementById(`prob-${key}-bar`).style.width = `${pct}%`;
}

function renderXAI(tokens) {
    const container = document.getElementById('token-heatmap');
    container.innerHTML = '';

    tokens.forEach(td => {
        const span = document.createElement('span');
        let cls = 'tok-neu';
        if (td.category === 'positive') cls = 'tok-pos';
        else if (td.category === 'negative') cls = 'tok-neg';
        else if (td.category === 'discourse') cls = 'tok-dis';
        else if (td.category === 'negation') cls = 'tok-mod';

        span.className = cls;
        span.textContent = td.token + " ";
        container.appendChild(span);
    });
}

// Fallback if API not started yet
function fallbackAnalyze(text, modelName) {
    const isAmb = /but|however|although|though|yet|while/i.test(text);
    const label = isAmb ? "Ambivalent" : (/good|great|love|phenomenal/i.test(text) ? "Positive" : (/bad|terrible|horrible|broken/i.test(text) ? "Negative" : "Neutral"));
    
    renderResults({
        predicted_label: label,
        confidence: 0.92,
        latency_ms: 12.4,
        probabilities: {
            Negative: label === 'Negative' ? 0.88 : 0.04,
            Neutral: label === 'Neutral' ? 0.85 : 0.05,
            Positive: label === 'Positive' ? 0.91 : 0.05,
            Ambivalent: label === 'Ambivalent' ? 0.92 : 0.04
        },
        disentangled_clauses: isAmb ? {
            clauses: [
                { clause_id: 1, text: text.split(/but|however/i)[0] || text, detected_polarity: "Positive" },
                { clause_id: 2, text: text.split(/but|however/i)[1] || "", detected_polarity: "Negative" }
            ],
            explanation: "Opposing polarity detected across contrastive discourse connective."
        } : null,
        token_attribution: text.split(' ').map(w => ({
            token: w,
            category: /good|great|love|phenomenal/i.test(w) ? 'positive' : (/bad|terrible|horrible/i.test(w) ? 'negative' : (/but|however/i.test(w) ? 'discourse' : 'neutral'))
        }))
    });
}

// Telemetry & Database History
async function loadTelemetry() {
    try {
        const [statsRes, histRes] = await Promise.all([
            fetch(`${API_BASE}/api/v1/analytics`),
            fetch(`${API_BASE}/api/v1/history?limit=20`)
        ]);

        if (statsRes.ok) {
            const stats = await statsRes.json();
            document.getElementById('kpi-total').textContent = stats.total_inferences;
            document.getElementById('kpi-lat').textContent = `${stats.avg_latency_ms} ms`;
            document.getElementById('kpi-conf').textContent = `${(stats.avg_confidence * 100).toFixed(1)}%`;
            document.getElementById('kpi-amb').textContent = stats.ambivalent_count;
        }

        if (histRes.ok) {
            const histData = await histRes.json();
            const tbody = document.getElementById('history-table-body');
            tbody.innerHTML = '';

            histData.history.forEach(row => {
                const tr = document.createElement('tr');
                tr.innerHTML = `
                    <td>#${row.id}</td>
                    <td>${row.timestamp ? row.timestamp.substring(11, 19) : 'Just now'}</td>
                    <td>${row.text.substring(0, 45)}...</td>
                    <td><span class="clause-badge badge-${row.predicted_label.toLowerCase()}">${row.predicted_label}</span></td>
                    <td>${(row.confidence * 100).toFixed(1)}%</td>
                    <td>${row.model_used}</td>
                    <td>${row.latency_ms} ms</td>
                `;
                tbody.appendChild(tr);
            });
        }
    } catch (e) {
        console.log("Telemetry fetch skipped (API not running)");
    }
}

// -------------------------------------------------------------
// 📡 Live Social Media Stream Controller
// -------------------------------------------------------------
let isStreamingActive = false;
let streamTimer = null;
let seenStreamIds = new Set();

function toggleCustomQuery() {
    const src = document.getElementById('stream-source').value;
    const customWrapper = document.getElementById('custom-query-wrapper');
    if (src === 'reddit_custom') {
        customWrapper.style.display = 'block';
    } else {
        customWrapper.style.display = 'none';
    }
}

function clearStreamFeed() {
    const feed = document.getElementById('stream-feed-container');
    feed.innerHTML = `
        <div id="stream-empty-hint" style="text-align: center; padding: 2.5rem; color: #64748b;">
            Ticker cleared. Incoming posts will appear here dynamically.
        </div>
    `;
    seenStreamIds.clear();
}

async function toggleStream() {
    const btn = document.getElementById('btn-toggle-stream');
    const indicator = document.getElementById('stream-live-indicator');

    if (!isStreamingActive) {
        // Start streaming
        let rawSrc = document.getElementById('stream-source').value;
        let source = "simulation";
        let query = "technology";

        if (rawSrc === 'simulation') {
            source = "simulation";
        } else if (rawSrc === 'reddit') {
            source = "reddit";
            query = "technology";
        } else if (rawSrc === 'reddit_gadgets') {
            source = "reddit";
            query = "gadgets";
        } else if (rawSrc === 'reddit_apple') {
            source = "reddit";
            query = "apple";
        } else if (rawSrc === 'reddit_custom') {
            source = "reddit";
            query = document.getElementById('stream-query').value.trim() || "technology";
        }

        const modelName = document.getElementById('stream-model').value;
        const interval = parseFloat(document.getElementById('stream-speed').value) || 2.5;

        try {
            const res = await fetch(`${API_BASE}/api/v1/stream/start`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    source: source,
                    query: query,
                    interval: interval,
                    model_name: modelName
                })
            });

            if (!res.ok) throw new Error("Stream start failed");

            isStreamingActive = true;
            btn.className = "btn-stream-stop";
            btn.innerHTML = "<span>⏹ Stop Live Stream</span>";
            indicator.style.display = "inline-flex";

            // Remove empty placeholder
            const hint = document.getElementById('stream-empty-hint');
            if (hint) hint.remove();

            // Start polling loop
            pollStream();
            streamTimer = setInterval(pollStream, 1500);

        } catch (err) {
            console.error(err);
            alert("Could not connect to FastAPI streaming engine. Ensure server is running!");
        }
    } else {
        // Stop streaming
        try {
            await fetch(`${API_BASE}/api/v1/stream/stop`, { method: 'POST' });
        } catch (e) {}

        isStreamingActive = false;
        clearInterval(streamTimer);
        streamTimer = null;

        btn.className = "btn-stream-start";
        btn.innerHTML = "<span>▶ Start Live Stream</span>";
        indicator.style.display = "none";
    }
}

async function pollStream() {
    try {
        const [statusRes, recentRes] = await Promise.all([
            fetch(`${API_BASE}/api/v1/stream/status`),
            fetch(`${API_BASE}/api/v1/stream/recent?limit=15`)
        ]);

        if (statusRes.ok) {
            const status = await statusRes.json();
            document.getElementById('stream-kpi-total').textContent = status.total_streamed;
            document.getElementById('stream-kpi-amb').textContent = status.ambivalent_count;
            document.getElementById('stream-kpi-pos').textContent = status.positive_count;
            document.getElementById('stream-kpi-neg').textContent = status.negative_count;
        }

        if (recentRes.ok) {
            const recent = await recentRes.json();
            const feed = document.getElementById('stream-feed-container');

            // Iterate newly arrived posts (in reverse chronological order)
            recent.posts.slice().reverse().forEach(post => {
                const uniqueKey = post.post_id || `${post.log_id}_${post.timestamp}`;
                if (!seenStreamIds.has(uniqueKey)) {
                    seenStreamIds.add(uniqueKey);

                    const card = document.createElement('div');
                    const labelLower = post.predicted_label.toLowerCase();
                    card.className = `stream-card stream-card-${labelLower}`;

                    // Opposing clauses HTML
                    let clausesHtml = '';
                    if (post.disentangled_clauses && post.disentangled_clauses.length > 0) {
                        clausesHtml = `<div class="inline-clauses">` +
                            post.disentangled_clauses.map(c =>
                                `<span class="clause-pill" style="border-left: 2px solid ${c.detected_polarity === 'Positive' ? '#10b981' : (c.detected_polarity === 'Negative' ? '#ef4444' : '#94a3b8')}">${c.detected_polarity}: "${c.text.substring(0, 32)}..."</span>`
                            ).join('') +
                            `</div>`;
                    }

                    card.innerHTML = `
                        <div class="stream-card-top">
                            <div class="stream-meta">
                                <span class="platform-tag">${post.platform || 'Social'}</span>
                                <span class="author-tag">${post.author || '@user'}</span>
                                <span class="time-tag">⏱️ ${post.timestamp}</span>
                            </div>
                            <span class="sentiment-badge badge-${labelLower}" style="font-size: 0.78rem; padding: 3px 10px;">
                                ${post.predicted_label} (${(post.confidence * 100).toFixed(1)}%)
                            </span>
                        </div>
                        <div class="stream-card-body">${escapeHtml(post.text)}</div>
                        <div class="stream-card-footer">
                            ${clausesHtml}
                            <div style="font-size: 0.78rem; color: #64748b;">Latency: ${post.latency_ms} ms</div>
                        </div>
                    `;

                    // Prepend card to top of feed
                    feed.insertBefore(card, feed.firstChild);

                    // Limit rendered feed items in DOM to 40 for max performance
                    if (feed.children.length > 40) {
                        feed.removeChild(feed.lastChild);
                    }
                }
            });
        }
    } catch (err) {
        console.log("Stream polling paused");
    }
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// Run initial check and analysis on load
window.addEventListener('DOMContentLoaded', () => {
    checkHealth();
    analyzeSentiment();
});

