import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches

os.makedirs("hybridsense_paper/figures", exist_ok=True)

# Set global matplotlib font settings to match IEEE papers (serif / Times-like)
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif', 'Georgia'],
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.titlesize': 12,
    'mathtext.fontset': 'stix'
})

# ==============================================================================
# FIGURE 1: Proposed System Methodology Flowchart
# ==============================================================================
def generate_fig1():
    fig, ax = plt.subplots(figsize=(8.0, 5.0), dpi=300)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 70)
    ax.axis('off')

    # Color palette
    box_blue = '#EBF3FA'
    border_blue = '#245C8D'
    box_teal = '#E6F6F4'
    border_teal = '#1A7A71'
    box_gold = '#FFF8E6'
    border_gold = '#A67300'
    box_purple = '#F4EEFB'
    border_purple = '#643799'

    def draw_box(x, y, w, h, title, subtitle, bg_color, border_color):
        rect = patches.FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=1.2,rounding_size=2.0",
            facecolor=bg_color,
            edgecolor=border_color,
            linewidth=1.4
        )
        ax.add_patch(rect)
        ax.text(x + w/2, y + h*0.64, title, ha='center', va='center',
                fontsize=9.2, fontweight='bold', color='#111111')
        ax.text(x + w/2, y + h*0.28, subtitle, ha='center', va='center',
                fontsize=7.6, color='#444444', style='italic')

    def draw_arrow(x1, y1, x2, y2, label=""):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", color='#2A2A2A', lw=1.3, shrinkA=3, shrinkB=3))
        if label:
            ax.text((x1+x2)/2, (y1+y2)/2 + 1.6, label, ha='center', va='bottom',
                    fontsize=7.5, color='#333333', fontweight='semibold')

    # Top Row: Ingestion -> Preprocessing -> Contrastive Mining
    draw_box(4, 52, 26, 12, "Social Stream Ingestion", "TweetEval (45,586 tweets) + Live Stream", box_blue, border_blue)
    draw_box(37, 52, 26, 12, "Negation-Preserving Preprocess", "Handle/URL Stripping, Lexical Normalization", box_blue, border_blue)
    draw_box(70, 52, 26, 12, "Clause Contrastive Mining", "Discourse Connectives & Dual Valence", box_gold, border_gold)

    draw_arrow(30, 58, 37, 58)
    draw_arrow(63, 58, 70, 58)

    # Middle Row: Embedding -> Conv1D -> BiLSTM -> Attention
    draw_arrow(83, 52, 83, 44)
    draw_box(68, 30, 28, 14, "Token Embedding & Conv1D", "Vocab: 20k, Dim: 128 | 128 Filters, k=3", box_teal, border_teal)
    draw_arrow(68, 37, 57, 37)
    draw_box(31, 30, 26, 14, "Bidirectional LSTM Layer", "Forward & Backward Context (128 Dim)", box_teal, border_teal)
    draw_arrow(31, 37, 23, 37)
    draw_box(4, 30, 19, 14, "Self-Attention", "Bahdanau Dynamic Pooling", box_teal, border_teal)

    # Bottom Row: Classification Head & Saliency / Telemetry
    draw_arrow(13.5, 30, 13.5, 20)
    draw_box(2, 6, 25, 14, "4-Class Classification", "Dense(64) + Softmax (Neg, Neu, Pos, Amb)", box_purple, border_purple)

    draw_arrow(13.5, 13, 34, 13)
    draw_box(34, 6, 28, 14, "XAI Polarity Attribution", "Token Saliency & Clause Disentangler", box_gold, border_gold)

    draw_arrow(62, 13, 70, 13)
    draw_box(70, 6, 26, 14, "Deployment & Telemetry", "FastAPI REST Server, SQLite & Streamlit", box_blue, border_blue)

    plt.tight_layout()
    fig.savefig("hybridsense_paper/figures/fig1_methodology.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("Generated Fig 1: Methodology Flowchart")

# ==============================================================================
# FIGURE 2: Text Preprocessing & Clause Disentanglement
# ==============================================================================
def generate_fig2():
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.2, 3.6), dpi=300)
    fig.patch.set_facecolor('#FFFFFF')

    # Subfigure (a): Raw Tweet Input
    ax1.set_xlim(0, 100)
    ax1.set_ylim(0, 40)
    ax1.axis('off')
    
    rect1 = patches.FancyBboxPatch((2, 4), 96, 32, boxstyle="round,pad=1.0,rounding_size=2.0",
                                   facecolor='#FFF6F6', edgecolor='#C0392B', linewidth=1.2)
    ax1.add_patch(rect1)
    ax1.text(5, 30, "(a) Raw Social Stream Tweet (Noisy & Multi-Aspect Polarity)", fontsize=9.2, fontweight='bold', color='#922B21')
    raw_tweet = '"@TechGuru The battery life on this new device is phenomenal!  http://t.co/xyz89\n  HOWEVER... the build quality is shockingly fragile and terrible! #disappointed"'
    ax1.text(5, 14, raw_tweet, fontsize=8.2, color='#222222', family='monospace')

    # Subfigure (b): Preprocessed & Disentangled
    ax2.set_xlim(0, 100)
    ax2.set_ylim(0, 48)
    ax2.axis('off')

    rect2 = patches.FancyBboxPatch((2, 2), 96, 44, boxstyle="round,pad=1.0,rounding_size=2.0",
                                   facecolor='#F4FAF8', edgecolor='#16A085', linewidth=1.2)
    ax2.add_patch(rect2)
    ax2.text(5, 40, "(b) Preprocessed & Clause-Disentangled Dual Representation", fontsize=9.2, fontweight='bold', color='#0E6655')
    
    c1 = "Clause 1 [Positive]:  \"the battery life on this new device is phenomenal\"  --> Valence: +0.72"
    cm = "Discourse Connective:   \"however\"  --> Contrastive Polarity Shift Marker"
    c2 = "Clause 2 [Negative]:  \"the build quality is shockingly fragile and terrible\" --> Valence: -0.68"
    
    ax2.text(5, 27, c1, fontsize=8.0, color='#1E8449', family='monospace', fontweight='semibold')
    ax2.text(5, 17, cm, fontsize=8.0, color='#B7950B', family='monospace', fontweight='semibold')
    ax2.text(5, 7, c2, fontsize=8.0, color='#C0392B', family='monospace', fontweight='semibold')

    plt.tight_layout()
    fig.savefig("hybridsense_paper/figures/fig2_preprocessing.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("Generated Fig 2: Preprocessing and Clause Disentanglement")

# ==============================================================================
# FIGURE 3: Hybrid CNN-BiLSTM-Attention Neural Architecture
# ==============================================================================
def generate_fig3():
    fig, ax = plt.subplots(figsize=(7.5, 4.6), dpi=300)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 80)
    ax.axis('off')

    layers = [
        (4, 68, 92, 8.5, "Input Sequence", "[w1, w2, w3, ..., wT]  (Fixed Length T = 80 tokens)", '#F2F4F4', '#566573'),
        (4, 56, 92, 8.5, "Embedding Layer", "Vocabulary V = 20,000 | Embedding Dimension d = 128", '#EBF5FB', '#2980B9'),
        (4, 44, 92, 8.5, "1D CNN Layer", "128 Filters, Kernel k = 3, ReLU  [Captures Local N-Gram Phrases]", '#E8F8F5', '#16A085'),
        (4, 32, 92, 8.5, "BiLSTM Layer", "Forward LSTM (64) + Backward LSTM (64) --> H_t in R^128", '#FEF9E7', '#F39C12'),
        (4, 20, 92, 8.5, "Bahdanau Attention", "Dynamic Query u_w --> Weights alpha_t --> Context Vector v in R^128", '#F5EEF8', '#8E44AD'),
        (4, 8, 92, 8.5, "Dense Classifier", "Dense(64, ReLU) + Dropout(0.4) --> Dense(4, Softmax) --> [Neg, Neu, Pos, Amb]", '#FDEDEC', '#C0392B')
    ]

    for x, y, w, h, title, sub, bg, border in layers:
        rect = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=1.0,rounding_size=1.5",
                                       facecolor=bg, edgecolor=border, linewidth=1.3)
        ax.add_patch(rect)
        ax.text(x + 2.5, y + h*0.5, title + " :", fontsize=8.8, fontweight='bold', color='#111111', va='center')
        ax.text(x + 24, y + h*0.5, sub, fontsize=7.8, color='#2B2B2B', va='center', family='sans-serif')

    # Connecting arrows
    for y_top in [68, 56, 44, 32, 20]:
        ax.annotate('', xy=(50, y_top - 3.5), xytext=(50, y_top),
                    arrowprops=dict(arrowstyle="->", color='#333333', lw=1.4))

    plt.tight_layout()
    fig.savefig("hybridsense_paper/figures/fig3_neural_architecture.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("Generated Fig 3: Neural Architecture Diagram")

# ==============================================================================
# FIGURE 4: Detection & Explainability Output
# ==============================================================================
def generate_fig4():
    fig, ax = plt.subplots(figsize=(7.5, 4.4), dpi=300)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 75)
    ax.axis('off')

    ax.text(50, 71, "Representative Real-Time Outputs of HybridSense System", ha='center', fontsize=10.5, fontweight='bold', color='#1B2631')

    cases = [
        (4, 51, 92, 16, "Case 1: Standard Positive Tweet", 
         "Input: \"The customer service team was genuinely wonderful and resolved my issue fast!\"\n"
         "Prediction: Positive (96.4%) | Top Saliency: wonderful (+0.84), genuinely (+0.61), fast (+0.52)",
         '#EAFAF1', '#27AE60'),
        (4, 33, 92, 16, "Case 2: Standard Negative Tweet", 
         "Input: \"Horrible flight experience, lost baggage and rude ground attendants.\"\n"
         "Prediction: Negative (95.1%) | Top Saliency: Horrible (-0.88), rude (-0.79), lost (-0.45)",
         '#FDEDEC', '#E74C3C'),
        (4, 15, 92, 16, "Case 3: Ambivalent Sentiment with Opposing Aspects (Clause Disentangled)", 
         "Input: \"The camera takes brilliant photos, but the overheating issue is completely unacceptable.\"\n"
         "Prediction: Ambivalent (92.8%) | Clause 1: brilliant (+0.76) | Pivot: but | Clause 2: unacceptable (-0.82)",
         '#FEF9E7', '#D4AC0D'),
    ]

    for x, y, w, h, title, content, bg, border in cases:
        rect = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=1.0,rounding_size=2.0",
                                       facecolor=bg, edgecolor=border, linewidth=1.3)
        ax.add_patch(rect)
        ax.text(x + 2.5, y + h - 4.2, title, fontsize=9.0, fontweight='bold', color='#111111')
        ax.text(x + 2.5, y + 3.8, content, fontsize=7.6, color='#222222', family='monospace')

    plt.tight_layout()
    fig.savefig("hybridsense_paper/figures/fig4_detection_output.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("Generated Fig 4: Qualitative Detection Output")

# ==============================================================================
# FIGURE 5: Training and Testing Accuracy Curve
# ==============================================================================
def generate_fig5():
    with open("results/history_hybrid.json", "r") as f:
        hist = json.load(f)
    
    epochs = np.arange(1, len(hist['accuracy']) + 1)
    train_acc = [acc * 100 for acc in hist['accuracy']]
    val_acc = [acc * 100 for acc in hist['val_accuracy']]

    fig, ax = plt.subplots(figsize=(5.5, 4.0), dpi=300)
    ax.plot(epochs, train_acc, color='#8B1E3F', marker='o', linewidth=1.8, label='Training Accuracy')
    ax.plot(epochs, val_acc, color='#2B4C7E', marker='s', linestyle='--', linewidth=1.8, label='Testing (Validation) Accuracy')
    
    ax.set_xlabel('Epoch', fontweight='bold')
    ax.set_ylabel('Accuracy (%)', fontweight='bold')
    ax.set_xlim(0.8, len(epochs) + 0.2)
    ax.set_ylim(35, 90)
    ax.set_xticks(epochs)
    ax.grid(True, linestyle=':', alpha=0.6, color='#888888')
    ax.legend(loc='lower right', framealpha=0.9, edgecolor='#CCCCCC')
    
    plt.tight_layout()
    fig.savefig("hybridsense_paper/figures/fig5_accuracy_curve.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("Generated Fig 5: Accuracy Curve")

# ==============================================================================
# FIGURE 6: Training and Testing Loss Curve
# ==============================================================================
def generate_fig6():
    with open("results/history_hybrid.json", "r") as f:
        hist = json.load(f)
    
    epochs = np.arange(1, len(hist['loss']) + 1)
    train_loss = hist['loss']
    val_loss = hist['val_loss']

    fig, ax = plt.subplots(figsize=(5.5, 4.0), dpi=300)
    ax.plot(epochs, train_loss, color='#C0392B', marker='o', linewidth=1.8, label='Training Loss')
    ax.plot(epochs, val_loss, color='#2980B9', marker='s', linestyle='--', linewidth=1.8, label='Testing (Validation) Loss')
    
    ax.set_xlabel('Epoch', fontweight='bold')
    ax.set_ylabel('Categorical Cross-Entropy Loss', fontweight='bold')
    ax.set_xlim(0.8, len(epochs) + 0.2)
    ax.set_ylim(0.2, 1.3)
    ax.set_xticks(epochs)
    ax.grid(True, linestyle=':', alpha=0.6, color='#888888')
    ax.legend(loc='upper right', framealpha=0.9, edgecolor='#CCCCCC')
    
    plt.tight_layout()
    fig.savefig("hybridsense_paper/figures/fig6_loss_curve.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("Generated Fig 6: Loss Curve")

# ==============================================================================
# FIGURE 7: Graphic Representation of Performance Comparison
# ==============================================================================
def generate_fig7():
    models = ['TF-IDF+LR', 'CNN', 'BiLSTM', 'DistilBERT', 'Proposed Hybrid']
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    
    scores = {
        'TF-IDF+LR': [62.48, 65.43, 62.48, 63.44],
        'CNN': [60.08, 65.50, 60.08, 61.34],
        'BiLSTM': [58.29, 62.96, 58.29, 59.59],
        'DistilBERT': [64.05, 68.44, 64.05, 65.20],
        'Proposed Hybrid': [59.41, 61.85, 59.41, 59.93]
    }

    x = np.arange(len(metrics))
    width = 0.16
    colors_list = ['#7F8C8D', '#E67E22', '#2980B9', '#8E44AD', '#27AE60']

    fig, ax = plt.subplots(figsize=(6.8, 4.4), dpi=300)
    for i, m_name in enumerate(models):
        offset = (i - 2) * width
        rects = ax.bar(x + offset, scores[m_name], width, label=m_name, color=colors_list[i], edgecolor='#333333', linewidth=0.7)

    ax.set_ylabel('Score (%)', fontweight='bold')
    ax.set_title('Graphic Representation of Performance Analysis Across Evaluated Models', fontweight='bold', fontsize=11, pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontweight='bold')
    ax.set_ylim(45, 75)
    ax.grid(True, axis='y', linestyle=':', alpha=0.6, color='#888888')
    ax.legend(loc='lower right', framealpha=0.9, edgecolor='#CCCCCC', ncol=2, fontsize=8)

    plt.tight_layout()
    fig.savefig("hybridsense_paper/figures/fig7_performance_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("Generated Fig 7: Performance Comparison Bar Chart")

if __name__ == '__main__':
    generate_fig1()
    generate_fig2()
    generate_fig3()
    generate_fig4()
    generate_fig5()
    generate_fig6()
    generate_fig7()
    print("All 7 IEEE figures successfully generated!")
