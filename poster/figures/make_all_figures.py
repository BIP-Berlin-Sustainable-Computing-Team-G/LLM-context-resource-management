#!/usr/bin/env python3
"""
Generate all poster figures for:
Context Windows and KV-Cache Reuse: Energy vs. Quality in Multi-Turn LLM Chat
Sustainable Computing - TU Berlin / Erasmus (Phase 1 Poster)
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.lines import Line2D
import fitz  # PyMuPDF

# Ensure output directory exists
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
POSTER_FIGS_DIR = SCRIPT_DIR
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
PAPERS_DIR = os.path.join(REPO_ROOT, "papers")

print(f"Generating poster figures in: {POSTER_FIGS_DIR}")

# -------------------------------------------------------------
# FIGURE 1: Analytical Token Growth (Quadratic vs. Linear)
# -------------------------------------------------------------
def make_fig1_token_growth():
    print("Generating Fig 1: Analytical Token Growth...")
    turns = np.arange(1, 26)
    S = 500   # System prompt tokens
    u = 100   # User message tokens
    a = 300   # Assistant reply tokens
    m = u + a # 400 tokens per round

    # Per turn prefilled tokens
    no_cache_per_turn = S + (turns - 1) * m + u
    cache_per_turn = np.array([S + u if t == 1 else m for t in turns])

    # Cumulative prefilled tokens
    no_cache_cumulative = np.cumsum(no_cache_per_turn)
    cache_cumulative = np.cumsum(cache_per_turn)

    # Styling setup
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.size': 13,
        'axes.labelsize': 14,
        'axes.titlesize': 16,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12,
        'figure.titlesize': 18
    })

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6.2), dpi=300)
    fig.patch.set_facecolor('#ffffff')

    # Palette
    c_nocache = '#c0392b'  # Dark Crimson
    c_cache = '#27ae60'    # Emerald Green
    c_fill = '#e8f8f5'     # Soft teal/green fill

    # --- SUBPLOT 1: Cumulative Tokens ---
    ax1.plot(turns, no_cache_cumulative / 1000, color=c_nocache, linewidth=3.2, 
             marker='o', markersize=6, label=r'Without Cache: $O(T^2)$ quadratic')
    ax1.plot(turns, cache_cumulative / 1000, color=c_cache, linewidth=3.2, 
             marker='s', markersize=6, label=r'With Prefix Cache: $O(T)$ linear')

    ax1.fill_between(turns, cache_cumulative / 1000, no_cache_cumulative / 1000, 
                     color=c_nocache, alpha=0.10)

    # Annotate Turn 20 ratio
    t20_idx = 19  # turn 20
    t20_val_nc = no_cache_cumulative[t20_idx] / 1000
    t20_val_c = cache_cumulative[t20_idx] / 1000
    ratio_20 = no_cache_cumulative[t20_idx] / cache_cumulative[t20_idx]

    ax1.annotate(f'Turn 20:\n88.0k vs 8.2k tokens\n({ratio_20:.1f}× fewer tokens)',
                 xy=(20, t20_val_c), xytext=(12, 62),
                 arrowprops=dict(arrowstyle="->", color='#2c3e50', lw=2, shrinkB=6),
                 bbox=dict(boxstyle="round,pad=0.5", facecolor='#fcf3cf', edgecolor='#f39c12', lw=1.5),
                 fontsize=12, fontweight='bold', color='#2c3e50')

    ax1.set_title("Cumulative Prefilled Tokens", pad=12, fontweight='bold', color='#2c3e50')
    ax1.set_xlabel("Conversation Turn (T)")
    ax1.set_ylabel("Cumulative Tokens Prefilled (× 1,000)")
    ax1.set_xlim(1, 25)
    ax1.set_ylim(0, 140)
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.legend(loc='upper left', framealpha=0.95)

    # --- SUBPLOT 2: Per-Turn Tokens ---
    ax2.plot(turns, no_cache_per_turn, color=c_nocache, linewidth=3.2, 
             marker='o', markersize=6, label='Without Cache: grows +400 tok/turn')
    ax2.plot(turns, cache_per_turn, color=c_cache, linewidth=3.2, 
             marker='s', markersize=6, label='With Prefix Cache: constant 400 tok/turn')

    ax2.set_title("Per-Turn Prefill Workload", pad=12, fontweight='bold', color='#2c3e50')
    ax2.set_xlabel("Conversation Turn (T)")
    ax2.set_ylabel("Tokens Prefilled in Current Turn")
    ax2.set_xlim(1, 25)
    ax2.set_ylim(0, 11000)
    ax2.grid(True, linestyle='--', alpha=0.5)
    ax2.legend(loc='upper left', framealpha=0.95)

    # Super title & footnote
    fig.suptitle("Theoretical Impact of KV-Cache Reuse on Multi-Turn Prefill", 
                 fontsize=19, fontweight='heavy', color='#1a252f', y=0.98)
    
    plt.figtext(0.5, 0.01, 
                "Analytical derivation: S = 500 (system), u = 100 (user), a = 300 (reply) tokens. "
                "Calculated values (Phase 1 hypothesis), not empirical measurements.", 
                ha="center", fontsize=11, fontstyle='italic', color='#566573')

    plt.tight_layout(rect=[0, 0.04, 1, 0.94])
    
    png_path = os.path.join(POSTER_FIGS_DIR, "fig1_token_growth.png")
    svg_path = os.path.join(POSTER_FIGS_DIR, "fig1_token_growth.svg")
    plt.savefig(png_path, dpi=300, facecolor=fig.get_facecolor(), edgecolor='none')
    plt.savefig(svg_path, format='svg', facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close()
    print(f" Saved: {png_path} and {svg_path}")


# -------------------------------------------------------------
# FIGURE 2: Lost in the Middle (Crop & Vector Reconstruction)
# -------------------------------------------------------------
def make_fig2_lost_in_the_middle():
    print("Generating Fig 2: Lost in the Middle...")
    pdf_path = os.path.join(PAPERS_DIR, "2024.tacl-1.9.pdf")
    
    # 1. Crop from paper PDF
    if os.path.exists(pdf_path):
        doc = fitz.open(pdf_path)
        page = doc[1] # Page 2 (0-indexed = 1)
        # Figure 1 bounding box on page 2 (in 72-dpi points)
        # Bounding box roughly: x0=50, y0=60, x1=270, y1=240
        rect = fitz.Rect(54, 70, 275, 230)
        pix = page.get_pixmap(clip=rect, dpi=300)
        paper_crop_path = os.path.join(POSTER_FIGS_DIR, "fig2_lost_in_the_middle_paper.png")
        pix.save(paper_crop_path)
        print(f" Extracted paper Figure 1: {paper_crop_path}")
        doc.close()
    else:
        print(f" Warning: {pdf_path} not found!")

    # 2. Modern Vector Plot of Lost in the Middle (Liu et al., Fig 1 / Fig 5)
    fig, ax = plt.subplots(figsize=(8, 6.2), dpi=300)
    fig.patch.set_facecolor('#ffffff')

    positions = np.array([1, 2, 3, 5, 8, 10, 12, 15, 18, 20])
    # Approximate GPT-3.5 20-doc accuracies from Fig 1 / Fig 5
    accuracies = np.array([73.8, 70.2, 65.4, 60.5, 57.1, 55.6, 56.2, 57.0, 60.1, 63.8])
    closed_book = 56.1

    # Regions
    ax.axvspan(1, 3.5, color='#d4efdf', alpha=0.45, label='Primacy Bias (Start: >70%)')
    ax.axvspan(7.5, 15.5, color='#fadbd8', alpha=0.55, label='Lost in the Middle (< Closed-book)')
    ax.axvspan(17.5, 20, color='#d4efdf', alpha=0.45, label='Recency Bias (End: ~64%)')

    # Curve
    ax.plot(positions, accuracies, color='#8e44ad', marker='o', markersize=8, 
            linewidth=3.2, label='GPT-3.5-Turbo (20 Docs, ~4k tokens)')

    # Closed-book baseline
    ax.axhline(closed_book, color='#c0392b', linestyle='--', linewidth=2.2, 
               label=f'Closed-Book Baseline ({closed_book}%, no context)')

    # Callout in the middle
    ax.annotate("Performance drops BELOW\nclosed-book baseline!\nContext distracts LLM reasoning", 
                xy=(10, 55.6), xytext=(6, 46.5),
                arrowprops=dict(arrowstyle="->", color='#c0392b', lw=2, shrinkB=6),
                bbox=dict(boxstyle="round,pad=0.45", facecolor='#fdedec', edgecolor='#e74c3c', lw=1.5),
                fontsize=11.5, fontweight='bold', color='#922b21')

    ax.set_title("Lost in the Middle: The U-Shaped Accuracy Curve", 
                 fontsize=16, fontweight='bold', pad=12, color='#1a252f')
    ax.set_xlabel("Position of Relevant Document in Context (1st = Start, 20th = End)", fontsize=13)
    ax.set_ylabel("Retrieval / QA Accuracy (%)", fontsize=13)
    ax.set_xlim(0.5, 20.5)
    ax.set_ylim(44, 78)
    ax.set_xticks([1, 5, 10, 15, 20])
    ax.set_xticklabels(['1st (Start)', '5th', '10th (Middle)', '15th', '20th (End)'])
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(loc='upper right', framealpha=0.95, fontsize=10.5)

    plt.figtext(0.5, 0.01, 
                "Source: Liu et al., TACL 2024 (doi:10.1162/tacl_a_00638). "
                "Demonstrates that retaining 100% full history does not guarantee fact recall.",
                ha="center", fontsize=10, fontstyle='italic', color='#566573')

    plt.tight_layout(rect=[0, 0.04, 1, 0.96])
    png_path = os.path.join(POSTER_FIGS_DIR, "fig2_lost_in_the_middle_vector.png")
    svg_path = os.path.join(POSTER_FIGS_DIR, "fig2_lost_in_the_middle_vector.svg")
    plt.savefig(png_path, dpi=300, facecolor=fig.get_facecolor(), edgecolor='none')
    plt.savefig(svg_path, format='svg', facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close()
    print(f" Saved: {png_path} and {svg_path}")


# -------------------------------------------------------------
# FIGURE 3: Context Management Trade-off & Cache Interaction
# -------------------------------------------------------------
def make_fig3_context_tradeoff():
    print("Generating Fig 3: Context Strategies Trade-off Diagram...")
    fig, ax = plt.subplots(figsize=(14, 8.2), dpi=300)
    fig.patch.set_facecolor('#ffffff')
    ax.set_xlim(0, 140)
    ax.set_ylim(0, 100)
    ax.axis('off')

    # Header
    ax.text(70, 95, "Context Strategies vs. KV-Cache Reuse Trade-Off", 
            ha='center', va='center', fontsize=18, fontweight='heavy', color='#1a252f')

    # Helper to draw strategy cards
    strategies = [
        {
            "y": 72,
            "title": "1. Full History (Naive Baseline)",
            "badge": "CACHE FRIENDLY (Stable Prefix)",
            "badge_col": "#27ae60",
            "bg": "#f4fcf6",
            "border": "#2ecc71",
            "blocks": [("H1", "#2ecc71"), ("H2", "#2ecc71"), ("H3", "#2ecc71"), ("H4", "#2ecc71"), ("New", "#3498db")],
            "pros": "• Prefix matches past turns perfectly → High cache hit rate\n• Zero summarisation compute overhead",
            "cons": "• VRAM footprint scales O(T) → early cache eviction\n• Severe 'Lost in the Middle' accuracy drops (Liu et al.)"
        },
        {
            "y": 48,
            "title": "2. Sliding Window (Last k Messages)",
            "badge": "CACHE HOSTILE (Prefix Shifts)",
            "badge_col": "#c0392b",
            "bg": "#fdf4f4",
            "border": "#e74c3c",
            "blocks": [("Dropped", "#bdc3c7"), ("H2", "#e74c3c"), ("H3", "#e74c3c"), ("H4", "#e74c3c"), ("New", "#3498db")],
            "pros": "• Strictly bounded VRAM & token budget\n• No 'Lost in the Middle' for recent turns",
            "cons": "• 100% CACHE MISS: Start token shifts every turn!\n• Total amnesia for facts mentioned before window"
        },
        {
            "y": 24,
            "title": "3. Context Summarisation (One-Time / Rolling)",
            "badge": "ENERGY OVERHEAD (Decode Tax)",
            "badge_col": "#d35400",
            "bg": "#fef9f3",
            "border": "#f39c12",
            "blocks": [("Summary", "#e67e22"), ("Recent", "#e74c3c"), ("New", "#3498db")],
            "pros": "• Compressed representation bounds sequence length\n• Keeps essential high-level facts in context",
            "cons": "• Rewriting summary destroys KV cache prefix\n• High decode energy tax on each summarisation call"
        }
    ]

    for s in strategies:
        y = s["y"]
        # Outer Card
        rect = patches.FancyBboxPatch((5, y - 9), 130, 18, boxstyle="round,pad=0.8",
                                      facecolor=s["bg"], edgecolor=s["border"], linewidth=2.0)
        ax.add_patch(rect)

        # Title & Badge
        ax.text(8, y + 6.5, s["title"], fontsize=13.5, fontweight='bold', color='#2c3e50', va='center')
        ax.text(85, y + 6.5, s["badge"], fontsize=10.5, fontweight='bold', color=s["badge_col"],
                bbox=dict(boxstyle="round,pad=0.3", facecolor='#ffffff', edgecolor=s["badge_col"], lw=1.2),
                va='center')

        # Visual Tokens / KV Blocks
        x_block = 8
        for b_text, b_color in s["blocks"]:
            b_patch = patches.Rectangle((x_block, y - 1.5), 11, 5.5, facecolor=b_color, 
                                        edgecolor='#2c3e50', linewidth=1.0)
            ax.add_patch(b_patch)
            ax.text(x_block + 5.5, y + 1.25, b_text, ha='center', va='center', 
                    color='#ffffff' if b_color != '#bdc3c7' else '#2c3e50', fontsize=9.5, fontweight='bold')
            x_block += 12.5

        # Pros and Cons
        ax.text(68, y - 0.5, s["pros"], fontsize=10.5, color='#1e8449', va='center')
        ax.text(102, y - 0.5, s["cons"], fontsize=10.5, color='#922b21', va='center')

    # Bottom Callout: Break-Even Question (RQ2 / O2)
    bottom_rect = patches.FancyBboxPatch((5, 2), 130, 9, boxstyle="round,pad=0.8",
                                         facecolor='#eaf2f8', edgecolor='#2980b9', linewidth=2.0)
    ax.add_patch(bottom_rect)
    ax.text(70, 6.5, 
            "Core Research Question (RQ2 / O2): What is the net energy overhead of summarization,\n"
            "and at what conversation length does it reach a break-even point compared to KV-cache reuse?",
            ha='center', va='center', fontsize=12, fontweight='bold', color='#1b4f72')

    png_path = os.path.join(POSTER_FIGS_DIR, "fig3_context_tradeoff.png")
    svg_path = os.path.join(POSTER_FIGS_DIR, "fig3_context_tradeoff.svg")
    plt.savefig(png_path, dpi=300, facecolor=fig.get_facecolor(), edgecolor='none')
    plt.savefig(svg_path, format='svg', facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close()
    print(f" Saved: {png_path} and {svg_path}")


# -------------------------------------------------------------
# FIGURE 4: System Architecture & Container Ablation Matrix
# -------------------------------------------------------------
def make_fig4_system_architecture():
    print("Generating Fig 4: Container Architecture Diagram...")
    fig, ax = plt.subplots(figsize=(15, 7.8), dpi=300)
    fig.patch.set_facecolor('#ffffff')
    ax.set_xlim(0, 150)
    ax.set_ylim(0, 100)
    ax.axis('off')

    # Header
    ax.text(75, 95, "Phase 2 Experiment Architecture: Test Harness & Container Ablation", 
            ha='center', va='center', fontsize=18, fontweight='heavy', color='#1a252f')

    # Column 1: Test Harness & Workload
    c1 = patches.FancyBboxPatch((5, 12), 35, 74, boxstyle="round,pad=0.8",
                                facecolor='#f8f9fa', edgecolor='#7f8c8d', linewidth=2.0)
    ax.add_patch(c1)
    ax.text(22.5, 82, "1. Conversation\nHarness", ha='center', va='center', 
            fontsize=13.5, fontweight='bold', color='#2c3e50')
    
    harness_items = [
        "Scripted Dialogues (T=1..40)",
        "Workload Task Mix:\n • Text Generation (Decode)\n • Code Writing\n • Tokenization / Math",
        "Quality Evaluation:\n • Fact recall planted at:\n   - Early (turns 1-3)\n   - Middle (turns 10-15)\n   - Recent (last 2-3)",
        "Turn-by-turn replayer\nvia OpenAI-compatible API"
    ]
    y_pos = 68
    for item in harness_items:
        ax.text(8, y_pos, item, fontsize=10.2, color='#34495e', va='top')
        y_pos -= 15

    # Arrows between C1 and C2
    ax.annotate("", xy=(46, 50), xytext=(40, 50),
                arrowprops=dict(arrowstyle="->", color='#2980b9', lw=3.0))

    # Column 2: Isolated Docker Containers (Ablation Matrix)
    c2 = patches.FancyBboxPatch((47, 12), 56, 74, boxstyle="round,pad=0.8",
                                facecolor='#eef7fb', edgecolor='#2980b9', linewidth=2.0)
    ax.add_patch(c2)
    ax.text(75, 82, "2. vLLM Docker Containers\n(Ablation Suite)", ha='center', va='center', 
            fontsize=13.5, fontweight='bold', color='#1b4f72')

    containers = [
        ("Container 0 (Baseline):", "vLLM, Prefix Caching OFF (Full Recompute)", "#95a5a6"),
        ("Container A (Caching Only):", "vLLM, Prefix Caching ON (--enable-prefix-caching)", "#27ae60"),
        ("Container B (Sliding Window):", "Fixed Window (last 5 msgs), Caching OFF", "#e67e22"),
        ("Container C (Summarisation):", "Rolling / Batch Summary Context Manager", "#d35400"),
        ("Container D (Mega-Optimized):", "Prefix Caching ON + Cache-Aware Window", "#8e44ad")
    ]
    y_c = 68
    for name, desc, col in containers:
        box = patches.FancyBboxPatch((50, y_c - 4), 50, 8.5, boxstyle="round,pad=0.4",
                                     facecolor='#ffffff', edgecolor=col, linewidth=1.6)
        ax.add_patch(box)
        ax.text(52, y_c + 2.2, name, fontsize=10.5, fontweight='bold', color=col)
        ax.text(52, y_c - 1.2, desc, fontsize=9.2, color='#2c3e50')
        y_c -= 11.5

    # Arrows between C2 and C3
    ax.annotate("", xy=(109, 50), xytext=(103, 50),
                arrowprops=dict(arrowstyle="->", color='#2980b9', lw=3.0))

    # Column 3: Hardware & Measurement
    c3 = patches.FancyBboxPatch((110, 12), 35, 74, boxstyle="round,pad=0.8",
                                facecolor='#f4fbf7', edgecolor='#27ae60', linewidth=2.0)
    ax.add_patch(c3)
    ax.text(127.5, 82, "3. Hardware &\nEnergy Monitoring", ha='center', va='center', 
            fontsize=13.5, fontweight='bold', color='#145a32')

    hw_items = [
        "GPU Hardware Server:\n • 2 × NVIDIA RTX 4080\n   (16 GB VRAM each)\n • Single GPU per run\n • BF16 Small LM (3-4B)",
        "Power & Energy Metrics:\n • CodeCarbon & NVML\n • Per-turn Energy (Joules)\n • Prefill vs Decode split\n • Static idle baseline subtracted",
        "Performance Latency:\n • TTFT & Inter-Token (ITL)\n • Cache hit/query counters",
        "Composite Metric (O4):\n • Energy per correct answer"
    ]
    y_hw = 68
    for item in hw_items:
        ax.text(113, y_hw, item, fontsize=10.2, color='#1e8449', va='top')
        y_hw -= 15.5

    # Footer note
    plt.figtext(0.5, 0.02, 
                "Phase 2 implementation testbed. Isolates the energy savings and accuracy costs "
                "of individual context optimizations against unoptimized and combined baselines.",
                ha="center", fontsize=10.5, fontstyle='italic', color='#566573')

    png_path = os.path.join(POSTER_FIGS_DIR, "fig4_container_architecture.png")
    svg_path = os.path.join(POSTER_FIGS_DIR, "fig4_container_architecture.svg")
    plt.savefig(png_path, dpi=300, facecolor=fig.get_facecolor(), edgecolor='none')
    plt.savefig(svg_path, format='svg', facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close()
    print(f" Saved: {png_path} and {svg_path}")


if __name__ == "__main__":
    make_fig1_token_growth()
    make_fig2_lost_in_the_middle()
    make_fig3_context_tradeoff()
    make_fig4_system_architecture()
    print("\nAll poster figures successfully generated in poster/figures/!")
