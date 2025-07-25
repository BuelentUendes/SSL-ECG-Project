import pandas as pd
import numpy as np
import json, ast
import matplotlib.pyplot as plt
from pathlib import Path
import seaborn as sns

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Color scheme for different models
COLORS = {
    # Supervised models
    'CNN': '#2E86C1',
    'TCN': '#E74C3C',
    'Transformer': '#6A5ACD',
    
    # SSL models
    'TS2Vec': '#2E86C1',
    'SoftTS2Vec': '#E74C3C',
    'TSTCC': '#6A5ACD',
    'SoftTSTCC': '#2ECC71',
    'SimCLR': '#FFA500'
}

# Marker styles for different models
MARKERS = {
    # Supervised models
    'CNN': 'o',
    'TCN': 's',
    'Transformer': '^',
    
    # SSL models
    'TS2Vec': 'o',
    'SoftTS2Vec': 's',
    'TSTCC': '^',
    'SoftTSTCC': 'D',
    'SimCLR': '*'
}

# Load significance maps
SIG_DIR = Path("../results/significance")
def load_sig(name: str):
    raw = json.load(open(SIG_DIR / f"{name}.json"))
    out = {}
    for frac_str, comp in raw.items():
        frac = float(frac_str)
        cleaned = {}
        for pair, val in comp.items():
            m1_full, m2_full = pair.split("|")
            # drop any classifier suffix
            m1 = m1_full.split("_")[0]
            m2 = m2_full.split("_")[0]
            cleaned[(m1, m2)] = val
        out[frac] = cleaned
    return out

SIGNIFICANCE = load_sig("supervised")    
SSL_LINEAR_SIGNIFICANCE = load_sig("ssl_linear")
SSL_MLP_SIGNIFICANCE  = load_sig("ssl_mlp")

# Load data
df = pd.read_csv(Path("../results/confidence_intervals/supervised_ci.csv"))
df["label_fraction"] = df["label_fraction"].astype(float)
fractions = sorted(df["label_fraction"].unique())
models = sorted(df["model"].unique())
n_models = len(models)
bar_width = 0.2
x = np.arange(len(fractions))

# ----------------------------------------------
# Plot for Supervised Models
# ----------------------------------------------

# Plot
fig, ax = plt.subplots(figsize=(12, 5), dpi=300)

# Plot each model
for i, model in enumerate(models):
    sub = df[df["model"] == model].sort_values("label_fraction")
    means = sub["f1_mean"].values
    yerr_lower = means - sub["f1_ci_lower"].values
    yerr_upper = sub["f1_ci_upper"].values - means

    positions = x + i * bar_width
    ax.errorbar(
        positions, means, yerr=[yerr_lower, yerr_upper],
        fmt=MARKERS.get(model, 'o'),
        capsize=5,
        label=model,
        color=COLORS.get(model, f'C{i}'),
        linestyle='None',
        markersize=8,
        markerfacecolor='white',
        markeredgewidth=2,
        capthick=2,
        elinewidth=2
    )

# Add significance asterisks
for idx, frac in enumerate(fractions):
    comparisons = SIGNIFICANCE.get(frac, {})
    base_x = x[idx]
    height = max(df[df["label_fraction"] == frac]["f1_ci_upper"]) + 2

    for j, (m1, m2) in enumerate(comparisons):
        if comparisons[(m1, m2)]:
            i1 = models.index(m1)
            i2 = models.index(m2)
            x1 = base_x + i1 * bar_width
            x2 = base_x + i2 * bar_width
            y = height + j * 3
            ax.plot([x1, x2], [y, y], color='black', linewidth=1)
            ax.text((x1 + x2)/2, y + 0.01, "*", ha='center', va='bottom', fontsize=14)

# Styling
ax.set_title("Supervised Models", fontsize=14, fontweight='bold', pad=20)
ax.set_xlabel("Proportion of Labeled Training Data", fontsize=12, fontweight='bold')
ax.set_ylabel("MF1 Score", fontsize=12, fontweight='bold')
ax.set_xticks(x + bar_width * (n_models - 1) / 2)
ax.set_xticklabels([f'{int(f*100)}%' for f in fractions], fontsize=12)
ax.tick_params(axis='both', which='major', labelsize=12) 
ax.grid(True, linestyle='--', alpha=0.3)
ax.set_facecolor('#FAFAFA')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_linewidth(0.5)
ax.spines['bottom'].set_linewidth(0.5)
ax.legend(frameon=True, fancybox=True, shadow=True, fontsize=12, loc='lower right')

plt.tight_layout()
plt.savefig("../results/error_bars_supervised.pdf", dpi=300)
plt.savefig("../results/error_bars_supervised.png", dpi=300)
plt.show()

# ----------------------------------------------
# Individual plot for SSL Linear Models
# ----------------------------------------------
# Create figure for SSL Linear
fig_linear, ax_linear = plt.subplots(figsize=(12, 6), dpi=300)

# Load and process Linear data
df_ssl_linear = pd.read_csv(Path("../results/confidence_intervals/ssl_linear_ci.csv"))
df_ssl_linear["label_fraction"] = df_ssl_linear["label_fraction"].astype(float)
fractions_ssl_linear = sorted(df_ssl_linear["label_fraction"].unique())

# Filter for SSL models in desired order
ssl_models = ['SimCLR', 'TS2Vec', 'SoftTS2Vec', 'TSTCC', 'SoftTSTCC']
df_ssl_linear = df_ssl_linear[df_ssl_linear["model"].isin(ssl_models)]
df_ssl_linear['model'] = pd.Categorical(df_ssl_linear['model'], categories=ssl_models, ordered=True)
df_ssl_linear = df_ssl_linear.sort_values('model')

n_models = len(ssl_models)
x_ssl = np.arange(len(fractions_ssl_linear)) * 1.5

# Plot Linear
for i, model in enumerate(ssl_models):
    sub = df_ssl_linear[df_ssl_linear["model"] == model].sort_values("label_fraction")
    means = sub["f1_mean"].values
    yerr_lower = means - sub["f1_ci_lower"].values
    yerr_upper = sub["f1_ci_upper"].values - means

    positions = x_ssl + i * bar_width
    ax_linear.errorbar(
        positions, means, yerr=[yerr_lower, yerr_upper],
        fmt=MARKERS.get(model, 'o'),
        capsize=5,
        label=model,
        color=COLORS.get(model, f'C{i}'),
        linestyle='None',
        markersize=8,
        markerfacecolor='white',
        markeredgewidth=2,
        capthick=2,
        elinewidth=2
    )

# Add significance asterisks for Linear plot
for idx, frac in enumerate(fractions_ssl_linear):
    comparisons = SSL_LINEAR_SIGNIFICANCE.get(frac, {})
    base_x = x_ssl[idx]
    relevant_data = df_ssl_linear[df_ssl_linear["label_fraction"] == frac]
    if not relevant_data.empty:
        height = max(relevant_data["f1_ci_upper"]) + 2
    else:
        height = ax_linear.get_ylim()[1] * 0.1

    for j, (m1, m2) in enumerate(comparisons):
        if comparisons[(m1, m2)]:
            i1 = ssl_models.index(m1)
            i2 = ssl_models.index(m2)
            x1 = base_x + i1 * bar_width
            x2 = base_x + i2 * bar_width
            y = height + j * 1.5
            ax_linear.plot([x1, x2], [y, y], color='black', linewidth=1)
            ax_linear.text((x1 + x2)/2, y + 0.01, "*", ha='center', va='bottom', fontsize=14)

# Style Linear plot
ax_linear.set_title("SSL Models with Linear Classifier", fontsize=14, fontweight='bold', pad=20)
ax_linear.set_xlabel("Proportion of Labeled Training Data", fontsize=12, fontweight='bold')
ax_linear.set_ylabel("MF1 Score", fontsize=12, fontweight='bold')
ax_linear.set_xticks(x_ssl + bar_width * (n_models - 1) / 2)
ax_linear.set_xticklabels([f'{int(f*100)}%' for f in fractions_ssl_linear], fontsize=12)
ax_linear.tick_params(axis='both', which='major', labelsize=12)
ax_linear.grid(True, linestyle='--', alpha=0.3)
ax_linear.set_facecolor('#FAFAFA')
ax_linear.spines['top'].set_visible(False)
ax_linear.spines['right'].set_visible(False)
ax_linear.spines['left'].set_linewidth(0.5)
ax_linear.spines['bottom'].set_linewidth(0.5)
ax_linear.legend(frameon=True, fancybox=True, shadow=True, fontsize=12, loc='lower right')

plt.tight_layout()
plt.savefig("../results/error_bars_ssl_linear.pdf", dpi=300, bbox_inches='tight')
plt.savefig("../results/error_bars_ssl_linear.png", dpi=300, bbox_inches='tight')
plt.show()


# ----------------------------------------------
# Individual plot for SSL MLP Models
# ----------------------------------------------
# Create figure for SSL MLP
fig_mlp, ax_mlp = plt.subplots(figsize=(12, 6), dpi=300)

# Load and process MLP data
df_ssl_mlp = pd.read_csv(Path("../results/confidence_intervals/ssl_mlp_ci.csv"))
df_ssl_mlp["label_fraction"] = df_ssl_mlp["label_fraction"].astype(float)
df_ssl_mlp = df_ssl_mlp[df_ssl_mlp["model"].isin(ssl_models)]
df_ssl_mlp['model'] = pd.Categorical(df_ssl_mlp['model'], categories=ssl_models, ordered=True)
df_ssl_mlp = df_ssl_mlp.sort_values('model')

# Plot MLP
for i, model in enumerate(ssl_models):
    sub = df_ssl_mlp[df_ssl_mlp["model"] == model].sort_values("label_fraction")
    means = sub["f1_mean"].values
    yerr_lower = means - sub["f1_ci_lower"].values
    yerr_upper = sub["f1_ci_upper"].values - means

    positions = x_ssl + i * bar_width
    ax_mlp.errorbar(
        positions, means, yerr=[yerr_lower, yerr_upper],
        fmt=MARKERS.get(model, 'o'),
        capsize=5,
        label=model,
        color=COLORS.get(model, f'C{i}'),
        linestyle='None',
        markersize=8,
        markerfacecolor='white',
        markeredgewidth=2,
        capthick=2,
        elinewidth=2
    )

# Add significance asterisks for MLP plot
for idx, frac in enumerate(fractions_ssl_linear):
    comparisons = SSL_MLP_SIGNIFICANCE.get(frac, {})
    base_x = x_ssl[idx]
    relevant_data = df_ssl_mlp[df_ssl_mlp["label_fraction"] == frac]
    if not relevant_data.empty:
        height = max(relevant_data["f1_ci_upper"]) + 2
    else:
        height = ax_mlp.get_ylim()[1] * 0.1

    for j, (m1, m2) in enumerate(comparisons):
        if comparisons[(m1, m2)]:
            i1 = ssl_models.index(m1)
            i2 = ssl_models.index(m2)
            x1 = base_x + i1 * bar_width
            x2 = base_x + i2 * bar_width
            y = height + j * 1.2
            ax_mlp.plot([x1, x2], [y, y], color='black', linewidth=1)
            ax_mlp.text((x1 + x2)/2, y + 0.01, "*", ha='center', va='bottom', fontsize=14)

# Style MLP plot
ax_mlp.set_title("SSL Models with MLP Classifier", fontsize=14, fontweight='bold', pad=20)
ax_mlp.set_xlabel("Proportion of Labeled Training Data", fontsize=12, fontweight='bold')
ax_mlp.set_ylabel("MF1 Score", fontsize=12, fontweight='bold')
ax_mlp.set_xticks(x_ssl + bar_width * (n_models - 1) / 2)
ax_mlp.set_xticklabels([f'{int(f*100)}%' for f in fractions_ssl_linear], fontsize=12)
ax_mlp.tick_params(axis='both', which='major', labelsize=12)
ax_mlp.grid(True, linestyle='--', alpha=0.3)
ax_mlp.set_facecolor('#FAFAFA')
ax_mlp.spines['top'].set_visible(False)
ax_mlp.spines['right'].set_visible(False)
ax_mlp.spines['left'].set_linewidth(0.5)
ax_mlp.spines['bottom'].set_linewidth(0.5)
ax_mlp.legend(frameon=True, fancybox=True, shadow=True, fontsize=12, loc='lower right')

plt.tight_layout()
plt.savefig("../results/error_bars_ssl_mlp.pdf", dpi=300, bbox_inches='tight')
plt.savefig("../results/error_bars_ssl_mlp.png", dpi=300, bbox_inches='tight')
plt.show()

# ----------------------------------------------
# Additional metric plots for SSL MLP Models
# ----------------------------------------------

metrics = {
    'accuracy': 'Accuracy',
    'auc': 'AUC-ROC',
    'pr_auc': 'PR-AUC'
}

for metric_prefix, metric_name in metrics.items():
    # Create figure for this metric
    fig_metric, ax_metric = plt.subplots(figsize=(12, 6), dpi=300)
    
    # Plot each model's metric
    for i, model in enumerate(ssl_models):
        sub = df_ssl_mlp[df_ssl_mlp["model"] == model].sort_values("label_fraction")
        means = sub[f"{metric_prefix}_mean"].values
        yerr_lower = means - sub[f"{metric_prefix}_ci_lower"].values
        yerr_upper = sub[f"{metric_prefix}_ci_upper"].values - means

        positions = x_ssl + i * bar_width
        ax_metric.errorbar(
            positions, means, yerr=[yerr_lower, yerr_upper],
            fmt=MARKERS.get(model, 'o'),
            capsize=5,
            label=model,
            color=COLORS.get(model, f'C{i}'),
            linestyle='None',
            markersize=8,
            markerfacecolor='white',
            markeredgewidth=2,
            capthick=2,
            elinewidth=2
        )

    # Style metric plot
    ax_metric.set_title(f"SSL Models with MLP Classifier - {metric_name}", 
                       fontsize=14, fontweight='bold', pad=20)
    ax_metric.set_xlabel("Proportion of Labeled Training Data", 
                        fontsize=12, fontweight='bold')
    ax_metric.set_ylabel(f"{metric_name} Score", 
                        fontsize=12, fontweight='bold')
    ax_metric.set_xticks(x_ssl + bar_width * (n_models - 1) / 2)
    ax_metric.set_xticklabels([f'{int(f*100)}%' for f in fractions_ssl_linear], 
                             fontsize=12)
    ax_metric.tick_params(axis='both', which='major', labelsize=12)
    ax_metric.grid(True, linestyle='--', alpha=0.3)
    ax_metric.set_facecolor('#FAFAFA')
    ax_metric.spines['top'].set_visible(False)
    ax_metric.spines['right'].set_visible(False)
    ax_metric.spines['left'].set_linewidth(0.5)
    ax_metric.spines['bottom'].set_linewidth(0.5)
    ax_metric.legend(frameon=True, fancybox=True, shadow=True, 
                    fontsize=12, loc='lower right')

    plt.tight_layout()
    plt.savefig(f"../results/error_bars_ssl_mlp_{metric_prefix}.pdf", 
                dpi=300, bbox_inches='tight')
    plt.savefig(f"../results/error_bars_ssl_mlp_{metric_prefix}.png", 
                dpi=300, bbox_inches='tight')
    plt.show()