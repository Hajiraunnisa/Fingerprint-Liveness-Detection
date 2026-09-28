import os
import matplotlib.pyplot as plt
import pandas as pd

os.makedirs("static/analytics", exist_ok=True)

# ============================================================
# MODEL RESULTS
# Stage 1 — CNN alone:           96.37%
# Stage 2 — CNN + GA + CNN head: 93.68%
# Stage 3 — CNN + GA + DT:       98.21%  ← deployed model
# ============================================================

models = [
    "CNN\nalone",
    "CNN + GA\n+ CNN head",
    "CNN + GA\n+ Decision Tree"
]

accuracy  = [96.37, 93.68, 98.21]
precision = [93.22, 91.48, 97.54]
recall    = [100.00, 96.31, 98.90]
f1        = [96.49, 93.83, 98.22]

# ============================================================
# SAVE CSV
# ============================================================

results = pd.DataFrame({
    "Model":     models,
    "Accuracy":  accuracy,
    "Precision": precision,
    "Recall":    recall,
    "F1 Score":  f1
})
results.to_csv("static/analytics/model_comparison.csv", index=False)
print(results)

# ============================================================
# COMBINED BAR CHART WITH SCORE LABELS
# ============================================================

fig, ax = plt.subplots(figsize=(12, 7))

x      = range(len(models))
width  = 0.18
colors = ["#4e91d9", "#f59e0b", "#22c55e", "#ef4444"]
metrics = [
    ("Accuracy",  accuracy),
    ("Precision", precision),
    ("Recall",    recall),
    ("F1 Score",  f1),
]

for idx, (label, values) in enumerate(metrics):
    offsets = [i + width * (idx - 1.5) for i in x]
    bars = ax.bar(offsets, values, width, label=label, color=colors[idx])
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.1,
            f"{val:.2f}",
            ha="center", va="bottom",
            fontsize=8, fontweight="bold", color="white"
        )

ax.set_xticks(list(x))
ax.set_xticklabels(models, fontsize=11)
ax.set_ylim(85, 103)
ax.set_ylabel("Score (%)", fontsize=11)
ax.set_title("Progressive Model Performance Comparison",
             fontsize=14, fontweight="bold", pad=14)
ax.legend(fontsize=10)
ax.grid(axis="y", linestyle="--", alpha=0.4)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig("static/analytics/model_comparison.png", dpi=300)
plt.close()

print("\n==============================")
print("BEST MODEL: CNN + GA + Decision Tree — 98.21%")
print("==============================")
print("model_comparison.png saved.")
