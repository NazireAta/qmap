import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the compiled results
df = pd.read_csv("placement_comparison.csv")

# Convert microseconds to milliseconds for cleaner chart labels
df["layout_synthesis_ms"] = df["layoutSynthesisTime"] / 1000

# Plotting settings
sns.set_theme(style="whitegrid")
fig, axes = plt.subplots(2, 3, figsize=(18, 10), sharey=False)
axes = axes.flatten()

# Filter and plot each unique benchmark family
benchmarks = df["benchmark"].unique()
strategy_labels = {0: "Trivial", 1: "Interaction Graph", 2: "Zone Affinity"}
df["Strategy"] = df["strategy_name"].map(strategy_labels)

for i, bench in enumerate(benchmarks):
    if i >= len(axes):
        break
    bench_df = df[df["benchmark"] == bench]
    
    sns.barplot(
        data=bench_df,
        x="n_qubits",
        y="layout_synthesis_ms",
        hue="Strategy",
        ax=axes[i],
        palette="muted"
    )
    axes[i].set_title(f"Synthesis Time: {bench.upper()}")
    axes[i].set_xlabel("Number of Qubits (n)")
    axes[i].set_ylabel("Time (ms)")

plt.tight_layout()
plt.savefig("placement_strategy_analysis.png", dpi=300)
print("Analysis plots saved as 'placement_strategy_analysis.png'!")