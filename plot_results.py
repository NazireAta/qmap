import ast
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime

# Load the compiled results
df = pd.read_csv("placement_comparison.csv")

# Convert microseconds to milliseconds for cleaner chart labels
df["layout_synthesis_ms"] = df["layoutSynthesisTime"] / 1000
df["routing_time_ms"] = df["layoutSynthesizerStatistics"].apply(
    lambda s: ast.literal_eval(s)["routingTime"] / 1000 if isinstance(s, str) else None
)

# Plotting settings
sns.set_theme(style="whitegrid")
fig, axes = plt.subplots(2, 3, figsize=(18, 10), sharey=False)
axes = axes.flatten()

# Filter and plot each unique benchmark family
benchmarks = df["benchmark"].unique()
strategy_labels = {
    0: "Trivial",
    1: "Geometric",
    2: "Activity-based",
    3: "Interaction-based",
}
df["Strategy"] = df["strategy_name"].map(strategy_labels)
fig.suptitle("Routing Time Comparison by Initial Placement Strategy", fontsize=16)
for i, bench in enumerate(benchmarks):
    if i >= len(axes):
        break
    bench_df = df[df["benchmark"] == bench]
    
    sns.barplot(
        data=bench_df,
        x="n_qubits",
        y="routing_time_ms",
        hue="Strategy",
        ax=axes[i],
        palette="muted"
    )
    axes[i].set_title(f"{bench.upper()}")
    axes[i].set_xlabel("Number of Qubits (n)")
    axes[i].set_ylabel("Routing Time (ms)")
plt.tight_layout(rect=[0, 0, 0.88, 1])
plt.tight_layout()
plt.savefig(f"{datetime.datetime.now().strftime('%d_%H-%M-%S')}_routing_times.png", dpi=300)
plt.close()

fig, axes = plt.subplots(2, 3, figsize=(18, 10), sharey=False)
axes = axes.flatten()
fig.suptitle("Layout Synthesis Time Comparison by Initial Placement Strategy", fontsize=16)
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
    axes[i].set_title(f"{bench.upper()}")
    axes[i].set_xlabel("Number of Qubits (n)")
    axes[i].set_ylabel("Layout Synthesis Time (ms)")
plt.tight_layout(rect=[0, 0, 0.88, 1])
plt.tight_layout()
plt.savefig(f"{datetime.datetime.now().strftime('%d_%H-%M-%S')}_layout_synthesis_times.png", dpi=300)
plt.close()
print("Analysis plots saved as 'layout_synthesis_times.png'!")