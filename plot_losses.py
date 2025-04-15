import numpy as np
import matplotlib.pyplot as plt

# Set Times New Roman as the font
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 10  # Adjust font size as needed

# Load the losses
strategies = [
    "random_sampling",
    "uncertainty_sampling",
    "density_weighted_sampling",
    "estimated_error_reduction",
    "query_by_committee",
    "ours",
]
losses_dict = {strategy: np.loadtxt(f"./results/Query_strategy_losses/{strategy}.txt") for strategy in strategies}

# Plot the losses
fig, ax = plt.subplots(figsize=(5, 5))

# Colors and markers for each strategy
colors = ["k", "b", "g", "r", "c", "m"]
markers = ["o", "v", "s", "D", "^", "P"]

for strategy, color, marker in zip(strategies, colors, markers):
    losses = losses_dict[strategy]
    iter_vec = np.arange(1, len(losses) + 1)
    ax.errorbar(iter_vec, losses, fmt=f"-{color}", label=strategy.replace("_", " ").title(), marker=marker, capsize=3)

# Set axis labels and legend
ax.set_xlabel("Iteration")
ax.set_ylabel("Loss")
ax.legend(loc="upper right", fontsize="small", ncol=1, fancybox=True)
ax.grid(True, linestyle=":", color="gray", alpha=0.7)
output_filename = "./results/Query_strategy_losses/query_strategies_losses.png"
plt.savefig(output_filename, dpi=300, bbox_inches="tight", format="png")
# Show the plot
plt.show()