import re
import matplotlib.pyplot as plt

# Your palette
colors = [
    "#6FA8DC",  # blue
    "#E06666",  # red
    "#93C47D",  # green
    "#F6B26B",  # orange
    "#B4A7D6",  # purple
    "#FFD966"   # yellow
]

with open("fcnn_training_data/log.txt", "r") as f:
    text = f.read()

pattern = re.compile(
    r"epoch\s+(\d+)\s+(\d+),\s*train loss:\s*([0-9.]+)\s*\n\s*validation loss:\s*([0-9.]+)",
    re.IGNORECASE
)

# Storage
data = {
    5: {"epoch": [], "train": [], "val": []},
    60: {"epoch": [], "train": [], "val": []},
}

# Parse file
for m in pattern.finditer(text):
    epoch = int(m.group(1))
    kind = int(m.group(2))
    train_loss = float(m.group(3))
    val_loss = float(m.group(4))

    if kind in data:
        data[kind]["epoch"].append(epoch)
        data[kind]["train"].append(train_loss)
        data[kind]["val"].append(val_loss)

# Styling
plt.rcParams.update({
    "font.size": 18,
    "axes.labelsize": 22,
    "axes.titlesize": 24,
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,
    "legend.fontsize": 18,
})

plt.figure(figsize=(12, 8))

# Pick 4 colours
c1 = colors[0]  # blue
c2 = colors[1]  # red
c3 = colors[2]  # green
c4 = colors[3]  # orange

# Type 5
plt.plot(
    data[5]["epoch"],
    data[5]["train"],
    color=c1,
    linewidth=3.5,
    label="Training Loss (2048)"
)

plt.plot(
    data[5]["epoch"],
    data[5]["val"],
    color=c2,
    linewidth=3.5,
    linestyle="--",
    label="Validation Loss (2048)"
)

# Type 60
plt.plot(
    data[60]["epoch"],
    data[60]["train"],
    color=c3,
    linewidth=3.5,
    label="Training Loss (4096)"
)

plt.plot(
    data[60]["epoch"],
    data[60]["val"],
    color=c4,
    linewidth=3.5,
    linestyle="--",
    label="Validation Loss (4096)"
)

plt.xlabel("Epoch")
plt.ylabel("Loss on z-normalised dataset")

# Thicker ticks
plt.tick_params(width=2, length=8)

# Thicker axes
ax = plt.gca()
for spine in ax.spines.values():
    spine.set_linewidth(2)

plt.legend()
plt.plot()

plt.tight_layout()
plt.show()