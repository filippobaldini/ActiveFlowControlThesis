import pandas as pd
import matplotlib.pyplot as plt

# Path to the CSV file
file_path = "drag_steps.csv"  # Make sure this path is correct

# Skip the first three lines (metadata)
df = pd.read_csv(file_path, skiprows=3)

# Convert columns to numeric
df["Step"] = pd.to_numeric(df["Step"])
df["Drag"] = pd.to_numeric(df["Drag"])

# Plot
plt.figure(figsize=(10, 5))
plt.plot(df["Step"], df["Drag"], marker="o", color="darkorange")
plt.title("Drag over Control Steps")
plt.xlabel("Step")
plt.ylabel("Drag Coefficient $C_D$")
plt.grid(True)
plt.tight_layout()
plt.show()
