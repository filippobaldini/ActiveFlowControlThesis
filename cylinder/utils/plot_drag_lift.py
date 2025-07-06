import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file
filename = "drag_steps.csv"  # change this to your actual file path
df = pd.read_csv(filename)

# Check the available columns
print("Available columns:", df.columns)

# Plot Drag
plt.figure()
plt.plot(df["Step"], df["Drag"], label="Drag", color="b")
plt.xlabel("Steps")
plt.ylabel("Drag")
plt.title("Drag vs Actuation Steps")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# Plot Lift
plt.figure()
plt.plot(df["Episode"], df["Cl"], label="Lift", color="r")
plt.xlabel("Episodes")
plt.ylabel("Lift")
plt.title("Lift vs Episodes")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
