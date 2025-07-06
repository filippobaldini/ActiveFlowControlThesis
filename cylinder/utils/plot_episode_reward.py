import pandas as pd
import matplotlib
import matplotlib.pyplot as plt


print("Pandas version:", pd.__version__)
print("Matplotlib version:", matplotlib.__version__)

# Replace 'episode_rewards.csv' with the path to your CSV file
df = pd.read_csv("episode_rewards.csv")

print(df.head())

episode_numbers = df["Episode"].to_numpy()
rewards = df["Reward"].to_numpy()

# Plot the data
plt.figure(figsize=(10, 6))
plt.plot(episode_numbers, rewards, linestyle="-", color="b", label="Reward")
plt.xlabel("Episode Number")
plt.ylabel("Reward")
plt.title("Reward vs. Episode Number")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
