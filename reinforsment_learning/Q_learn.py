import pandas as pd
import numpy as np
import random

# Load dataset
df = pd.read_csv("traffic_data.csv")

# Encode states and actions
state_map = {"LowTraffic": 0, "MediumTraffic": 1, "HighTraffic": 2}
action_map = {"Green": 0, "Red": 1}

df["State"] = df["State"].map(state_map)
df["Action"] = df["Action"].map(action_map)

# Q-table (3 states x 2 actions)
Q = np.zeros((3, 2))

# Parameters
alpha = 0.1
gamma = 0.9
epsilon = 0.2

# Training
episodes = 100

for ep in range(episodes):
    for i in range(len(df)):
        state = df.iloc[i]["State"]
        action = df.iloc[i]["Action"]
        reward = df.iloc[i]["Reward"]

        # Q-learning update
        Q[state, action] = Q[state, action] + alpha * (
            reward + gamma * np.max(Q[state]) - Q[state, action]
        )

# Accuracy calculation
correct = 0

for i in range(len(df)):
    state = int(df.iloc[i]["State"])
    actual_action = int(df.iloc[i]["Action"])
    predicted_action = np.argmax(Q[state])

    if actual_action == predicted_action:
        correct += 1

accuracy = (correct / len(df)) * 100

print("Accuracy:", round(accuracy, 2), "%")

# Prediction for new data
new_state = "HighTraffic"
state_val = state_map[new_state]

predicted_action = np.argmax(Q[state_val])

action_reverse = {0: "Green", 1: "Red"}

print("\nNew Data Prediction:")
print("State:", new_state)
print("Best Action:", action_reverse[predicted_action])