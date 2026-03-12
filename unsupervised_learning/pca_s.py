import pandas as pd
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder, StandardScaler

print("Running PCA Algorithm...")

# Load dataset
data = pd.read_csv("pesticides.csv")

# Convert categorical columns to numbers
le = LabelEncoder()

data["Domain"] = le.fit_transform(data["Domain"])
data["Area"] = le.fit_transform(data["Area"])
data["Element"] = le.fit_transform(data["Element"])
data["Item"] = le.fit_transform(data["Item"])
data["Unit"] = le.fit_transform(data["Unit"])

# Features
X = data.drop("Value", axis=1)

# Standardize data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

print("Original Shape:", X.shape)
print("Reduced Shape:", X_pca.shape)

# Visualization
plt.scatter(X_pca[:, 0], X_pca[:, 1])

plt.title("PCA Visualization of Pesticides Dataset")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")

plt.show()
