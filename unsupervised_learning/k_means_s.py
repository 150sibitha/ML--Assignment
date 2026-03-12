import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder

print("Running K-Means Clustering...")

# Load dataset
data = pd.read_csv("pesticides.csv")

# Convert categorical columns to numbers
le = LabelEncoder()

data["Domain"] = le.fit_transform(data["Domain"])
data["Area"] = le.fit_transform(data["Area"])
data["Element"] = le.fit_transform(data["Element"])
data["Item"] = le.fit_transform(data["Item"])
data["Unit"] = le.fit_transform(data["Unit"])

# Features (remove target if needed)
X = data.drop("Value", axis=1)

# Apply K-Means
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X)

# Cluster labels
clusters = kmeans.labels_

print("Cluster Labels (First 10):", clusters[:10])

# Add cluster column
data["Cluster"] = clusters

# Visualization using first two features
plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=clusters)

plt.title("K-Means Clustering of Pesticides Dataset")
plt.xlabel(X.columns[0])
plt.ylabel(X.columns[1])

plt.show()
