import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Step 1: Load the .npy file
features = np.load('path_to_your_file.npy')

# Step 2: Preprocess the data (if necessary)
# Here you might normalize or scale your features if required

# Step 3: Perform clustering using K-Means
n_clusters = 3  # Specify the number of clusters you want
kmeans = KMeans(n_clusters=n_clusters)
kmeans.fit(features)
labels = kmeans.labels_

# Step 4: Visualize the clusters (if features are 2D or 3D)
plt.scatter(features[:, 0], features[:, 1], c=labels, cmap='viridis')
plt.title('K-Means Clustering')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()

# add logic to find distance from cluster center and use that metric to find anomalies
# Step 5: Identify anomalies based on cluster distance
cluster_centers = kmeans.cluster_centers_
distances = np.array([np.linalg.norm(features[i] - cluster_centers[labels[i]]) for i in range(len(features))])
anomaly_indices = np.argsort(distances)[-5:]  # Get the top 5 anomalies
print("Anomaly indices:", anomaly_indices)