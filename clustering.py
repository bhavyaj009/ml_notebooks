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

# add logic to show roc curve with input labels provided
# Step 6: Evaluate anomaly detection performance (optional)
# If you have ground truth labels for anomalies, you can evaluate the performance using ROC curve or other metrics
# For example, if you have a list of true labels for anomalies, you can compute the ROC curve as follows:

import numpy as np

# Example: Total number of labels and anomaly indices
total_labels = 10
anomaly_indices = [1, 4, 5, 9]

# Initialize true_labels with zeros
true_labels = np.zeros(total_labels, dtype=int)

# Set anomalies in true_labels
true_labels[anomaly_indices] = 1

print("True Labels:", true_labels)
true_labels = np.array([0, 1, 0, 0, 1, 1, 0, 0, 0, 1])  # Example true labels (0: normal, 1: anomaly)
scores = -distances  # Use negative distances as anomaly scores
from sklearn.metrics import roc_curve, auc
fpr, tpr, _ = roc_curve(true_labels, scores)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc="lower right")
plt.show()