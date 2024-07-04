import numpy as np
import networkx as nx
from sklearn.feature_extraction import image
from sklearn.metrics.pairwise import cosine_similarity
from skimage.util import view_as_blocks
from skimage import io, color

# Load and preprocess image
image_path = 'path_to_image.jpg'
img = io.imread(image_path)
img_gray = color.rgb2gray(img)
patch_size = (8, 8)

# Divide image into patches
patches = view_as_blocks(img_gray, block_shape=patch_size).reshape(-1, patch_size[0] * patch_size[1])

# Compute cosine similarity between patches
similarity_matrix = cosine_similarity(patches)

# Create a graph from the similarity matrix
G = nx.from_numpy_array(similarity_matrix)

# Function to update the graph based on user feedback
def update_graph_with_feedback(G, defective_nodes):
    for node in defective_nodes:
        # Mark the node as defective by assigning a high anomaly score
        G.nodes[node]['anomaly_score'] = 1.0  # High score for defective nodes
        
        # Adjust the edge weights connected to defective nodes
        for neighbor in G.neighbors(node):
            G[node][neighbor]['weight'] = 0.0  # Decrease similarity
        
    return G

# List of nodes marked as defective by the user
defective_nodes = [10, 23, 45]  # Example indices of defective nodes

# Update the graph with user feedback
G = update_graph_with_feedback(G, defective_nodes)

# Recompute PageRank to get updated anomaly scores
pagerank_scores = nx.pagerank(G, weight='weight')
anomaly_scores = np.array(list(pagerank_scores.values()))

# Identify top anomalies
num_anomalies = 5
anomaly_indices = np.argsort(anomaly_scores)[:num_anomalies]

# Visualize anomalies
fig, axes = plt.subplots(1, num_anomalies, figsize=(15, 5))
for i, idx in enumerate(anomaly_indices):
    row = idx // (img_gray.shape[1] // patch_size[1])
    col = idx % (img_gray.shape[1] // patch_size[1])
    anomaly_patch = img[row*patch_size[0]:(row+1)*patch_size[0], col*patch_size[1]:(col+1)*patch_size[1]]
    axes[i].imshow(anomaly_patch, cmap='gray')
    axes[i].axis('off')
plt.show()
