"""
K-Means Clustering Implementation
Based on GeeksforGeeks guide, adapted to use data.txt file
"""

import numpy as np
import matplotlib.pyplot as plt

# Step 1: Load data from data.txt file
def load_data(filename):
    """
    Load 2D data points from a text file.
    Each line should contain two space-separated floating-point values.
    """
    data = []
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if line:  # Skip empty lines
                parts = line.split()
                if len(parts) >= 2:
                    x, y = float(parts[0]), float(parts[1])
                    data.append([x, y])
    return np.array(data)

# Step 2: Define Euclidean Distance
def distance(p1, p2):
    """
    Calculate Euclidean distance between two points.
    """
    return np.sqrt(np.sum((p1 - p2) ** 2))

# Step 3: Assign clusters
def assign_clusters(X, clusters, k):
    """
    Assign each data point to the nearest centroid.
    """
    for idx in range(X.shape[0]):
        dist = []
        curr_x = X[idx]
        
        for i in range(k):
            dis = distance(curr_x, clusters[i]['center'])
            dist.append(dis)
        
        curr_cluster = np.argmin(dist)
        clusters[curr_cluster]['points'].append(curr_x)
    
    return clusters

# Step 4: Update clusters
def update_clusters(X, clusters, k):
    """
    Recalculate centroids based on assigned points.
    """
    for i in range(k):
        points = np.array(clusters[i]['points'])
        if points.shape[0] > 0:
            new_center = points.mean(axis=0)
            clusters[i]['center'] = new_center
            clusters[i]['points'] = []
    return clusters

# Step 5: Predict cluster for data points
def pred_cluster(X, clusters, k):
    """
    Predict cluster assignment for each data point.
    """
    pred = []
    for i in range(X.shape[0]):
        dist = []
        for j in range(k):
            dist.append(distance(X[i], clusters[j]['center']))
        pred.append(np.argmin(dist))
    return pred

# Step 6: Initialize random centroids
def initialize_centroids(X, k, random_seed=23):
    """
    Initialize k random centroids within the data range.
    """
    np.random.seed(random_seed)
    clusters = {}
    
    # Get data range for better initialization
    x_min, x_max = X[:, 0].min(), X[:, 0].max()
    y_min, y_max = X[:, 1].min(), X[:, 1].max()
    
    for idx in range(k):
        # Random center within data range
        center = np.array([
            np.random.uniform(x_min, x_max),
            np.random.uniform(y_min, y_max)
        ])
        cluster = {
            'center': center,
            'points': []
        }
        clusters[idx] = cluster
    
    return clusters

# Main K-Means algorithm
def kmeans(X, k, max_iters=100, random_seed=23, tolerance=1e-4):
    """
    Perform K-Means clustering on the dataset.
    
    Parameters:
    - X: numpy array of data points (n_samples, n_features)
    - k: number of clusters
    - max_iters: maximum number of iterations
    - random_seed: random seed for reproducibility
    - tolerance: convergence threshold
    
    Returns:
    - clusters: dictionary with cluster centers and assignments
    - predictions: cluster assignments for each data point
    - iterations: number of iterations performed
    """
    # Initialize centroids
    clusters = initialize_centroids(X, k, random_seed)
    
    prev_centers = None
    
    for iteration in range(max_iters):
        # Assign points to nearest centroids
        clusters = assign_clusters(X, clusters, k)
        
        # Update centroids
        clusters = update_clusters(X, clusters, k)
        
        # Check for convergence
        if prev_centers is not None:
            # Calculate change in centroids
            max_change = 0
            for i in range(k):
                change = distance(clusters[i]['center'], prev_centers[i])
                max_change = max(max_change, change)
            
            if max_change < tolerance:
                print(f"Converged after {iteration + 1} iterations")
                break
        
        # Store current centers for next iteration
        prev_centers = [clusters[i]['center'].copy() for i in range(k)]
    
    # Final prediction
    predictions = pred_cluster(X, clusters, k)
    
    return clusters, predictions, iteration + 1

# Visualization functions
def plot_data(X, title="Dataset"):
    """
    Plot the raw data points.
    """
    plt.figure(figsize=(10, 8))
    plt.grid(True)
    plt.scatter(X[:, 0], X[:, 1], alpha=0.6, s=20)
    plt.title(title)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()

def plot_initial_centroids(X, clusters, k):
    """
    Plot data points with initial random centroids.
    """
    plt.figure(figsize=(10, 8))
    plt.grid(True)
    plt.scatter(X[:, 0], X[:, 1], alpha=0.6, s=20, c='blue', label='Data Points')
    
    for i in range(k):
        center = clusters[i]['center']
        plt.scatter(center[0], center[1], marker='*', c='red', s=300, 
                   edgecolors='black', linewidths=2, label='Centroids' if i == 0 else '')
    
    plt.title('Data Points with Initial Random Centroids')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.show()

def plot_clusters(X, clusters, predictions, k, title="K-Means Clustering Result"):
    """
    Plot data points colored by cluster assignment with final centroids.
    """
    plt.figure(figsize=(10, 8))
    plt.grid(True)
    
    # Plot data points colored by cluster
    scatter = plt.scatter(X[:, 0], X[:, 1], c=predictions, cmap='viridis', 
                         alpha=0.6, s=20)
    
    # Plot centroids
    for i in range(k):
        center = clusters[i]['center']
        plt.scatter(center[0], center[1], marker='^', c='red', s=400,
                   edgecolors='black', linewidths=2, label='Centroids' if i == 0 else '')
    
    plt.colorbar(scatter, label='Cluster')
    plt.title(title)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.show()

# Main execution
if __name__ == "__main__":
    # Load data from data.txt
    print("Loading data from data.txt...")
    X = load_data('data.txt')
    print(f"Loaded {X.shape[0]} data points with {X.shape[1]} features")
    
    # Set number of clusters (you can change this)
    k = 3
    
    # Step 1: Plot raw data
    print("\nStep 1: Plotting raw data...")
    plot_data(X, "Raw Dataset from data.txt")
    
    # Step 2: Initialize centroids
    print(f"\nStep 2: Initializing {k} random centroids...")
    clusters = initialize_centroids(X, k, random_seed=23)
    
    # Step 3: Plot initial centroids
    print("Step 3: Plotting data with initial centroids...")
    plot_initial_centroids(X, clusters, k)
    
    # Step 4: Run K-Means algorithm
    print(f"\nStep 4: Running K-Means algorithm with k={k}...")
    clusters, predictions, iterations = kmeans(X, k, max_iters=100, random_seed=23)
    print(f"Algorithm completed in {iterations} iterations")
    
    # Step 5: Plot final results
    print("\nStep 5: Plotting final clustering results...")
    plot_clusters(X, clusters, predictions, k, f"K-Means Clustering (k={k})")
    
    # Print cluster centers
    print("\nFinal Cluster Centers:")
    for i in range(k):
        center = clusters[i]['center']
        print(f"Cluster {i}: ({center[0]:.4f}, {center[1]:.4f})")
    
    # Count points in each cluster
    unique, counts = np.unique(predictions, return_counts=True)
    print("\nPoints per cluster:")
    for cluster_id, count in zip(unique, counts):
        print(f"Cluster {cluster_id}: {count} points")

