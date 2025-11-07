# K-Means Clustering Implementation

This project implements the K-Means clustering algorithm from scratch, following the GeeksforGeeks guide but adapted to work with your `data.txt` file.

## Overview

K-Means clustering is an unsupervised machine learning algorithm that groups data points into k clusters based on their similarity. The algorithm works by:

1. **Initialization**: Randomly selecting k cluster centroids
2. **Assignment**: Assigning each data point to the nearest centroid
3. **Update**: Recalculating centroids based on assigned points
4. **Repeat**: Iterating until convergence or maximum iterations reached

## Files

- `kmeans_clustering.py`: Main implementation of the K-Means algorithm
- `data.txt`: Dataset containing 5000 2D data points (x, y coordinates)

## How to Use

### Basic Usage

```python
python kmeans_clustering.py
```

This will:
1. Load data from `data.txt`
2. Initialize k=3 clusters
3. Run the K-Means algorithm
4. Display visualizations showing:
   - Raw data points
   - Initial centroids
   - Final clustering results

### Customizing the Algorithm

You can modify the script to change:

- **Number of clusters (k)**: Change the `k` variable in the main section
- **Random seed**: Modify `random_seed` parameter for reproducibility
- **Maximum iterations**: Adjust `max_iters` parameter
- **Convergence tolerance**: Change `tolerance` parameter

Example:
```python
k = 5  # Use 5 clusters instead of 3
clusters, predictions, iterations = kmeans(X, k, max_iters=200, random_seed=42)
```

## Algorithm Steps (Following GeeksforGeeks Guide)

### Step 1: Import Libraries
- `numpy`: For numerical operations and distance calculations
- `matplotlib`: For plotting data and results

### Step 2: Load Data
Instead of using `make_blobs` from sklearn, we load data from `data.txt`:
- Each line contains two space-separated floating-point values
- Data is converted to a numpy array

### Step 3: Initialize Random Centroids
- Randomly select k points within the data range as initial centroids
- Store centroids in a dictionary structure

### Step 4: Define Distance Function
- Euclidean distance: `√((x₁-x₂)² + (y₁-y₂)²)`

### Step 5: Assignment Step
- For each data point, calculate distance to all centroids
- Assign point to the nearest centroid

### Step 6: Update Step
- Recalculate each centroid as the mean of all points assigned to it
- Clear the points list for next iteration

### Step 7: Iteration
- Repeat assignment and update steps until:
  - Centroids stop changing (convergence)
  - Maximum iterations reached

### Step 8: Visualization
- Plot raw data
- Plot initial centroids
- Plot final clusters with color-coded assignments

## Key Functions

- `load_data(filename)`: Loads 2D data from text file
- `distance(p1, p2)`: Calculates Euclidean distance
- `initialize_centroids(X, k)`: Randomly initializes k centroids
- `assign_clusters(X, clusters, k)`: Assigns points to nearest centroids
- `update_clusters(X, clusters, k)`: Updates centroids based on assignments
- `kmeans(X, k, ...)`: Main algorithm that runs until convergence
- `pred_cluster(X, clusters, k)`: Predicts cluster for each point

## Differences from GeeksforGeeks Example

1. **Data Loading**: Uses `data.txt` instead of `make_blobs`
2. **Initialization**: Centroids initialized within data range (not fixed -2 to 2)
3. **Convergence Check**: Added tolerance-based convergence detection
4. **Visualization**: Enhanced plots with better labels and legends

## Requirements

```bash
pip install numpy matplotlib
```

## Output

The script generates three plots:
1. Raw dataset visualization
2. Initial centroids with data points
3. Final clustering result with color-coded clusters

It also prints:
- Number of iterations until convergence
- Final cluster center coordinates
- Number of points in each cluster

