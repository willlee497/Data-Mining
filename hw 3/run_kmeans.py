import numpy as np

def load_data(file_path='places.txt'):
    """
    Load 2D points (longitude, latitude) from places.txt
    """
    return np.loadtxt(file_path, delimiter=',')

def initialize_centroids_kmeans_pp(data, k):
    """
    Initialize centroids using k means algorithm
    """
    n_samples, _ = data.shape
    centroids = np.zeros((k, data.shape[1]))
    
    #pick the first centroid randomly
    centroids[0] = data[np.random.randint(n_samples)]
    
    #initialize remaining centroids
    for i in range(1, k):
        #compute squared distances to nearest existing centroid
        distances = np.min(np.linalg.norm(data[:, None] - centroids[:i], axis=2)**2, axis=1)
        #select next centroid with probability proportional to distance
        probabilities = distances / distances.sum()
        cumulative_probs = np.cumsum(probabilities)
        r = np.random.rand()
        centroids[i] = data[np.searchsorted(cumulative_probs, r)]
    
    return centroids

def assign_clusters(data, centroids):
    """
    assign each data point to the nearest centroid
    """
    distances = np.linalg.norm(data[:, None] - centroids, axis=2)
    return np.argmin(distances, axis=1)

def update_centroids(data, labels, k):
    """
    update centroids as the mean of assigned points
    """
    new_centroids = np.zeros((k, data.shape[1]))
    for i in range(k):
        cluster_points = data[labels == i]
        if len(cluster_points) > 0:
            new_centroids[i] = cluster_points.mean(axis=0)
        else:
            #if a cluster lost all points, reinitialize randomly
            new_centroids[i] = data[np.random.randint(data.shape[0])]
    return new_centroids

def kmeans(data, k=3, max_iters=100, tol=1e-6):
    """
    Run k means clustering
    """
    centroids = initialize_centroids_kmeans_pp(data, k)
    for iteration in range(max_iters):
        labels = assign_clusters(data, centroids)
        new_centroids = update_centroids(data, labels, k)
        
        #check for convergence
        if np.allclose(centroids, new_centroids, atol=tol):
            break
        centroids = new_centroids
    return labels, centroids

def save_clusters(labels, file_path='clusters.txt'):
    """
    save cluster labels to file in the required format
    """
    with open(file_path, 'w') as f:
        for idx, label in enumerate(labels):
            f.write(f"{idx} {label}\n")

#Main execution
if __name__ == "__main__":
    data = load_data('places.txt')
    labels, centroids = kmeans(data, k=3)
    save_clusters(labels, 'clusters.txt')
    print("clusters.txt file generated with k-means labels.")

