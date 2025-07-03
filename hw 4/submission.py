# Submit this file to Gradescope
from typing import List
# you may use other Python standard libraries, but not data
# science libraries, such as numpy, scikit-learn, etc.
import math, sys

class Solution:
  def hclus_single_link(self, X: List[List[float]], K: int) -> List[int]:
    """Single link hierarchical clustering
    Args:
      - X: 2D input data
      - K: the number of output clusters
    Returns:
      A list of integers (range from 0 to K - 1) that represent class labels.
      The number does not matter as long as the clusters are correct.
      For example: [0, 0, 1] is treated the same as [1, 1, 0]"""
    # implement this function
    return self._hclus(X, K, mode=0)

  def hclus_average_link(self, X: List[List[float]], K: int) -> List[int]:
    """Complete link hierarchical clustering"""
    # implement this function
    return self._hclus(X, K, mode=2)
  def hclus_complete_link(self, X: List[List[float]], K: int) -> List[int]:
    """Average link hierarchical clustering"""
    # implement this function
    return self._hclus(X, K, mode=1)
  def _hclus(self, X: List[List[float]], K: int, mode: int) -> List[int]:
    def dist(a: List[float], b: List[float]) -> float:
        return math.hypot(a[0] - b[0], a[1] - b[1])

    N = len(X)
    # initialize each point as its own cluster
    clusters: List[List[int]] = [[i] for i in range(N)]

    # merge until K clusters remain
    while len(clusters) > K:
        best_i, best_j, best_d = 0, 1, float('inf')
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                # compute distance between clusters
                dists = [dist(X[p], X[q]) for p in clusters[i] for q in clusters[j]]
                if mode == 0:
                    d = min(dists)
                elif mode == 1:
                    d = max(dists)
                else:
                    d = sum(dists) / len(dists)
                if d < best_d:
                    best_d, best_i, best_j = d, i, j
        # merge j into i
        clusters[best_i].extend(clusters[best_j])
        clusters.pop(best_j)

    # assign labels
    labels = [0] * N
    for lbl, cl in enumerate(clusters):
        for idx in cl:
            labels[idx] = lbl
    return labels


if __name__ == '__main__':
    data = sys.stdin.read().strip().split()
    if not data:
        sys.exit(0)
    it = iter(data)
    #parse header: N points, K clusters, M linkage mode
    N = int(next(it))
    K = int(next(it))
    M = int(next(it))
    #read points
    X: List[List[float]] = []
    for _ in range(N):
        lon = float(next(it)); lat = float(next(it))
        X.append([lon, lat])
    sol = Solution()
    #select method based on M
    if M == 0:
        labels = sol.hclus_single_link(X, K)
    elif M == 1:
        labels = sol.hclus_complete_link(X, K)
    else:
        labels = sol.hclus_average_link(X, K)
    #output labels
    out = sys.stdout
    for i, lbl in enumerate(labels):
        out.write(f"{i} {lbl}\n")