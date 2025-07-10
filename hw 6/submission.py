# Submit this file to Gradescope
import math
from typing import Dict, List, Tuple
# You may use any built-in standard Python libraries
# You may NOT use any non-standard Python libraries such as numpy, scikit-learn, etc.

num_C = 7 # Represents the total number of classes

class Solution:
  
  def prior(self, X_train: List[List[int]], Y_train: List[int]) -> List[float]:
    """Calculate the prior probabilities of each class
    Args:
      X_train: Row i represents the i-th training datapoint
      Y_train: The i-th integer represents the class label for the i-th training datapoint
    Returns:
      A list of length num_C where num_C is the number of classes in the dataset
    """
    # implement this function
    n = len(Y_train)
    pokemon = [0.0] * num_C
    for y in Y_train:
      pokemon[y - 1] += 1
    for c in range(num_C):
      pokemon[c] = (pokemon[c] + 0.1) / (n + 0.1 * num_C)
    return pokemon #pokemon represent priors

  def label(self, X_train: List[List[int]], Y_train: List[int], X_test: List[List[int]]) -> List[int]:
    """Calculate the classification labels for each test datapoint
    Args:
      X_train: Row i represents the i-th training datapoint
      Y_train: The i-th integer represents the class label for the i-th training datapoint
      X_test: Row i represents the i-th testing datapoint
    Returns:
      A list of length M where M is the number of datapoints in the test set
    """
    # implement this function
    N = len(Y_train)
    D = len(X_train[0])

    #determine unique values for each attribute
    unique_vals = [{} for _ in range(D)]
    for j in range(D):
      vals = set(x[j] for x in X_train)
      unique_vals[j] = vals
    #for legs attribute at index 12, ensure correct possible values
    unique_vals[12] = set([0, 2, 4, 5, 6, 8])

    #prior P(y)
    class_counts = [0] * num_C
    for y in Y_train:
      class_counts[y - 1] += 1
    priors = [(class_counts[c] + 0.1) / (N + 0.1 * num_C) for c in range(num_C)]

    #likelihoods P(x_i | y)
    likelihood = [{} for _ in range(D)]
    for j in range(D):
      likelihood[j] = {c: {} for c in range(num_C)}
      for c in range(num_C):
        for val in unique_vals[j]:
          count = 0
          for i in range(N):
            if Y_train[i] == c + 1 and X_train[i][j] == val:
              count += 1
          denom = class_counts[c] + 0.1 * len(unique_vals[j])
          prob = (count + 0.1) / denom
          likelihood[j][c][val] = prob

    predictions = []
    for x in X_test:
      log_probs = []
      for c in range(num_C):
        logp = math.log(priors[c])
        for j in range(D):
          val = x[j]
          #if unseen value, apply smoothing uniformly
          if val in likelihood[j][c]:
            logp += math.log(likelihood[j][c][val])
          else:
            denom = class_counts[c] + 0.1 * len(unique_vals[j])
            logp += math.log(0.1 / denom)
        log_probs.append(logp)
      best_class = log_probs.index(max(log_probs)) + 1
      predictions.append(best_class)

    return predictions
