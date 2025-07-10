from typing import List
import math
class Node:

  """
  This class, Node, represents a single node in a decision tree. It is designed to store information about the tree
  structure and the specific split criteria at each node. It is important to note that this class should NOT be
  modified as it is part of the assignment and will be used by the autograder.

  The attributes of the Node class are:
  - split_dim: The dimension/feature along which the node splits the data (-1 by default, indicating uninitialized)
  - split_point: The value used for splitting the data at this node (-1 by default, indicating uninitialized)
  - label: The class label assigned to this node, which is the majority label of the data at this node. If there is a tie,
    the numerically smaller label is assigned (-1 by default, indicating uninitialized)
  - left: The left child node of this node (None by default). Either None or a Node object.
  - right: The right child node of this node (None by default) Either None or a Node object.
  """

  def __init__(self):
    self.split_dim = -1
    self.split_point = -1
    self.label = -1
    self.left = None
    self.right = None


class Solution:
  """
  Example usage of the Node class to build a decision tree using a custom method called split_node():

  # In the fit method, create the root node and call the split_node() method to build the decision tree
    self.root = Node()
    self.split_node(self.root, data, ..., depth=0)

  def split_node(self, node, data, ..., depth):
      # Your implementation to calculate split_dim, split_point, and label for the given node and data
      # ...

      # Assign the calculated values to the node
      node.split_dim = split_dim
      node.split_point = split_point
      node.label = label

      # Recursively call split_node() for the left and right child nodes if the current node is not a leaf node
      # Remember, a leaf node is one that either only has data from one class or one that is at the maximum depth
      if not is_leaf:
          left_child = Node()
          right_child = Node()

          split_node(left_child, left_data, ..., depth+1)
          split_node(right_child, right_data, ..., depth+1)
  """
  def __init__(self):
    self.root = None

  def split_info(self, data: List[List[float]], label: List[int], split_dim: int, split_point: float) -> float:
    """
    Compute the information needed to classify a dataset if it's split
    with the given splitting dimension and splitting point, i.e. Info_A in the slides.

    Parameters:
    data (List[List]): A nested list representing the dataset.
    label (List): A list containing the class labels for each data point.
    split_dim (int): The dimension/attribute index to split the data on.
    split_point (float): The value at which the data should be split along the given dimension.

    Returns:
    float: The calculated Info_A value for the given split. Do NOT round this value
    """
    left_labels = [label[i] for i in range(len(data)) if data[i][split_dim] <= split_point]
    right_labels = [label[i] for i in range(len(data)) if data[i][split_dim] > split_point]
    total = len(label)
    left_weight = len(left_labels) / total
    right_weight = len(right_labels) / total
    info = 0.0
    if left_labels:
        info += left_weight * self.entropy(left_labels)
    if right_labels:
        info += right_weight * self.entropy(right_labels)
    return info

  def fit(self, train_data: List[List[float]], train_label: List[int]) -> None:
    self.root = Node()
    self.build_tree(self.root, train_data, train_label, depth=0)
    """
    Fit the decision tree model using the provided training data and labels.

    Parameters:
    train_data (List[List[float]]): A nested list of floating point numbers representing the training data.
    train_label (List[int]): A list of integers representing the class labels for each data point in the training set.

    This method initializes the decision tree model by creating the root node. It then builds the decision tree starting 
    from the root node
    
    It is important to note that for tree structure evaluation, the autograder for this assignment
    first calls this method. It then performs tree traversals starting from the root node in order to check whether 
    the tree structure is correct. 
    
    So it is very important to ensure that self.root is assigned correctly to the root node
    
    It is best to use a different method (such as in the example above) to build the decision tree.
    """

  def classify(self, train_data: List[List[float]], train_label: List[int], test_data: List[List[float]]) -> List[int]:
    """
    Classify the test data using a decision tree model built from the provided training data and labels.
    This method first fits the decision tree model using the provided training data and labels by calling the
    'fit()' method.

    Parameters:
    train_data (List[List[float]]): A nested list of floating point numbers representing the training data.
    train_label (List[int]): A list of integers representing the class labels for each data point in the training set.
    test_data (List[List[float]]): A nested list of floating point numbers representing the test data.

    Returns:
    List[int]: A list of integer predictions, which are the label predictions for the test data after fitting
               the train data and labels to a decision tree.
    """
    self.fit(train_data, train_label)
    return [self.predict_single(self.root, x) for x in test_data]

  """
  Students are encouraged to implement as many additional methods as they find helpful in completing
  the assignment. These methods can be implemented either as class methods of the Solution class or as
  global methods, depending on design preferences.

  For instance, one essential method that must be implemented is a method to build out the decision tree recursively.
  """
  def entropy(self, labels):
    total = len(labels)
    if total == 0:
      return 0.0
    count = {}
    for l in labels:
      count[l] = count.get(l, 0) + 1
    ent = 0.0
    for c in count.values():
      p = c / total
      ent -= p * math.log2(p)
    return ent
  
  def predict_single(self, node, x):
    while node.split_dim != -1:
        if x[node.split_dim] <= node.split_point:
            node = node.left
        else:
            node = node.right
    return node.label


  def build_tree(self, node, data, labels, depth):
    node.label = self.majority_label(labels)
    if depth == 2 or len(set(labels)) == 1:
      return
    current_entropy = self.entropy(labels)
    best_info = float('inf')
    best_dim = -1
    best_split = None

    for d in range(len(data[0])):
        attr_vals = [row[d] for row in data]
        splits = self.candidate_splits(attr_vals)
        for s in splits:
            info = self.split_info(data, labels, d, s)
            gain = current_entropy - info
            if info < best_info or \
              (info == best_info and d < best_dim) or \
              (info == best_info and d == best_dim and s < best_split):
                best_info = info
                best_dim = d
                best_split = s

    if best_dim == -1:
      return  # No split improves information gain

    node.split_dim = best_dim
    node.split_point = best_split

    left_data, left_labels = [], []
    right_data, right_labels = [], []
    for i in range(len(data)):
      if data[i][best_dim] <= best_split:
          left_data.append(data[i])
          left_labels.append(labels[i])
      else:
          right_data.append(data[i])
          right_labels.append(labels[i])

    node.left = Node()
    self.build_tree(node.left, left_data, left_labels, depth + 1)
    node.right = Node()
    self.build_tree(node.right, right_data, right_labels, depth + 1)

  def majority_label(self, labels):
    count = {}
    for l in labels:
      count[l] = count.get(l, 0) + 1
    max_count = max(count.values())
    majority = [k for k, v in count.items() if v == max_count]
    return min(majority)  # Tie-breaker: smallest label

  def candidate_splits(self, values):
    unique_vals = sorted(set(values))
    splits = []
    for i in range(len(unique_vals) - 1):
      mid = (unique_vals[i] + unique_vals[i + 1]) / 2
      splits.append(mid)
    return splits