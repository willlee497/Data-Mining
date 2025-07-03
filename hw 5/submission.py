# Submit this file to Gradescope
from typing import Dict, List, Tuple
# you may use other Python standard libraries, but not data
# science libraries, such as numpy, scikit-learn, etc.
import math, sys

class Solution:

  def confusion_matrix(self, true_labels: List[int], pred_labels: List[int]) -> Dict[Tuple[int, int], int]:
    """Calculate the confusion matrix and return it as a sparse matrix in dictionary form.
    Args:
      true_labels: list of true labels
      pred_labels: list of predicted labels
    Returns:
      A dictionary of (true_label, pred_label): count
    """
    cm: Dict[Tuple[int, int], int] = {}
    for t, p in zip(true_labels, pred_labels):
        key = (t, p)
        cm[key] = cm.get(key, 0) + 1
    return cm

  def jaccard(self, true_labels: List[int], pred_labels: List[int]) -> float:
    """Calculate the Jaccard index.
    Args:
      true_labels: list of true cluster labels
      pred_labels: list of predicted cluster labels
    Returns:
      The Jaccard index. Do NOT round this value.
    """
    n = len(true_labels)
    tp = 0  #pairs in same true and same pred
    union = 0  #pairs in same true or same pred
    for i in range(n):
        for j in range(i + 1, n):
            same_true = (true_labels[i] == true_labels[j])
            same_pred = (pred_labels[i] == pred_labels[j])
            if same_true and same_pred:
                tp += 1
            if same_true or same_pred:
                union += 1
    return tp / union if union > 0 else 0.0   


  def nmi(self, true_labels: List[int], pred_labels: List[int]) -> float:

    """Calculate the normalized mutual information.
    Args:
      true_labels: list of true cluster labels
      pred_labels: list of predicted cluster labels
    Returns:
      The normalized mutual information. Do NOT round this value.
    """
    n = len(true_labels)
    #count occurrences
    true_counts: Dict[int, int] = {}
    pred_counts: Dict[int, int] = {}
    joint_counts: Dict[Tuple[int, int], int] = {}
    for t, p in zip(true_labels, pred_labels):
        true_counts[t] = true_counts.get(t, 0) + 1
        pred_counts[p] = pred_counts.get(p, 0) + 1
        joint_key = (t, p)
        joint_counts[joint_key] = joint_counts.get(joint_key, 0) + 1

    #entropy of true labels
    H_true = 0.0
    for count in true_counts.values():
        prob = count / n
        H_true -= prob * math.log(prob)
    #entropy of predicted labels
    H_pred = 0.0
    for count in pred_counts.values():
        prob = count / n
        H_pred -= prob * math.log(prob)

    #mutual info
    MI = 0.0
    for (t, p), count in joint_counts.items():
        joint_prob = count / n
        pt = true_counts[t] / n
        pp = pred_counts[p] / n
        MI += joint_prob * math.log(joint_prob / (pt * pp))

    #normalize
    if H_true > 0 and H_pred > 0:
        return MI / math.sqrt(H_true * H_pred)
    return 0.0
  

if __name__ == '__main__':
  data = sys.stdin.read().strip().split()
  if not data:
    sys.exit(0)
  it = iter(data)
  test_id = int(next(it))
  true_labels: List[int] = []
  pred_labels: List[int] = []
  for a, b in zip(it, it):
    true_labels.append(int(a))
    pred_labels.append(int(b))
  sol = Solution()
  if test_id == 0:
    #Jaccard 
    print(sol.jaccard(true_labels, pred_labels))
  elif test_id == 1:
    #NMI
    print(sol.nmi(true_labels, pred_labels))
  else:
    #confusion matrix
    cm = sol.confusion_matrix(true_labels, pred_labels)
    for i, j in sorted(cm.keys()):
        print(f"{i} {j} {cm[(i,j)]}")