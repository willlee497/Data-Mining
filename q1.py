import random
import math
import numpy as np
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import datasets

# Do not import any other libraries

def get_model(method):
    """
    return classifier
    method: one of "DecisionTreeClassifier", "GaussianNB", "LogisticRegression",
            "RandomForestClassifier", "XGBClassifier".
    """
    model = None
    # Implement your code to return the appropriate model with the specified parameters here
    # Do NOT change the return statement
    if method == "DecisionTreeClassifier":
        model = DecisionTreeClassifier(max_depth=10, random_state=42)
    elif method == "GaussianNB":
        model = GaussianNB()
    elif method == "LogisticRegression":
        model = model = LogisticRegression(
        penalty='l2',
        solver='lbfgs',
        random_state=42,
        max_iter=1000
    )
    elif method == "RandomForestClassifier":
        model = RandomForestClassifier(max_depth=15, n_estimators=250, random_state=42)
    elif method == "XGBClassifier":
        model = XGBClassifier(max_depth=7, random_state=42, eval_metric='mlogloss')
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return model

def get_splits(n, k, seed):
    """
    partition indices [0..n-1] into k disjoint, almos equal sized lists, 
    randomized by seed
    """
    splits = None
    # Implement your code to construct the splits here
    # Do NOT change the return statement
    idx = list(range(n))
    rnd = random.Random(seed)
    rnd.shuffle(idx)
    #determine fold sizes
    base = n // k
    remainder = n % k
    splits = []
    start = 0
    for i in range(k):
        size = base + (1 if i < remainder else 0)
        splits.append(idx[start:start+size])
        start += size
    return splits

def my_cross_val(method, X, y, splits):
    """
    perform k fold cross validation using pre computed splits
    return error rates (# wrong / total) for each 
    """
    errors = []
    # Implement your code to construct the list of errors here
    # Do NOT change the return statement
    X_arr = np.array(X)
    y_arr = np.array(y)
    for fold in splits:
        test_idx = fold
        # training indices are all others
        train_idx = [i for i in range(len(X_arr)) if i not in test_idx]
        X_train, y_train = X_arr[train_idx], y_arr[train_idx]
        X_test, y_test = X_arr[test_idx], y_arr[test_idx]
        model = get_model(method)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        err = np.mean(preds != y_test)
        errors.append(err)
    
    return np.array(errors)

if __name__ == "__main__":
    #sanity check using digits dataset and 5fold splits
    digits = datasets.load_digits()
    X, y = digits.data, digits.target
    n_samples = len(X)
    k = 5
    seed = 42

    splits = get_splits(n_samples, k, seed)
    methods = ["DecisionTreeClassifier", "GaussianNB", "LogisticRegression",
               "RandomForestClassifier", "XGBClassifier"]
    for m in methods:
        errs = my_cross_val(m, X, y, splits)
        print(f"{m}: mean error = {errs.mean():.4f}, std = {errs.std():.4f}")