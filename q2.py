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
    model = None
    # Implement your code to return the appropriate model with the specified parameters here
    # This is the same as Q1
    # Do NOT change the return statement
    if method == "DecisionTreeClassifier":
        model = DecisionTreeClassifier(max_depth=10, random_state=42)
    elif method == "GaussianNB":
        model = GaussianNB()
    elif method == "LogisticRegression":
        model = LogisticRegression(
            penalty='l2',
            solver='lbfgs',
            random_state=42,
            multi_class='multinomial'
        )
    elif method == "RandomForestClassifier":
        model = RandomForestClassifier(max_depth=15, n_estimators=250, random_state=42)
    elif method == "XGBClassifier":
        model = XGBClassifier(
            max_depth=7,
            random_state=42,
            eval_metric='mlogloss'
        )
    else:
        raise ValueError(f"Unknown method: {method}")
    return model

def my_train_test(method, X, y, pi, k):
    X_arr = np.array(X)
    y_arr = np.array(y)
    n = len(X_arr)
    errors = []
    # Implement your code to construct the list of errors here
    # Do NOT change the return statement
    for _ in range(k):
        # randomly select training indices
        indices = list(range(n))
        train_size = int(pi * n)
        train_idx = random.sample(indices, train_size)
        test_idx = [i for i in indices if i not in train_idx]
        X_train, y_train = X_arr[train_idx], y_arr[train_idx]
        X_test, y_test = X_arr[test_idx], y_arr[test_idx]
        model = get_model(method)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        err = np.mean(preds != y_test)
        errors.append(err)
    return np.array(errors)

if __name__ == "__main__":
    #quick sanity check on the digits dataset
    digits = datasets.load_digits()
    X, y = digits.data, digits.target

    #test each method
    methods = ["DecisionTreeClassifier", "GaussianNB", "LogisticRegression",
               "RandomForestClassifier", "XGBClassifier"]
    for m in methods:
        errs = my_train_test(m, X, y, pi=0.75, k=5)
        print(f"{m}: mean error = {errs.mean():.4f}, std = {errs.std():.4f}")