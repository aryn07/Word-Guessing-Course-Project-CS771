import numpy as np
from collections import Counter
import math

# Helper function to get bigrams from a word


def get_bigrams(word):
    return sorted(set(word[i:i+2] for i in range(len(word) - 1)))[:5]

# Custom Decision Tree Node


class TreeNode:
    def __init__(self, depth=0, max_depth=10, min_samples_split=2):
        self.depth = depth
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.is_leaf = False
        self.prediction = None
        self.best_bigram = None
        self.children = {}

    def set_leaf(self, prediction):
        self.is_leaf = True
        self.prediction = prediction

    def split(self, X, y):
        if self.depth >= self.max_depth or len(set(y)) == 1 or len(y) < self.min_samples_split:
            self.set_leaf(Counter(y).most_common(1)[0][0])
            return

        current_entropy = self.entropy(y)
        best_info_gain = 0
        best_bigram = None

        candidate_bigrams = self.select_candidate_bigrams(X, top_k=10)

        for bigram in candidate_bigrams:
            info_gain = self.information_gain(X, y, bigram, current_entropy)
            if info_gain > best_info_gain:
                best_info_gain = info_gain
                best_bigram = bigram

        if best_bigram is None:
            self.set_leaf(Counter(y).most_common(1)[0][0])
            return

        self.best_bigram = best_bigram
        self.children = {
            0: TreeNode(self.depth + 1, self.max_depth, self.min_samples_split),
            1: TreeNode(self.depth + 1, self.max_depth, self.min_samples_split)
        }
        X_left, y_left, X_right, y_right = self.split_data(X, y, best_bigram)

        if len(y_left) == 0 or len(y_right) == 0:
            self.set_leaf(Counter(y).most_common(1)[0][0])
            return

        self.children[0].split(X_left, y_left)
        self.children[1].split(X_right, y_right)

    def information_gain(self, X, y, bigram, current_entropy):
        X_left, y_left, X_right, y_right = self.split_data(X, y, bigram)
        if len(y_left) == 0 or len(y_right) == 0:
            return 0
        p_left = len(y_left) / len(y)
        p_right = len(y_right) / len(y)
        gain = current_entropy - \
            (p_left * self.entropy(y_left) + p_right * self.entropy(y_right))
        return gain

    def entropy(self, y):
        counter = Counter(y)
        return -sum((count / len(y)) * math.log2(count / len(y)) for count in counter.values())

    def split_data(self, X, y, bigram):
        X_left = [x for x in X if bigram not in x]
        y_left = [y[i] for i, x in enumerate(X) if bigram not in x]
        X_right = [x for x in X if bigram in x]
        y_right = [y[i] for i, x in enumerate(X) if bigram in x]
        return X_left, y_left, X_right, y_right

    def select_candidate_bigrams(self, X, top_k=10):
        bigram_counts = Counter(bigram for x in X for bigram in x)
        return [bigram for bigram, _ in bigram_counts.most_common(top_k)]

    def predict(self, x):
        if self.is_leaf:
            return self.prediction
        if self.best_bigram not in x:
            return self.children[0].predict(x)
        else:
            return self.children[1].predict(x)


class DecisionTree:
    def __init__(self, max_depth=10, min_samples_split=2):
        self.root = TreeNode(max_depth=max_depth,
                             min_samples_split=min_samples_split)

    def fit(self, X, y):
        self.root.split(X, y)

    def predict(self, X):
        return [self.root.predict(x) for x in X]

# Function to fit the model with the given words


def my_fit(words, max_depth=70, min_samples_split=2):
    bigrams = [get_bigrams(word) for word in words]
    tree = DecisionTree(max_depth=max_depth,
                        min_samples_split=min_samples_split)
    tree.fit(bigrams, words)
    return tree

# Function to predict the word(s) based on the given bigram list


def my_predict(tree, bigram_list):
    return tree.predict([bigram_list])
