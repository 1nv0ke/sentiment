"""
This module includes methods for training and predicting using decision trees.
"""
from __future__ import division
import numpy as np


def calculate_information_gain(data, labels):
    """
    Computes the information gain on label probability for each feature in data
    
    :param data: d x n matrix of d features and n examples
    :type data: ndarray
    :param labels: n x 1 vector of class labels for n examples
    :type labels: array
    :return: d x 1 vector of information gain for each feature (H(y) - H(y|x_d))
    :rtype: array
    """
    all_labels = np.unique(labels)
    num_classes = len(all_labels)

    class_count = np.zeros(num_classes)

    d, n = data.shape

    full_entropy = 0
    for c in range(num_classes):
        class_count[c] = np.sum(labels == all_labels[c])
        if class_count[c] > 0:
            class_prob = class_count[c] / n
            full_entropy -= class_prob * np.log(class_prob)

    # print("Full entropy is %d\n" % full_entropy)

    gain = full_entropy * np.ones((1, d))
    num_x = np.asarray(data.sum(1)).T
    prob_x = num_x / n
    prob_not_x = 1 - prob_x

    for c in range(num_classes):
        # print("Computing contribution of class %d." % c)
        num_y = np.sum(labels == all_labels[c])
        # this next line sums across the rows of data, multiplied by the
        # indicator of whether each column's label is c. It counts the number
        # of times each feature is on among examples with label c
        num_y_and_x = np.asarray(data[:, labels == all_labels[c]].sum(1)).T

        # Prevents Python from outputting a divide-by-zero warning
        with np.errstate(invalid='ignore'):
            prob_y_given_x = num_y_and_x / num_x
        prob_y_given_x[num_x == 0] = 0

        nonzero_entries = prob_y_given_x > 0
        if np.any(nonzero_entries):
            with np.errstate(invalid='ignore', divide='ignore'):
                cond_entropy = - np.multiply(np.multiply(prob_x, prob_y_given_x), np.log(prob_y_given_x))
            gain[nonzero_entries] -= cond_entropy[nonzero_entries]

        # The next lines compute the probability of y being c given x = 0 by
        # subtracting the quantities we've already counted
        # num_y - num_y_and_x is the number of examples with label y that
        # don't have each feature, and n - num_x is the number of examples
        # that don't have each feature
        with np.errstate(invalid='ignore'):
            prob_y_given_not_x = (num_y - num_y_and_x) / (n - num_x)
        prob_y_given_not_x[n - num_x == 0] = 0

        nonzero_entries = prob_y_given_not_x > 0
        if np.any(nonzero_entries):
            with np.errstate(invalid='ignore', divide='ignore'):
                cond_entropy = - np.multiply(np.multiply(prob_not_x, prob_y_given_not_x), np.log(prob_y_given_not_x))
            gain[nonzero_entries] -= cond_entropy[nonzero_entries]

    return gain.ravel()


def decision_tree_train(train_data, train_labels, params):
    """
    Trains a decision tree to classify data using the entropy decision criterion.
     
    :param train_data: d x n numpy matrix (ndarray) of d binary features for n examples
    :type train_data: ndarray
    :param train_labels: length n numpy vector with integer labels
    :type train_labels: array_like
    :param params: learning algorithm parameter dictionary. Must include a 'max_depth' value
    :type params: dict
    :return: dictionary encoding the learned decision tree
    :rtype: dict
    """
    max_depth = params['max_depth']

    labels = np.unique(train_labels)
    num_classes = labels.size

    model = recursive_tree_train(train_data, train_labels, depth=0, max_depth=max_depth, num_classes=num_classes)
    return model


def recursive_tree_train(data, labels, depth, max_depth, num_classes):
    """
    Helper function to recursively build a decision tree by splitting the data by a feature
    
    :param data: d x n numpy matrix (ndarray) of d binary features for n examples
    :type data: ndarray
    :param labels: length n numpy array with integer labels
    :type labels: array_like
    :param depth: current depth of the decision tree node being constructed
    :type depth: int
    :param max_depth: maximum depth to expand the decision tree to 
    :type max_depth: int
    :param num_classes: number of classes in the classification problem
    :type num_classes: int
    :return: dictionary encoding the learned decision tree node
    :rtype: dict
    """
    class_count = np.zeros(num_classes)
    d, n = data.shape

    for c in range(num_classes):
        class_count[c] = np.count_nonzero(labels == c)

    prediction = class_count.argmax()
    max_class_count = class_count[prediction]

    node = dict()

    # check if either max_depth was reached or subset is pure
    if depth == max_depth or max_class_count == n:
        node['prediction'] = prediction
        return node

    # otherwise, split data
    gain = calculate_information_gain(data, labels)
    best_word = gain.argmax()
    best_gain = gain[best_word]

    # check if none of the words provide any information gain
    if best_gain == 0:
        node['prediction'] = prediction
        return node

    # split on best_word
    true_feature = np.asarray(data[best_word, :].todense()).ravel() > 0
    true_indices = true_feature.nonzero()[0]
    false_indices = np.logical_not(true_feature).nonzero()[0]

    left_data = data[:, true_indices]
    left_labels = labels[true_indices]
    right_data = data[:, false_indices]
    right_labels = labels[false_indices]

    node['left'] = recursive_tree_train(left_data, left_labels, depth + 1, max_depth, num_classes)
    node['right'] = recursive_tree_train(right_data, right_labels, depth + 1, max_depth, num_classes)
    node['split_feature'] = best_word

    return node


def decision_tree_predict(data, model):
    """
    Predicts most likely label given computed decision tree in model
    
    :param data: d x n ndarray of d binary features for n examples
    :type data: ndarray
    :param model: learned decision tree model
    :type model: dict
    :return: length n numpy array of the predicted class labels
    :rtype: array_like
    """

    d, n = data.shape
    labels = np.zeros(n)

    if 'prediction' in model:
        # at a leaf of the tree
        labels[:] = model['prediction']
    else:
        # recurse further down tree
        true_feature = np.asarray(data[model['split_feature'], :].todense()).ravel() > 0
        left_indices = true_feature.nonzero()[0]
        right_indices = np.logical_not(true_feature).nonzero()[0]
        left_data = data[:, left_indices]
        right_data = data[:, right_indices]

        labels[left_indices] = decision_tree_predict(left_data, model['left'])
        labels[right_indices] = decision_tree_predict(right_data, model['right'])

    return labels

