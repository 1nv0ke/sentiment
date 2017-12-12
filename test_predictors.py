import numpy as np
import random
import copy
from parse_dictionary import parse_tweet_json
from parse_dataset import get_labels_from_csv, parse_tweets, split_dataset
from decision_tree import calculate_information_gain, decision_tree_train, decision_tree_predict
from naive_bayes import naive_bayes_train, naive_bayes_predict
from mlp import mlp_train, mlp_predict, logistic, nll
from kernelsvm import kernel_svm_train, kernel_svm_predict
from crossval import cross_validate
from scipy.sparse import dok_matrix, csc_matrix

# _________________________________________________________________________________________________

def load_all_data():
    """
    Load data from raw training/test data.

    :return: tuple containing num_words, num_training, num_testing, train_data, test_data, train_labels, test_labels
    :rtype: tuple
    """

    train_data_ijv = np.loadtxt('data/train.data', dtype=int)
    test_data_ijv = np.loadtxt('data/test.data', dtype=int)

    # load labels and convert to zero-indexed
    train_labels = np.loadtxt('data/train.label') + 1
    test_labels = np.loadtxt('data/test.label') + 1

    num_training = train_labels.shape[0]
    num_testing = test_labels.shape[0]

    # convert to zero-indexing
    train_data_ijv[:, :2] -= 1
    test_data_ijv[:, :2] -= 1

    assert train_data_ijv.min() >= 0, "Indexing correction created a negative index"
    assert test_data_ijv.min() >= 0, "Indexing correction created a negative index"

    max_word = max(train_data_ijv[:, 1].max(), test_data_ijv[:, 1].max())

    num_words = max_word + 1

    return num_words, num_training, num_testing, train_data_ijv, test_data_ijv, train_labels, test_labels

# _________________________________________________________________________________________________

def multi_class_to_binary(labels):
    n = labels.shape[0]
    keys = np.unique(labels)
    num_keys = keys.shape[0]
    bin_labels = np.ones((n, num_keys)) * (-1)
    for i in range(num_keys):
        bin_labels[:, i] = (labels == keys[i]) * 2 - 1
    return keys, bin_labels

# _________________________________________________________________________________________________

def test_all_classifiers():

    num_words, num_training, num_testing, train_data_ijv, test_data_ijv, train_labels, test_labels = load_all_data()

    # Set up data matrices for Naive Bayes and Decision Tree
    train_data = dok_matrix((num_words, num_training), dtype=bool)
    for (col, row, val) in train_data_ijv:
        train_data[row, col] = (val > 0)

    test_data = dok_matrix((num_words, num_testing), dtype=bool)
    for (col, row, val) in test_data_ijv:
        test_data[row, col] = (val > 0)

    train_data = csc_matrix(train_data)
    test_data = csc_matrix(test_data)

    d = None # maximum number of features

    # Filter features by information gain
    gain = calculate_information_gain(train_data, train_labels)

    # sort features by calculated information gain
    ranks = gain.argsort()[::-1]
    train_data = train_data[ranks[:d], :]
    test_data = test_data[ranks[:d], :]

    # Try naive Bayes with hard-coded alpha value
    nb_params = {'alpha': 0.1}
    nb_model = naive_bayes_train(train_data, train_labels, nb_params)

    # Compute training accuracy
    nb_train_predictions = naive_bayes_predict(train_data, nb_model)
    nb_train_accuracy = np.mean(nb_train_predictions == train_labels)
    print("Naive Bayes training accuracy: %f" % nb_train_accuracy)

    # Compute testing accuracy
    nb_test_predictions = naive_bayes_predict(test_data, nb_model)
    nb_test_accuracy = np.mean(nb_test_predictions == test_labels)
    print("Naive Bayes testing accuracy: %f" % nb_test_accuracy)

    # Try decision tree with hard-coded maximum depth
    dt_params = {'max_depth': 32}
    dt_model = decision_tree_train(train_data, train_labels, dt_params)

    # Compute training accuracy
    dt_train_predictions = decision_tree_predict(train_data, dt_model)
    dt_train_accuracy = np.mean(dt_train_predictions == train_labels)
    print("Decision tree training accuracy: %f" % dt_train_accuracy)

    # Compute testing accuracy
    dt_test_predictions = decision_tree_predict(test_data, dt_model)
    dt_test_accuracy = np.mean(dt_test_predictions == test_labels)
    print("Decision tree testing accuracy: %f" % dt_test_accuracy)


    # Set up data matrices for Multi-layer Perceptron and kernel SVM
    train_data = np.zeros((num_words, num_training), dtype=int)
    for (col, row, val) in train_data_ijv:
        train_data[row, col] = val
    test_data = np.zeros((num_words, num_testing), dtype=int)
    for (col, row, val) in test_data_ijv:
        test_data[row, col] = val
    classes, train_labels = multi_class_to_binary(train_labels)
    num_class = classes.shape[0]

    num_folds = 4
    structures = [[20]] #[[100], [100, 10], [10, 10], [10, 100], [100, 100]]
    lambda_vals = [0.1]#[0.01, 0.1, 1]

    params = {
        'max_iter': 400,
        'squash_function': logistic,
        'loss_function': nll
    }

    best_params = []
    best_score = 0

    cv_score = np.zeros((num_class,))
    for i in range(len(structures)):
        for j in range(len(lambda_vals)):
            params['num_hidden_units'] = structures[i]
            params['lambda'] = lambda_vals[j]
            for k in range(num_class):
                cv_score[k], models = cross_validate(mlp_train, mlp_predict, train_data, train_labels[:, k], num_folds, params)
            if np.mean(cv_score) > best_score:
                best_score = np.mean(cv_score)
                best_params = copy.copy(params)

    mlp_test_scores = np.zeros((test_labels.shape[0], num_class))
    for k in range(num_class):
        mlp_model = mlp_train(train_data, train_labels[:, k], best_params)
        _, mlp_test_scores[:, k], _, _ = mlp_predict(test_data, mlp_model)
    mlp_test_predictions = np.argmax(mlp_test_scores, axis=1)
    mlp_test_accuracy = np.mean(mlp_test_predictions == test_labels)
    print("MLP testing accuracy: %f" % (mlp_test_accuracy))
    print("with structure %s and lambda = %f" % (repr(best_params['num_hidden_units']), best_params['lambda']))

    svm_params = {'kernel': 'rbf', 'C': 1.0, 'sigma': 0.01}
    svm_test_scores = np.zeros((test_labels.shape[0], num_class))
    for k in range(num_class):
        rbf_svm_model = kernel_svm_train(train_data, train_labels[:, k], svm_params)
        _, svm_test_scores[:, k] = kernel_svm_predict(test_data, rbf_svm_model)
    svm_test_predictions = np.argmax(svm_test_scores, axis=1)
    svm_test_accuracy = np.mean(svm_test_predictions == test_labels)
    print("SVM (RBF kernel) test accuracy: %f" % (svm_test_accuracy))

# _________________________________________________________________________________________________

if __name__=='__main__':

    # Generate dictionary for bag of words
    parse_tweet_json('./data/raw/tw_with_id.json',
                     './data/dict/words.dict',
                     './data/dict/emojis.dict',
                     max_word_count=2000, max_emoji_count=500)

    # Select test set
    indices = get_labels_from_csv()
    label_cnt = len(indices)
    folds = 4
    testset = sorted(random.sample(range(1, label_cnt + 1), label_cnt / folds))

    # Test without emoji
    print('******Test all classifiers with plain words******')
    parse_tweets(indices=indices, include_emoji=False)
    split_dataset(testset=testset)
    test_all_classifiers()

    # Test with emoji
    print('******Test all classifiers with plain words and emoji******')
    parse_tweets(indices=indices, include_emoji=True)
    split_dataset(testset=testset)
    test_all_classifiers()

# _________________________________________________________________________________________________
