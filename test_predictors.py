import numpy as np
from decision_tree import calculate_information_gain, decision_tree_train, decision_tree_predict
from naive_bayes import naive_bayes_train, naive_bayes_predict
from mlp import mlp_train, mlp_predict, logistic, nll
from kernelsvm import kernel_svm_train, kernel_svm_predict
from crossval import cross_validate
from load_all_data import load_all_data
import copy

def multi_class_to_binary(labels):
    n = labels.shape[0]
    keys = np.unique(labels)
    num_keys = keys.shape[0]
    bin_labels = np.ones((n, num_keys)) * (-1)
    for i in range(num_keys):
        bin_labels[:, i] = (labels == keys[i]) * 2 - 1
    return keys, bin_labels

num_words, num_training, num_testing, train_data, test_data, train_labels, test_labels = load_all_data()

d = 5000 # maximum number of features

# Filter features by information gain
gain = calculate_information_gain(train_data, train_labels)

# sort features by calculated information gain
ranks = gain.argsort()[::-1]
train_data = train_data[ranks[:d], :]
test_data = test_data[ranks[:d], :]

# Try naive Bayes with hard-coded alpha value
nb_params = {'alpha': 1.0}
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
dt_params = {'max_depth': 16}
dt_model = decision_tree_train(train_data, train_labels, dt_params)

# Compute training accuracy
dt_train_predictions = decision_tree_predict(train_data, dt_model)
dt_train_accuracy = np.mean(dt_train_predictions == train_labels)
print("Decision tree training accuracy: %f" % dt_train_accuracy)

# Compute testing accuracy
dt_test_predictions = decision_tree_predict(test_data, dt_model)
dt_test_accuracy = np.mean(dt_test_predictions == test_labels)
print("Decision tree testing accuracy: %f" % dt_test_accuracy)

num_folds = 5
structures = [[400], [200, 100], [200, 200], [200, 400], [400, 400]]
lambda_vals = [0.01, 0.1, 1]

params = {
    'max_iter': 100,
    'squash_function': logistic,
    'loss_function': nll
}

best_params = []
best_score = 0

train_data = train_data.toarray()
classes, train_labels = multi_class_to_binary(train_labels)
num_class = classes.shape[0]
test_data = test_data.toarray()

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

svm_params = {'kernel': 'rbf', 'C': 1.0, 'sigma': 0.1}
svm_test_scores = np.zeros((test_labels.shape[0], num_class))
for k in range(num_class):
    rbf_svm_model = kernel_svm_train(train_data, train_labels[:, k], svm_params)
    _, svm_test_scores[:, k] = kernel_svm_predict(test_data, rbf_svm_model)
svm_test_predictions = np.argmax(svm_test_scores, axis=1)
svm_test_accuracy = np.mean(svm_test_predictions == test_labels)
print("SVM (RBF kernel) test accuracy: %f" % (svm_test_accuracy))
