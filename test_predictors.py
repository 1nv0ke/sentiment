import numpy as np
from decision_tree import calculate_information_gain, decision_tree_train, decision_tree_predict
from naive_bayes import naive_bayes_train, naive_bayes_predict
from load_all_data import load_all_data

num_words, num_training, num_testing, train_data, test_data, train_labels, test_labels = load_all_data()

d = 5000 # maximum number of features

# Filter features by information gain
gain = calculate_information_gain(train_data, train_labels)
print gain

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
