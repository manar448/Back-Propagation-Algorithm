import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# Initialize Weight and bias
def Generate_weights_bias(hidden_layers, neurons, num_features, num_classes, bias):
    np.random.seed(42)
    weights = []
    biases = []
    num_input_neurons = num_features

    for num_Of_layer in range(hidden_layers + 1):
        # if user need to add bias
        if bias == 1:
            # Input layer to the first hidden layer
            if num_Of_layer == 0:
                matrix_weight = np.random.rand(num_input_neurons, neurons[0])
                new_bias = np.random.rand(neurons[0])
                num_input_neurons = neurons[0]

            # Hidden layers
            elif num_Of_layer < hidden_layers:
                matrix_weight = np.random.rand(num_input_neurons, neurons[num_Of_layer])
                new_bias = np.random.rand(neurons[num_Of_layer])
                num_input_neurons = neurons[num_Of_layer]

            # Last hidden layer to output layer
            elif num_Of_layer == hidden_layers:
                matrix_weight = np.random.rand(num_input_neurons, num_classes)
                new_bias = np.random.rand(num_classes)
        # if user don't need to add bias
        else:
            # Input layer to the first hidden layer
            if num_Of_layer == 0:
                matrix_weight = np.random.rand(num_input_neurons, neurons[0])
                new_bias = np.zeros((1, neurons[0]))
                num_input_neurons = neurons[0]

            # Hidden layers
            elif num_Of_layer < hidden_layers:
                matrix_weight = np.random.rand(num_input_neurons, neurons[num_Of_layer])
                np.zeros((1, neurons[num_Of_layer]))
                new_bias = np.zeros((1, neurons[num_Of_layer]))
                num_input_neurons = neurons[num_Of_layer]

            # Last hidden layer to output layer
            elif num_Of_layer == hidden_layers:
                matrix_weight = np.random.rand(num_input_neurons, num_classes)
                new_bias = np.zeros((1, num_classes))

        weights.append(matrix_weight)
        biases.append(new_bias)

    return weights, biases


# Activation Functions (Sigmoid, Hyperbolic Tangent)
def activation_func(activation, x):
    # Choice of activation function
    if activation == 'Sigmoid':
        return 1 / (1 + np.exp(- 0.6 * x))
    elif activation == 'Hyperbolic Tangent':
        return np.tanh(x)


# Derivative of Activation Functions
def activation_func_derivative(activation, x):
    if activation == 'Sigmoid':
        return activation_func(activation, x) * (1 - activation_func(activation, x))
    elif activation == 'Hyperbolic Tangent':
        return 1 - np.tanh(x) ** 2


# Algorithm
def Backpropagation_Algo(X_train, y_train, weights, bias, learning_rate, activation_function):
    num_layers = len(weights)
    output_layer = X_train
    inputs_layer = []
    outputs = [output_layer]

    # first forward step
    for i in range(num_layers):
        # calc net
        layer_input = np.dot(output_layer, weights[i]) + bias[i]
        inputs_layer.append(layer_input)
        output_layer = activation_func(activation_function, layer_input)
        outputs.append(output_layer)

    # backward step
    # calculate local gradient for output layer
    error = y_train - output_layer
    gradient = [error * activation_func_derivative(activation_function, inputs_layer[-1])]

    # calculate gradient for each hidden layer
    for i in range(num_layers - 1):
        delta_i = np.dot(gradient[-1], weights[-i - 1].T) * activation_func_derivative(activation_function,
                                                                                       inputs_layer[-i - 2])
        gradient.append(delta_i)

    gradient.reverse()

    # update weights and bias
    for i in range(num_layers):
        weights[i] += learning_rate * np.dot(outputs[i].T, gradient[i])
        bias[i] += learning_rate * np.sum(gradient[i], axis=0)

    return weights, bias


# forward step
def forward(X, weights, bias, activation_function):
    output_layer = X
    outputs = []
    # two loops that loop the weights and bias lists
    for i_weight, i_bias in zip(weights, bias):
        input_layer = np.dot(output_layer, i_weight) + i_bias
        output_layer = activation_func(activation_function, input_layer)
        outputs.append(output_layer)

    return outputs


# Train Algorithm
def train_algo(X_train, y_train, num_features, num_classes, num_hidden_layers, neurons, learning_rate,
               epochs, activation_function, bias):
    # Initialize weight and bias
    weights, biases = Generate_weights_bias(num_hidden_layers, neurons, num_features, num_classes, bias)

    # loop for each epoch
    for epoch in range(epochs):
        weights, biases = Backpropagation_Algo(X_train, y_train, weights, biases, learning_rate, activation_function)

    return weights, biases


# Predict outputs
def predict(X_test, weights, bias, activation_function):
    outputs = forward(X_test, weights, bias, activation_function)[-1]
    y_predict = np.argmax(outputs, axis=1)

    return y_predict


# Calculate Accuracy and show confusion matrix
def Calc_accuracy(num_classes, y_predict, y_actual):
    # Initialize the confusion matrix with zeros
    confusion_matrix = np.zeros((num_classes, num_classes))

    # Loop about each sample in the dataset
    for i in range(len(y_actual)):
        actual_class = y_predict[i]
        predicted_class = y_actual[i]

        if predicted_class == actual_class:
            confusion_matrix[actual_class - 1][predicted_class - 1] += 1

    # Print the confusion matrix
    print("Confusion Matrix:")
    print(confusion_matrix)
    print("\n")

    # Calculate accuracy
    accuracy = np.trace(confusion_matrix) / len(y_actual)

    print("Accuracy: {:.2f}%".format(accuracy * 100))

    # Create confusion matrix dataframe
    confusion_matrix_df = pd.DataFrame(confusion_matrix,
                                       index=['Actual Class 1', 'Actual Class 2', 'Actual Class 3'],
                                       columns=['Predicted Class 1', 'Predicted Class 2', 'Predicted Class 3'])

    # Plot heatmap
    sns.heatmap(confusion_matrix_df, annot=True, cmap='YlGnBu')
    plt.show()

    return accuracy
