import numpy as np
from network import *

def train(train_X, train_Y, epochs, lr, layers=[4, 5, 1], activate=['R', 'S']):
    # initiation of neural netowrk parameters
    params_w, params_b = init(layers)

    losses = []
    accuracies = []
    
    # performing calculations for subsequent iterations
    for i in range(epochs):
        # step forward
        y_pred, activations, outputs = forward_pass(train_X, params_w, params_b, layers, activate)
        
        # monitor loss and accuracy and keep a record of them.
        loss = cross_entropy_loss(y_pred, train_Y)
        losses.append(loss)
        accuracy = accuracy_metric(y_pred, train_Y)
        accuracies.append(accuracy)
        
        # back prop to calculate the gradients
        gradients = backward_pass(y_pred, train_Y, activations, outputs, params_w, params_b)

        # update the weights and biases
        params_w, params_b = param_updates(params_w, params_b, gradients, lr)
        
        print('Loss for epoch {} : {}, accuracy is {}'.format(i+1, loss, accuracy))

    return params_w, params_b

def test(val_X, val_Y, layers=[4, 5, 1], activate=['R', 'S']):
    # initiation of neural net parameters
    params_w, params_b = init(layers)

    accuracies = []

    # step forward
    y_pred, activations, outputs = forward_pass(val_X, params_w, params_b, layers, activate)
    
    # calculating metrics and saving them in history
    accuracy = accuracy_metric(y_pred, val_Y)
    accuracies.append(accuracy)
    
    print('Accuracy is {}'.format(accuracy))