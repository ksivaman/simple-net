import numpy as np
from network import backward_pass, param_updates, cross_entropy_loss, forward_pass, accuracy_metric

def train(train_X, train_Y, nn_architecture, epochs, lr):
    # initiation of neural net parameters
    params_w, params_b = init()

    losses = []
    accuracies = []
    
    # performing calculations for subsequent iterations
    for i in range(epochs):
        # step forward
        y_pred, activations, outputs = forward_pass(X, params_values, nn_architecture)
        
        # calculating metrics and saving them in history
        loss = cross_entropy_loss(y_pred, train_Y)
        losses.append(loss)
        accuracy = accuracy_metric(y_pred, train_Y)
        accuracies.append(accuracy)
        
        # step backward - calculating gradient
        gradients = backward_pass(y_pred, train_Y, activations, outputs, params_w, params_b)

        # updating model state
        params_values = param_updates(params_w, params_b, gradients, lr)
        
        print('Loss for epoch {} : {}, accuracy is {}'.format())

    return params_w, params_b