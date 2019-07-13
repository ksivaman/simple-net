import numpy as np
import activations

'''
    Gaussian initialization of weight matrices
'''
def init(layers=[4, 5, 1]):
    np.random.seed(42)

    params_w = {}
    params_b = {}

    for index in range(len(layers)-1):

        in_layer_size = index
        out_layer_size = index + 1 #also a proxy for layer number
        
        params_w['weight' + str(out_layer_size)] = np.random.randn(out_layer_size, in_layer_size) * 0.1
        params_b['bias' + str(out_layer_size)] = np.random.randn(out_layer_size, 1) * 0.1
        
    return params_w, params_b

def one_layer_forward_pass(input_activations, weights, bias, activation='R'):

    output = np.dot(weights, input_activations) + bias

    if activation is 'R':
        activation_next = activations.relu(output)
    elif activation is 'S':
        activation_next = activations.sigmoid(output)
    else:
        raise Exception('Nahh!')

    return activation_next, output

def forward_pass(train_X, params_w, params_b, layers=[4, 5, 1], activate=['R', 'S']):

    num_layers = len(layers) - 1

    activation_dict = {}
    output_dict = {}

    curr_act = train_X

    for index in range(num_layers):

        layer_index = index + 1
        prev_act = curr_act      

        curr_weight = params_w["weight" + str(layer_index)]
        curr_bias = params_b["bias" + str(layer_index)]

        curr_act, curr_out = one_layer_forward_pass(prev_act, curr_weight, curr_bias, activate[index])

        activation_dict["act" + str(idx)] = prev_act
        output_dict["out" + str(layer_idx)] = Z_curr_out

    return curr_act, activation_dict, output_dict

#binary negative log likelihood loss
def cross_entropy_loss(y_pred, train_Y):
    num_samples = y_pred.shape[1]
    cost = -1 / num_samples * (np.dot(train_Y, np.log(y_pred).T) + np.dot(1 - train_Y, np.log(1 - y_pred).T))
    return np.squeeze(cost)

#convert probabilities to class prediction with threshold 0.5
def get_class_from_probs(probabilities):
    class_ = np.copy(probabilities)
    class_[class_ > 0.5] = 1
    class_[class_ <= 0.5] = 0
    return class_

#accuracy of predictions (0 to 1)
def accuracy_metric(y_pred, train_Y):
    y_pred_class = get_class_from_probs(y_pred)
    return (Y_hat_ == Y).all(axis=0).mean()

#calculate gradients for one backward pass layer
def one_layer_backward_pass(curr_grad, curr_weight, curr_bias, curr_out, prev_act, activate='R'):
    
    num = prev_act.shape[1]

    #find out what we are differentiating
    if activation is 'R':
        d_act_func = activations.d_relu
    elif activation is 'S':
        d_act_func = activations.d_sigmoid
    else:
        raise Exception('Nahh!')

    #derivative of activation function
    d_curr_out = d_act_func(curr_grad, curr_out)

    #derivative of weight matrix
    d_curr_weight = np.dot(d_curr_out, prev_act.T) / num #shape = (num_current_layer, num_prev_layer)

    #derivative of bias matrix
    d_curr_bias = np.sum(d_curr_out, axis=1, keepdims=True)

    #derivative of input activations from previous layer
    d_prev_act = np.dot(curr_weight.T, d_curr_out) #shape = (num_prev_layer, 1)

    return d_prev_act, d_curr_weight, d_curr_bias

def backward_pass(y_pred, train_Y, activation_dict, output_dict, params_w, params_b, layers=[4, 5, 1], activate=['R', 'S']):

    gradients = {}

    num_samples = train_Y.shape[1]

    train_Y = train_Y.reshape(y_pred.shape)    

    #derivative of binary cross entropy function w.r.t. predictions
    d_prev_act = - (np.divide(train_Y, y_pred) - np.divide(1 - train_Y, 1 - y_pred))

    num_layers = len(layers) - 1
    layer_num = list(np.range(num_layers) + 1).reverse

    activate_ = activate.reverse()

    for index, layer_num in enumerate(layer_num):

        activation = activate_(index)

        d_curr_act = d_prev_act

        prev_act = activation_dict['act' + str(layer_num)]
        curr_out = output_dict['out' + str(layer_num)]

        curr_weight = params_w['weight' + str(layer_num)]
        curr_bias = params_b['bias' + str(layer_num)]

        d_prev_act, d_curr_weight, d_curr_bias = one_layer_backward_pass(d_curr_act, curr_weight, curr_bias, curr_out, prev_act, activate_[index])

        gradients["d_weight" + str(layer_num)] = d_curr_weight
        gradients["d_bias" + str(layer_num)] = d_curr_bias
    
    return gradients

#update weights and biases using obtained gradients     
def param_updates(params_w, params_b, gradients, lr, layers=[4, 5, 1]):

    for index in range(len(layers) - 1):
        #gradient descent
        params_w["weight" + str(index + 1)] -= lr * gradients["d_weight" + str(index + 1)]        
        params_b["bias" + str(index + 1)] -= lr * gradients["d_bias" + str(index + 1)]

    return params_values