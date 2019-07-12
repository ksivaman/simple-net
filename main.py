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

        curr_act, curr_out = single_layer_forward_propagation(prev_act, curr_weight, curr_bias, activate[index])

        activation_dict["act" + str(idx)] = prev_act
        output_dict["Z" + str(layer_idx)] = Z_curr_out

    return curr_act, activation_dict, output_dict




