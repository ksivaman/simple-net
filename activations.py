import numpy as np

#sigmoid activation
def sigmoid(input):
    return 1/(1 + np.exp(-input))

#relu activation
def relu(input):
    return np.maximum(input, 0)

#derivate of a sigmoid w.r.t. input
def d_sigmoid(d_init, out):
    sig = sigmoid(out)
    return d_init * sig * (1 - sig)

#derivate of a relu w.r.t. input
def d_relu(d_init, out):
    d = np.array(d_init, copy = True)
    d[out < 0] = 0.
    return d
