import numpy as np
from train import train, test

def train_val_split(X, Y, train_percent=0.8):

    '''
        Function takes in the training data as input and returns
        a training validation split based on a given percentage.
    '''

    num_points = X.shape[0]

    train_size = int(num_points * 100 * train_percent // 100)

    inds = np.arange(num_points)
    np.random.shuffle(inds)

    train_inds = inds[:train_size]
    val_inds = inds[train_size: ]

    train_X = X[train_inds, :]
    val_X = X[val_inds, :]

    train_Y = Y[train_inds]
    val_Y = Y[val_inds]

    return train_X, train_Y, val_X, val_Y

def parse_txt(fname, num_features=4, num_targets=1, num_points=1372):
    
    '''
        Read data from a text file and generate arrays 
        ready to be fed into the network as inputs.

        Each line in the text file is separated by a
        newline, and represents a data point.
        Features in a line are separated by blank space 
        and the last data point is the target.

    '''

    X = np.empty((num_points, num_features), dtype=float)
    Y = np.empty(num_points, dtype=int)

    with open(fname) as f:
        for index, line in enumerate(f):
            line = line.rstrip('\n')
            data = line.split(',')


            X[index, :] = np.asarray(data[:-1])
            Y[index] = np.asarray(data[num_features])

    return X, Y

epochs = 100
lr = 0.1

X, Y = parse_txt('data/data_banknote_authentication.txt')
train_X, train_Y, val_X, val_Y = train_val_split(X, Y)

params_w, params_b = train(train_X.T, train_Y.T, epochs, lr)

#uncomment following line for testing the model

# test(val_X.T, val_Y.T)