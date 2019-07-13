import numpy as np
# from train import train

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

def train_val_split(train_X, train_Y, train_percent=0.8):

    '''
        Function takes in the training data as input and returns
        a training validation split based on a given percentage.
    '''

train_X, train_Y = parse_txt('data/data_banknote_authentication.txt')

