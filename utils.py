import numpy as np


# How to generate batch?
# Do we want to naively iterate through all examples
# Or maybe sample uniformly from each class
def batch_generator(data, batch_size):
    """
    Generates the next batch
    """
    X, y = data

    if X.shape[0] != y.shape[0]:
        raise Exception("non matching dimensions for X ({}) and y ({})".format(
            X.shape[0], y.shape[0]))

    size = X.shape[0]
    
    i = 0
    while True:
        if i + batch_size <= size:
            yield X[i:i + batch_size], y[i:i + batch_size]
            i += batch_size
        else:
            if i < size:
                to_yield = X[i:size], y[i:size]
                num_left_to_yield = batch_size - (size - i)
                
                i = 0
                yield (np.concatenate((to_yield[0], X[i:i + num_left_to_yield]), axis=0),
                    np.concatenate((to_yield[1], y[i:i + num_left_to_yield]), axis=0))
                i += num_left_to_yield

            else:
                i = 0

    

def load_word_vectors(path):
    word_vectors = np.load(path)
    return word_vectors

