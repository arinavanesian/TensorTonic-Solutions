import numpy as np
def min_max_scaling(data):
    """
    Scale each column of the data matrix to the [0, 1] range.
    """
    # Write code here
    data = np.asarray(data, dtype =float)
    x_min = np.min(data, axis = 0, keepdims = True)
    x_max =np.max(data, axis = 0, keepdims = True)
    diff = x_max -x_min
    diff[diff==0] = 1.0
    return ((data - x_min)/diff).tolist()
    