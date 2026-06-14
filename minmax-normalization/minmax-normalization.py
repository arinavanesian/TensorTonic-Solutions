import numpy as np

def minmax_scale(X, axis=0, eps=1e-12):
    """
    Scale X to [0,1]. If 2D and axis=0 (default), scale per column.
    Return np.ndarray (float).
    """
    # Write code here
    # shape_cols =X.shape[1]
    # for each c in range(shape_cols):
    #     col_data = X[:, x]
    #     x_min = np.min(col_data)
    #     x_max = np.max(col_data)
    #     for each x in X[:, c]:
    #         x =  (x-x_min_)/(x_max - x_min)
    #         X_new.append(x)
    # return X_new
    # X_new = list()
    X = np.asarray(X, dtype = float)
    x_min = np.min(X, axis = axis, keepdims=True)
    x_max = np.max(X, axis =axis,  keepdims=True)
    diff = x_max - x_min
    diff[diff == 0] = eps
    return (X- x_min )/diff
