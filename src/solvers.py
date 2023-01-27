"""
Contains sparsity promoting solvers
1. STRidge
"""
from sklearn import preprocessing
import numpy as np

def STRidge(X, y, lam, tol, maxit=1000, W=None, standardize = False, print_flag = False, \
            thresh_nonzero = None):
    """
    Sequential Threshold Ridge Regression algorithm.
    NOTE: this assumes y is single column
    thresh_nonzero: vector which is the same length as columns in X0 which has
    0 where the feature is not to be thresholded and 1 where it is thresholded
    """

    n,d = X.shape

    #Data standardiation: important for ridge regression because the penalty on the coefficients is uniform
    #Make columns of X to be zero mean and unit variance
    #Make y data to be zero mean (but not unit variance)
    if standardize:
        X_std = preprocessing.scale(X)
        y_std = preprocessing.scale(y, with_std=False)
    else:
        X_std = X
        y_std = y

    #set default weights matrix
    if W is None:
        W = np.ones((d,1))
    else:
        #keep only the diagonal of the weights Matrix
        W = np.diagonal(W).reshape(d,1)

    #set default threshold vector (all terms one if all the coefficients are to be included
    # in the thresholding)
    if thresh_nonzero is None:
        thresh_nonzero = np.ones((d,1))

    # Get the standard ridge esitmate
    w = np.linalg.lstsq(X_std.T@X_std + lam*np.diag(W)@np.diag(W), X_std.T@y_std, rcond=-1)[0]
    num_relevant = d

    # Thresholding loop
    for j in range(maxit):

        if print_flag:
            print("iter, num_relev = ", j, num_relevant)

        # Figure out which items to cut out
        smallinds = np.where(abs(w) - tol*thresh_nonzero < 0)[0]
        biginds = [i for i in range(d) if i not in smallinds]

        # If nothing changes then stop
        if num_relevant == len(biginds):
            if print_flag: print("breaking")
            break
        else:
            num_relevant = len(biginds)

        # Also make sure we didn't just lose all the coefficients
        if len(biginds) == 0:
            if j == 0:
                if print_flag: print("All coeffs < tolerance at 1st iteration!")
                return np.zeros((d,1))
            else:
                if print_flag: print("All coeffs < tolerance at %i iteration!" %(j))
                #break -- this statement will just keep coefficients at previous iteration
                return np.zeros((d,1)) # -- this statement will just return zeros

        # New guess
        w[smallinds] = 0
        w[biginds] = np.linalg.lstsq(X_std[:, biginds].T@X_std[:, biginds] + lam*np.diag(W[biginds])@np.diag(W[biginds]),X_std[:, biginds].T@y_std, rcond=-1)[0]


    # Now that we have the sparsity pattern, use standard least squares to get w
    if biginds != []: w[biginds] = np.linalg.lstsq(X[:, biginds],y, rcond=-1)[0]

    return w
