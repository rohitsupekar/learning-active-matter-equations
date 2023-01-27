import numpy as np
from sklearn.preprocessing import scale
from sklearn.preprocessing import StandardScaler
import matplotlib
import matplotlib.pyplot as plt

def find_error(X, y, w):
    """
    Returns || Xw - y ||_2^2 (squared error)
    """
    return np.linalg.norm(X@w - y, ord=2)**2

def get_lambda_lims(X, y, eps):
    """
    Returns the limits of the regularization path:
    lambda_max = (1/N) max(X.T @ y)
    lambda_min = eps * lambda_max
    """
    n = X.shape[0]
    lambda_max = (1/n)*np.linalg.norm(X.T @ y, np.inf)
    lambda_min = eps*lambda_max

    return lambda_min, lambda_max

def scale_X_y(X,y):
    """
    Scales columns of matrix X to have zero mean and unit variance
    Scales column vector y to have zero mean
    """

    #scaledX = scale(X)
    #scaledy = scale(y, with_std=False)

    scalerX = StandardScaler()
    scalery = StandardScaler(with_std=False)

    scaledX = scalerX.fit_transform(X)
    scaledy = scalery.fit_transform(y)

    return scaledX, scaledy


def pde_string(w, rhs_description, ut = 'u_t', print_imag=False):
    """
    Prints a pde based on:
    w: weights vector
    rhs_description: a list of strings corresponding to the entries in w
    ut: string descriptor of the time derivative
    print_imag: whether to print the imaginary part of the weights
    Returns:
    pde: string with the pde
    """
    pde = ut + ' = '
    first = True
    for i in range(len(w)):
        if w[i] != 0:
            if not first:
                pde = pde + ' + '
            if print_imag==False:
                pde = pde + "(%.5f)" % (w[i].real) + rhs_description[i] + "\n "
            else:
                pde = pde + "(%.5f %0.5fi)" % (w[i].real, w[i].imag) + rhs_description[i] + "\n "
            first = False

    return pde

def is_pareto_efficient(costs):
    """
    Find the pareto-efficient points
    :param costs: An (n_points, n_costs) array
    :return: A (n_points, ) boolean array, indicating whether each point is Pareto efficient
    """
    is_efficient = np.ones(costs.shape[0], dtype = bool)
    for i, c in enumerate(costs):
        if is_efficient[i]:
            is_efficient[is_efficient] = np.any(costs[is_efficient]<c, axis=1)  # Keep any point with a lower cost
            is_efficient[i] = True  # And keep self
    return is_efficient

def print_pde(w, rhs_description, ut = 'u_t', print_imag=False):
    """Prints the pde string"""
    print(pde_string(w, rhs_description, ut, print_imag=False))

def find_pareto_front(obj1, obj2, order_axis=1, plot_fig=True, \
        xlabel='Log(loss)', ylabel='Complexity', file_name='pareto_front.pdf'):
    """
    Plots the Pareto front between values the lists of obj1 and obj2
    INPUT:
    obj1, obj2: Objectives over which to find the Pareto front
    order_axis: which axis to to use for order the indices
    plot_fig: True or False to make a figure
    pareto_file: location for saving the figure
    xlabel, ylabel: labels for the axes
    Returns:
    inds of the pareto front sorted according to the order_axis
    """

    #find the pareto front
    obj1_col = np.expand_dims(np.array(obj1), axis=1)
    obj2_col = np.expand_dims(np.array(obj2), axis=1)

    costs = np.hstack([obj1_col, obj2_col])
    inds = is_pareto_efficient(costs)
    pareto_obj1 = obj1_col[inds].flatten()
    pareto_obj2 = obj2_col[inds].flatten()

    pareto_inds = np.arange(0, costs.shape[0], dtype=int)[inds]

    if plot_fig:
        plt.figure(figsize=(8,3), dpi=300)
        plt.subplot(121)
        plt.scatter(obj1, obj2, 10, color='k')
        plt.title('All the solutions', fontsize=10)
        plt.xlabel(xlabel); plt.ylabel(ylabel)

        plt.subplot(122)
        plt.scatter(pareto_obj1, pareto_obj2, 10, color='k')
        plt.title('Pareto Front')
        plt.xlabel(xlabel); plt.ylabel(ylabel)

        plt.tight_layout()
        plt.savefig(file_name)

    #Order the PDEs as per the error and print
    if order_axis==1:
        inds = np.argsort(pareto_obj1)
    else:
        inds = np.argsort(pareto_obj2)

    sorted_pareto_inds = pareto_inds[inds]

    return sorted_pareto_inds

def find_relaxed_intersection(*sets, q=0):
    """
    This function finds the q-relaxed intersection set of the sets supplied in
    *sets as a list.
    """
    n = len(sets)
    #form union
    union = set.union(*sets)
    q_relaxed_set = []
    score = []

    for i, elem in enumerate(union):
        count = 0
        for s in sets:
            if elem not in s: count += 1
        if count <= q:
            q_relaxed_set.append(elem)
            score.append(1-count/n)

    return q_relaxed_set, score

def find_IC(sq_error, complexity, n, use='aic'):
    """
    Find AIC or BIC using lists for error and complexity
    n: number of data points
    """
    IC = []
    for (RSS, k) in zip(sq_error, complexity):
        if use=='aic':
            #IC = n*np.log(RSS/n) + 2*k + 2*(k)*(k+1)/(n-k-1)
            ic = n*np.log(RSS/n) + 2*k
        else:
            ic = n*np.log(RSS/n) + 2*k*np.log(n)

        IC.append(ic)

    return IC
