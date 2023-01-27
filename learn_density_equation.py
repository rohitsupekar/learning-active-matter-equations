import os
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["HDF5_USE_FILE_LOCKING"]="FALSE"

import sys
sys.path.append("src/") # add source
import numpy as np
import h5py
import sys
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy.io as sc
import itertools
import seaborn as sns
import pdelearn as pdel
from funcs import *
from sklearn.model_selection import KFold
import glob
from scipy.fftpack import diff
from tqdm import tqdm

data_path = "data/"
run_name = 'density_equation'
#data file name
file_name = '_reconstruction_cheb_cheb_cheb_space100_time50.mat'

pdelearn_path = '%s/PDElearn' %(data_path)
save_path = '%s/PDElearn/%s' %(data_path, run_name)
if os.path.exists(pdelearn_path) == False: os.mkdir(pdelearn_path)
if os.path.exists(save_path) == False: os.mkdir(save_path)

import logging
#reload
import importlib
importlib.reload(logging)

logging.basicConfig(format='%(asctime)s:%(name)s:%(levelname)s: %(message)s',
                    level=logging.INFO, \
                    handlers=[logging.FileHandler(save_path + '/log.out', mode = 'w'), \
                    logging.StreamHandler()])

logger = logging.getLogger(__name__)
logger.info('Logging Started for run: %s' %(run_name))

########## Load data ##########
#sub_sampling factor
sample_fac = 1

data = h5py.File("%s/%s" %(data_path, file_name), 'r')

x, y = np.array(data['x']).flatten()[::sample_fac], np.array(data['y']).flatten()[::sample_fac]
t = np.array(data['t']).flatten()
dx = x[2] - x[1]
dy = y[2] - y[1]
dt = t[2] - t[1]
nx, ny, nt = len(x), len(y), len(t)

logger.info('Loaded data')

#load all data and change (y, x, t) to (t, y, x) to be compatible with this code
def cax(var):
    """cax = change axes order"""
    return np.moveaxis(var, [0, 1, 2], [-2, -1, -3])

rho = cax(data['rho_'])
px = cax(data['vx_'])
py = cax(data['vy_'])

rho_t = cax(data['rho_t'])
px_t = cax(data['vx_t'])
py_t = cax(data['vy_t'])

#rescale density to convert to area fraction, new density doesn't have unit
Dc = 4.8*10**(-3); #particle diameter
Ac = (np.pi/4)*Dc**2; #particle area
rho = rho*Ac;
rho_t = rho_t*Ac;

##### Set parameters for learning #####
num_data = int(50000)
num_cores = 4
n_folds = 2
n_repeats = 100
order=0
seed0 = np.random.randint(0,100)
stab_thresh = 0.6
num_iters = 100
algo = 'stridge'
nlam1 = 40
nlam2 = 1

if algo != 'stridge':
    nlam2 = 1

##### Set parameters to define the learning data window ####
frac_cut = 0.05 #fraction of domain to cut at the edges

xmin_learn, xmax_learn = frac_cut*np.max(x), (1 - frac_cut)*np.max(x)
ymin_learn, ymax_learn = frac_cut*np.max(y), (1 - frac_cut)*np.max(y)
tmin_learn = 0.2
tmax_learn = 1.6
rhomin_learn = 0.0

logger.info('Constructing data for library terms')
#### Find derivatives #####
#note that the data are stored as rho(y,x) and not rho(x,y)!
# _, rho_y, rho_x = np.gradient(rho, dt, dy, dx, edge_order=2)
# _, rho_xy, rho_xx = np.gradient(rho_x, dt, dy, dx, edge_order=2)
# _, rho_yy, rho_yx = np.gradient(rho_y, dt, dy, dx, edge_order=2)
#
# _, px_y, px_x = np.gradient(px, dt, dy, dx, edge_order=2)
# _, px_xy, px_xx = np.gradient(px_x, dt, dy, dx, edge_order=2)
# _, px_yy, px_yx = np.gradient(px_y, dt, dy, dx, edge_order=2)
#
# _, py_y, py_x = np.gradient(py, dt, dy, dx, edge_order=2)
# _, py_xy, py_xx = np.gradient(py_x, dt, dy, dx, edge_order=2)
# _, py_yy, py_yx = np.gradient(py_y, dt, dy, dx, edge_order=2)


#functions to find laplacian, divergence and grad
#dt, dx, dy already need to be defined above

def lap(f):
    _, f_y, f_x = np.gradient(f, dt, dy, dx, edge_order=2)
    _, _, f_xx = np.gradient(f_x, dt, dy, dx, edge_order=2)
    _, f_yy, _ = np.gradient(f_y, dt, dy, dx, edge_order=2)
    return f_xx + f_yy

def div(fx, fy):
    _, _, fx_x = np.gradient(fx, dt, dy, dx, edge_order=2)
    _, fy_y, _ = np.gradient(fy, dt, dy, dx, edge_order=2)
    return fx_x + fy_y

def grad(f):
    _, f_y, f_x = np.gradient(f, dt, dy, dx, edge_order=2)
    return f_x, f_y

## library terms ###
data_raw={}

data_raw['rho'] = rho

data_raw['rho_t'] = rho_t

data_raw['Lap(rho)'] = lap(rho)

data_raw['Div(v)'] = div(px, py)

data_raw['Div(rho v)'] = div(rho*px, rho*py)

data_raw['Lap(rho^2)'] = lap(rho**2)

data_raw['Lap(|v|^2)'] = lap(px**2 + py**2)

data_raw['Div(rho^2 v)'] = div((rho**2)*px, (rho**2)*py)

data_raw['Lap(rho^3)'] = lap(rho**3)

data_raw['Div(v|v|^2)'] = div(px*(px**2 + py**2), py*(px**2 + py**2))

p2_x, p2_y = grad(px**2 + py**2)
data_raw['Div(rho Grad(|v|^2))'] = div(rho*p2_x, rho*p2_y)

rho_x, rho_y = grad(rho)
data_raw['Div(|v|^2 Grad(rho))'] = div((px**2 + py**2)*rho_x, (px**2 + py**2)*rho_y)

##### Create data dictionaries ######
tt, yy, xx = np.meshgrid(t, y, x, indexing='ij')

inds = np.where((xx >= xmin_learn) & (xx <= xmax_learn) & (yy >= ymin_learn) & (yy <= ymax_learn) \
                    & (tt>tmin_learn) & (tt <= tmax_learn) & (rho >= rhomin_learn))
num_data = min([num_data, len(inds[0])])

rand= np.random.RandomState(seed=seed0)
rand_inds = rand.choice(len(inds[0]), num_data, replace=False)

#parameters
pars = {}
pars['num_data'] = num_data
pars['num_cores'] = num_cores
pars['n_folds'] = n_folds
pars['n_repeats'] = n_repeats
pars['order'] = order
pars['seed0'] = seed0
pars['stab_thresh'] = stab_thresh
pars['num_iters'] = num_iters
pars['algo'] = algo
pars['nlam1'] = nlam1
pars['nlam2'] = nlam2
pars['tmin'] = tmin_learn
pars['tmax'] = tmax_learn

#data in column vector
data_cv={}

for key in data_raw.keys():
    temp = np.expand_dims(data_raw[key][inds],axis=1)
    data_cv[key] = temp[rand_inds, :]

data_raw['1'] = np.ones_like(data_raw['rho'])
data_cv['1'] = np.ones_like(data_cv['rho'])

#delete to free up memory
del data_raw

n = data_cv['1'].shape[0]

#define features
features = ['Div(v)', 'Lap(rho)', 'Div(rho v)', 'Lap(rho^2)', 'Lap(|v|^2)', \
            'Div(rho^2 v)', 'Lap(rho^3)', 'Div(v|v|^2)', 'Div(rho Grad(|v|^2))', 'Div(|v|^2 Grad(rho))']

logger.info('Library Terms:' + str(features))

#reload
import importlib
importlib.reload(pdel)

model = pdel.PDElearn('rho', 'rho_t', features, data_cv, poly_order=order, \
                      print_flag = False, sparse_algo=algo, \
                      path=save_path)

#hyper-parameters to sweep
lambda_min, lambda_max = get_lambda_lims(*scale_X_y(model.Theta, model.ft), 0.01)
#additional changes to the lambdas
lambda_min = lambda_min * 1;
lambda_max = lambda_max * 1;

lam1_arr = np.logspace(np.log10(lambda_min), np.log10(lambda_max), nlam1)
lam2_arr = [0.0]

#cross validate
model.run_cross_validation(lam1_arr, lam2_arr, n_cores=num_cores, n_folds=n_folds, n_repeats=n_repeats, maxit=num_iters);

#intersection
model.find_intersection_of_folds(thresh=stab_thresh, plot_hist=False)

#stability selection
coeffs_all, error_all, _, complexity = model.select_stable_components(thresh=stab_thresh, plot_stab=True)

logger.info('PDEs found are shown below:')
model.print_pdes(coeffs_all, error_all, complexity=complexity, file_name_end='stability')

#save separate files for the stable PDEs
coeffs = coeffs_all
for i, arr in enumerate(coeffs):
    file_name = save_path + '/' + 'pde_stability_%.2d.txt' %(i+1)
    with open(file_name, 'w') as f:
        count = 0
        for s in features:
            if s in model.Theta_desc:
                print('%0.10f\t%s' %(arr[model.Theta_desc.index(s)], s), file=f)
                count += 1
            else:
                print('%0.10f\t%s' %(0, s), file=f)
