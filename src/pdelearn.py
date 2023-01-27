import numpy as np
import logging
import pandas as pd
import time
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import Parallel, delayed
import pickle as pkl
from tqdm import tqdm
import copy
from sklearn import preprocessing
from solvers import *
from funcs import *
from sklearn.model_selection import RepeatedKFold, KFold

logger = logging.getLogger(__name__)

class PDElearn:

    def __init__(self, f_desc, ft_desc, features, data_raw, poly_order=0, sparse_algo='stridge', \
                    print_flag=False, path='.'):
        """
        f, ft: string descriptors of the field variable
        features: list with string descriptors of all the features
        data_raw: dictioary of data for each descriptor in features
        poly_order: largest order of f to construct the feature matrix
        sparse_algo: which sparsity promoting algo to use
            currently implemented 'STRidge', 'IHTd'
        path: various text/pdf files will be saved at path + qualifier . extension
        """
        self.ft_desc = ft_desc
        self.f_desc= f_desc
        self.features = features
        self.P = poly_order
        self.print_flag = print_flag
        self.path = path

        if sparse_algo.lower() not in ['stridge']:
            self.sparse_algo = 'stridge' #default
        else:
            self.sparse_algo = sparse_algo.lower()

        #build the feature matrix and get descriptors
        self.Theta, self.Theta_desc = self.create_feature_matrix(data_raw, P=poly_order)

    def create_feature_columns(self, data_raw):
        """
        Makes column vectors of elements in data_raw and stores in data_cv
        Also adds features that are multiplications of the base features
        (indicated with a * in the descriptor)
        INPUT:
        data_raw: dictionary of the raw non-columnar data
        """
        data_cv = {}
        data_cv[self.f_desc] = np.expand_dims(data_raw[self.f_desc].flatten(), axis=1)
        data_cv[self.ft_desc] = np.expand_dims(data_raw[self.ft_desc].flatten(), axis=1)

        for key in self.features:
            if key in data_raw:
                data_cv[key] = np.expand_dims(data_raw[key].flatten(), axis=1)
            elif '*' in key:
                split_desc = key.split('*')
                if split_desc[0] in data_raw and split_desc[1] in data_raw:
                    field1 = np.expand_dims(data_raw[split_desc[0]].flatten(), axis=1)
                    field2 = np.expand_dims(data_raw[split_desc[1]].flatten(), axis=1)
                    data_cv[key] = field1*field2
                else:
                    logger.error('`%s` not in data_raw!' %(key))
            else:
                logger.error('`%s` not in data_raw!' %(key))

        return data_cv

    def create_feature_matrix(self, data_raw, P=3):
        """
        Makes a feature matrix by first converting the features to column vectors
        and then stacking these columns, also stores the field and the time derivative of the field
        INPUT:
        data_raw: dictionary of the raw non-columnar data (could be columnar too)
        split: the fraction of data to use for training. The rest is used for validation.
        """
        data_cv = self.create_feature_columns(data_raw)

        f_desc, ft_desc = self.f_desc, self.ft_desc
        f, ft = data_cv[f_desc], data_cv[ft_desc]
        self.f, self.ft, self.P = f, ft, P

        Th_list = []
        Theta_desc = [] #description of the columns
        W_list = []

        for key in self.features:
            for p in range(P+1):
                #don't include the constant term in the features
                if key == '1' and p==0:
                    continue

                Th_list.append(np.multiply(data_cv[key], self.f**p))
                if p == 0:
                    Theta_desc.append(key)
                else:
                    if key=='1':
                        Theta_desc.append('%s^%i' %(f_desc, p))
                    else:
                        Theta_desc.append('%s^%i %s' %(f_desc, p, key))

        #feature matrix
        Theta = np.hstack(Th_list)

        n, d = Theta.shape
        logger.debug('Created (n_data, n_features) = (%i, %i)' %(n,d))

        return Theta, Theta_desc

    def create_weight_matrix(self, weightFac=0, w_list=None):
        """ Makes the weights matrix W"""
        #set base weights if w_list is unspecified
        if w_list is None:
            #if weight factor unspecified
            if weightFac==0:
                w_list = {key:1 for key in self.features}
            else:
                w_list = {key:weightFac for key in self.features}

        W_list = []
        for key in self.features:
            for p in range(self.P+1):
                #don't include the constant term in the features
                if key == '1' and p==0:
                    continue
                #add the base weight and the additional weight due to the nonlinearity
                if key=='1' and p>0:
                    #special treatment for terms like rho^1, rho^2 etc
                    W_list.append(w_list[key] + weightFac*(p-1))
                elif '*' in key:
                    #add an additional weightfac if the feature itself has two terms
                    W_list.append(w_list[key] + weightFac*p + weightFac)
                else:
                    W_list.append(w_list[key] + weightFac*p)

        #weights matrix
        self.W = np.diag(W_list)
        return self.W

    def get_sparse_solver(self):
        """
        Return a wrapper for the sparse solver
        """
        #build the thresh_nonzero vector based the entries in self.forced_features
        if hasattr(self, 'forced_features'):
            temp_vec = np.array([0. if key in self.forced_features else 1. \
                for key in self.Theta_desc])
            thresh_nonzero = temp_vec[:, np.newaxis]
        else:
            thresh_nonzero = None

        if hasattr(self, 'w_list'):
            w_list = self.w_list
        else:
            w_list = None

        W = self.create_weight_matrix(w_list=w_list)

        if self.sparse_algo == 'stridge':
            find_sparse_coeffs = lambda X, y, lam1, lam2, maxit: \
                STRidge(X, y, lam2, lam1, maxit, W=W, thresh_nonzero=thresh_nonzero, print_flag = self.print_flag)
        elif self.sparse_algo == 'ihtd':
            find_sparse_coeffs = lambda X, y, lam1, lam2, maxit: \
                IHTd(X, y, lam1, maxit, 100, 10**-10, htp_flag=1, print_flag = self.print_flag)
        else:
            pass #implement lasso

        logger.info('Sparse solver selected: %s' %(self.sparse_algo))

        return find_sparse_coeffs

    def run_cross_validation(self, lam1_arr, lam2_arr, n_cores=1, n_folds=4, \
        n_repeats=1, random_state=None, maxit=1000, plot_folds=False):
        """
        Performs cross_validation
        lam1_arr: array with values for tau for STRidge and lambda for LASSO and IHTd
        lam2_arr: array with values for lambda for STRidge (ignored for LASSO and IHTd)
        num_cores: cores to be used for running in parallel
        n_folds, n_repeats: parameters for k-fold cross validation
        maxit: max iterations for the solver
        """

        if self.sparse_algo != 'stridge' and len(lam2_arr) != 1:
            logger.error('Since solver is not STRidge, size of lambda2 array must be 1! Exiting..')
            return

        self.lam1_arr = lam1_arr #store lambda values to use in stability selection plots
        self.lam2_arr = lam2_arr

        k_fold = RepeatedKFold(n_splits=n_folds, n_repeats=n_repeats, random_state=random_state)
        tot_folds = k_fold.get_n_splits()

        n, d = self.Theta.shape

        train_inds_list, test_inds_list = [None]*tot_folds, [None]*tot_folds

        #do the regular k_fold split of n_folds > 1 otherwise set train_inds and
        #test_inds (both) to the entire dataset
        for k, (train_inds, test_inds) in enumerate(k_fold.split(self.Theta)) \
            if tot_folds !=1 else enumerate([(np.arange(n), np.arange(n))]):

            train_inds_list[k] = train_inds
            test_inds_list[k] = test_inds

        find_sparse_coeffs = self.get_sparse_solver()

        logger.info('Running algorithm on subsamples: %i folds, %i repeats' %(n_folds, n_repeats))

        #function that does the computation for each fold
        def run_fold(train_inds, test_inds):

            #scaling training set
            XTrainStd, yTrainStd = scale_X_y(self.Theta[train_inds], self.ft[train_inds])
            #scaling testing set
            XTestStd, yTestStd = scale_X_y(self.Theta[test_inds], self.ft[test_inds])

            #initialize lists for coeffs and error
            CoeffsList = [np.zeros((d,1)) for i in range(len(lam1_arr)*len(lam2_arr))]
            ErrorList = [0.0 for i in range(len(lam1_arr)*len(lam2_arr))]

            count = 0
            #sweep through hyperparameters and construct a list
            #lam2 has to be in the outer loop because of the way the stability path
            #is constructed in select_stable_components()
            for lam2 in lam2_arr:
                for lam1 in lam1_arr:
                    w_train = find_sparse_coeffs(XTrainStd, yTrainStd, lam1, lam2, maxit)
                    test_error = find_error(XTestStd, yTestStd, w_train)
                    CoeffsList[count][:] = w_train
                    ErrorList[count] = test_error
                    count += 1

            return CoeffsList, ErrorList

            print('*** Done ***')

        output = Parallel(n_jobs=n_cores)\
                    (delayed(run_fold)(train_inds_list[i], test_inds_list[i]) \
                    for i in tqdm(range(tot_folds)))

        self.coeffs_folds = [i[0] for i in output]
        self.error_folds = [i[1] for i in output]

        #plot the error and complexity of all the PDEs in each fold
        if plot_folds:
            for k, lst in enumerate(self.coeffs_folds):
                fig = plt.figure(figsize=(5,3))
                for i, arr in enumerate(lst):
                    tup = tuple(np.where(arr[:,0]==0., 0., 1.))
                    complexity = np.sum(self.W @ (np.array(tup)[:, np.newaxis]))
                    error = self.error_folds[k][i]
                    plt.scatter(np.log10(error), complexity, 10, 'k')
                plt.xlabel(r'$\log(Loss)$')
                plt.ylabel('Complexity')
                plt.title('Fold %i' %(k))
                plt.tight_layout()
                plt.savefig('%s/fold_%i.pdf' %(self.path, k))
                plt.close(fig)

        logger.info('Cross Validation done!')

        return self.coeffs_folds, self.error_folds

    def find_intersection_of_folds(self, thresh=0.8, plot_hist=False):
        """
        Finds the intersection of the PDEs from multiple folds using Python sets.
        Each PDE is represented by a tuple like (0, 1, 1, ...., 0) where
        0 indicates a term being present and 1 otherwise.
        These tuples are used as keys for the dictionaries for storing the
        corresponding actual coefficients, the squared errors and the complexity of the PDE.
        Note that each fold has mutliple repeated PDEs in the
        hyperparameter-sweeping. Using the sets avoids such repeated PDEs.
        INPUT:
        thresh: a number out of 1 that indicates the percentage of folds in which
        the PDEs are supposed to belong
        """
        if not hasattr(self, 'coeffs_folds'):
            logger.error('Execute PDElearn.run_cross_validation() first!')
            return

        logger.info('Finding the intersection set of PDEs from the folds!')

        n, d = self.Theta.shape

        coeffs_folds = self.coeffs_folds
        error_folds = self.error_folds
        n_folds = len(coeffs_folds)

        tup_sets = [set() for i in range(n_folds)]
        coeff_dict = [{} for i in range(n_folds)]
        error_dict = [{} for i in range(n_folds)]

        for k, lst in enumerate(coeffs_folds):
            for i, arr in enumerate(lst):
                tup = tuple(np.where(arr[:,0]==0., 0., 1.))
                tup_sets[k].add(tup)
                coeff_dict[k][tup] = arr
                error_dict[k][tup] = error_folds[k][i]

        #find q-intersection of the pdes
        q = int(np.floor((1-thresh)*n_folds)) #number of sets to leave out for intersection
        tup_sets_, score_ = find_relaxed_intersection(*tup_sets, q=q)

        #sort the tuples and the scores in terms of number of terms
        n_coeffs_list = [sum(tup) for tup in tup_sets_]
        tup_sets_all = [x for _, x in sorted(zip(n_coeffs_list, tup_sets_))]
        score_all = [x for _, x in sorted(zip(n_coeffs_list, score_))]

        if plot_hist:
            plt.figure(figsize=(5,3), dpi=200)
            plt.bar(np.arange(1,len(score_all)+1), np.sort(score_all)[::-1])
            plt.plot([1, len(score_all)+1], [thresh, thresh], 'r--')
            plt.xlabel('PDE #'); plt.ylabel('Score');
            plt.tight_layout(); plt.savefig(self.path + '/scores.pdf');
            logger.info('Score plot saved at %s' %(self.path + '/scores.pdf'))

        #get the coeffs and the errors
        coeffs_all, error_all, num_terms_all, complexity_all = [], [], [], []

        #if set is not empty
        if tup_sets_all is []:
            logger.error('q-relaxed intersection set is empty for q=%i\nExiting..' %(q))
            return

        for i, (tup, score) in enumerate(zip(tup_sets_all, score_all)):

            #check if the PDE has no terms
            if sum(tup) == 0:
                continue

            #find the average test error
            error_new = np.mean([error_dict[k][tup] \
                    for k in range(n_folds) if tup in tup_sets[k]], axis=0)

            #refit coefficients to full data
            NonZeroInds = np.nonzero(np.array(tup))[0]
            coeffs_new = np.zeros((d,1))
            #refit coefficients only if some are non-zero
            if len(NonZeroInds) != 0:
                coeffs_new[NonZeroInds] = np.linalg.lstsq(self.Theta[:, NonZeroInds], self.ft, rcond=-1)[0]

            num_terms = np.sum(np.array(tup))
            complexity = np.sum(self.W @ (np.array(tup)[:, np.newaxis]))

            #append
            coeffs_all.append(coeffs_new)
            error_all.append(error_new)
            num_terms_all.append(num_terms)
            complexity_all.append(complexity)

        self.coeffs = coeffs_all
        self.errors = error_all
        self.scores = score_all
        self.num_terms = num_terms_all
        self.complexity = complexity_all

        return coeffs_all, error_all, score_all, num_terms_all, complexity_all

    def select_stable_components(self, thresh=0.8, plot_stab=False):
        """
        This function calculates the stability score for each term in the dictionary
        for every value of the hyperparamters lambda and tau.
        The returned PDE contains all those terms with stability value >=thresh
        """
        if not hasattr(self, 'coeffs_folds'):
            logger.error('First run CrossValidate()!')
            return

        coeffs_folds = self.coeffs_folds
        error_folds = self.error_folds

        n_folds = len(coeffs_folds)
        n_coeffs = len(coeffs_folds[0][0])

        is_term_present = [None]*n_folds
        for i, lst in enumerate(coeffs_folds):
            temp = [None]*len(lst)

            for j, arr in enumerate(lst):
                temp[j] = np.where(arr[:,0]==0., 0., 1.)

            is_term_present[i] = temp

        nlam1, nlam2 = len(self.lam1_arr), len(self.lam2_arr)

        stability = np.zeros((n_coeffs, nlam1*nlam2))

        for i in range(nlam1*nlam2):
            sum_ = np.zeros_like(is_term_present[0][0])
            for j in range(n_folds):
                sum_ += is_term_present[j][i]
            stability[:, i] = sum_/n_folds

        if plot_stab:
            #save new plot for each element in lam2_arr
            for j in range(nlam2):
                plt.figure(figsize=(5,3), dpi=200)
                for i in range(n_coeffs):
                    plt.plot(-np.log10(self.lam1_arr/np.max(self.lam1_arr)), stability[i,j*nlam1:(j+1)*nlam1], \
                            label = '%s' %(self.Theta_desc[i]), alpha=0.6)
                plt.title('Stability Path ($\lambda_2 = %0.4f$)' %(self.lam2_arr[j]));
                plt.xlabel(r'$-\log(\lambda/\lambda_{max})$');
                plt.ylabel('Stability Score')
                plt.legend(frameon=False, fontsize=3, loc='center left', bbox_to_anchor=(1, 0.5))
                plt.tight_layout();
                plt.savefig(self.path + '/stability_path%.2d.pdf' %(j))

        #find all the unique PDEs
        tup_sets = set()

        #scan through all pairs of lam1, lam2
        for i in range(nlam1*nlam2):
            tup = tuple(np.where(stability[:, i]>=thresh, 1.0, 0.))

            #add the PDE tuple if non-zero number of terms are above the threshold
            if sum(tup) != 0:
                tup_sets.add(tup)

        coeffs_all, error_all, num_terms_all, complexity_all = [], [], [], []

        #convert the set of tuples to a list and sort according to number of terms
        tups_list = list(tup_sets)
        n_coeffs_list = [sum(tup) for tup in tups_list]
        tups_list_sorted = [x for _, x in sorted(zip(n_coeffs_list, tups_list))]

        for tup in tups_list_sorted:

            #find the new coeffs on full data
            coeffs = np.zeros((n_coeffs, 1))
            NonZeroInds = np.nonzero(np.array(tup))[0]
            #refit coefficients only if some are non-zero
            if len(NonZeroInds) != 0:
                coeffs[NonZeroInds] = np.linalg.lstsq(self.Theta[:, NonZeroInds], self.ft, rcond=-1)[0]
            sq_error = find_error(self.Theta, self.ft, coeffs)

            num_terms = np.sum(np.array(tup))
            complexity = np.sum(self.W @ (np.array(tup)[:, np.newaxis]))

            coeffs_all.append(coeffs)
            error_all.append(sq_error)
            num_terms_all.append(num_terms)
            complexity_all.append(complexity)

        return coeffs_all, error_all, num_terms_all, complexity_all

    def print_features(self):
        """Print features and weights assigned"""
        frame_dict = {'%s' %(self.ft_desc): self.Theta_desc, 'Weights': np.diagonal(self.W)}
        data_frame = pd.DataFrame(frame_dict)
        print(data_frame)

    def store_actual_coeffs(self, coeffs_dict):
        """
        Create an actual coefficient vector
        INPUT: coeffs_dict -- dictionary containing the actual coeffs
        """
        if not hasattr(self, 'Theta_desc'):
            logger.error('Create Feature Matrix First! Exiting...')
            return

        coeffs_actual = np.zeros((len(self.Theta_desc), 1))

        count = 0
        for key in self.Theta_desc:
            if key in coeffs_dict.keys():
                    coeffs_actual[count, 0] = coeffs_dict[key]
            count += 1

        self.coeffs_actual = coeffs_actual
        return coeffs_actual

    def find_pareto(self, plot_fig=False):
        """
        Find the Pareto Front of the intersection set
        """
        if not hasattr(self, 'coeffs'):
            logger.error('Run cross validation and intersection first! Exiting..')
            return

        #find the pareto front
        ParetoInds = find_pareto_front(np.log10(self.errors), self.complexity, \
                        plot_fig=plot_fig, file_name=self.path + '/pareto.pdf', \
                        xlabel='Log(loss)', ylabel='Complexity')
        self.pareto_coeffs = [self.coeffs[i] for i in ParetoInds]
        self.pareto_errors = [self.errors[i] for i in ParetoInds]
        self.pareto_scores = [self.scores[i] for i in ParetoInds]
        self.pareto_complexity = [self.complexity[i] for i in ParetoInds]

    def print_pdes(self, coeffs, error, file_name_end='', **kwargs):
        """ Print PDEs corresponding to the list of coeffs and the error
        kwargs: gets in the addtional values to be added to the text file
        """
        file_name = self.path + '/pdes_' + file_name_end + '.txt'
        with open(file_name, "w") as text_file:
            for i, arr in enumerate(coeffs):
                print('Log(loss) = %0.6f' %(np.log10(error[i])), file=text_file)
                print('Log(loss) = %0.6f' %(np.log10(error[i])))

                #print the additional fields
                if kwargs is not None:
                    for key, val in kwargs.items():
                        print('%s = %0.3f' %(key, val[i]), file=text_file)
                        print('%s = %0.3f' %(key, val[i]))

                pde_text = pde_string(arr, self.Theta_desc, ut=self.ft_desc)
                print(pde_text, file=text_file)
                print(pde_text)
