"""Code to run all the experiments on fMRI datasets.

Author: Sandro Vega-Pons, Emanuele Olivetti
"""

import os
import numpy as np
from load_data import load_1000_funct_connectome
from load_data import load_schizophrenia_data, load_kernel_matrix
from classif_and_ktst import compute_rbf_kernel_matrix, apply_svm
from classif_and_ktst import apply_ktst, plot_null_distribution
from classif_and_ktst import compute_svm_score_nestedCV
from classif_and_ktst import balanced_accuracy_scoring
from kernel_embedding import DCE_embedding, DR_embedding, WL_K_embedding
from kernel_embedding import SP_K_embedding
from joblib import Parallel, delayed
import pickle


def simple_experiment(K, y, n_folds=5, iterations=10000, subjects=False,
                      verbose=True, data_name='', plot=True,
                      random_state=None):
    """
    """
    acc, acc_null, p_value = apply_svm(K, y, n_folds=n_folds,
                                       iterations=iterations,
                                       subjects=subjects,
                                       verbose=verbose,
                                       random_state=random_state)
    if plot:
        plot_null_distribution(acc, acc_null, p_value, data_name=data_name,
                               stats_name='Bal_Acc')
    if verbose:
        print ''

    mmd2u, mmd2u_null, p_value = apply_ktst(K, y, iterations=iterations,
                                            subjects=subjects,
                                            verbose=verbose)
    if plot:
        plot_null_distribution(mmd2u, mmd2u_null, p_value, data_name=data_name,
                               stats_name='$MMD^2_u$')


def experiment_schizophrenia_data(data_path='data', n_folds=5,
                                  iterations=10000,
                                  verbose=True, plot=True, random_state=None):
    """Run the experiments on the Schizophrenia dataset

    Parameters:
    ----------
    data_path: string
        Path to the folder containing the dataset.
    n_folds: int
        The number of folds in a StratifiedKFold cross-validation
    iterations: int
        Number of iterations to compute the null distribution of
        balanced_accuracy and MMD^2_u
    verbose: bool
    plot: bool
        Whether to plot the results of the statistical tests.

    """
    name = 'Schizophrenia'
    if verbose:
        print '\nWorking on %s dataset...' % name
        print '-----------------------'
    X, y = load_schizophrenia_data(data_path, verbose=verbose)

    # DCE Embedding
    if verbose:
        print '\n### Results for DCE_Embedding ###'

    X_dce = DCE_embedding(X)
    K_dce = compute_rbf_kernel_matrix(X_dce)
    simple_experiment(K_dce, y, n_folds=n_folds, iterations=iterations,
                      verbose=verbose, data_name=name + '_dce', plot=plot,
                      random_state=random_state)

    # DR Embedding
    if verbose:
        print '\n### Results for DR_Embedding ###'

    X_dr = DR_embedding(X)
    K_dr = compute_rbf_kernel_matrix(X_dr)
    simple_experiment(K_dr, y, n_folds=n_folds, iterations=iterations,
                      verbose=verbose, data_name=name+'_dr',plot=plot,
                      random_state=random_state)

    # WL Kernel based Embedding
    if verbose:
        print '\n### Results for WL_K_Embedding ###'
    th = 0.2
    K_wl = WL_K_embedding(X, th)
    simple_experiment(K_wl, y, n_folds=n_folds,
                      iterations=iterations,
                      verbose=verbose, data_name=name+'_wl', plot=plot,
                      random_state=random_state)

    # SP Kernel based Embedding
    if verbose:
        print '\n### Results for SP_K_Embedding ###'
    th = 0.2
    K_sp = SP_K_embedding(X, th)
    simple_experiment(K_sp, y, n_folds=n_folds,
                      iterations=iterations,
                      verbose=verbose, data_name=name+'_sp', plot=plot,
                      random_state=random_state)


def experiment_1000_func_conn_data(data_path='data', location='all', n_folds=5,
                                   iterations=10000, verbose=True, plot=True,
                                   random_state=None):
    """
    Run the experiments on the 1000_functional_connectome data.

    Parameters:
    ----------
    data_path: string
        Path to the folder containing the dataset.
    location: string
        If location=='all' we run the experiments for all locations, otherwise
        only the selected location is used.
    n_folds: int
        The number of folds in a StratifiedKFold cross-validation
    iterations: int
        Number of iterations to compute the null distribution of
        balanced_accuracy and MMD^2_u
    verbose: bool
    plot: bool
        Whether to plot the results of the statistical tests.
    """
    if location == 'all':
        locs = os.listdir(os.path.join(data_path, 'Functional_Connectomes',
                                       'Locations'))
        if verbose:
            print("")
            print("We will analyze the following datasets:")
            print("%s \n" % '\n'.join(locs))

    else:
        locs = [location]

    for name in locs:
        if verbose:
            print('')
            print('Working on %s dataset...' % name)
            print('-----------------------')

        X, y = load_1000_funct_connectome(data_path, name, verbose=verbose)

        # DCE Embedding
        if verbose:
            print('')
            print('### Results for DCE_Embedding ###')

        X_dce = DCE_embedding(X)
        K_dce = compute_rbf_kernel_matrix(X_dce)
        simple_experiment(K_dce, y, n_folds=n_folds,
                          iterations=iterations,
                          verbose=verbose, data_name=name+'_dce', plot=plot,
                          random_state=random_state)

        # DR Embedding
        if verbose:
            print('')
            print('### Results for DR_Embedding ###')

        X_dr = DR_embedding(X)
        K_dr = compute_rbf_kernel_matrix(X_dr)
        simple_experiment(K_dr, y, n_folds=n_folds, iterations=iterations,
                          verbose=verbose, data_name=name+'_dr', plot=plot,
                          random_state=random_state)

        # # WL Kernel based Embedding
        # if verbose:
        #     print('')
        #     print('### Results for WL_K_Embedding ###')

        # th = 0.2
        # K_wl = WL_K_embedding(X, th)
        # simple_experiment(K_wl, y, n_folds=n_folds,
        #                   iterations=iterations,
        #                   verbose=verbose, data_name=name+'_wl', plot=plot,
        #                   random_state=random_state)


def experiment_precomputed_matrix(data_path='data', study='wl_kernel',
                                  n_folds=5,
                                  iterations=10000, verbose=True, plot=True,
                                  random_state=None):
    """Run the experiments on the Uri dataset. For simplicity, already
    computed kernel matrices are used.

    Paramters:
    ---------
    data_path: string
        Path to the folder containing the dataset.
    study: string
        Name of the study (kernel method) to be used, e.g. 'wl_kernel' should
        contain the kernel matrix computed with the WL kernel.
    n_folds: int
        The number of folds in a StratifiedKFold cross-validation
    iterations: int
        Number of iterations to compute the null distribution of
        balanced_accuracy and MMD^2_u
    verbose: bool
    plot: bool
        Whether to plot the results of the statistical tests.

    """
    name = 'Uri_' + study
    if verbose:
        print('')
        print('Working on %s dataset...' % name)
        print('-----------------------')

    K, y = load_kernel_matrix(data_path, study, verbose=verbose)

    # kernel based embedding
    simple_experiment(K, y, n_folds=n_folds, iterations=iterations,
                      subjects=True,
                      verbose=verbose, data_name=name, plot=plot,
                      random_state=random_state)


def check_instability_classification(X, y, location='', n_folds=5,
                                     iterations=10000, verbose=True, reps=100,
                                     seed=0):
    """
    """
    if verbose:
        print 'Computing embeddings...'

    X_dce = DCE_embedding(X)
    K_dce = compute_rbf_kernel_matrix(X_dce)
    X_dr = DR_embedding(X)
    K_dr = compute_rbf_kernel_matrix(X_dr)
    X_wl = WL_K_embedding(X, th=0.2)
    K_wl = compute_rbf_kernel_matrix(X_wl)

    np.random.seed(seed)
    seeds = np.random.randint(0, 10000000, reps)

    accs_dce = np.zeros(reps)
    accs_dr = np.zeros(reps)
    accs_wl = np.zeros(reps)
    pvals_dce = np.zeros(reps)
    pvals_dr = np.zeros(reps)
    pvals_wl = np.zeros(reps)

    if verbose:
        print 'Computing null distributions...'

    param_grid = [{'C': np.logspace(-5, 5, 25)}]
    yis = [np.random.permutation(y) for i in range(iterations)]
    acc_null_dce = Parallel(n_jobs=-1)(delayed(compute_svm_score_nestedCV)(K_dce, yis[i], n_folds, scoring=balanced_accuracy_scoring, param_grid=param_grid) for i in range(iterations))
    acc_null_dr  = Parallel(n_jobs=-1)(delayed(compute_svm_score_nestedCV)(K_dr , yis[i], n_folds, scoring=balanced_accuracy_scoring, param_grid=param_grid) for i in range(iterations))
    acc_null_wl  = Parallel(n_jobs=-1)(delayed(compute_svm_score_nestedCV)(K_wl , yis[i], n_folds, scoring=balanced_accuracy_scoring, param_grid=param_grid) for i in range(iterations))

    for i, s in enumerate(seeds):
        # if verbose:
        #     print 'Repetition number: % s, seed: %s' % (i, s)

        rs = np.random.RandomState(s)
        # DCE Embedding
        acc_dce = compute_svm_score_nestedCV(K_dce, y, n_folds,
                                             scoring=balanced_accuracy_scoring,
                                             random_state=rs,
                                             param_grid=param_grid)

        p_value_dce = max(1.0/iterations, (acc_null_dce > acc_dce).sum()
                          / float(iterations))
        accs_dce[i] = acc_dce
        pvals_dce[i] = p_value_dce
        # if verbose:
        #     print "DCE => acc: %s, p_value: %s" %(acc_dce, p_value_dce)

        # DR Embedding
        acc_dr = compute_svm_score_nestedCV(K_dr, y, n_folds,
                                            scoring=balanced_accuracy_scoring,
                                            random_state=rs,
                                            param_grid=param_grid)
        p_value_dr = max(1.0/iterations, (acc_null_dr > acc_dr).sum()
                         / float(iterations))
        accs_dr[i] = acc_dr
        pvals_dr[i] = p_value_dr
        # if verbose:
        #     print "DR => acc: %s, p_value: %s" % (acc_dr, p_value_dr)

        # WL Kernel
        acc_wl = compute_svm_score_nestedCV(K_wl, y, n_folds,
                                            scoring=balanced_accuracy_scoring,
                                            random_state=rs,
                                            param_grid=param_grid)
        p_value_wl = max(1.0/iterations, (acc_null_wl > acc_wl).sum()
                         / float(iterations))
        accs_wl[i] = acc_wl
        pvals_wl[i] = p_value_wl
        # if verbose:
        #     print "WL => acc: %s, p_value: %s" %(acc_wl, p_value_wl)

    # saving the results
    if not os.path.isdir('./results_dic'):
        os.mkdir('./results_dic')

    save_dir = './results_dic'
    if verbose:
        print 'Saving results at %s' % save_dir

    res_dce = {}
    res_dce['null_distribution'] = np.array(acc_null_dce)
    res_dce['accuracies'] = accs_dce
    res_dce['p_values'] = pvals_dce

    res_dr = {}
    res_dr['null_distribution'] = np.array(acc_null_dr)
    res_dr['accuracies'] = accs_dr
    res_dr['p_values'] = pvals_dr

    res_wl = {}
    res_wl['null_distribution'] = np.array(acc_null_wl)
    res_wl['accuracies'] = accs_wl
    res_wl['p_values'] = pvals_wl

    pickle.dump(res_dce, open(os.path.join(save_dir,
                                           '%s_dce.pkl' % location),
                              'wb'))
    pickle.dump(res_dr, open(os.path.join(save_dir,
                                          '%s_dre.pkl' % location),
                             'wb'))
    pickle.dump(res_wl, open(os.path.join(save_dir,
                                          '%s_wl.pkl' % location),
                             'wb'))

    if verbose:
        print 'DCE p-values: min:%s, mean:%s, max:%s' % (np.min(pvals_dce),
                                                         np.mean(pvals_dce),
                                                         np.max(pvals_dce))
        print 'DR  p-values: min:%s, mean:%s, max:%s' % (np.min(pvals_dr),
                                                         np.mean(pvals_dr),
                                                         np.max(pvals_dr))
        print 'WL  p-values: min:%s, mean:%s, max:%s' % (np.min(pvals_wl),
                                                         np.mean(pvals_wl),
                                                         np.max(pvals_wl))

    return accs_dce, pvals_dce, accs_dr, pvals_dr, accs_wl, pvals_wl


# def reformatting_results():
#     open_path = './results'
#     save_path = './results_dic'
#     dirs = os.listdir(open_path)
#     dirs.sort()
#     for f in dirs:
#         null_dist, accs, pvals = pickle.load(open(os.path.join('./results', f),
#                                                   'rb'))
#         res = {}
#         res['null_distribution'] = np.array(null_dist)
#         res['accuracies'] = accs
#         res['p_values'] = pvals
#         pickle.dump(res, open(os.path.join(save_path, f), 'wb'))


if __name__ == '__main__':
    # Script to run all the experiments

    # numpy seed
    seed = 0
    random_state = np.random.RandomState(seed)

    # Path to the folder containing all datasets
    data_path = 'data'

    # Verbosity
    verbose = True

    # Whether to plot the results of test statistics, their
    # null-distributions and p-values.
    plot = True

    # whether to use the schizophrenia dataset in the experiments
    schizophrenia_data = False

    # whether to use the 1000_functional_connectoms dataset in the experiments
    connectome_data = False

    # Location to be used inside the 1000_functional_connectome
    # (e.g. location='Leipzig'). If 'all', experiments are run on all
    # locations.
    location = 'Leipzig'
    location = 'all'

    # Whether to use the Uri dataset. We will use already computed
    # kernel matrices on the Uri dataset. There will be a kernel
    # matrix for each kernel, e.g. WL and Shortest-path.
    uri_data = False

    # Number of iterations to compute the null distribution.
    iterations = 10000

    # Number of folds in the Stratified cross-validation.
    n_folds = 5

    if schizophrenia_data:
        experiment_schizophrenia_data(data_path, n_folds, iterations, verbose,
                                      plot, random_state)

    if connectome_data:
        experiment_1000_func_conn_data(data_path, location, n_folds,
                                       iterations,
                                       verbose, plot, random_state)

    if uri_data:
        studies = os.listdir(os.path.join(data_path, 'precomputed_kernels'))
        for std in studies:
            experiment_precomputed_matrix(data_path, std, n_folds, iterations,
                                          verbose, plot, random_state)

    # whether to run the experiments to analyze the instability of the
    # classification based test
    check_instability = True

    if check_instability:
        locs = os.listdir(os.path.join(data_path, 'Functional_Connectomes',
                                       'Locations'))
        locs.sort()
        repetitions = 100

        # # Working on the functional connectome data
        # for name in locs:
        #     if verbose:
        #         print '\nWorking on %s dataset...' % name
        #         print '-----------------------'
        #     X, y = load_1000_funct_connectome(data_path, name,
        #                                       verbose=verbose)
        #     check_instability_classification(X, y, location=name,
        #                                      n_folds=n_folds,
        #                                      iterations=iterations,
        #                                      verbose=verbose,
        #                                      reps=repetitions, seed=seed)

        # Working on the schizophrenia data
        if verbose:
            print '\nWorking on Schizophrenia dataset...'
        X, y = load_schizophrenia_data(data_path, verbose=verbose)
        check_instability_classification(X, y, location='Schizophrenia',
                                         n_folds=n_folds,
                                         iterations=iterations,
                                         verbose=verbose,
                                         reps=repetitions, seed=seed)

        # # Working on URI data
        # X, y = load_kernel_matrix()
        # check_instability_classification(X, y, n_folds=n_folds,
        #                                  iterations=iterations,
        #                                  verbose=verbose,
        #                                  reps=repetitions, seed=seed)
