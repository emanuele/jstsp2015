"""Classification-based test and kernel two-sample test.

Author: Sandro Vega-Pons, Emanuele Olivetti.
"""

import os
import numpy as np
from sklearn.metrics import pairwise_distances, confusion_matrix
from sklearn.metrics import pairwise_kernels
from sklearn.svm import SVC
from sklearn.cross_validation import StratifiedKFold, KFold, cross_val_score
from sklearn.grid_search import GridSearchCV
from kernel_two_sample_test import MMD2u, compute_null_distribution
from kernel_two_sample_test import compute_null_distribution_given_permutations
import matplotlib.pylab as plt
from joblib import Parallel, delayed


def compute_rbf_kernel_matrix(X):
    """Compute the RBF kernel matrix with sigma2 as the median pairwise
    distance.
    """
    sigma2 = np.median(pairwise_distances(X, metric='euclidean'))**2
    K = pairwise_kernels(X, X, metric='rbf', gamma=1.0/sigma2, n_jobs=-1)
    return K


def balanced_accuracy_scoring(clf, X, y):
    """Scoring function that computes the balanced accuracy to be used
    internally in the cross-validation procedure.
    """
    y_pred = clf.predict(X)
    conf_mat = confusion_matrix(y, y_pred)
    bal_acc = 0.
    for i in range(len(conf_mat)):
        bal_acc += (float(conf_mat[i, i])) / np.sum(conf_mat[i])

    bal_acc /= len(conf_mat)
    return bal_acc


def compute_svm_cv(K, y, C=100.0, n_folds=5,
                   scoring=balanced_accuracy_scoring):
    """Compute cross-validated score of SVM with given precomputed kernel.
    """
    cv = StratifiedKFold(y, n_folds=n_folds)
    clf = SVC(C=C, kernel='precomputed', class_weight='auto')
    scores = cross_val_score(clf, K, y,
                             scoring=scoring, cv=cv)
    return scores.mean()


def compute_svm_subjects(K, y, n_folds=5):
    """
    """
    cv = KFold(len(K)/2, n_folds)
    scores = np.zeros(n_folds)
    for i, (train, test) in enumerate(cv):
        train_ids = np.concatenate((train, len(K)/2+train))
        test_ids = np.concatenate((test, len(K)/2+test))
        clf = SVC(kernel='precomputed')
        clf.fit(K[train_ids, :][:, train_ids], y[train_ids])
        scores[i] = clf.score(K[test_ids, :][:, train_ids], y[test_ids])

    return scores.mean()


def permutation_subjects(y):
    """Permute class labels of Contextual Disorder dataset.
    """
    y_perm = np.random.randint(0, 2, len(y)/2)
    y_perm = np.concatenate((y_perm, np.logical_not(y_perm).astype(int)))
    return y_perm


def permutation_subjects_ktst(y):
    """Permute class labels of Contextual Disorder dataset for KTST.
    """
    yp = np.random.randint(0, 2, len(y)/2)
    yp = np.concatenate((yp, np.logical_not(yp).astype(int)))
    y_perm = np.arange(len(y))
    for i in range(len(y)/2):
        if yp[i] == 1:
            y_perm[i] = len(y)/2+i
            y_perm[len(y)/2+i] = i
    return y_perm


def compute_svm_score_nestedCV(K, y, n_folds,
                               scoring=balanced_accuracy_scoring,
                               random_state=None,
                               param_grid=[{'C': np.logspace(-5, 5, 25)}]):
    """Compute cross-validated score of SVM using precomputed kernel.
    """
    cv = StratifiedKFold(y, n_folds=n_folds, shuffle=True,
                         random_state=random_state)
    scores = np.zeros(n_folds)
    for i, (train, test) in enumerate(cv):
        cvclf = SVC(kernel='precomputed')
        y_train = y[train]
        cvcv = StratifiedKFold(y_train, n_folds=n_folds,
                               shuffle=True,
                               random_state=random_state)
        clf = GridSearchCV(cvclf, param_grid=param_grid, scoring=scoring,
                           cv=cvcv, n_jobs=1)
        clf.fit(K[train, :][:, train], y_train)
        # print clf.best_params_
        scores[i] = clf.score(K[test, :][:, train], y[test])

    return scores.mean()


def apply_svm(K, y, n_folds=5, iterations=10000, subjects=False, verbose=True,
              random_state=None):
    """
    Compute the balanced accuracy, its null distribution and the p-value.

    Parameters:
    ----------
    K: array-like
        Kernel matrix
    y: array_like
        class labels
    cv: Number of folds in the stratified cross-validation
    verbose: bool
        Verbosity

    Returns:
    -------
    acc: float
        Average balanced accuracy.
    acc_null: array
        Null distribution of the balanced accuracy.
    p_value: float
         p-value
    """
    # Computing the accuracy
    param_grid = [{'C': np.logspace(-5, 5, 20)}]
    if subjects:
        acc = compute_svm_subjects(K, y, n_folds)
    else:
        acc = compute_svm_score_nestedCV(K, y, n_folds, param_grid=param_grid,
                                         random_state=random_state)
    if verbose:
        print("Mean balanced accuracy = %s" % (acc))
        print("Computing the null-distribution.")

    # Computing the null-distribution

    # acc_null = np.zeros(iterations)
    # for i in range(iterations):
    #     if verbose and (i % 1000) == 0:
    #         print(i),
    #         stdout.flush()

    #     y_perm = np.random.permutation(y)
    #     acc_null[i] = compute_svm_score_nestedCV(K, y_perm, n_folds,
    #                                              param_grid=param_grid)

    # if verbose:
    #     print ''

    # Computing the null-distribution
    if subjects:
        yis = [permutation_subjects(y) for i in range(iterations)]
        acc_null = Parallel(n_jobs=-1)(delayed(compute_svm_subjects)(K, yis[i], n_folds) for i in range(iterations))
    else:
        yis = [np.random.permutation(y) for i in range(iterations)]
        acc_null = Parallel(n_jobs=-1)(delayed(compute_svm_score_nestedCV)(K, yis[i], n_folds, scoring=balanced_accuracy_scoring, param_grid=param_grid) for i in range(iterations))
    # acc_null = Parallel(n_jobs=-1)(delayed(compute_svm_cv)(K, yis[i], C=100., n_folds=n_folds) for i in range(iterations))

    p_value = max(1.0 / iterations, (acc_null > acc).sum()
                  / float(iterations))
    if verbose:
        print("p-value ~= %s \t (resolution : %s)" % (p_value, 1.0/iterations))

    return acc, acc_null, p_value


def apply_ktst(K, y, iterations=10000, subjects=False, verbose=True):
    """
    Compute MMD^2_u, its null distribution and the p-value of the
    kernel two-sample test.

    Parameters:
    ----------
    K: array-like
        Kernel matrix
    y: array_like
        class labels
    verbose: bool
        Verbosity

    Returns:
    -------
    mmd2u: float
        MMD^2_u value.
    acc_null: array
        Null distribution of the MMD^2_u
    p_value: float
         p-value
    """
    assert len(np.unique(y)) == 2, 'KTST only works on binary problems'

    # Assuming that the first m rows of the kernel matrix are from one
    # class and the other n rows from the second class.
    m = len(y[y == 0])
    n = len(y[y == 1])
    mmd2u = MMD2u(K, m, n)
    if verbose:
        print("MMD^2_u = %s" % mmd2u)
        print("Computing the null distribution.")
    if subjects:
        perms = [permutation_subjects_ktst(y) for i in range(iterations)]
        mmd2u_null = compute_null_distribution_given_permutations(K, m, n,
                                                                  perms,
                                                                  iterations)
    else:
        mmd2u_null = compute_null_distribution(K, m, n, iterations,
                                               verbose=verbose)

    p_value = max(1.0/iterations, (mmd2u_null > mmd2u).sum()
                  / float(iterations))
    if verbose:
        print("p-value ~= %s \t (resolution : %s)" % (p_value, 1.0/iterations))

    return mmd2u, mmd2u_null, p_value


def plot_null_distribution(stats, stats_null, p_value, data_name='',
                           stats_name='$MMD^2_u$', save_figure=True):
    """Plot the observed value for the test statistic, its null
    distribution and p-value.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111)
    prob, bins, patches = plt.hist(stats_null, bins=50, normed=True)
    ax.plot(stats, prob.max()/30, 'w*', markersize=15,
            markeredgecolor='k', markeredgewidth=2,
            label="%s = %s" % (stats_name, stats))

    ax.annotate('p-value: %s' % (p_value),
                xy=(float(stats), prob.max()/9.),  xycoords='data',
                xytext=(-105, 30), textcoords='offset points',
                bbox=dict(boxstyle="round", fc="1."),
                arrowprops={"arrowstyle": "->",
                            "connectionstyle": "angle,angleA=0,angleB=90,rad=10"},
                )
    plt.xlabel(stats_name)
    plt.ylabel('p(%s)' % stats_name)
    plt.legend(numpoints=1)
    plt.title('Data: %s' % data_name)

    if save_figure:
        save_dir = 'figures'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        stn = 'ktst' if stats_name == '$MMD^2_u$' else 'clf'
        fig_name = os.path.join(save_dir, '%s_%s.pdf' % (data_name, stn))
        fig.savefig(fig_name)
