"""Simulation estimating Type I and Type II error of CBT and KTST.

Author: Sandro Vega-Pons, Emanuele Olivetti
"""

import numpy as np
from sklearn.metrics.pairwise import pairwise_distances
from kernel_two_sample_test import MMD2u, compute_null_distribution
from sklearn.svm import SVC
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import StratifiedKFold, cross_val_score
# from multiprocessing import cpu_count
from joblib import Parallel, delayed

# Temporarily stop warnings to cope with the too verbose sklearn
# GridSearchCV.score warning:
import warnings
warnings.simplefilter("ignore")


# boundaries for seeds generation during parallel processing:
MAX_INT = np.iinfo(np.uint32(1)).max
MIN_INT = np.iinfo(np.uint32(1)).min


def estimate_pvalue(score_unpermuted, scores_null):
    iterations = len(scores_null)
    p_value = max(1.0/iterations, (scores_null > score_unpermuted).sum() /
                  float(iterations))
    return p_value


def compute_svm_score(K, y, n_folds, scoring='accuracy', random_state=0):
    cv = StratifiedKFold(y, n_folds=n_folds, shuffle=True,
                         random_state=random_state)
    clf = SVC(C=1.0, kernel='precomputed')
    scores = cross_val_score(clf, K, y, scoring=scoring, cv=cv, n_jobs=1)
    score = scores.mean()
    return score


def compute_svm_score_nestedCV(K, y, n_folds, scoring='accuracy',
                               random_state=None,
                               param_grid=[{'C': np.logspace(-5, 5, 20)}]):
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
        clf.fit(K[:, train][train, :], y_train)
        scores[i] = clf.score(K[test, :][:, train], y[test])

    return scores.mean()


if __name__ == '__main__':

    np.random.seed(0)
    print("JSTSP Simulation Experiments.")
    nA = 20  # size of class A
    nB = 20  # size of class B
    d = 5  # number of dimensions
    # separation between the two normally-distributed classes:
    delta = 0.75
    twist = np.ones(d)
    print("nA = %s" % nA)
    print("nB = %s" % nB)
    print("d = %s" % d)
    print("delta = %s" % delta)
    print("twist = %s" % twist)

    muA = np.zeros(d)
    muB = np.ones(d) * delta
    covA = np.eye(d)
    covB = np.eye(d) * twist

    seed_data = 0  # random generation of data
    rng_data = np.random.RandomState(seed_data)

    seed_ktst = 0  # random permutations of KTST
    rng_ktst = np.random.RandomState(seed_ktst)

    seed_cv = 0  # random splits of cross-validation
    rng_cv = np.random.RandomState(seed_cv)

    svm_param_grid = [{'C': np.logspace(-5, 5, 20)}]
    # svm_param_grid = [{'C': np.logspace(-3, 2, 10)}]

    repetitions = 100
    print("This experiments will be repeated on %s randomly-sampled datasets."
          % repetitions)

    scores = np.zeros(repetitions)
    p_value_scores = np.zeros(repetitions)
    mmd2us = np.zeros(repetitions)
    p_value_mmd2us = np.zeros(repetitions)
    for r in range(repetitions):
        print("")
        print("Repetition %s" % r)

        A = rng_data.multivariate_normal(muA, covA, size=nA)
        B = rng_data.multivariate_normal(muB, covB, size=nB)

        X = np.vstack([A, B])
        y = np.concatenate([np.zeros(nA), np.ones(nB)])

        distances = pairwise_distances(X, metric='euclidean')
        sigma2 = np.median(distances) ** 2.0
        K = np.exp(- distances * distances / sigma2)
        # K = X.dot(X.T)

        iterations = 10000
        mmd2u_unpermuted = MMD2u(K, nA, nB)
        print("mmd2u: %s" % mmd2u_unpermuted)
        mmd2us[r] = mmd2u_unpermuted

        mmd2us_null = compute_null_distribution(K, nA, nB, iterations,
                                                random_state=rng_ktst)
        p_value_mmd2u = estimate_pvalue(mmd2u_unpermuted, mmd2us_null)
        print("mmd2u p-value: %s" % p_value_mmd2u)
        p_value_mmd2us[r] = p_value_mmd2u

        scoring = 'accuracy'
        n_folds = 5
        iterations = 1
        # score_unpermuted = compute_svm_score_nestedCV(K, y, n_folds,
        #                                               scoring=scoring,
        #                                               random_state=rng_cv)

        rngs = [np.random.RandomState(rng_cv.randint(low=MIN_INT, high=MAX_INT)) for i in range(iterations)]
        scores_unpermuted = Parallel(n_jobs=-1)(delayed(compute_svm_score_nestedCV)(K, y, n_folds, scoring, rngs[i], param_grid=svm_param_grid) for i in range(iterations))
        score_unpermuted = np.mean(scores_unpermuted)
        print("accuracy: %s" % score_unpermuted)
        scores[r] = score_unpermuted

        # print("Doing permutations:"),
        iterations = 100
        scores_null = np.zeros(iterations)

        # for i in range(iterations):
        #     if (i % 10) == 0:
        #         print(i)

        #     yi = rng_cv.permutation(y)
        #     scores_null[i] = compute_svm_score_nestedCV(K, yi, n_folds,
        #                                                 scoring=scoring,
        #                                                 random_state=rng_cv)

        rngs = [np.random.RandomState(rng_cv.randint(low=MIN_INT, high=MAX_INT)) for i in range(iterations)]
        yis = [np.random.permutation(y) for i in range(iterations)]
        scores_null = Parallel(n_jobs=-1)(delayed(compute_svm_score_nestedCV)(K, yis[i], n_folds, scoring, rngs[i], param_grid=svm_param_grid) for i in range(iterations))
        p_value_score = estimate_pvalue(score_unpermuted, scores_null)
        p_value_scores[r] = p_value_score

        print("%s p-value: %s" % (scoring, p_value_score))

        p_value_threshold = 0.05
        mmd2u_power = (p_value_mmd2us[:r+1] <= p_value_threshold).mean()
        scores_power = (p_value_scores[:r+1] <= p_value_threshold).mean()
        print("p_value_threshold: %s" % p_value_threshold)
        print("Partial results - MMD2u: %s , %s: %s" %
              (mmd2u_power, scoring, scores_power))

    print("")
    print("FINAL RESULTS:")
    p_value_threshold = 0.1
    print("p_value_threshold: %s" % p_value_threshold)
    mmd2u_power = (p_value_mmd2us <= p_value_threshold).mean()
    scores_power = (p_value_scores <= p_value_threshold).mean()
    print("MMD2u Power: %s" % mmd2u_power)
    print("%s Power: %s" % (scoring, scores_power))
    print("")
    p_value_threshold = 0.05
    print("p_value_threshold: %s" % p_value_threshold)
    mmd2u_power = (p_value_mmd2us <= p_value_threshold).mean()
    scores_power = (p_value_scores <= p_value_threshold).mean()
    print("MMD2u Power: %s" % mmd2u_power)
    print("%s Power: %s" % (scoring, scores_power))
    print("")
    p_value_threshold = 0.01
    print("p_value_threshold: %s" % p_value_threshold)
    mmd2u_power = (p_value_mmd2us <= p_value_threshold).mean()
    scores_power = (p_value_scores <= p_value_threshold).mean()
    print("MMD2u Power: %s" % mmd2u_power)
    print("%s Power: %s" % (scoring, scores_power))
    print("")
