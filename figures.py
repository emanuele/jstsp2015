"""Code to generate figures of the manuscript.

Author: Sandro Vega-Pons, Emanuele Olivetti
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import scipy.stats as ss


sites = ["Beijing_Zang",
         "Berlin_Margulies",
         "Cambridge_Buckner",
         "Cleveland",
         "Dallas",
         "ICBM",
         "Leipzig",
         "Milwaukee",
         "NewYork",
         "Oulu",
         "Oxford",
         "SaintLouis",
         "Schizophrenia",
         "Uri"]


# Results of the p-value KTST:

DCE = [0.0004,
       0.0092,
       0.0001,
       0.4346,
       0.5451,
       0.1422,
       0.0085,
       0.057,
       0.1128,
       0.0001,
       0.0111,
       0.9218,
       0.004]


DRE = [0.0001,
       0.054,
       0.0108,
       0.7052,
       0.714,
       0.2102,
       0.0089,
       0.0365,
       0.1708,
       0.1484,
       0.1048,
       0.9378,
       0.008]


WL = [0.0399,
      0.6185,
      0.2577,
      0.3433,
      0.1226,
      0.5177,
      0.5873,
      0.1204,
      0.151,
      0.3319,
      0.1984,
      0.6759,
      0.0290,
      0.01]


SP = [0.003,
      0.9106,
      0.0472,
      0.6589,
      0.0572,
      0.9835,
      0.0057,
      0.0629,
      0.171,
      0.5175,
      0.1712,
      0.578,
      0.0682,
      0.0165]


if __name__ == '__main__':

    p_value_threshold = 0.05
    xmin, xmax = [0.00005, 1.1]
    ymin, ymax = [0.00005, 1.1]

    plt.interactive(True)

    DCE = np.array(DCE)
    DRE = np.array(DRE)
    WL = np.array(WL)
    SP = np.array(SP)

    np.set_printoptions(precision=3, suppress=True)
    plt.figure()
    pvalues_cbt_all = []
    for name, data, symb in [('DCE', DCE, 'ro'), ('DRE', DRE, 'ks'),
                             ('WL', WL, 'bv'), ('SP', SP, 'gx')]:
        accuracies = []
        p_values_acc = []
        for location in sites:
            filename = "results_dic" + os.path.sep + "%s_%s.pkl" % (location, name.lower())
            if location == 'Uri' and (name == 'DCE' or name == 'DRE'):
                print('Skipping Uri dataset for %s' % name)
                continue

            tmp = pickle.load(open(filename))
            accuracies.append(tmp['accuracies'])
            p_values_acc.append(tmp['p_values'])

        accuracies = np.array(accuracies)
        p_values_acc = np.array(p_values_acc)
        p_values_acc_median = np.median(p_values_acc, 1)
        pvalues_cbt_all.append(p_values_acc_median)
        p_values_acc_min = p_values_acc_median - np.array(p_values_acc).min(1)
        p_values_acc_max = np.array(p_values_acc).max(1) - p_values_acc_median
        print(name)
        print("median balanced accuracy")
        print(np.median(accuracies, 1)[:, None])
        print("p-value KTST , p-value CBT")
        print(np.array([data, p_values_acc_median]).T)
        print("")

        plt.plot(data, p_values_acc_median, symb, markersize=10, label=name)
        plt.plot([p_value_threshold, p_value_threshold], [ymin, ymax], 'g--')
        plt.plot([xmin, xmax], [p_value_threshold, p_value_threshold], 'g--')

    plt.ylabel('CBT $p$-value')
    plt.xlabel('KTST $p$-value')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlim([xmin, xmax])
    plt.ylim([ymin, ymax])
    plt.legend(loc='upper left')
    plt.savefig('pvalues_KTST_vs_CBT.pdf')

    for name, data, symb in [('DCE', DCE, 'ro'), ('DRE', DRE, 'ks'),
                             ('WL', WL, 'bv'), ('SP', SP, 'gx')]:
        print(name)
        accuracies = []
        p_values_acc = []
        for location in sites:
            filename = "results_dic" + os.path.sep + "%s_%s.pkl" % (location, name.lower())
            if location == 'Uri' and (name == 'DCE' or name == 'DRE'):
                print('Skipping Uri dataset for %s' % name)
                continue

            tmp = pickle.load(open(filename))
            accuracies.append(tmp['accuracies'])
            p_values_acc.append(tmp['p_values'])

        pva = np.array(p_values_acc)

        plt.figure()
        plt.boxplot(pva.T)
        plt.plot([0, pva.shape[0]+1], [p_value_threshold, p_value_threshold],
                 'g--')
        sites_short = [site.split('_')[0] for site in sites]
        sites_short[sites_short.index('Schizophrenia')] = 'Schizophr.'
        sites_short[sites_short.index('Uri')] = 'Contex.Dis.'
        plt.xticks(range(1, pva.shape[0] + 1), sites_short, rotation=30)
        plt.ylabel('%s CBT $p$-value' % name)
        plt.yscale('log')
        plt.ylim([ymin, ymax])
        plt.savefig('pvalues_CBT_%s_boxplot.pdf' % name)

    print("")
    print("Correlation between pvalues of KTST and CBT:")
    pvalues_all = np.vstack([np.concatenate([DCE, DRE, WL, SP]),
                             np.concatenate(pvalues_cbt_all)]).T
    print("Pearson r:")
    print(np.corrcoef(pvalues_all.T))
    spearmanr = ss.spearmanr(pvalues_all)[0]
    print("Spearman r = %s" % spearmanr)

    np.random.seed(0)
    iterations = 10000
    null_distribution = np.zeros(iterations)
    for i in range(iterations):
        idx = np.random.permutation(range(pvalues_all.shape[0]))
        pvalues_all_permuted = np.vstack([pvalues_all[idx, 0],
                                          pvalues_all[:, 1]]).T
        null_distribution[i] = ss.spearmanr(pvalues_all_permuted)[0]

    print("p-value (permutations) = %s" % (null_distribution >
                                           spearmanr).mean())
