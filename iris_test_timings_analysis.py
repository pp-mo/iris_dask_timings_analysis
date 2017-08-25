#
# Analyse test timing differences between two different iris commits.
#
# Begins with files piped from iris full test runs.
# This has to use unittest discover, as it doesn't work with nose.
# Done like this :
#   $ IRIS_TEST_TIMINGS=1 python -m unittest discover -v iris.tests >myfile.txt 2>&1
#
# ALSO while you're about it, you may also need to plugin the test-data
# E.G. $ export OVERRIDE_TEST_DATA_REPOSITORY=~/git/iris-test-data/test_data
#

import os
import os.path

import numpy as np
import numpy.lib.recfunctions as nprec

from nprec_utils import nprec_from_name_column_pairs, nprec_print
from log_timings_utils import testdir_timings

#
# =========================== ACTUAL CORE ANALYSIS ===========================
#

#
# Read two sets of timing testfiles: 'before' and 'after' a change.
# These are referred to as "basis" and "probe".
#

def compare_timings(basis_dirpath, basis_name, probe_dirpath, probe_name,
                    custom_excludes=['test_coding_standards', 'plot']):

    basis_tests, basis_means, basis_sds = testdir_timings(
        basis_dirpath, exclude_matches=custom_excludes)
    probe_tests, probe_means, probe_sds = testdir_timings(
        probe_dirpath, exclude_matches=custom_excludes)


    # Calculate string dtype wide enough to store any test names.
    max_strlen = max(max(len(testname) for testname in tests)
                     for tests in (basis_tests, probe_tests))
    test_str_dtype = '|S{}'.format(max_strlen + 4)
    basis_tests = np.array(basis_tests, dtype=test_str_dtype)
    probe_tests = np.array(probe_tests, dtype=test_str_dtype)

    # Make structured arrays
    basis = nprec_from_name_column_pairs(
        ('test', basis_tests),
        ('mean', basis_means),
        ('sd', basis_sds))

    probe = nprec_from_name_column_pairs(
        ('test', probe_tests),
        ('mean', probe_means),
        ('sd', probe_sds))

    #import matplotlib.pyplot as plt
    #plt.plot(basis_means, basis_sds, '.', color='red', label=basis_name)
    #plt.plot(probe_means, probe_sds, '.', color='blue', label=probe_name)
    #plt.legend()
    #
    #plt.show()

    #
    # Report on similarities and differences in the set of tests recorded.
    #
    basis_testset = set(basis['test'])
    probe_testset = set(probe['test'])
    common_tests_set = basis_testset & probe_testset
    basis_not_probe_tests = basis_testset - probe_testset
    probe_not_basis_tests = probe_testset - basis_testset
    for name, testset in [('common', common_tests_set),
                          (probe_name + '-only', probe_not_basis_tests),
                          (basis_name + '-only', basis_not_probe_tests)]:
        msg = 'n tests in {} = {}'
        print msg.format(name, len(testset))

    # Prune both to the common set of tests and combine into a single structure.
    basis_prune_mask = np.array([testname in common_tests_set
                                 for testname in basis['test']])
    basis = basis[basis_prune_mask]

    probe_prune_mask = np.array([testname in common_tests_set
                                for testname in probe['test']])
    probe = probe[probe_prune_mask]

    assert np.all(basis['test'] == probe['test'])
    basis_t_name = 't_' + basis_name
    probe_t_name = 't_' + probe_name
    data = nprec_from_name_column_pairs(
        ('test', basis['test']),
        (basis_t_name, basis['mean']),
        (probe_t_name, probe['mean']))

    # Compute all the comparison measures.
    avs = 0.5 * (probe['mean'] + basis['mean'])
    data = nprec.rec_append_fields(data, 't_avg', avs)

    # Crudely calculate an 'average' error level ("serr") between the two.
    # (N.B. not really right if different numbers of results in each set).
    ref_serrs = np.sqrt(0.5 * (probe['sd'] ** 2 + basis['sd'] ** 2))

    # Add a fraction of the average --> lower limit on the sds.
    # -- this stops results with very small sds dominating.
    SD_MIN_FRAC = 0.02  # 2% for now ?
    ref_serrs += SD_MIN_FRAC * data['t_avg']

    data = nprec.rec_append_fields(data, 't_serr', ref_serrs)

    # Calculate:
    #   * difference: diff = probe_mean - basis_mean
    #   * 'reference' error level: ref_relerr  = sd / average
    #   * relative error: relerr = diff / average
    ref_relerrs = ref_serrs / avs
    data = nprec.rec_append_fields(data, 'ref_relerr', ref_relerrs)
    diffs = data[probe_t_name] - data[basis_t_name]
    data = nprec.rec_append_fields(data, 't_diff', diffs)
    relerrs = diffs / ref_serrs
    data = nprec.rec_append_fields(data, 'relerr', relerrs)

    # Calculate the magic 'significance estimator': sig = relerr / ref_relerr
    sigs = np.abs(relerrs / ref_relerrs)
    data = nprec.rec_append_fields(data, 'SIG', sigs)

    # Add a column showing overall "slowdown factor" (ratio).
    slowage_factor = data[probe_t_name] / data[basis_t_name]
    data = nprec.rec_append_fields(data, 'n_slower', slowage_factor)

    # Sort the whole lot on the significance statistic, downwards.
    sort_sig_inds = np.argsort(data['SIG'])[::-1]
    data = data[sort_sig_inds]

    # Print the top of the pile.
    n_print = 25
    print_specs = [('test', 190),
                   basis_t_name, probe_t_name, 'n_slower',
                   'SIG', 'relerr']

    def rep_across(msg, width=300):
        n_msg = len(msg)
        n_rep = 1 + (width // n_msg)
        return ((msg + ' ::: ') * n_rep)[:width]

    print
    print rep_across('Top {} SIGs:'.format(n_print))
    nprec_print(data[:n_print], print_specs)

    #print
    #print 'Bottom SIGs:'
    ## N.B. need to distinguish where diffs are sensibly > 0.0...
    #tst = data[np.abs(data['t_diff']) > 0.0000005]
    #nprec_print(tst[-n_print:], print_specs)

    print
    print 'Silly ones? : abs(t_diff) < 1uSec  '
    tst = data[np.abs(data['t_diff']) < 1.0e-6]
    print 'NUMBER = ', len(tst)

    print
    print rep_across('Best {} FASTER ones:'.format(n_print))
    tst = data[data['t_diff'] < 0]
    nprec_print(tst[:n_print], print_specs)


    print
    print rep_across('Worst {} SLOWER ones:'.format(n_print))
    tst = data[data['t_diff'] > 0]
    nprec_print(tst[:(n_print * 4)], print_specs)


    """
    What we happen to find in this many worst-slower ones (top 100)...
     * the lowest SIG is about 1/10th the biggest : 3861 ~ 371.5
     * typical 'slowness' factors drop from typically 10-20 at the top of the table
       to around 1.3-2.0 at the bottom.

    So, think high-SIGs are a reasonable place to look for any serious problems.

    Let's bin those results by Test-CLASS to see *where* problems may lie ...

    """
    n_sel = 100
    # Split apart testname + testclass name, and record as separate columns.
    tst = tst[:n_sel]
    test_names = [test.split('.')[-1]
                  for test in tst['test']]
    test_classes = ['.'.join(test.split('.')[:-1])
                    for test in tst['test']]
    tst = nprec.rec_append_fields(tst, 'testclass', test_classes)
    tst = nprec.rec_append_fields(tst, 'testname', test_names)

    # Against each test, record the worst-case SIG value for its testclass.
    worst_sigs = tst['SIG'].copy()
    for testclass in set(test_classes):
        members_mask = (tst['testclass'] == testclass)
        worst_sig = np.max(tst[members_mask]['SIG'])
        worst_sigs[members_mask] = worst_sig

    tst = nprec.rec_append_fields(tst, 'worst-sig', worst_sigs)

    #
    # Sort by these to show :
    #  * all worst tests ..
    #  * .. grouped into testclasses ..
    #  * .. worst testclasses first
    #
    tst.sort(order=('worst-sig', 'SIG'))
    tst = tst[::-1]
    print_specs2 = [('testclass', 190), ('testname', 60),
                   basis_t_name, probe_t_name, 'n_slower', 'relerr',
                   'SIG', 'worst-sig']

    print
    msg = 'Top {} by SIG in TestClasses.'.format(n_sel)
    print rep_across(msg)
    nprec_print(tst, print_specs2)

    ## Plot all results: log(SIG) / log(t)
    #import matplotlib.pyplot as plt
    #plt.plot(np.log10(data['t_avg']), np.log10(data['SIG']), '.', color='red', label='basis')
    #plt.title('log-SIG v. log-t')
    #plt.legend()
    #
    #plt.show()


if __name__ == '__main__':
    notebooks_dirpath = './results'
    basis_dirpath = os.path.join(notebooks_dirpath,
                                'results_temp_dask_basis_c3dd97d')
    probe_dirpath = os.path.join(notebooks_dirpath,
                                 'results_temp_dask_b559c71')

    basis_name = 'basis'
    probe_name = 'dask'

    custom_excludes = ['test_coding_standards', 'plot']

    compare_timings(basis_dirpath, basis_name, probe_dirpath, probe_name,
                    custom_excludes)

