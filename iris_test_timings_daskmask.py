#
# Tests to see if any timings significantly change with dask masked-arrays.
#

import numpy as np
import numpy.lib.recfunctions as nprec

from log_timings_utils import testdir_all_test_times
from nprec_utils import nprec_from_name_column_pairs, nprec_print
from iris_test_timings_analysis import compare_timings


def show_excursions(logs_dirpath, n_top_print=10):
    #
    # Quick thing to check the timing outliers in a set of timings-test logs.
    #
    print '\nLooking for timing outliers in {}\n'.format(logs_dirpath)
    filenames, testnames, file_test_times = \
        testdir_all_test_times(logs_dirpath)

    time_mean = np.mean(file_test_times, axis=0)
    time_std = np.std(file_test_times, axis=0, ddof=1)
    time_range = (np.max(file_test_times, axis=0) -
                  np.min(file_test_times, axis=0))
    # Create a sanitised version of the results, skipping those with zero stdev
    time_std_masked = np.ma.masked_array(time_std)
    time_std_masked[time_std <= 0.0] = 1.0
    time_std_masked[time_std <= 0.0] = np.ma.masked

    # Calculate a figure of merit = peak-2-peak / s.d.
    time_p2p_sds = time_range / time_std

    stats = nprec_from_name_column_pairs(
        ('testname', testnames),
        ('extremity-ratio', time_p2p_sds),
        ('time', time_mean),
        ('range', time_range),
        ('stdev', time_std)
        )

#    # HACK!!
#    # This made NO reasonable difference...
#    stats = stats[stats['time'] > 0.001]

    stats.sort(order='extremity-ratio')
    stats = stats[::-1]

    n_top_print = 10
    print_specs = [('testname', 190),
                   'extremity-ratio', 'time', 'range', 'stdev']

    print 'WORST-{} ONLY ...'.format(n_top_print)
    nprec_print(stats[:n_top_print], print_specs)


if __name__ == '__main__':
    basis_dirpath = './results/results_20170825_master_v20a0_7ec22fa3f'
    probe_dirpath = './results/results_20170825_dask-mask-array_2b54b7b0c'
    basis_name = 'basis'
    probe_name = 'dask-mask'

#    # Attempt to show up any outlying values was not interesting.
#    show_excursions(basis_dirpath)
#    show_excursions(probe_dirpath)

    custom_excludes = ['test_coding_standards', 'plot']
    compare_timings(basis_dirpath, basis_name, probe_dirpath, probe_name,
                    custom_excludes)

