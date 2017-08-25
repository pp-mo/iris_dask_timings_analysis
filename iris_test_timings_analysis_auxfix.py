#
# Analyse timing differences on dask branch before + after fixing AuxFactories.
#  NOTE: reconstructed code, re-cast in terms of generic operation.
#
import os
import os.path

from iris_test_timings_analysis import compare_timings


notebooks_dirpath = './results'

basis_dirpath = os.path.join(notebooks_dirpath,
                            'results_temp_dask_b559c71')
basis_name = 'basis'

probe_dirpath = os.path.join(notebooks_dirpath,
                            'results_20170626_auxfix_8dab554bf')
probe_name = 'auxfix'

custom_excludes = ['test_coding_standards', 'plot', 'aux_factory']

compare_timings(basis_dirpath, 'basis', probe_dirpath, probe_name,
                custom_excludes)
