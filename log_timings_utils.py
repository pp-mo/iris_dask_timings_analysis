import glob
import os
import os.path

import numpy as np

def testdir_results_files(dirpath):
    # List the ".tst" files in the directory.
    # E.G. dir = 'notebooks_link/results/results_temp_dask_basis_c3dd97d'
    # E.G. filename = 'testlog_dask_basis_c3dd97d_01_20170523.txt'
    return glob.glob(os.path.join(dirpath, '*.txt'))

def get_timing_lines(source):
    # Grab *valid* test outout lines
    lines = [line.strip() for line in source.readlines()
             if 'TEST TIMING' in line]
    # Filter for valid lines only (N.B. some can be corrupted by interleaved test output)
    lines = [line for line in lines if line.startswith('TEST') and line.endswith('sec.')]
    return lines

def get_file_timing_lines(filepath):
    with open(filepath) as open_file:
        result = get_timing_lines(open_file)
    return result

def line_to_testname_and_duration(line):
    # line e.g. : 'TEST TIMING -- "path.path.path.<testclass>.<testname>" took :   <num>  sec.'
    _, testname, rest = line.split('"')
    # rest e.g. " took :   <num>  sec."
    _took, _colon, num, _secs = rest.split() 
    duration = float(num)
    return testname, duration

def filepath_test_timings(filepath):
    # Return {testname: timing} for one file.
    test_timings = {}
    for line in get_file_timing_lines(filepath):
        testname, duration = line_to_testname_and_duration(line)
        test_timings[testname] = duration
    return test_timings

def dirpath_raw_results(dirpath):
    # Return {filename: {testname: timing}} for all files in directory.
    return {os.path.basename(filepath): filepath_test_timings(filepath)
            for filepath in testdir_results_files(dirpath)}

def dirpath_combined_results(dirpath):
    #
    # Suck in + translate all test output files from one directory.
    # Work out which tests are common to all, and report on those not.
    # Return :
    #    (filenames, testnames, file_test_timings, report_lines)
    #    Where:
    #        'filenames' is a sorted list of filenames (not paths).
    #        'testnames' is a sorted list of the tests present in all files.
    #        'file_test_timings' is {file: {test: duration}}
    #            where 'file' and 'test' range over 'filenames' and 'testnames'
    #        'report_lines' is a list of strings
    #
    raw_results = dirpath_raw_results(dirpath)
    file_testname_sets = {}
    common_testnames = None
    all_testnames = set()
    for filepath, results in raw_results.iteritems():
        file_tests_set = set(results.keys())
        file_testname_sets[filepath] = file_tests_set
        all_testnames |= file_tests_set
        if common_testnames is None:
            common_testnames = file_tests_set
        else:
            common_testnames &= file_tests_set
    # Build report message lines.
    report_lines = []
    msg = 'In directory {}'.format(dirpath)
    report_lines.append(msg)
    msg = '  n-files = {}'.format(len(raw_results))
    report_lines.append(msg)
    msg = '  n testnames common to all = {}'.format(len(common_testnames))
    report_lines.append(msg)
    non_common_testnames = all_testnames - common_testnames
    msg = '  n testnames not in all = {} '.format(len(non_common_testnames))
    report_lines.append(msg)

    # Construct an account of the missing ones...
    filenames_sorted = sorted(raw_results.keys())
    def files_presence_string(testname):
        return ''.join(('1' if testname in file_testname_sets[filepath]
                        else '0')
                       for filepath in filenames_sorted)
    test_presence_patterns = {
        testname: files_presence_string(testname)
        for testname in non_common_testnames}
    all_presence_patterns = set(test_presence_patterns.values())
    presence_patterns_occur = [
        (this_pattern, len([0 for testname, a_pattern
                            in test_presence_patterns.iteritems()
                            if a_pattern == this_pattern]))
        for this_pattern in all_presence_patterns]
    presence_patterns_occur = sorted(
        presence_patterns_occur, key=lambda (pattern, n): n)
    for pattern, occurs in presence_patterns_occur[::-1]:
        msg = '    presence pattern {} occurs for {} tests'
        msg = msg.format(pattern, occurs)
        report_lines.append(msg)

    # Construct sorted list of testnames to return.
    testnames_sorted = sorted(common_testnames)

    # Construct 'pruned' timings result dicts to return.
    def pruned_test_timings(test_timings):
        return {testname: test_timings[testname]
                for testname in common_testnames}
    file_test_timings = {
        filepath: pruned_test_timings(raw_results[filepath])
        for filepath in raw_results.keys()}

    return filenames_sorted, testnames_sorted, file_test_timings, report_lines


def testdir_timings(dirpath,
                    exclude_matches=['test_coding_standards', 'plot']):

    """

    Analyse test timing log outputs in the directory and return arrays of
    aggregated timing data.

    Args:

    * dirpath (string):
       path of directory to seach for logfiles ('*.txt').

    * exclude_matches (list of string):
        sub-strings defining tests excluded from the results.
        Tests whose names contain one of these are removed from the analysis.

    Note: any tests not present in all input files are also excluded.

    Returns :

       testnames, testtime_means, testtime_sds

    where:
       * 'testnames' is a (sorted) array of test identity strings.
         NOTE: only those common to all logfiles.
       * 'testtime_means' is an array of mean test times, in seconds.
       * 'testtime_sds' is a corresponding array of standard deviations.

    """
    filenames, testnames, file_test_timings, report_lines = \
        dirpath_combined_results(dirpath)

    print '\n'.join(report_lines)

    # Make a big array of all the results
    all_file_test_times = np.array([[file_test_timings[filename][testname]
                                     for testname in testnames]
                                    for filename in filenames])

    # Remove some known troublemakers + ones we have no basic interest in ...
    testnames = np.array(testnames)

    def testnames_exclude_mask(name_finder):
        return np.array([name_finder in testname
                         for testname in testnames])

    for name_finder in exclude_matches:
        exclude_mask = testnames_exclude_mask(name_finder)
        testnames = testnames[~exclude_mask]
        all_file_test_times = all_file_test_times[:, ~exclude_mask]

    time_mean = np.mean(all_file_test_times, axis=0)
    time_std = np.std(all_file_test_times, axis=0, ddof=1)
    time_var = time_std ** 2

    return testnames, time_mean, time_std

