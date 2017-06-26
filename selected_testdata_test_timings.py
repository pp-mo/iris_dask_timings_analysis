from selected_testdata_test_names import selected_testdata_test_names

import os
import os.path

import numpy as np
import numpy.lib.recfunctions as nprec

from nprec_utils import nprec_from_name_column_pairs, nprec_print
from log_timings_utils import testdir_timings

notebooks_dirpath = './results'
#basis_dirpath = os.path.join(notebooks_dirpath,
#                            'results_20170626_latest_cb2d53c77')
basis_dirpath = os.path.join(notebooks_dirpath,
                            'results_temp_dask_b559c71')
testnames, means, sds = testdir_timings(basis_dirpath)

class Ss(object):
    def __getitem__(self, keys):
        return keys

ss = Ss()

def dotsel(name_string, keys):
    bits = name_string.split('.')
    bits = bits[keys]
    return '.'.join(bits)

def sl(iterable):
    return '\n'.join(str(x) for x in iterable)

def ind(string, n_indent=2):
    s_indent = ' ' * n_indent
    return '\n'.join(s_indent + x for x in string.split('\n'))

#sel_egs = selected_testdata_test_names[:25]
#print 'Select examples: '
#print ind(sl(sel_egs))
#print '  cut --> '
#print ind(sl([dotsel(sel, ss[:-1]) for sel in sel_egs]), '    ')
#
#print
#data_egs = testnames[:25]
#print 'Data examples: '
#print ind(sl(data_egs))
#print '  cut --> '
#print ind(sl([dotsel(data, ss[:-2]) for data in data_egs]), '    ')

test_classes = [dotsel(name, ss[:-1]) for name in testnames]
data = nprec_from_name_column_pairs(
    ('fullname', testnames),
    ('test_class', test_classes),
    ('time', means))

selected_test_classes = sorted(list(set(selected_testdata_test_names)))
data_inds = np.array([test_class in selected_test_classes
                      for test_class in test_classes])
selected_data = data[data_inds]
print 'N selected test classes = ', len(selected_test_classes)
print 'N selected tests = ', len(selected_data)

#print
n_print = 50
#print 'Top {} tests, by time taken...'.format(n_print)
#selected_data = selected_data[np.argsort(selected_data['time'])][::-1]
#
#print nprec_print(selected_data[:n_print],
#                  names_widths_formats=[('fullname', 150), 'time'])

#
# Group by test classes + work out each class total
#
class_totals_map = {}
for class_name in selected_test_classes:
    data_inds = selected_data['test_class'] == class_name
    class_data = selected_data[data_inds]
    class_totals_map[class_name] = np.sum(class_data['time'])

#
# Form new array + sort on class
#
class_totals = [class_totals_map[class_name]
                for class_name in selected_data['test_class']]

class_sorted_data = nprec_from_name_column_pairs(
    ('fullname', selected_data['fullname']),
    ('test_class', selected_data['test_class']),
    ('test_name', [dotsel(name, ss[-1:])
                   for name in selected_data['fullname']]),
    ('class_total', class_totals),
    ('time', selected_data['time']))

class_sorted_data = class_sorted_data[np.argsort(class_totals)[::-1]]

total_time = np.sum(class_totals_map.values())
print
print 'Total extra time taken = ', total_time




# Apply a significance threshold to 'n% of total time'.
percent_thresh = 90.0

accum_time_thresh = total_time * percent_thresh / 100.0
test_time_accums = np.cumsum(class_sorted_data['time'])
index_of_first_time_over = np.where(test_time_accums > accum_time_thresh)[0][0]
class_thresh = class_sorted_data[index_of_first_time_over]['test_class']
index_of_last_in_thresh_class = np.where(
    class_sorted_data['test_class'] == class_thresh)[0][-1]

# trim to that point + show those results ...
trimmed_data = class_sorted_data[:index_of_last_in_thresh_class + 1]

print
print 'Tests taking {}% of time are {} of {}'.format(
    int(percent_thresh), len(trimmed_data), len(class_sorted_data))
print
print 'The {}%-of-time tests, sorted by total-class time :'.format(
    int(percent_thresh))

max_classname_len = max(len(x) for x in trimmed_data['test_class'])
max_testname_len = max(len(x) for x in trimmed_data['test_name'])
print nprec_print(
    trimmed_data,
    [('test_class', max_classname_len + 2),
     'class_total',
     ('test_name', max_testname_len + 2),
     'time'])

#
# NOW compare that to older pre-dask results ...
#
predask_dirpath = os.path.join(notebooks_dirpath,
                               'results_temp_dask_basis_c3dd97d')
predask_names, predask_means, predask_sds = testdir_timings(predask_dirpath)
predask_classes = [dotsel(name, ss[:-1]) for name in predask_names]
key_classes = list(set(trimmed_data['test_class']))
# Get these sorted into total-time significance order...
key_class_totals = [class_totals_map[classname] for classname in key_classes]
key_classes = np.array(key_classes)
key_classes = key_classes[np.argsort(key_class_totals)[::-1]]

predask_totals = {}
for key_class in key_classes:
    predask_totals[key_class] = sum(
        mean
        for mean, classname in zip(predask_means, predask_classes)
        if classname == key_class)

print
print 'Total difference of these 90% w.r.t. old "pre-dask":'
new_majority_total = sum(key_class_totals)
old_majority_total = sum(predask_totals.values())
print '  selected "{:2d}%" tests total time = {:9.6f}'.format(
    int(percent_thresh), new_majority_total)
print '  equivalent pre-dask total time  =  {:9.6f}'.format(
    old_majority_total)

# Now print a comparison table..
print
print 'Per-class differences to old "pre-dask":'
fmt = '{}: new={:9.6f} old-ref={:9.6f} diff={:12.9} relative={:12.9f}'
for key_class in key_classes:
    this_total = class_totals_map[key_class]
    predask_total = predask_totals[key_class]
    print fmt.format(key_class.rjust(max_classname_len),
                     this_total,
                     predask_total,
                     this_total - predask_total,
                     this_total / predask_total)
