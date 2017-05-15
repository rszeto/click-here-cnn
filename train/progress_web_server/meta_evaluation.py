import os
import sys
import re
import numpy as np
import glob

# Import global variables
SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.dirname(os.path.dirname(SCRIPT_DIR)))
from global_variables import *

# Array of tuples specifying name for value, regex to use to find value, and whether to reverse-sort
display_info = [
    ('Mean accuracy', 'Mean accuracy: (([0-9]|\.)+)', True),
    ('Mean MedErr', 'Mean medErr: (([0-9]|\.)+)', False)
]
# Tuple specifying value to define overall performance with
overall_perf_info = display_info[0]

# Create mapping from models (i.e. weights for a given architecture) to values extracted from result file
def get_model_values_map(only_include_experiments=None):
    model_values_map = {}
    exp_names = only_include_experiments if only_include_experiments else os.listdir(g_experiments_root_folder)

    for exp_name in exp_names:
        # Get experiment number from name (first 6 characters)
        exp_num = int(exp_name[:6])
        # Get list of evaluation result files
        eval_file_paths = glob.glob(os.path.join(g_experiments_root_folder, exp_name, 'evaluation', 'acc_mederr_*.txt'))

        for eval_file_path in eval_file_paths:
            # Get iteration number
            iter_num = int(re.findall('\d+', os.path.basename(eval_file_path))[0])

            # Find the display info in the evaluation file
            with open(eval_file_path, 'r') as f:
                lines = f.read()
            cur_display_info_dict = {}
            for name, regex, reverse_sort in display_info:
                m = re.search(regex, lines, re.MULTILINE)
                if m is not None:
                    cur_display_info_dict[name] = float(m.group(1))
            
            # Add to model-values map
            model_values_map[(exp_num, iter_num)] = cur_display_info_dict

    return model_values_map

def sort_models_by_indiv_perf(model_values_map, perf_info):
    '''
    Sort the models for the given experiment by the given performance information
    :param model_values_map: The evaluation values for the experiment of interest
    :param perf_info: The display_info tuple associated with the metric of interest
    '''
    name, regex, reverse_sort = perf_info
    model_keys = []
    cur_values = []

    # Extract required value from each evaluation
    for model_key, values in model_values_map.iteritems():
        model_keys.append(model_key)
        cur_values.append(values[name])
    # Get indexes of sorted values
    cur_values_arr = np.array(cur_values)
    sorted_indexes = np.argsort(cur_values_arr)
    if reverse_sort:
        sorted_indexes = sorted_indexes[::-1]

    # Return list of tuples with experiment number, iter number, and value
    ret = []
    for i in sorted_indexes:
        exp_num, iter_num = model_keys[i]
        value = cur_values[i]
        ret.append((exp_num, iter_num, value))
    return ret

def sort_exps_by_overall_perf(model_values_map, overall_perf_info):
    name, regex, reverse_sort = overall_perf_info
    # Map from experiment to best iteration
    exp_best_iter_map = {}
    # Map from experiment to best value of performance metric
    exp_best_overall_perf_map = {}
    for model_key, values in model_values_map.iteritems():
        # Skip if there aren't enough values (overall performance was probably not reported)
        exp_num, iter_num = model_key
        best_exp_overall_perf = exp_best_overall_perf_map.get(exp_num, None)
        # Get performance metric of current experiment and iter num
        exp_overall_perf = values[name]
        # Set as best performance if experiment wasn't seen
        if best_exp_overall_perf is None:
            exp_best_overall_perf_map[exp_num] = exp_overall_perf
            exp_best_iter_map[exp_num] = iter_num
        elif reverse_sort:
            # Keep the higher value
            if exp_overall_perf > best_exp_overall_perf:
                exp_best_overall_perf_map[exp_num] = exp_overall_perf
                exp_best_iter_map[exp_num] = iter_num
        else:
            # Keep the lower value
            if exp_overall_perf < best_exp_overall_perf:
                exp_best_overall_perf_map[exp_num] = exp_overall_perf
                exp_best_iter_map[exp_num] = iter_num
    
    # Sort experiments by performance
    exp_nums = []
    best_overall_perfs = []
    for exp_num, best_overall_perf in exp_best_overall_perf_map.iteritems():
        exp_nums.append(exp_num)
        best_overall_perfs.append(best_overall_perf)

    # Get indexes of sorted values
    best_overall_perfs_arr = np.array(best_overall_perfs)
    sorted_indexes = np.argsort(best_overall_perfs_arr)
    if reverse_sort:
        sorted_indexes = sorted_indexes[::-1]

    ret = []
    for i in sorted_indexes:
        exp_num = exp_nums[i]
        iter_num = exp_best_iter_map[exp_num]
        best_overall_perf = best_overall_perfs[i]
        ret.append((exp_num, iter_num, best_overall_perf))
    return ret
