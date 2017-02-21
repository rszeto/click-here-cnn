import os
import re
import pdb
import numpy as np

SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))
EXP_ROOT = os.path.join(SCRIPT_DIR, '..', '..', 'experiments')
EXCLUDE_EXPERIMENTS = [
    '000001_02-03-2017_17:58:13',
    '000002_02-03-2017_17:58:36',
    '000003_02-03-2017_19:58:35'
]

# Array of tuples specifying name for value, index in the values array, and whether to reverse-sort
display_info = [('Mean accuracy', 6, True), ('Mean medErr', 7, False)]
# Tuple specifying value to define overall performance with
overall_perf_info = display_info[0]

# Create mapping from models (i.e. weights for a given architecture) to values extracted from result file
def get_model_values_map(exclude_experiments=None, only_include_experiments=None):
    model_values_map = {}

    if only_include_experiments:
        exp_names = only_include_experiments
    else:
        exp_names = os.listdir(EXP_ROOT)
        if exclude_experiments:
            exp_names = filter(lambda x: not x in exclude_experiments, exp_names)

    for exp_name in exp_names:
        if exclude_experiments and exp_name in exclude_experiments:
            continue

        # Get experiment number from name (first 6 characters)
        exp_num = int(exp_name[:6])
        # Path to evaluation results for this experiment
        exp_eval_path = os.path.join(EXP_ROOT, exp_name, 'evaluation')
        for acc_mederr_file_name in os.listdir(exp_eval_path):
            # Skip arguments file
            if acc_mederr_file_name == 'evalAcc_args.txt':
                continue
            # Get iteration number
            iter_num = int(re.findall('\d+', acc_mederr_file_name)[0])

            # Find all instances of XX.XXXX in the file
            with open(os.path.join(exp_eval_path, acc_mederr_file_name), 'r') as f:
                lines = f.read()
            values = map(lambda x: float(x), re.findall('\d+\.\d\d\d\d', lines))
            
            # Add to model-values map
            model_values_map[(exp_num, iter_num)] = values

    return model_values_map

def sort_models_by_indiv_perf(model_values_map, perf_info):
    label, index, reverse = perf_info
    model_keys = []
    cur_values = []

    # Extract required value from each evaluation
    for model_key, values in model_values_map.iteritems():
        model_keys.append(model_key)
        cur_values.append(values[index])
    # Get indexes of sorted values
    cur_values_arr = np.array(cur_values)
    sorted_indexes = np.argsort(cur_values_arr)
    if reverse:
        sorted_indexes = sorted_indexes[::-1]

    # Return list of tuples with experiment number, iter number, and value
    ret = []
    for i in sorted_indexes:
        exp_num, iter_num = model_keys[i]
        value = cur_values[i]
        ret.append((exp_num, iter_num, value))
    return ret

def sort_exps_by_overall_perf(model_values_map, overall_perf_info):
    label, index, reverse = overall_perf_info
    # Map from experiment to best iteration
    exp_best_iter_map = {}
    # Map from experiment to best value of performance metric
    exp_best_overall_perf_map = {}
    for model_key, values in model_values_map.iteritems():
        exp_num, iter_num = model_key
        best_exp_overall_perf = exp_best_overall_perf_map.get(exp_num, None)
        # Get performance metric of current experiment and iter num
        exp_overall_perf = values[index]
        # Set as best performance if experiment wasn't seen
        if best_exp_overall_perf is None:
            exp_best_overall_perf_map[exp_num] = exp_overall_perf
            exp_best_iter_map[exp_num] = iter_num
        elif reverse:
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
    if reverse:
        sorted_indexes = sorted_indexes[::-1]

    ret = []
    for i in sorted_indexes:
        exp_num = exp_nums[i]
        iter_num = exp_best_iter_map[exp_num]
        best_overall_perf = best_overall_perfs[i]
        ret.append((exp_num, iter_num, best_overall_perf))
    return ret

if __name__ == '__main__':
    model_values_map = get_model_values_map(only_include_experiments=['000027_02-20-2017_19:47:36'])

    # # Find best performing architecture + iter combos
    # for perf_info in display_info:
    #     model_perf_tuples = sort_models_by_indiv_perf(model_values_map, perf_info)
    #     print(perf_info[0])
    #     for exp_num, iter_num, value in model_perf_tuples[:5]:
    #         print('\t%f (experiment %d, iter %d)' % (value, exp_num, iter_num))

    # Find best performing model; each model uses the best weights    
    for perf_info in display_info:
        print('Models sorted by: %s' % perf_info[0])
        overall_perf_tuples = sort_exps_by_overall_perf(model_values_map, perf_info)
        for exp_num, iter_num, best_overall_perf in overall_perf_tuples[:5]:
            print('\t%f (experiment %d, iter %d)' % (best_overall_perf, exp_num, iter_num))