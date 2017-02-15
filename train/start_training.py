import argparse
import os
import subprocess
import glob
import pdb
import re

SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))
EXP_ROOT = os.path.join(SCRIPT_DIR, '..', 'experiments')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_jobs', type=int, default=-1, help='Only run the first NUM_JOBS jobs')
    parser.add_argument('--exp_num', type=int, default=[], nargs='+', help='Only run given numbered experiments')
    parser.add_argument('--resume', action='store_true', help='Resume experiments from the last iteration. Only used if exp_num is given')
    args = parser.parse_args()

    # Default case - run all jobs
    if args.num_jobs == -1 and args.exp_num == []:
        # Go through all experiments
        for exp_folder_name in sorted(os.listdir(EXP_ROOT)):
            full_exp_path = os.path.join(EXP_ROOT, exp_folder_name)
            if not os.path.exists(os.path.join(full_exp_path, 'NOT_STARTED')):
                print('Starting running experiment %s' % exp_folder_name)
                subprocess.call(['/bin/bash', os.path.join(full_exp_path, 'model', 'train.sh')])

    # Number of jobs to run was specified
    elif args.num_jobs != -1 and args.exp_num == []:
        # Go through all experiments
        for exp_folder_name in sorted(os.listdir(EXP_ROOT))[:args.num_jobs]:
            full_exp_path = os.path.join(EXP_ROOT, exp_folder_name)
            if not os.path.exists(os.path.join(full_exp_path, 'NOT_STARTED')):
                print('Starting running experiment %s' % exp_folder_name)
                subprocess.call(['/bin/bash', os.path.join(full_exp_path, 'model', 'train.sh')])

    # Which experiments to run was specified
    elif args.num_jobs == -1 and args.exp_num != []:
        for i, exp_num in enumerate(args.exp_num):
            # Find folder(s) starting with formatted experiment name
            full_exp_paths = glob.glob(os.path.join(EXP_ROOT, '%06d*' % exp_num))
            if len(full_exp_paths) > 0:
                full_exp_path = full_exp_paths[0]
                exp_folder_name = os.path.basename(full_exp_path)
                if args.resume:
                    # Find last snapshot
                    full_snapshot_paths = glob.glob(os.path.join(full_exp_path, 'snapshots', '*.solverstate'))
                    iter_nums = [int(re.search('.?(\d+).solverstate', x).group(1)) for x in full_snapshot_paths]
                    max_iter_num = max(iter_nums)
                    # Resume training
                    print('Resuming experiment %s from iteration %d' % (exp_folder_name, max_iter_num))
                    solver_state_path = os.path.join(full_exp_path, 'snapshots', 'snapshot_iter_%d.solverstate' % max_iter_num)
                    subprocess.call(['/bin/bash', os.path.join(full_exp_path, 'model', 'resume.sh'), solver_state_path])
                else:
                    print('Starting running experiment %s' % exp_folder_name)
                    subprocess.call(['/bin/bash', os.path.join(full_exp_path, 'model', 'train.sh')])
            else:
                print('Couldn\'t find experiment %d' % exp_num)

    # Prevent both optional argument types
    else:
        print('Cannot specify both number of jobs and experiment IDs')
