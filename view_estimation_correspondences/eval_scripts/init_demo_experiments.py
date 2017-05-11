import os
import sys

# Import global variables
sys.path.append(os.path.abspath('../..'))
from global_variables import *

for root, dirs, files in os.walk(g_demo_experiments_root_folder):
    for file in files:
        if file == 'evalAcc_args.txt':
            # Get file contents
            evalAcc_args_path = os.path.join(root, file)
            with open(evalAcc_args_path, 'r') as f:
                file_content = f.read()
            # Replace project root and test LMDB paths
            file_content = file_content.replace('[[G_RENDER4CNN_ROOT_FOLDER]]', g_render4cnn_root_folder)
            file_content = file_content.replace('[[G_CORRESP_PASCAL_TEST_LMDB_FOLDER]]', g_corresp_pascal_test_lmdb_folder)
            # Rewrite the file contents
            with open(evalAcc_args_path, 'w') as f:
                f.write(file_content)
