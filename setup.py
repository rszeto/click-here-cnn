import os
from global_variables import *

# matlab global variable file
mf = open(os.path.join(g_render4cnn_root_folder, 'global_variables.m'), 'w')

mf.write("g_matlab_kde_folder = '%s';\n" % (g_matlab_kde_folder))
mf.write("g_view_statistics_folder = '%s';\n" % (g_view_statistics_folder))
mf.write("g_view_distribution_folder = '%s';\n" % (g_view_distribution_folder))
mf.write("g_truncation_statistics_folder = '%s';\n" % (g_truncation_statistics_folder))
mf.write("g_truncation_distribution_folder = '%s';\n" % (g_truncation_distribution_folder))

mf.write("g_pascal3d_root_folder = '%s';\n" % (g_pascal3d_root_folder))

mf.write("g_cls_names = {'aeroplane','bicycle','boat','bottle','bus','car','chair','diningtable','motorbike','sofa','train','tvmonitor'};\n")
mf.write("g_detection_results_folder = '%s';\n" % (g_detection_results_folder))

# train.sh
with open(os.path.join(g_corresp_model_root_folder, 'train.sh'), 'r') as f:
    train_script_contents = f.read()
# Replace variables in script
train_script_contents = train_script_contents.replace('[[CAFFE]]', g_caffe_path)
with open(os.path.join(g_corresp_model_root_folder, 'train.sh'), 'w') as f:
    f.write(train_script_contents)
# Make script executable
os.chmod(os.path.join(g_corresp_model_root_folder, 'train.sh'), 0744)
