PBS_SCRIPT="/z/home/szetor/Documents/DENSO_VCA/RenderForCNN/pbs_scripts/train.pbs"
# Full path to initial weights. Set to "NONE" if training from scratch
export INITIAL_WEIGHTS="/z/home/szetor/Documents/DENSO_VCA/RenderForCNN/experiments/000097_03-20-2017_10:50:10/snapshots/snapshot_iter_150000.caffemodel"
# Name of the folder containing the current experiment
export EXPERIMENT_FOLDER_NAME="000099_03-23-2017_15:02:29"
# Root path of the experiments
export EXPERIMENTS_ROOT="/z/home/szetor/Documents/DENSO_VCA/RenderForCNN/experiments"

# cd to pbs_out folder so logs appear there
cd "$(dirname $PBS_SCRIPT)/pbs_out"
qsub $PBS_SCRIPT -N "$EXPERIMENT_FOLDER_NAME"
