PBS_SCRIPT="/z/home/szetor/Documents/DENSO_VCA/RenderForCNN/pbs_scripts/train.pbs"
# Full path to initial weights. Set to "NONE" if training from scratch
export INITIAL_WEIGHTS="[[INITIAL_WEIGHTS]]"
# Name of the folder containing the current experiment
export EXPERIMENT_FOLDER_NAME="[[EXPERIMENT_FOLDER_NAME]]"
# Root path of the experiments
export EXPERIMENTS_ROOT="[[EXPERIMENTS_ROOT]]"
# Root path of the snapshots
export SNAPSHOTS_ROOT="[[SNAPSHOTS_ROOT]]"

# cd to pbs_out folder so logs appear there
cd "$(dirname $PBS_SCRIPT)/pbs_out"
qsub $PBS_SCRIPT -N "$EXPERIMENT_FOLDER_NAME"
