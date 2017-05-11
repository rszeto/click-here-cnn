PBS_SCRIPT="/z/home/szetor/Documents/DENSO_VCA/RenderForCNN/pbs_scripts/train.pbs"
# Full path to solver state
export SOLVER_STATE="$1"
# Name of the folder containing the current experiment
export EXPERIMENT_FOLDER_NAME="000089_03-12-2017_10:58:01"
# Root path of the experiments
export EXPERIMENTS_ROOT="/z/home/szetor/Documents/DENSO_VCA/RenderForCNN/experiments"

# cd to pbs_out folder so logs appear there
cd "$(dirname $PBS_SCRIPT)/pbs_out"
qsub $PBS_SCRIPT -N "$EXPERIMENT_FOLDER_NAME"
