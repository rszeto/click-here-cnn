if [ "$#" -ne 1 ]; then
	echo "Usage: ./resume_model.sh solver_state"
	exit 1
fi

# Full path to solver state
export SOLVER_STATE="$1"
# Name of the folder containing the current experiment
export EXPERIMENT_FOLDER_NAME="[[EXPERIMENT_FOLDER_NAME]]"
# Root path of the experiments
export EXPERIMENTS_ROOT="[[EXPERIMENTS_ROOT]]"

TRAIN_SCRIPT="[[TRAIN_ROOT]]/train.sh"
"$TRAIN_SCRIPT"
