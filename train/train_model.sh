if [ "$#" -gt 0 ]; then
	echo "Usage: ./train_model.sh"
	exit 1
fi

# Full path to initial weights. Set to "NONE" if training from scratch
export INITIAL_WEIGHTS="[[INITIAL_WEIGHTS]]"
# Name of the folder containing the current experiment
export EXPERIMENT_FOLDER_NAME="[[EXPERIMENT_FOLDER_NAME]]"
# Root path of the experiments
export EXPERIMENTS_ROOT="[[EXPERIMENTS_ROOT]]"

TRAIN_SCRIPT="[[TRAIN_ROOT]]/train.sh"
"$TRAIN_SCRIPT"
