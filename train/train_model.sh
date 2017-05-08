# Handle flag to queue on PBS server
if [ "$#" -eq 1 ]; then
	if [ "$1" == "--queue_pbs" ]; then
		QUEUE_PBS=1
	else
		echo "Usage: ./train_model.sh [--queue_pbs]"
		exit 1
	fi
elif [ "$#" -gt 1 ]; then
	echo "Usage: ./train_model.sh [--queue_pbs]"
	exit 1
fi

# Full path to initial weights. Set to "NONE" if training from scratch
export INITIAL_WEIGHTS="[[INITIAL_WEIGHTS]]"
# Name of the folder containing the current experiment
export EXPERIMENT_FOLDER_NAME="[[EXPERIMENT_FOLDER_NAME]]"
# Root path of the experiments
export EXPERIMENTS_ROOT="[[EXPERIMENTS_ROOT]]"
# Path to PBS and training scripts
PBS_SCRIPT_DIR="[[PBS_SCRIPT_DIR]]"

if [ "$QUEUE_PBS" ]; then
	PBS_SCRIPT="$PBS_SCRIPT_DIR/train.pbs"
	# cd to pbs_out folder so logs appear there
	cd "$(dirname $PBS_SCRIPT)/pbs_out"
	qsub "$PBS_SCRIPT" -N "$EXPERIMENT_FOLDER_NAME"
else
	TRAIN_SCRIPT="$PBS_SCRIPT_DIR/train.sh"
	"$TRAIN_SCRIPT"
fi
