# Handle flag to queue on PBS server
if [ "$#" -eq 2 ]; then
	if [ "$2" == "--queue_pbs" ]; then
		QUEUE_PBS=1
	else
		echo "Usage: ./resume_model.sh solver_state [--queue_pbs]"
		exit 1
	fi
elif [ "$#" -gt 2 ]; then
	echo "Usage: ./resume_model.sh solver_state [--queue_pbs]"
	exit 1
fi

# Full path to solver state
export SOLVER_STATE="$1"
# Name of the folder containing the current experiment
export EXPERIMENT_FOLDER_NAME="[[EXPERIMENT_FOLDER_NAME]]"
# Root path of the experiments
export EXPERIMENTS_ROOT="[[EXPERIMENTS_ROOT]]"

if [ "$QUEUE_PBS" ]; then
	PBS_SCRIPT="/z/home/szetor/Documents/DENSO_VCA/RenderForCNN/pbs_scripts/train.pbs"
	# cd to pbs_out folder so logs appear there
	cd "$(dirname $PBS_SCRIPT)/pbs_out"
	qsub "$PBS_SCRIPT" -N "$EXPERIMENT_FOLDER_NAME"
else
	TRAIN_SCRIPT="/z/home/szetor/Documents/DENSO_VCA/RenderForCNN/pbs_scripts/train.sh"
	"$TRAIN_SCRIPT"
fi
