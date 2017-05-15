#PBS -j oe
#PBS -l walltime=7:00:00:00
#PBS -l nodes=1:ppn=1:gpus=1
#PBS -S /bin/bash
#PBS -m abe
#PBS -M szetor@umich.edu
#PBS -v INITIAL_WEIGHTS,EXPERIMENT_FOLDER_NAME,EXPERIMENTS_ROOT,SOLVER_STATE

# Set up Caffe and prereqs
module load cuda cudnn opencv gflags glog boost
CAFFE="[[CAFFE]]/bin/caffe"

# Make sure experiment variables were set
if [ -z $EXPERIMENT_FOLDER_NAME ]; then
	echo "Error: EXPERIMENT_FOLDER_NAME undefined"
	exit 1
elif [ -z $EXPERIMENTS_ROOT ]; then
	echo "Error: EXPERIMENTS_ROOT undefined"
	exit 1
fi

# Make sure only one of INITIAL_WEIGHTS, SOLVER_STATE is defined
if [ -z "$INITIAL_WEIGHTS" ] && [ -z "$SOLVER_STATE" ]; then
	echo "Error: INITIAL_WEIGHTS and SOLVER_STATE are undefined. Please specify one"
	exit 1
elif [ -n "$INITIAL_WEIGHTS" ] && [ -n "$SOLVER_STATE" ]; then
	echo $INITIAL_WEIGHTS
	echo $SOLVER_STATE
	echo "Error: Both INITIAL_WEIGHTS and SOLVER_STATE are defined. Please specify one"
	exit 1
fi

# Derive relevant paths from environment
EXPERIMENT_FULL_PATH="$EXPERIMENTS_ROOT/$EXPERIMENT_FOLDER_NAME"
MODEL_PATH="$EXPERIMENT_FULL_PATH/model"
PROGRESS_PATH="$EXPERIMENT_FULL_PATH/progress"
SNAPSHOTS_PATH="$EXPERIMENT_FULL_PATH/snapshots"

# Indicate that experiment is running. Put PBS job ID in RUNNING file
rm "$EXPERIMENT_FULL_PATH/NOT_STARTED"
rm "$EXPERIMENT_FULL_PATH/KILLED"
rm "$EXPERIMENT_FULL_PATH/ERROR"
echo $PBS_JOBID > "$EXPERIMENT_FULL_PATH/RUNNING"
# Get progress log path. Enumerate to avoid overwriting
NUM_PROGRESS_LOGS=`find "$PROGRESS_PATH" -name "progress*.log" | wc -l`
if [ "$NUM_PROGRESS_LOGS" -eq 0 ]; then
	PROGRESS_LOG_PATH="$PROGRESS_PATH/progress.log"
else
	PROGRESS_LOG_PATH="$PROGRESS_PATH/progress_$NUM_PROGRESS_LOGS.log"
fi

# Start training
mkdir -p "$SNAPSHOTS_PATH"
cd "$SNAPSHOTS_PATH"
if [ -n "$INITIAL_WEIGHTS" ]; then
	$CAFFE train --solver="$MODEL_PATH/solver.prototxt" --weights="$INITIAL_WEIGHTS" \
			2>&1 | tee "$PROGRESS_LOG_PATH"
else
	$CAFFE train --solver="$MODEL_PATH/solver.prototxt" --snapshot="$SOLVER_STATE" \
			2>&1 | tee "$PROGRESS_LOG_PATH"
fi

# Capture status of Caffe
CAFFE_STATUS=${PIPESTATUS[0]}
rm "$EXPERIMENT_FULL_PATH/RUNNING"
if [ $CAFFE_STATUS -ne 0 ]; then
	touch "$EXPERIMENT_FULL_PATH/ERROR"
else
	touch "$EXPERIMENT_FULL_PATH/DONE"
fi
