# Experiment root location
SCRIPT_DIR="$( cd "$(dirname "$0")" ; pwd -P )"
EXP_ROOT="$SCRIPT_DIR/../experiments"
# Evaluation script location
EVAL_SCRIPT_PATH="/z/home/szetor/Documents/DENSO_VCA/RenderForCNN/view_estimation_correspondences/eval_scripts/evaluateAcc.py"
# Number of evaluations to run simultaneously
NUM_EVAL_PROCS=3

# Kill child processes on SIGINT
trap trap_sigint SIGINT
function trap_sigint() {
	echo "Killing child processes"
	# Wait on remaining processes in the queue
	while [ ${#PID_ARR[@]} -gt 0 ]; do
		kill "${PID_ARR[0]}"
		# Remove first element in queue
		len="${#PID_ARR[@]}"
		for i in `seq 1 $((len - 1))`; do
			PID_ARR[$((i-1))]=${PID_ARR[$i]}
		done
		unset PID_ARR[$((len - 1))]
	done
	exit 1
}

# Loop forever
while :; do
	# count=0
	for EXP_NAME in `ls $EXP_ROOT`; do
		EXP_PATH="$EXP_ROOT/$EXP_NAME"
		EVALUATION_PATH="$EXP_PATH/evaluation"
		SNAPSHOTS_PATH="$EXP_PATH/snapshots"

		for WEIGHTS_PATH in `find $SNAPSHOTS_PATH -name "*.caffemodel"`; do
			# Extract iteration number. Assume snapshot file name is "snapshot_iter_###.caffemodel",
			# and get the number after the underscore
			ITER_NUM=`basename $WEIGHTS_PATH | sed -r 's/.*_([0-9]*)\..*/\1/g' `
			# Evaluate metrics if evaluation file doesn't exist
			RESULTS_PATH="$EVALUATION_PATH/acc_mederr_$ITER_NUM.txt"
			if ! [ -e "$RESULTS_PATH" ]; then
				# Get required arguments for eval script as array
				EVAL_ARGS_FILE="$EVALUATION_PATH/evalAcc_args.txt"
				mapfile EVAL_ARGS_ARR < "$EVAL_ARGS_FILE"
				# Remove newlines from arguments
				for i in $(seq 0 $(expr ${#EVAL_ARGS_ARR[@]} - 1)); do
					EVAL_ARGS_ARR["$i"]=`echo -n ${EVAL_ARGS_ARR["$i"]}`
				done
				# Replace second argument (weights file) with weights file
				EVAL_ARGS_ARR[1]="$WEIGHTS_PATH"
				# Add output file argument
				EVAL_ARGS_ARR[${#EVAL_ARGS_ARR[@]}]="--output_file"
				EVAL_ARGS_ARR[${#EVAL_ARGS_ARR[@]}]="$RESULTS_PATH"

				python "$EVAL_SCRIPT_PATH" "${EVAL_ARGS_ARR[@]}" &
				PID="$!"

				# "$SCRIPT_DIR/test_proc.sh" "$count" &
				# (( count += 1 ))
				# PID="$!"

				# Add PID to queue
				PID_ARR[${#PID_ARR[@]}]="$PID"
				# If queue is full, wait for a process to finish
				if [ ${#PID_ARR[@]} -ge "$NUM_EVAL_PROCS" ]; then
					# Wait on the first process in the queue
					wait "${PID_ARR[0]}"
					# Remove first element in queue
					len="${#PID_ARR[@]}"
					for i in `seq 1 $((len - 1))`; do
						PID_ARR[$((i-1))]=${PID_ARR[$i]}
					done
					unset PID_ARR[$((len - 1))]
				fi
			fi
		done
	done

	# Wait on remaining processes in the queue
	while [ ${#PID_ARR[@]} -gt 0 ]; do
		wait "${PID_ARR[0]}"
		# Remove first element in queue
		len="${#PID_ARR[@]}"
		for i in `seq 1 $((len - 1))`; do
			PID_ARR[$((i-1))]=${PID_ARR[$i]}
		done
		unset PID_ARR[$((len - 1))]
	done

	# Wait an hour for new snapshots to appear
	echo "Sleeping"
	sleep 3600
done