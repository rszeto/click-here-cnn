PARSE_LOG_SCRIPT="/z/home/szetor/Documents/DENSO_VCA/RenderForCNN/view_estimation_correspondences/eval_scripts/parse_log.sh"
PLOT_LOSSES_SCRIPT="/z/home/szetor/Documents/DENSO_VCA/RenderForCNN/view_estimation_correspondences/eval_scripts/plot_losses.py"
# Experiment root location
SCRIPT_DIR="$( cd "$(dirname "$0")" ; pwd -P )"
EXP_ROOT="$SCRIPT_DIR/../experiments"

# Go through all experiments
for EXP_FOLDER_NAME in `ls $EXP_ROOT`; do
	PROGRESS_PARSE_FAILED=0
	# Only plot if experiment was run and wasn't flagged for plot skip
	FULL_EXP_PATH="$EXP_ROOT/$EXP_FOLDER_NAME"
	# Delete DO_NOT_PLOT if experiment is running. Needed if the experiment was restarted.
	if [ -e "$FULL_EXP_PATH/RUNNING" ]; then
		rm "$FULL_EXP_PATH/DO_NOT_PLOT"
	fi
	if ! [ -e "$FULL_EXP_PATH/DO_NOT_PLOT" ] && ! [ -e "$FULL_EXP_PATH/NOT_STARTED" ]; then
		# Generate tables
		PROGRESS_LOG_PATHS="$FULL_EXP_PATH/progress/progress*.log"
		for PROGRESS_LOG_PATH in $PROGRESS_LOG_PATHS; do
			"$PARSE_LOG_SCRIPT" "$PROGRESS_LOG_PATH"
			if [ $? -ne 0 ]; then
				echo "Problem parsing log at $PROGRESS_LOG_PATH"
				PROGRESS_PARSE_FAILED=1
				break
			fi
		done
		if [ $PROGRESS_PARSE_FAILED -eq 1 ]; then
			echo "Problem generating loss/accuracy tables for experiment $EXP_FOLDER_NAME"
			continue
		else
			# Generate plots
			echo "Generating plots for experiment $EXP_FOLDER_NAME"
			python "$PLOT_LOSSES_SCRIPT" "$FULL_EXP_PATH" --savePath "$FULL_EXP_PATH/progress"
			if [ $? -ne 0 ]; then
				echo "Problem generating plots for experiment $EXP_FOLDER_NAME"
			fi
		fi
		# If experiment is done or killed, add flag to skip plotting next time
		if [ -e "$FULL_EXP_PATH/DONE" ] || [ -e "$FULL_EXP_PATH/KILLED" ] || [ -e "$FULL_EXP_PATH/ERROR" ]; then
			touch "$FULL_EXP_PATH/DO_NOT_PLOT"
		fi
	fi
done
