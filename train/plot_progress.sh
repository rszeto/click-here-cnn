PARSE_LOG_SCRIPT="/z/home/szetor/Documents/DENSO_VCA/RenderForCNN/view_estimation_correspondences/eval_scripts/parse_log.sh"
PLOT_LOSSES_SCRIPT="/z/home/szetor/Documents/DENSO_VCA/RenderForCNN/view_estimation_correspondences/eval_scripts/plot_losses.py"
# Experiment root location
SCRIPT_DIR="$( cd "$(dirname "$0")" ; pwd -P )"
EXP_ROOT="$SCRIPT_DIR/../experiments"

# Go through all experiments
for EXP_FOLDER_NAME in `ls $EXP_ROOT`; do
	# Skip snapshots dir
	if [ "$EXP_FOLDER_NAME" == "snapshots" ]; then
		continue
	fi

	# Only plot if experiment was run and wasn't flagged for plot skip
	FULL_EXP_PATH="$EXP_ROOT/$EXP_FOLDER_NAME"
	if ! [ -e "$FULL_EXP_PATH/DO_NOT_PLOT" ] && ! [ -e "$FULL_EXP_PATH/NOT_STARTED" ]; then
		# Generate tables
		PROGRESS_LOG_PATH="$FULL_EXP_PATH/progress/progress.log"
		"$PARSE_LOG_SCRIPT" "$PROGRESS_LOG_PATH"
		if [ $? -eq 0 ]; then
			# Generate plots
			echo "Generating plots for experiment $EXP_FOLDER_NAME"
			python "$PLOT_LOSSES_SCRIPT" "$PROGRESS_LOG_PATH" --savePath "$FULL_EXP_PATH/progress"
			if [ $? -eq 0 ]; then
				# Stitch plots together
				montage -geometry 800 "$FULL_EXP_PATH/progress/trainLoss.png" \
					"$FULL_EXP_PATH/progress/testLoss.png" \
					"$FULL_EXP_PATH/progress/testAcc.png" \
					"$FULL_EXP_PATH/progress/plots.png"
			else
				echo "Problem generating plots for experiment $EXP_FOLDER_NAME"
			fi
		else
			echo "Problem generating loss/accuracy tables for experiment $EXP_FOLDER_NAME"
		fi
		# If experiment is done or killed, add flag to skip plotting next time
		if [ -e "$FULL_EXP_PATH/DONE" ] || [ -e "$FULL_EXP_PATH/KILLED" ] || [ -e "$FULL_EXP_PATH/ERROR" ]; then
			touch "$FULL_EXP_PATH/DO_NOT_PLOT"
		fi
	fi
done
