TRAIN_LOG_PATH=$1
SAVE_PLOT_DIR="$(dirname $TRAIN_LOG_PATH)"

# get the dirname of the script
DIR="$( cd "$(dirname "$0")" ; pwd -P )"

"$DIR/parse_log.sh" "$TRAIN_LOG_PATH" \
	&& python "$DIR/plot_losses.py" "$TRAIN_LOG_PATH" --savePath "$SAVE_PLOT_DIR" \
	&& montage -geometry 800 "$SAVE_PLOT_DIR/trainLoss.png" \
		"$SAVE_PLOT_DIR/testLoss.png" \
		"$SAVE_PLOT_DIR/testAcc.png" \
		"$SAVE_PLOT_DIR/plots.png"
