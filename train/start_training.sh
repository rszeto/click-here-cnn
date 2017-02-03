# Experiment root location
SCRIPT_DIR="$( cd "$(dirname "$0")" ; pwd -P )"
EXP_ROOT="$SCRIPT_DIR/../experiments"

# Go through all experiments
for EXP_FOLDER_NAME in `ls $EXP_ROOT`; do
	# Skip snapshots dir
	if [ "$EXP_FOLDER_NAME" == "snapshots" ]; then
		continue
	fi

	# Run experiment if NOT_STARTED file exists
	FULL_EXP_PATH="$EXP_ROOT/$EXP_FOLDER_NAME"
	if [ -e "$FULL_EXP_PATH/NOT_STARTED" ]; then
		echo "Started running experiment $EXP_FOLDER_NAME"
		"$FULL_EXP_PATH/model/train.sh"
	fi
done