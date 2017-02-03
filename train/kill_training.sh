for JOB_ID in "$@"; do
	# Check script args
	if [ -z $JOB_ID ]; then
		echo "JOB_ID undefined"
		continue
	fi

	# Check that the job is running
	qstat $JOB_ID &> /dev/null
	QSTAT_STATUS=$?
	if [ $QSTAT_STATUS -ne 0 ]; then
		echo "Could not find job $JOB_ID"
		continue
	fi

	# Experiment root location
	SCRIPT_DIR="$( cd "$(dirname "$0")" ; pwd -P )"
	EXP_ROOT="$SCRIPT_DIR/../experiments"
	# Assume experiment folder name is first 26 chars of job name
	JOB_NAME_LINE=`qstat -f $JOB_ID | grep "Job_Name"`
	EXP_FOLDER_NAME=${JOB_NAME_LINE:15:26}
	# Get experiment folder path
	EXP_FOLDER_PATH="$EXP_ROOT/$EXP_FOLDER_NAME"

	# Look for RUNNING file
	if [ -e "$EXP_FOLDER_PATH/RUNNING" ]; then
		qdel $JOB_ID
		rm "$EXP_FOLDER_PATH/RUNNING"
		touch "$EXP_FOLDER_PATH/KILLED"
		echo "Killed experiment $EXP_FOLDER_NAME (job $JOB_ID)"
	else
		echo "Experiment $EXP_FOLDER_NAME (job $JOB_ID) is not running"
		continue
	fi
done