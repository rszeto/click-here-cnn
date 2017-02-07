# Status of the job. Can be NOT_STARTED, RUNNING, ERROR, KILLED
STATUS="$1"
# Date string in format accepted by `date`
DATE_STR="$2"
# Flag for whether the full path to the experiment folder should be printed
PRINT_FULL_EXP_PATH="$3"

# Make sure status and date are provided
if [ -z "$STATUS" ] || [ -z "$DATE_STR" ]; then
    echo "Usage: <status> <date string> [-f]"
    exit 1
fi
# Make sure, if third argument exists, that it's the full-path flag
if ! [ -z "$PRINT_FULL_EXP_PATH" ] && ! [ "$PRINT_FULL_EXP_PATH" == "-f" ]; then
    echo "Usage: <status> <date string> [-f]"
    exit 1
fi

EXP_ROOT="/z/home/szetor/Documents/DENSO_VCA/RenderForCNN/experiments"
STATUS_FILES=`find "$EXP_ROOT" -newermt "$(date "+%Y-%m-%d %H:%M:%S" -d "$DATE_STR")" -type f -print | grep "$STATUS" | sort`
for FILE in $STATUS_FILES; do
    if [ -z "$PRINT_FULL_EXP_PATH" ]; then
        echo "$(basename $(dirname $FILE))"
    else
        echo "$(dirname $FILE)"
    fi
done
