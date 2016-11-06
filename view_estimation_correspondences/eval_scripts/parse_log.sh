#!/bin/bash

# get the dirname of the script
DIR="$( cd "$(dirname "$0")" ; pwd -P )"

# Require at least one argument
if [ "$#" -lt 1 ]; then
	echo "Usage parse_log.sh /path/to/your.log [savePath]"
	exit
fi

# Require that the log file exists
if [ ! -e $1 ]; then
	echo "Log file does not exist"
	exit 1
fi

# Save new logs in the log's directory if not specified
SAVE_PATH=$2
if [ -z $SAVE_PATH ]; then
	SAVE_PATH="$(dirname $1)"
fi
LOG=`basename $1`
# Save lines containing "test" to test_lines
grep 'Test' $1 > test_lines
# Extract iteration numbers from test_lines
echo '#Iters' > test_iters
grep 'Iteration ' test_lines | sed  's/.*Iteration \([[:digit:]]*\).*/\1/g' >> test_iters
# Extract angle accuracies
echo 'Acc_Azimuth' > test_acc_az
grep 'accuracy_azimuth' test_lines | awk '{print $11}' >> test_acc_az
echo 'Acc_Elevation' > test_acc_el
grep 'accuracy_elevation' test_lines | awk '{print $11}' >> test_acc_el
echo 'Acc_Tilt' > test_acc_ti
grep 'accuracy_tilt' test_lines | awk '{print $11}' >> test_acc_ti
# Extract angle losses
echo 'Loss_Azimuth' > test_loss_az
grep 'loss_azimuth' test_lines | awk '{print $11}' >> test_loss_az
echo 'Loss_Elevation' > test_loss_el
grep 'loss_elevation' test_lines | awk '{print $11}' >> test_loss_el
echo 'Loss_Tilt' > test_loss_ti
grep 'loss_tilt' test_lines | awk '{print $11}' >> test_loss_ti

# Extracting elapsed seconds during test
# Extract line containing start time
grep '] Solving ' $1 > test_time_lines
# Extract lines where test phase is run
grep 'Testing net' $1 >> test_time_lines
$DIR/extract_seconds.py test_time_lines test_secs

# Generating test file
paste test_iters test_secs test_acc_az test_acc_el test_acc_ti test_loss_az test_loss_el test_loss_ti | column -t > "$SAVE_PATH/$LOG.test"
rm test_lines test_iters test_secs test_acc_az test_acc_el test_acc_ti test_loss_az test_loss_el test_loss_ti test_time_lines

# Extract lines related to training
grep ', loss = \|Train net output\|, lr = ' $1 > train_lines
# Get loss lines, which contain the iteration numbers
echo '#Iters' > train_iters
grep ', loss = ' train_lines | sed  's/.*Iteration \([[:digit:]]*\).*/\1/g' >> train_iters
# # Extract angle losses
echo 'Loss_Azimuth' > train_loss_az
grep 'loss_azimuth' train_lines | awk '{print $11}' >> train_loss_az
echo 'Loss_Elevation' > train_loss_el
grep 'loss_elevation' train_lines | awk '{print $11}' >> train_loss_el
echo 'Loss_Tilt' > train_loss_ti
grep 'loss_tilt' train_lines | awk '{print $11}' >> train_loss_ti
echo 'Learning_Rate' > train_lr
grep ', lr = ' train_lines | awk '{print $9}' >> train_lr

# Extracting elapsed seconds during training
# Extract line containing start time
grep '] Solving ' $1 > train_time_lines
# Extract lines where training phase is run
grep ', loss = ' $1 >> train_time_lines
# Extracting elapsed seconds
$DIR/extract_seconds.py train_time_lines train_secs

# Generating training file
paste train_iters train_secs train_lr train_loss_az train_loss_el train_loss_ti | column -t > "$SAVE_PATH/$LOG.train"
rm train_lines train_iters train_secs train_lr train_loss_az train_loss_el train_loss_ti train_time_lines