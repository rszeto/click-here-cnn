# TRAIN_LOG_PATH=/z/home/szetor/Documents/DENSO_VCA/RenderForCNN/train/a/temp_syn/train_syn.log

TRAIN_LOG_PATH=$1

# get the dirname of the script
DIR="$( cd "$(dirname "$0")" ; pwd -P )"

"$DIR/parse_log.sh" "$TRAIN_LOG_PATH" && python "$DIR/plot_losses.py" "$TRAIN_LOG_PATH" --savePath "$(dirname $TRAIN_LOG_PATH)"
