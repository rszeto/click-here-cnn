PBS_SCRIPT="/z/home/szetor/Documents/DENSO_VCA/RenderForCNN/pbs_scripts/train.pbs"
# Full path to initial weights. Set to "NONE" if training from scratch
export INITIAL_WEIGHTS=""

# Extract the model name and mode from this directory and script name
MODEL_ROOT_DIR=$(cd $(dirname $0); pwd)
export MODEL_NAME=$(basename $MODEL_ROOT_DIR)
# Cut out ".sh\n" from the script name
export MODE=$(basename $0 | rev | cut -c 4- | rev)

cd $(dirname $PBS_SCRIPT)
qsub $PBS_SCRIPT -N "train/$MODEL_NAME/$MODE/bidoof"