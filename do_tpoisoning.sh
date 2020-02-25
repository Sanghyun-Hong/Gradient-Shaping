#!/bin/bash

#!/bin/bash

# ----------------------------------------------------------------
#  Task (Do targeted poisoning attacks)
# ----------------------------------------------------------------

# check the dataset name
if [ "$1" == "" ]; then
  echo "Error: specify the dataset, abort."
  exit 1
fi

# check the attack mode is set
if [ "$2" == "" ]; then
  echo "Error: specify the attack mode (oneshot or multipoison), abort."
  exit 1
fi


# common configurations
DATASET=$1
ATTMODE=$2


# Purchase-100 datasets
if [ "$DATASET" == "purchases" ]; then
  DATAPTH="datasets/purchases/purchase_100_20k_data.npz"
  POISONN=100
  # 3 to 4 (we don't have label names)
  BACLASS=3
  PTCLASS=4

  # [Against LR - logistic regressions]
  POISONF="5.0_1_2000_2e-05_3.5_4e-05_100_0.6_1e-10"
  NETWORK="lr"
  NETPATH=(
    # [experiments w. the vanilla model]
    models/purchases/vanilla_lr_100_100_0.04/best
    # [experiments w. DP-models (only use the clipping norm)]
    models/purchases/dp_lr_100_100_0.08_1000.0_1e-05_0.1_0.0/epoch_95_acc_0.7086_eps_inf
    models/purchases/dp_lr_100_100_0.08_1000.0_1e-05_1.0_0.0/epoch_70_acc_0.7174_eps_inf
    models/purchases/dp_lr_100_100_0.08_1000.0_1e-05_4.0_0.0/epoch_80_acc_0.7193_eps_inf
    models/purchases/dp_lr_100_100_0.08_1000.0_1e-05_8.0_0.0/epoch_99_acc_0.7177_eps_inf
    # [experiments w. DP-models (only use the noise multiplier)]
    models/purchases/dp_lr_100_100_0.08_1000.0_1e-05_4.0_0.1/epoch_2_acc_0.1804_eps_1973.8648
    models/purchases/dp_lr_100_100_0.08_1000.0_1e-05_4.0_0.025/epoch_3_acc_0.4758_eps_248918.0676
    models/purchases/dp_lr_100_100_0.08_1000.0_1e-05_4.0_0.0025/epoch_14_acc_0.6692_eps_123129195.5084
    models/purchases/dp_lr_100_100_0.08_1000.0_1e-05_4.0_0.00025/epoch_67_acc_0.7117_eps_58959660714.0866
  )

  # [Shallow-MLP]
  POISONF="5.0_1_2000_0.0002_3.5_4e-05_100_0.6_1e-10"
  NETWORK="shallow-mlp"
  NETPATH=(
    # [experiments w. the vanilla model]
    models/purchases/vanilla_shallow-mlp_100_100_0.001/best
    # [experiments w. DP-models (only use the clipping norm)]
    models/purchases/dp_shallow-mlp_100_100_0.002_1000.0_1e-05_0.1_0.0/epoch_86_acc_0.7246_eps_inf
    models/purchases/dp_shallow-mlp_100_100_0.002_1000.0_1e-05_1.0_0.0/epoch_94_acc_0.7231_eps_inf
    models/purchases/dp_shallow-mlp_100_100_0.002_1000.0_1e-05_4.0_0.0/epoch_73_acc_0.7311_eps_inf
    models/purchases/dp_shallow-mlp_100_100_0.002_1000.0_1e-05_8.0_0.0/epoch_70_acc_0.7252_eps_inf
    # [experiments w. DP-models (only use the noise multiplier)]
    models/purchases/dp_shallow-mlp_100_100_0.002_1000.0_1e-05_4.0_0.25/epoch_12_acc_0.2920_eps_157.0624
    models/purchases/dp_shallow-mlp_100_100_0.002_1000.0_1e-05_4.0_0.1/epoch_11_acc_0.5069_eps_10338.1745
    models/purchases/dp_shallow-mlp_100_100_0.002_1000.0_1e-05_4.0_0.025/epoch_17_acc_0.6445_eps_1409998.4468
    models/purchases/dp_shallow-mlp_100_100_0.002_1000.0_1e-05_4.0_0.0025/epoch_44_acc_0.7076_eps_386977224.8923
  )

# unknown case
else
  echo "Error: unknown dataset - $1"
  exit 1
fi
# ----------------------------------------------------------------


# ----------------------------------------------------------------
#  Split into multiple scripts
# ----------------------------------------------------------------
TOTTASKS=1        # for each network, split total attacks in N tasks
TOTPROCS=1        # for each task, the number of processes will be used
# ----------------------------------------------------------------


# ----------------------------------------------------------------
#  Run the script (for each task)
# ----------------------------------------------------------------
NET_COUNT=0
for EACH_NETPATH in ${NETPATH[@]}; do

  # : skip if it's not the path of interest
  if [[ $EACH_NETPATH != *"$NETWORK"* ]]; then
    continue
  fi

  # : set the privacy flag
  if [[ $EACH_NETPATH == *"vanilla"* ]]; then
    PRIVACY=False
  else
    PRIVACY=True
  fi

  # : compose the candidate location
  IFS='/' read -ra EACH_NETTOKS <<< "$EACH_NETPATH"
  TOKSLEN=${#EACH_NETTOKS[@]}
  SAMPATH="samples/$DATASET/"
  POISOND="datasets/poisons/targeted/clean-labels/$DATASET/"
  for (( tidx=0; tidx<$TOKSLEN; tidx++)); do
    # :: skip the 'models/<dataset>' part
    if (( tidx < 2 )); then
      continue
    # :: the first token to append
    elif (( tidx == 2 )); then
      SAMPATH="${SAMPATH}${EACH_NETTOKS[tidx]}"
      POISOND="${POISOND}${EACH_NETTOKS[tidx]}"
    # :: the rest to append
    else
      SAMPATH="${SAMPATH}_${EACH_NETTOKS[tidx]}"
      POISOND="${POISOND}_${EACH_NETTOKS[tidx]}"
    fi
  done
  SAMPATH="${SAMPATH}/${SAMFILE}"
  POISOND="${POISOND}/${POISONF}"

  # : for this specific task, create the slurm script for each...
  for task_idx in $(seq 1 $TOTTASKS); do

    # :: script prefix
    if [ "$ATTMODE" == "oneshot" ]; then
      script_prefix=`echo "$DATASET-net-$NET_COUNT-tpoisoning-one" | tr '[:upper:]' '[:lower:]'`
    elif [ "$ATTMODE" == "multipoison" ]; then
      script_prefix=`echo "$DATASET-net-$NET_COUNT-tpoisoning-mul" | tr '[:upper:]' '[:lower:]'`
    else
      echo "Error: unknown attack mode - $ATTMODE"
      exit 1
    fi

    # :: create a script to run
    echo \#!/bin/bash   > $script_prefix-$task_idx-of-${TOTTASKS}.sh

    # :: run the script (separate vanilla/privacy model)
    if [ $PRIVACY = True ]; then
      echo python3 do_tpoisoning.py \
        --dataset=$DATASET \
        --datapth=$DATAPTH \
        --poisond=$POISOND \
        --poisonn=$POISONN \
        --samples=$SAMPATH \
        --b-class=$BACLASS \
        --t-class=$PTCLASS \
        --network=$NETWORK \
        --netpath=$EACH_NETPATH \
        --privacy \
        --attmode=$ATTMODE \
        --cur-task=$task_idx \
        --tot-task=$TOTTASKS \
        --tot-proc=$TOTPROCS \
        >> $script_prefix-$task_idx-of-${TOTTASKS}.sh

    else
      echo python3 do_tpoisoning.py \
        --dataset=$DATASET \
        --datapth=$DATAPTH \
        --poisond=$POISOND \
        --poisonn=$POISONN \
        --samples=$SAMPATH \
        --b-class=$BACLASS \
        --t-class=$PTCLASS \
        --network=$NETWORK \
        --netpath=$EACH_NETPATH \
        --attmode=$ATTMODE \
        --cur-task=$task_idx \
        --tot-task=$TOTTASKS \
        --tot-proc=$TOTPROCS \
        >> $script_prefix-$task_idx-of-${TOTTASKS}.sh

    fi

    # : make the script runnable
    chmod +x "$script_prefix-$task_idx-of-${TOTTASKS}.sh"

    exit

  done
  # c.f. this is the end of "for task_idx..."

  # increase the net counter
  NET_COUNT=$((NET_COUNT+1))

done
# ----------------------------------------------------------------
