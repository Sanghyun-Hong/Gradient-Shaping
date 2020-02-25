#!/bin/bash


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


# configurations for each dataset
DATASET=$1
ATTMODE=$2

# ----------------------------------------------------------------
#  Purchase-100 cases
# ----------------------------------------------------------------
if [ "$DATASET" == "purchases" ]; then
  DATAPTH="datasets/purchases/purchase_100_20k_data.npz"
  POISONF="5.0_1_2000_2e-05_3.5_4e-05_100_0.6_1e-10"
  SAMFILE=""

  # [LR Model w. the vanilla optimizer]
  NETWORK="lr"
  NETPATH="models/purchases/vanilla_lr_100_100_0.04/best"
  T_INDEX=2252    # [Info: vanilla-lr (2252, 1) - 16th]
  P_INDEX=1

  # [LR Model w. the DP optimizer]
  # NETWORK="lr"
  # NETPATH="models/purchases/dp_lr_100_100_0.08_1000.0_1e-05_4.0_0.025/epoch_3_acc_0.4758_eps_248918.0676"
  # T_INDEX=2252    # [Info: DP-lr (2252, 1)]
  # P_INDEX=1

  # attack classes
  B_CLASS=3
  T_CLASS=4

# unknown case
else
  echo "Error: unknown dataset - $1"
  exit 1
fi



# ----------------------------------------------------------------
#  Pre-processings
#   - skip if it's not the path of interest
#   - set the privacy flag
#   - compose the locations
# ----------------------------------------------------------------
# : skip
if [[ $NETPATH != *"$NETWORK"* ]]; then
  echo "Error: [$NETPATH] is not from [$NETWORK]"
  exit
fi

# : set the privacy flag
if [[ $NETPATH == *"vanilla"* ]]; then
  PRIVACY=False
else
  PRIVACY=True
fi

# : compose the candidate location
IFS='/' read -ra each_nettoks <<< "$NETPATH"
TOKSLEN=${#each_nettoks[@]}
SAMPATH="samples/$DATASET/"
POISOND="datasets/poisons/targeted/clean-labels/$DATASET/"
for (( tidx=0; tidx<$TOKSLEN; tidx++)); do
  # :: skip the 'models/<dataset>' part
  if (( tidx < 2 )); then
    continue
  # :: the first token to append
  elif (( tidx == 2 )); then
    SAMPATH="${SAMPATH}${each_nettoks[tidx]}"
    POISOND="${POISOND}${each_nettoks[tidx]}"
  # :: the rest to append
  else
    SAMPATH="${SAMPATH}_${each_nettoks[tidx]}"
    POISOND="${POISOND}_${each_nettoks[tidx]}"
  fi
done
SAMPATH="${SAMPATH}/${SAMFILE}"
POISOND="${POISOND}/${POISONF}"


# ----------------------------------------------------------------
#  Run the script
# ----------------------------------------------------------------

# : run
if [ "$PRIVACY" = True ] ; then
  echo "python3 analyze_tpoisoning.py \
    --dataset=$DATASET \
    --datapth=$DATAPTH \
    --poisond=$POISOND \
    --samples=$SAMPATH \
    --b-class=$B_CLASS \
    --t-class=$T_CLASS \
    --network=$NETWORK \
    --netpath=$NETPATH \
    --privacy \
    --t-index=$T_INDEX \
    --p-index=$P_INDEX"

  python3 analyze_tpoisoning.py \
    --dataset=$DATASET \
    --datapth=$DATAPTH \
    --poisond=$POISOND \
    --samples=$SAMPATH \
    --b-class=$B_CLASS \
    --t-class=$T_CLASS \
    --network=$NETWORK \
    --netpath=$NETPATH \
    --privacy \
    --t-index=$T_INDEX \
    --p-index=$P_INDEX

else
  echo "python3 analyze_tpoisoning.py \
    --dataset=$DATASET \
    --datapth=$DATAPTH \
    --poisond=$POISOND \
    --samples=$SAMPATH \
    --b-class=$B_CLASS \
    --t-class=$T_CLASS \
    --network=$NETWORK \
    --netpath=$NETPATH \
    --t-index=$T_INDEX \
    --p-index=$P_INDEX"

  python3 analyze_tpoisoning.py \
    --dataset=$DATASET \
    --datapth=$DATAPTH \
    --poisond=$POISOND \
    --samples=$SAMPATH \
    --b-class=$B_CLASS \
    --t-class=$T_CLASS \
    --network=$NETWORK \
    --netpath=$NETPATH \
    --t-index=$T_INDEX \
    --p-index=$P_INDEX \

fi
