#!/bin/bash


# check the dataset name
if [ "$1" == "" ]; then
  echo "Error: specify the dataset, abort."
  exit 1
fi

# check the GPU has pinned
if [ "$2" == "" ]; then
  echo "Error: specify the GPU to run, abort."
  exit 1
fi


# common configurations
DATASET=$1
GPU_NUM=$2


# FashionMNIST
if [ "$DATASET" == "fashion_mnist" ]; then
  DATAPTH=""
  POISOND="datasets/backdoors/$DATASET"
  POISONC="4"
  POISONR=(0.01)
  POISONS=(1 4 7 10 14)

  # [BadNet]
  NETWORK="badnet"
  NETPATH="models/fashion_mnist/vanilla_badnet_100_80_0.01/best_model.h5"

# unknown case
else
  echo "Error: unknown dataset - $1"
  exit 1
fi


# ----------------------------------------------------------------
#  Run for each model location
# ----------------------------------------------------------------
for each_ratio in ${POISONR[@]}; do
for each_tsize in ${POISONS[@]}; do

  # : configure the location where poisons are
  comp_pfile=$POISOND"/"$POISONC"_"$each_ratio"_"$each_tsize".pkl"

  # : run analysis
  echo "python3 analyze_insertion.py \
    --pin-gpu=$GPU_NUM \
    --dataset=$DATASET \
    --datapth=$DATAPTH \
    --poisonp=$comp_pfile \
    --network=$NETWORK \
    --netpath=$NETPATH"

  python3 analyze_insertion.py \
    --pin-gpu=$GPU_NUM \
    --dataset=$DATASET \
    --datapth=$DATAPTH \
    --poisonp=$comp_pfile \
    --network=$NETWORK \
    --netpath=$NETPATH

done
done
