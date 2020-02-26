#!/bin/bash


# check the dataset name
if [ "$1" == "" ]; then
  echo "Error: specify the dataset, abort."
  exit 1
fi

# check the GPU has pinned
if [ "$2" == "" ]; then
  echo "Error: specify the interpolation mode (one/multi), abort."
  exit 1
fi


# common configurations
DATASET=$1
INTMODE=$2


# FashionMNIST
if [ "$DATASET" == "fashion_mnist" ]; then
  DATAPTH=""
  IPRATIO=(0.1 0.2 0.4 0.6 0.8 1.0)   # no meaning on 0.0 (it's clean...)
  NETWORK="shallow-mlp"
  NETBASE="models/fashion_mnist/vanilla_shallow-mlp_100_100_0.04/best"

# unknown case
else
  echo "Error: unknown dataset - $1"
  exit 1
fi


# ----------------------------------------------------------------
#  Run for each model location
# ----------------------------------------------------------------
for each_alpha in ${IPRATIO[@]}; do

  # : run
  echo "python3 analyze_collision.retrain.py \
    --dataset=$DATASET \
    --datapth=$DATAPTH \
    --network=$NETWORK \
    --netbase=$NETBASE \
    --imode=$INTMODE \
    --alpha=$each_alpha"

  python3 analyze_collision.retrain.py \
    --dataset=$DATASET \
    --datapth=$DATAPTH \
    --network=$NETWORK \
    --netbase=$NETBASE \
    --imode=$INTMODE \
    --alpha=$each_alpha

done
