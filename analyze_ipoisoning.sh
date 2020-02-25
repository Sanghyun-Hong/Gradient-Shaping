#!/bin/bash


# check the dataset name
if [ "$1" == "" ]; then
  echo "Error: specify the dataset, abort."
  exit 1
fi


# common configurations
DATASET=$1


# subset [FashionMNIST 3/4] of the entire FashionMNIST
if [ "$DATASET" == "subtask" ]; then
  DATAPTH="datasets/subtasks/fashion_mnist_3_4.1.0.pkl"
  POISPTH=(
    # [Random label-flipping]
    datasets/poisons/indiscriminate/label-flips/fashion_mnist_3_4.1.0_random_0.4.pkl
    # [Poisons formulated by Steinhardt et al.]
    datasets/poisons/indiscriminate/slabs/fmnist_34_attack_eps40_quantile65_rho_slab_v7.mat
  )
  NETWORK="lr"
  NETBASE="models/subtask/fashion_mnist_3_4/vanilla_lr_300_40_0.01/best"

  # [Privacy]
  PRIVACY=False
  DPDELTA=1e-5
  NORMCLP=(0.0)
  EPSILON=1000.0
  NOISEML=(0.0)

# unknown case
else
  echo "Error: unknown dataset - $1"
  exit 1
fi


# ----------------------------------------------------------------
#  Run for each model location
# ----------------------------------------------------------------
for each_nclip in ${NORMCLP[@]}; do
for each_noise in ${NOISEML[@]}; do
for each_ppath in ${POISPTH[@]}; do

  # : run
  if [ "$PRIVACY" = True ] ; then

    # :: compute the proper noise
    comp_noise=`bc <<<"scale=8; $each_noise / $each_nclip"`

    # :: run with privacy
    echo "python3 analyze_ipoisoning.py \
      --dataset=$DATASET \
      --datapth=$DATAPTH \
      --poisonp=$each_ppath \
      --network=$NETWORK \
      --netbase=$NETBASE \
      --privacy \
      --epsilon=$EPSILON \
      --delta=$DPDELTA \
      --nclip=$each_nclip \
      --noise=$comp_noise"

    python3 analyze_ipoisoning.py \
      --dataset=$DATASET \
      --datapth=$DATAPTH \
      --poisonp=$each_ppath \
      --network=$NETWORK \
      --netbase=$NETBASE \
      --privacy \
      --epsilon=$EPSILON \
      --delta=$DPDELTA \
      --nclip=$each_nclip \
      --noise=$comp_noise
  else
    echo "python3 analyze_ipoisoning.py \
      --dataset=$DATASET \
      --datapth=$DATAPTH \
      --poisonp=$each_ppath \
      --network=$NETWORK \
      --netbase=$NETBASE"

    python3 analyze_ipoisoning.py \
      --dataset=$DATASET \
      --datapth=$DATAPTH \
      --poisonp=$each_ppath \
      --network=$NETWORK \
      --netbase=$NETBASE
  fi

done
done
done
