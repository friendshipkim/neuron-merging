#!/bin/bash

LeNet_300_100_original(){
echo -e "\nLeNet_300_100 original\n"
python main.py \
   --arch LeNet_300_100 \
   --dataset FashionMNIST \
   --pretrained saved_models/LeNet_300_100.original.300_100.bias.pth.tar \
   --evaluate
}

LeNet_300_100_prune(){
echo -e "\nLeNet_300_100 prune $1 $2\n"
python main.py \
   --arch LeNet_300_100 \
   --dataset FashionMNIST \
   --retrain \
   --target ip \
   --criterion $1 \
   --model-type prune \
   --pretrained saved_models/LeNet_300_100.original.300_100.bias.pth.tar \
   --pruning-ratio $2 \
   --evaluate
}

LeNet_300_100_merge(){
echo -e "\nLeNet_300_100 merge $1 $2\n"
python main.py \
   --arch LeNet_300_100 \
   --dataset FashionMNIST \
   --retrain \
   --target ip \
   --criterion $1 \
   --model-type merge\
   --pretrained saved_models/LeNet_300_100.original.300_100.bias.pth.tar \
   --pruning-ratio $2 \
   --threshold 0.45 \
   --evaluate
}


help() {
    echo "LeNet_300_100_FashionMNIST.sh [OPTIONS]"
    echo "    -h			help."
    echo "    -t ARG		model type: original | prune | merge (default: original)."
    echo "    -c ARG		criterion : l1-norm | l2-norm | l2-GM. (default: l1-norm)"
    echo "    -r ARG 		pruning ratio : (default: 0.5)."
    exit 0
}

model_type=original
criterion=l1-norm
pruning_ratio=0.5

while getopts "t:c:r:h" opt
do
  case $opt in
	  t) model_type=$OPTARG
      ;;
    c) criterion=$OPTARG
      ;;
	  r) pruning_ratio=$OPTARG
      ;;
    h) help ;;
    ?) help ;;
  esac
done


LeNet_300_100_"$model_type" $criterion $pruning_ratio