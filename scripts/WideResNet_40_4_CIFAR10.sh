#!/bin/bash



WideResNet_40_4_original(){
echo -e "\nWideResNet_40_4 original\n"
python main.py \
   --arch WideResNet \
   --depth-wide [40,4] \
   --dataset cifar10 \
   --pretrained saved_models/WideResNet.cifar10.original.40_4.pth.tar \
   --evaluate
}

WideResNet_40_4_prune(){
echo -e "\nWideResNet_40_4 prune $1 $2\n"
python main.py \
   --arch WideResNet \
   --depth-wide [40,4] \
   --dataset cifar10 \
   --retrain \
   --target conv \
   --criterion $1 \
   --model-type prune \
   --pretrained saved_models/WideResNet.cifar10.original.40_4.pth.tar \
   --pruning-ratio $2 \
   --evaluate
}

WideResNet_40_4_merge(){
echo -e "\nWideResNet_40_4 merge $1 $2\n"
python main.py \
   --arch WideResNet \
   --depth-wide [40,4] \
   --dataset cifar10 \
   --retrain \
   --target conv \
   --criterion $1 \
   --model-type merge \
   --pretrained saved_models/WideResNet.cifar10.original.40_4.pth.tar \
   --pruning-ratio $2 \
   --threshold 0.1 \
   --lamda 0.8 \
   --evaluate 
}


help() {
    echo "WideResNet_40_4_CIFAR10.sh [OPTIONS]"
    echo "    -h		help."
    echo "    -t ARG    model type: original | prune | merge (default: original)."
    echo "    -c ARG    criterion : l1-norm | l2-norm | l2-GM (default: l1-norm)."
    echo "    -r ARG    pruning ratio : (default: 0.2)."
    exit 0
}

model_type=original
criterion=l1-norm
pruning_ratio=0.2

while getopts "t:c:r:h" opt
do
    case $opt in
	t) model_type=$OPTARG
          echo "model_type: $model_type"
          ;;
        c) criterion=$OPTARG
          ;;
	r) pruning_ratio=$OPTARG
          echo "pruning_ratio: $pruning_ratio"
          ;;
        h) help ;;
        ?) help ;;
    esac
done


WideResNet_40_4_$model_type $criterion $pruning_ratio