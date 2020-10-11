#!/bin/bash



ResNet56_original(){
echo -e "\nResNet56 original\n"
python main.py \
   --arch ResNet \
   --depth-wide 56 \
   --dataset cifar10 \
   --pretrained saved_models/ResNet.cifar10.original.56.pth.tar \
   --evaluate
}

ResNet56_prune(){
echo -e "\nResNet56 prune $1 $2\n"
python main.py \
   --arch ResNet \
   --depth-wide 56 \
   --dataset cifar10 \
   --retrain \
   --target conv \
   --criterion $1 \
   --model-type prune \
   --pretrained saved_models/ResNet.cifar10.original.56.pth.tar \
   --pruning-ratio $2 \
   --evaluate 
}

ResNet56_merge(){
echo -e "\nResNet56 merge $1 $2\n"
python main.py \
   --arch ResNet \
   --depth-wide 56 \
   --dataset cifar10 \
   --retrain \
   --target conv \
   --criterion $1 \
   --model-type merge \
   --pretrained saved_models/ResNet.cifar10.original.56.pth.tar \
   --pruning-ratio $2 \
   --threshold 0.1 \
   --lamda 0.85 \
   --evaluate 
}


help() {
    echo "ResNet56_CIFAR10.sh [OPTIONS]"
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
          ;;
        c) criterion=$OPTARG
          ;;
	r) pruning_ratio=$OPTARG
          ;;
        h) help ;;
        ?) help ;;
    esac
done


ResNet56_$model_type $criterion $pruning_ratio