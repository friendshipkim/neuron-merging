#!/bin/bash



VGG16_original(){
echo -e "\nVGG16 original\n"
python main.py \
   --arch VGG \
   --dataset cifar10 \
   --pretrained saved_models/VGG.cifar10.original.pth.tar \
   --evaluate
}

VGG16_prune(){
echo -e "\nVGG16 prune $1\n"
python main.py \
   --arch VGG \
   --dataset cifar10 \
   --retrain \
   --target conv \
   --criterion $1 \
   --model-type prune \
   --pretrained saved_models/VGG.cifar10.original.pth.tar \
   --evaluate
}

VGG16_merge(){
echo -e "\nVGG16 merge $1\n"
python main.py \
   --arch VGG \
   --dataset cifar10 \
   --retrain \
   --target conv \
   --criterion $1 \
   --model-type merge \
   --pretrained saved_models/VGG.cifar10.original.pth.tar \
   --threshold 0.1 \
   --lamda 0.85 \
   --evaluate
}


help() {
    echo "VGG16_CIFAR10.sh [OPTIONS]"
    echo "    -h		help."
    echo "    -t ARG    model type: original | prune | merge (default: original)."
    echo "    -c ARG    criterion : l1-norm | l2-norm | l2-GM (default: l1-norm)."
    exit 0
}

model_type=original
criterion=l1-norm

while getopts "t:c:h" opt
do
    case $opt in
	t) model_type=$OPTARG
          ;;
        c) criterion=$OPTARG
          ;;
        h) help ;;
        ?) help ;;
    esac
done


VGG16_$model_type $criterion