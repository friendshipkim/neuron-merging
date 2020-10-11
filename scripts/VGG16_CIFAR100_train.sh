#!/bin/bash

VGG16_original(){
python main.py \
   --arch VGG \
   --dataset cifar100
}

VGG16_original