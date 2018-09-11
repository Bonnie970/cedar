#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --exclusive
#SBATCH --mem=100G
#SBATCH --time=00:30:00
#SBATCH --account=def-jjclark

source activate mypython3
python ./train.py --predict /home/bonniehu/projects/def-jjclark/bonniehu/FasterRCNN/COCO/train2014-head250/ --load ./COCO-R101FPN-MaskRCNN-BetterParams-0.npz --config MODE_FPN=True BACKBONE.RESNET_NUM_BLOCK=[3,4,23,3] TEST.RESULT_SCORE_THRESH=1e-4 FRCNN.BBOX_REG_WEIGHTS=[20,20,10,10]

# predict 1 image 
# python ./train.py --predict COCO/val2014/COCO_val2014_000000123415.jpg --load ./COCO-R101FPN-MaskRCNN-BetterParams.npz --config MODE_FPN=True BACKBONE.RESNET_NUM_BLOCK=[3,4,23,3] TEST.RESULT_SCORE_THRESH=1e-4 FRCNN.BBOX_REG_WEIGHTS=[20,20,10,10]
