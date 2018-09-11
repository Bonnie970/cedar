#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --exclusive
#SBATCH --mem=25G
#SBATCH --time=1:00:00
#SBATCH --account=def-jjclark

source activate mypython3
# ISSUE: model does not work, suppose to work much better than detectron 
#python ./train.py --evaluate output.json --load ./COCO-R101FPN-MaskRCNN-BetterParams-1.npz --config MODE_FPN=True FPN.CASCADE=True BACKBONE.RESNET_NUM_BLOCK=[3,4,23,3] TEST.RESULT_SCORE_THRESH=1e-4 PREPROC.TRAIN_SHORT_EDGE_SIZE=[640,800] TRAIN.LR_SCHEDULE=[420000,500000,540000]

# so far works the best 
python ./train.py --evaluate output.json --load ./COCO-R101FPN-MaskRCNN-BetterParams-0.npz --config MODE_FPN=True BACKBONE.RESNET_NUM_BLOCK=[3,4,23,3] TEST.RESULT_SCORE_THRESH=1e-4 FRCNN.BBOX_REG_WEIGHTS=[20,20,10,10]
