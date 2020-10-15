#!/bin/bash
batchs=16
DISPLAY_PORT=${2:-7000}
GPU=$1
lr=0.0002
loadSize=256
fineSize=256
L1=100
model=SID
G='RESNEXT'
checkpoint='/nfs/bigneuron/add_disk0/hieule/checkpoints_PAMI/'
datasetmode=shadowparam
dataroot='/nfs/bigneuron/add_disk0/hieule/data/datasets/ISTD_Dataset/train/'
trainmask=${dataroot}'/train_B' 
param_path=${dataroot}'/train_params'
NAME="${model}_G${G}_${datasetmode}"

OTHER="--save_epoch_freq 100 --niter 500 --niter_decay 2000"

CMD="python ../SID_train.py --loadSize ${loadSize} \
    --randomSize
    --name ${NAME} \
    --dataroot  ${dataroot}\
    --checkpoints_dir ${checkpoint} \
    --fineSize $fineSize --model $model\
    --batch_size $batchs --display_port ${DISPLAY_PORT} --display_server http://bigiris.cs.stonybrook.edu \
    --randomSize --keep_ratio --phase train_  --gpu_ids ${GPU} --lr ${lr} \
    --lambda_L1 ${L1}
    --dataset_mode $datasetmode\
    --mask_train $trainmask \
    --param_path $param_path \
    --netG $G\
    $OTHER
"
echo $CMD
eval $CMD

