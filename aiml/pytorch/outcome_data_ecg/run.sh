#!/usr/bin/env bash
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=${PYTHONPATH}:$(pwd)/../../..

mkdir -p log

batch_size=128
master_csv=data_raw_trop6_phys
reload_from_checkpoint=False
num_epochs=100
lm=10
lr=1e-3
for seed in {0..49}
do
#  for lr in 5e-3 1e-3 5e-4
  for prefill_feature in 0 1
  do
    identifier=outcome_ecg_lm${lm}_lr${lr}_b${batch_size}_s${seed}
    echo ${identifier}
    echo ${master_csv}
    if [ ${lr} = '1e-3' ]
    then
      end_lr=1e-5
    elif [ ${lr} = '5e-3' ]
    then
      end_lr=5e-5
    elif [ ${lr} = '1e-2' ]
    then
      end_lr=1e-4
    fi
    echo "${end_lr}"
    python3 -u train.py --num_workers 4 \
                        --seed ${seed} \
                        --batch_size ${batch_size} \
                        --save_path ../data/${identifier} \
                        --save_interval 50 \
                        --display_interval 50 \
                        --num_epochs ${num_epochs} \
                        --learning_rate ${lr} \
                        --learning_rate_end "${end_lr}" \
                        --optimizer sgd \
                        --lm ${lm} \
                        --reload_from_checkpoint ${reload_from_checkpoint} \
                        --prefill_feature ${prefill_feature} \
                        --master_csv ${master_csv}  2>&1 | tee log/${identifier}.txt
  done
done
