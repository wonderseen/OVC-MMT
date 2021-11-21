#!/bin/bash

# Example to Run the multimodal model code on the Multi30K DE Dataset
GPU_IDs=(1 2 3)
order=(1 2 3)
for idx in ${!GPU_IDs[*]}
do
        GPU_ID=${GPU_IDs[${idx}]}
        echo "GPU ID is ${GPU_ID}."
        export CUDA_VISIBLE_DEVICES=${GPU_ID}
        trained_model_path="saves/Multi30K_OVC_Lm_Lv_de_${order[${idx}]}"
        mkdir -p ${trained_model_path}
        log_file="${trained_model_path}/model.log"
        nohup python -u OVC_train.py --data_path data/Multi30K_DE \
                --loss_w 0.9 \
                --trained_model_path ${trained_model_path} \
                --sr en \
                --tg de \
                > ${log_file} 2>&1 &
        echo ${log_file}
done

