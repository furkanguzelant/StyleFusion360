#!/bin/bash
model="easy-khair-180-gpc0.8-trans10-025000.pkl"
target_img="../../style_img"
Head_Back_MODEL="../../pretrained_models/back_head-230000.th"

# # Step2: Generate Head_Back

torchrun --master_port 14040 inference.py \
--model_config ./model_lib/ControlNet/models/cldm_v15_reference_only_pose_enable_PC.yaml \
--test_dataset back_head_generation \
--control_mode controlnet_important \
--local_image_dir ../../style_img/back \
--resume_dir ${Head_Back_MODEL} \
--control_type GAN_Generated \
--inference_image_path ${target_img} \
--nSample 1 \
--condition_path ../data/cam_condition/sphere32 \
