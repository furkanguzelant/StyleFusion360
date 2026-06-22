#!/bin/bash
model="easy-khair-180-gpc0.8-trans10-025000.pkl"
noise_dir="../data/3DNoise"
target_img="../data/input_image"  
out_dir="../data/Back_Head"

# change the path to your own
PANO_HEAD_MODEL="../../pretrained_models/easy-khair-180-gpc0.8-trans10-025000.pkl"
Head_Back_MODEL="../../pretrained_models/back_head-230000.th"
Diff360_MODEL="../../pretrained_models/model_state-340000.th"


# # Step2: Genertate Head_Back

torchrun --master_port 14033 inference.py \
--model_config ./model_lib/ControlNet/models/cldm_v15_reference_only_pose_enable_PC.yaml \
--test_dataset back_head_generation \
--control_mode controlnet_important \
--local_image_dir ${out_dir} \
--resume_dir ${Head_Back_MODEL} \
--control_type GAN_Generated \
--inference_image_path ${target_img} \
--nSample 1 \
--condition_path ../data/cam_condition/sphere32 \
--save_video \