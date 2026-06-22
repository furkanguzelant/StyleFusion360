#!/bin/bash
model="easy-khair-180-gpc0.8-trans10-025000.pkl"
noise_dir="../data/3DNoise"
target_img="../data/input_image"  

# change the path to your own
PANO_HEAD_MODEL="../../pretrained_models/easy-khair-180-gpc0.8-trans10-025000.pkl"
Head_Back_MODEL="../../pretrained_models/back_head-230000.th"
Diff360_MODEL="../../pretrained_models/model_state-800.th"

# # # Step3: Generate Video
torchrun --master_port 14031 inference_style.py \
--model_config ./model_lib/ControlNet/models/cldm_v15_reference_only_temporal_pose.yaml \
--test_dataset full_head_clean_inference_final_face \
--control_mode controlnet_important \
--local_image_dir ../output \
--resume_dir ${Diff360_MODEL} \
--control_type GAN_Generated \
--inference_image_path ../data \
--nSample 16 \
--condition_path ../data/cam_condition/sphere32 \
--denoise_from_guidance \
--initial_image_path ${noise_dir} \
--style_image_front ../../style_img/front/joker.jpg \
--style_image_back ../../style_img/back/joker.jpg \
--style_mask ../../masks/joker_mouth_mask.png



