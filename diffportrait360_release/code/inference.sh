#!/bin/bash
# Step 0.0 :Put the you own image to test under folder data/input_image

# Step 0.1: Using 3DDFA_V2_cropping to get camera pose and crop the image to the right format

# change the paths to your own
model="easy-khair-180-gpc0.8-trans10-025000.pkl"
out="../../data/3DNoise"
target_img="../../data/input_image"

PANO_HEAD_MODEL="../../../pretrained_models/easy-khair-180-gpc0.8-trans10-025000.pkl"
# Step1: PanoHead 3D aware noise generation
# cd to 3DNoise Generation folder in order to get the 3D aware noise from PanoHead PTI
cd 3DNoise
python projector_withseg.py \
--outdir=${out} \
--num_steps 200 \
--target_img=${target_img} \
--network ${model} \
--camera_json ${target_img}/dataset.json \
--network ${PANO_HEAD_MODEL}

noise_dir="../data/3DNoise"
back_head_dir="../data/Back_Head"

Head_Back_MODEL="../../pretrained_models/back_head-230000.th"
Diff360_MODEL="../../pretrained_models/model_state-340000.th"
target_img="../data/input_image"
# # Step 2.1: Generate Content Head_Back
cd ..
torchrun --master_port 14033 inference.py \
--model_config ./model_lib/ControlNet/models/cldm_v15_reference_only_pose_enable_PC.yaml \
--test_dataset back_head_generation \
--control_mode controlnet_important \
--local_image_dir ${back_head_dir} \
--resume_dir ${Head_Back_MODEL} \
--control_type GAN_Generated \
--inference_image_path ${target_img} \
--nSample 1 \
--condition_path ../data/cam_condition/sphere32 \
--save_video \

# # Step 2.2: Generate Style Head_Back
model="easy-khair-180-gpc0.8-trans10-025000.pkl"
target_img="../../style_img/front"

torchrun --master_port 14034 inference.py \
--model_config ./model_lib/ControlNet/models/cldm_v15_reference_only_pose_enable_PC.yaml \
--test_dataset back_head_generation \
--control_mode controlnet_important \
--local_image_dir ../../style_img/back \
--resume_dir ${Head_Back_MODEL} \
--control_type GAN_Generated \
--inference_image_path ${target_img} \
--nSample 1 \
--condition_path ../data/cam_condition/sphere32 \


target_img="../data/input_image"  
# change the path to your own
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
#--style_mask ../../masks/joker_mouth_mask.png
