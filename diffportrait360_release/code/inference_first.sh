#!/bin/bash
# export CUDA_VISIBLE_DEVICES=6
# Step 0.0 :Put the you own image to test under folder data/input_image

# Step 0.1: Using 3DDFA_V2_cropping to get camera pose and crop the image to the right format

# change the paths to your own
model="easy-khair-180-gpc0.8-trans10-025000.pkl"
out="../data/3DNoise"
target_img="../data/input_image"

PANO_HEAD_MODEL="../../pretrained_models/easy-khair-180-gpc0.8-trans10-025000.pkl"
Head_Back_MODEL="../../pretrained_models/back_head-230000.th"
Diff360_MODEL="../../pretrained_models/model_state-340000.th"

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
