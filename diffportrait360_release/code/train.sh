#!/bin/bash
#SBATCH -A EUHPC_A02_031
#SBATCH -p boost_usr_prod
#SBATCH -N 1                            # 1 node
#SBATCH --time 04:00:00                  # format: HH:MM:SS
#SBATCH --ntasks-per-node=4             # 4 tasks out of 32
#SBATCH --gres=gpu:2                    # 1 gpus per node out of 4
#SBATCH --mem=123000                    # memory per node out of 494000MB (481GB)
#SBATCH --job-name=joker_style_t40           # job name
#SBATCH --error=./logs/joker_style_t40.err             # standard error file
#SBATCH --output=./logs/joker_style_t40.out            # standard output file
# export CUDA_VISIBLE_DEVICES=6
# Step 0.0 :Put the you own image to test under folder sample_data/input_image

export HF_HOME=/leonardo_work/EUHPC_A02_031/hfhub2
module load python/3.10.8--gcc--11.3.0
module load gcc
module load cuda
module load openblas
module load openmpi

source /leonardo_work/EUHPC_A02_031/furkan_env/bin/activate

export HF_TOKEN=${HF_TOKEN:-}

# Step 0.1: Using 3DDFA_V2_cropping to get camera pose and crop the image to the right format

model="easy-khair-180-gpc0.8-trans10-025000.pkl"
noise_dir="/leonardo_work/EUHPC_A02_031/DiffPortrait360/joker/dataset/3D_Noise"
target_img="/leonardo_work/EUHPC_A02_031/DiffPortrait360/joker/dataset/input_image"

# change the path to your own
PANO_HEAD_MODEL="/leonardo_work/EUHPC_A02_031/DiffPortrait360/pretrained_models/easy-khair-180-gpc0.8-trans10-025000.pkl"
Head_Back_MODEL="/leonardo_work/EUHPC_A02_031/DiffPortrait360/pretrained_models/back_head-230000.th"
Diff360_MODEL="/leonardo_work/EUHPC_A02_031/DiffPortrait360/pretrained_models/model_state-340000.th"


# # check sample_data/Back_Head folder to see if the result is correct
# # # Step3: Generate Video

torchrun --master_port 14031 train.py \
--model_config ./model_lib/ControlNet/models/cldm_v15_reference_only_temporal_pose.yaml \
--test_dataset full_head_clean_inference_final_face \
--control_mode controlnet_important \
--local_image_dir /leonardo_work/EUHPC_A02_031/DiffPortrait360/result/motion_module_train \
--resume_dir ${Diff360_MODEL} \
--control_type GAN_Generated \
--inference_image_path ../sample_data \
--train_image_path /leonardo_work/EUHPC_A02_031/DiffPortrait360/joker/dataset \
--nSample 32 \
--condition_path ../sample_data/cam_condition/sphere32 \
--denoise_from_guidance \
--initial_image_path ${noise_dir} \



