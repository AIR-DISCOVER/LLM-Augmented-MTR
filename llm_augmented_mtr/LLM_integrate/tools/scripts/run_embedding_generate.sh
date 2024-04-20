CUDA_VISIBLE_DEVICES=6,7 \
bash scripts/generate_feature_vector.sh 2 \
--cfg_file cfgs/generate_feature_vector.yaml \
--ckpt ../../output/waymo/mtr+20_percent_data/mtr+20/ckpt/checkpoint_epoch_29.pth \
--extra_tag encoder_20 \
--batch_size 40

# CUDA_VISIBLE_DEVICES=6,7 \
# bash scripts/generate_feature_vector.sh 2 \
# --cfg_file cfgs/generate_feature_vector.yaml \
# --ckpt ../../output/waymo/mtr+100_percent_data/mtr+100/ckpt/checkpoint_epoch_29.pth \
# --extra_tag encoder_100 \
# --batch_size 40

# ../../output/waymo/mtr+20_percent_data/mtr+100/ckpt/checkpoint_epoch_29.pth
# ../../output/waymo/mtr+20_percent_data/mtr+20/ckpt/checkpoint_epoch_29.pth