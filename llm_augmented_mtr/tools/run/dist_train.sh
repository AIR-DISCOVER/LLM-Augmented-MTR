CUDA_VISIBLE_DEVICES=5,7 \
bash scripts/dist_train.sh 2 \
--cfg_file cfgs/waymo/mtr+100_percent_data.yaml \
--batch_size 6 \
--epochs 30 \
--extra_tag mtr+100_percent

# This file helps you train the origin MTR (https://github.com/sshaoshuai/MTR)
# paramters meaning
# CUDA_VISIBLE_DEVICES => GPU number on your server to conduct the validation tast
# --cfg_file [DO NOT CHANGE] => the configuration file path
# --batch_size => how many samples of data will be processed in one iteration, for example, if you have 2 GPU and batchsize = 20, then for each GPU, 10 data samples will be processed in one iteration.
# --extra_tag => the folder name to save log file of this task