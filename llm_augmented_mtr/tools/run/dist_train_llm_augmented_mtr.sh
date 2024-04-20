CUDA_VISIBLE_DEVICES=1,2,3,4,7 \
bash scripts/dist_train.sh 5 \
--cfg_file cfgs/waymo/mtr+100_percent_data_llm_augmented.yaml \
--batch_size 80 \
--epochs 30 \
--extra_tag llm_augmented_mtr+100_percent

# This file helps you train our LLM-Augmented-MTR
# paramters meaning
# CUDA_VISIBLE_DEVICES => GPU number on your server to conduct the validation tast
# --cfg_file [DO NOT CHANGE] => the configuration file path
# --batch_size => how many samples of data will be processed in one iteration, for example, if you have 2 GPU and batchsize = 20, then for each GPU, 10 data samples will be processed in one iteration.
# --extra_tag => the folder name to save log file of this task