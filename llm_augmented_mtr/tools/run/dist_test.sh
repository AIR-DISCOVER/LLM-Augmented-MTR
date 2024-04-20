CUDA_VISIBLE_DEVICES=4,6 \
bash scripts/dist_test.sh 2 \
--cfg_file cfgs/waymo/mtr+100_percent_data.yaml \
--ckpt ../output/waymo/mtr+100_percent_data_/mtr+100/ckpt/checkpoint_epoch_1.pth \
--batch_size 20 \
--extra_tag test_mtr+100_percent \
--dataset_type test

# This file helps you test the model's ability on the test set of WOMD
# you can use the result.pkl file to submit on WOMD leaderboar! see /submit_to_waymo
# paramters meaning
# CUDA_VISIBLE_DEVICES => GPU number on your server to conduct the validation tast
# --cfg_file [DO NOT CHANGE] => the configuration file path
# --ckpt => the check point file path
# --batch_size => how many samples of data will be processed in one iteration, for example, if you have 2 GPU and batchsize = 20, then for each GPU, 10 data samples will be processed in one iteration.
# --extra_tag => the folder name to save log file of this task
# --dataset_type [DO NOT CHANGE] => which part of WOMD will provided for dataloader