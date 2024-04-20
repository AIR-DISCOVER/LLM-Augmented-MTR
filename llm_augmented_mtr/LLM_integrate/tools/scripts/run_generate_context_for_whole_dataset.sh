dataset_type_array=("train" "valid" "test")

for dataset_type in ${dataset_type_array[@]}
do
    CUDA_VISIBLE_DEVICES=3,4 \
    bash scripts/generate_context.sh 2 \
    --cfg_file cfgs/generate_context.yaml \
    --ckpt ../../output/waymo/mtr+100_percent_data/mtr+100/ckpt/checkpoint_epoch_29.pth \
    --extra_tag encoder_100 \
    --batch_size 2 \
    --retrieval_database ../LLM_output/context_file/llm_output_context_with_encoder_100.pkl \
    --retrieval_window_size 4 \
    --dataset_type $dataset_type
done

# `retrieval_database` stands for how many nearest neighbors will be saved for one agent