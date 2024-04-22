import pickle
import os
from map_utils import *

    
def main():
    # Refer to "Dataset Preparation" in LLM-Augmented MTR, use validation
    ori_val_dir = '../data/waymo/mtr_processed/processed_scenarios_validation/'
    val_path_list = [os.path.join(ori_val_dir, f) for f in os.listdir(ori_val_dir)]
    data_path_list = val_path_list
    
    # Your output path
    output_dir = 'LLM-Augmented-MTR/llm_augmented_prompt/tc_map/output'
    random_val_list_path = "LLM-Augmented-MTR/llm_augmented_prompt/data/random_val_list.txt"
    os.makedirs(output_dir, exist_ok=True)
    
    print('====> Start processing data')

    # Customized data range
    s_idx = 0
    end_idx = 50
    # Random sampling
    with open(random_val_list_path, "r") as file:
        random_val_list = eval(file.read())
    for idx, random_id in enumerate(random_val_list[s_idx:end_idx]):
        print("scenario_num is: ", s_idx + idx + 1, ", scenario_idx is: ", random_id)
        with open(data_path_list[random_id], 'rb') as f:
            scenario_info = pickle.load(f)
        gen_falg = vis_frame(scenario_info, output_dir)
        
        if gen_falg == 0:
            break
    

if __name__ == "__main__":
    main()