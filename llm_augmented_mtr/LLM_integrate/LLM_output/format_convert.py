# This file convert files of LLM's output into 1) info file and 2) context info file
# so that the model can easily load and process LLM's output data
# <=== input: LLM's output (json)
# 1. data cleaning
# 2. data merge (merge all file into one dict)
# 3. generate info file via scenario_id and track_index_to_predict
# 4. generate context file
# ===> output: 1) info file and 2) context file

import json
import os
import pickle
from tqdm import tqdm
import data_cleaning_dict


def generate_info_file(llm_output_dict, cur_dir_name):
    print("===> Begin generate info file!")
    validation_set_path = "llm_augmented_mtr/data/waymo/mtr_processed/processed_scenarios_validation"
    info_file_save_path = os.path.join(cur_dir_name, "info_file", "llm_used_dataset_info.pkl")
    
    info_data = []
    for scenario_id, context_data in tqdm(llm_output_dict.items()):
        track_indexes = context_data["track_indexes"]
        # element in list `info`
        info_element = {}
        # open corresponding dataset file
        with open(os.path.join(validation_set_path, f"sample_{scenario_id}.pkl"), 'rb') as scenario_data_file:
            scenario_data = pickle.load(scenario_data_file)
        # scenario_data's keys
        # ['track_infos', 'dynamic_map_infos', 'map_infos', 'scenario_id', 'timestamps_seconds', 'current_time_index', 'sdc_track_index', 'objects_of_interest', 'tracks_to_predict']
        # keys of element in info_file
        # # # # # # # # # # # # # # # # # # # # # # # # # #['scenario_id', 'timestamps_seconds', 'current_time_index', 'sdc_track_index', 'objects_of_interest', 'tracks_to_predict']
        info_element['scenario_id'] = scenario_data['scenario_id']
        info_element['timestamps_seconds'] = scenario_data['timestamps_seconds']
        info_element['current_time_index'] = scenario_data['current_time_index']
        info_element['sdc_track_index'] = scenario_data['sdc_track_index']
        info_element['objects_of_interest'] = scenario_data['objects_of_interest']
        # for `tracks_to_predict`, as llm may use only part of tracks_to_predict, we need to cut this
        # initialize
        info_element['tracks_to_predict'] = {
            'track_index': [],
            'difficulty': [],
            'object_type': []
        }
        info_element['tracks_to_predict']['track_index'] = track_indexes
        for track_index_of_info_element in track_indexes:
            for index_of_track_index, track_index_of_scenario_data in enumerate(scenario_data['tracks_to_predict']['track_index']):
                if track_index_of_info_element == track_index_of_scenario_data:
                    info_element['tracks_to_predict']['difficulty'].append(scenario_data['tracks_to_predict']['difficulty'][index_of_track_index])
                    info_element['tracks_to_predict']['object_type'].append(scenario_data['tracks_to_predict']['object_type'][index_of_track_index])
                    
        info_data.append(info_element)
        
    with open(info_file_save_path, 'wb') as info_file:
        pickle.dump(info_data, info_file)
    
    print(f"<=== info file has been saved to {info_file_save_path}!")

def generate_context_file(llm_output_dict, cur_dir_name):
    print("===> Begin generate context file")
    context_file_save_path = os.path.join(cur_dir_name, "context_file", "llm_output_context.pkl")
    # directly use our merged llm output data
    with open(context_file_save_path, 'wb') as context_file:
        pickle.dump(llm_output_dict, context_file)
    print(f"<=== context file has been saved to {context_file_save_path}")

def main():
    current_dirname = os.path.dirname(os.path.abspath(__file__))
    
    llm_output_filename_list = os.listdir(os.path.join(current_dirname, "raw_data"))
    
    llm_output_data_merge = {}
    
    for llm_output_filename in llm_output_filename_list:
        llm_output_filepath = os.path.join(current_dirname, "raw_data", llm_output_filename)

        with open(llm_output_filepath, 'r') as llm_output_file:
            llm_output_data = json.load(llm_output_file)
        # 1. data clean
        for scenario_id, llm_output_data_scenario in llm_output_data.items():
            # for each ego-car
            for i, _ in enumerate(llm_output_data_scenario["track_ids"]):
                llm_output_data_scenario["track_ids"][i] = int(llm_output_data_scenario["track_ids"][i])
                for j, intention in enumerate(llm_output_data_scenario["track_values"][i]):
                    llm_output_data_scenario["track_values"][i][j] = data_cleaning_dict.intention_clean_dict[intention]
                for j, affordance in enumerate(llm_output_data_scenario["track_affordances"][i]):
                    llm_output_data_scenario["track_affordances"][i][j] = data_cleaning_dict.affordance_clean_dict[affordance]
                for j, scenario in enumerate(llm_output_data_scenario["track_scenarios"][i]):
                    llm_output_data_scenario["track_scenarios"][i][j] = data_cleaning_dict.scenario_clean_dict[scenario]

        # 2. data merge
        for scenario_id, llm_output_data_scenario in llm_output_data.items():
            if scenario_id in llm_output_data_merge.keys():
                for per_track_index in llm_output_data_scenario["track_ids"]:
                    llm_output_data_merge[scenario_id]["track_indexes"].append(per_track_index)
                for per_track_intention in llm_output_data_scenario["track_values"]:
                    llm_output_data_merge[scenario_id]["track_intentions"].append(per_track_intention)
                for per_track_affordance in llm_output_data_scenario["track_affordances"]:
                    llm_output_data_merge[scenario_id]["track_affordances"].append(per_track_affordance)
                for per_track_scenario in llm_output_data_scenario["track_scenarios"]:
                    llm_output_data_merge[scenario_id]["track_scenarios"].append(per_track_scenario)
            else:
                llm_output_data_merge[scenario_id] = {
                    "track_indexes": llm_output_data_scenario["track_ids"],
                    "track_intentions": llm_output_data_scenario["track_values"],
                    "track_affordances": llm_output_data_scenario["track_affordances"],
                    "track_scenarios": llm_output_data_scenario["track_scenarios"]
                }
            
    # now we have collected all context information into `llm_output_data_merge`
    # in order to add these data into MTR model, we need to generate 1) info file and 2) context info file, as dataloader need these
    # we better follow the structure of origin info file in the preprocessed waymo motion dataset
    # you can see an example info file in /info_file/info_example.txt
    
    # 3. generate info file via scenario_id and track_index_to_predict
    generate_info_file(llm_output_data_merge, current_dirname)
    
    # 4. generate context file
    generate_context_file(llm_output_data_merge, current_dirname)

if __name__ == "__main__":
    main()