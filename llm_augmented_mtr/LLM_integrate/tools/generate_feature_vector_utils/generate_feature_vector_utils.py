import pickle
import time

import numpy as np
import torch
import tqdm
import os
import torch.distributed as dist
from mtr.utils import common_utils
from LLM_integrate.embedding.embedding_format import embedding_dict, intention_type_decode_dict, affordance_type_decode_dict, scenario_type_decode_dict

def save_feature_vector(embedding_dict, embedding_file_dir, extra_tag):
    embedding_file_path = os.path.join(embedding_file_dir, f"llm_output_context_with_{extra_tag}.pkl")
    
    with open(embedding_file_path, 'wb') as embedding_file:
        pickle.dump(embedding_dict,embedding_file)

def find_k_nearest_map_embedding(k, map_feature, map_pos, map_mask):
    # map_feature shape (#agent, #polyline, 256)
    # map_pos     shape (#agent, #polyline, 3)
    # map_mask    shape (#agent, #polyline)
    # set the invaild map polyline's coordinate to infinity
    map_pos[~map_mask] = 100000000
    num_of_agent, num_of_polyline, _ = map_pos.shape
    # (#agent, #polyline)
    # distance = (map_pos[:,:,:2] - torch.zeros(num_of_agent, num_of_polyline, 2).to(map_pos.device)).norm(dim=-1)
    distance = map_pos[:,:,:2].to(map_pos.device).norm(dim=-1)
    # (#agent, k)
    topk_dist, polyline_idxs = distance.topk(k=k, dim=-1, largest=False)
    # bool matrix of shape (#agent, k)
    polyline_mask = (topk_dist > 100000000)
    # (#num_of_agent,) => (#num_of_agent, 1) => (#num_of_agent, k)
    agent_idx = torch.arange(num_of_agent).view(-1, 1).expand(-1, k)
    # (#agent, k, 256)
    need_map_feature = map_feature[agent_idx, polyline_idxs]
    
    need_map_feature[polyline_mask.unsqueeze(-1).expand(-1, -1, 256)] = 0.0
    # (#agent, k*256)
    need_map_feature = need_map_feature.view(num_of_agent, -1)
    
    return need_map_feature

def covert_onehot_context_into_text(track_intentions, track_affordances, track_scenarios):
    track_intentions_text = []
    track_affordances_text = []
    track_scenarios_text = []
    
    # we use weighted one hot encode for intention
    # encoded_data[encode_dict[value]] = len(data) - index
    for onehot_intention in track_intentions:
        # get the number of intentions
        num_of_intention = 0
        for code in onehot_intention:
            if int(code) != 0:
                num_of_intention += 1
        track_intention_text_single = ["" for _ in range(num_of_intention)]
        
        # decode the code and get intention text
        for index, code in enumerate(onehot_intention):
            if int(code) != 0:
                rank = num_of_intention - code
                track_intention_text_single[rank] = intention_type_decode_dict[index]
        
        # insert this decode vector
        track_intentions_text.append(track_intention_text_single)
    
    # we use one hot encode for affordance and scenario
    # encoded_data[encode_dict[value]] = 1
    for onehot_affordance in track_affordances:
        track_affordance_text_single = []
        # decode the code and get affordance text
        for index, code in enumerate(onehot_affordance):
            if int(code) != 0:
                track_affordance_text_single.append(affordance_type_decode_dict[index])
        # insert this decode vector
        track_affordances_text.append(track_affordance_text_single)
    
    for onehot_scenario in track_scenarios:
        track_scenario_text_single = []
        for index, code in enumerate(onehot_scenario):
            if int(code) != 0:
                track_scenario_text_single.append(scenario_type_decode_dict[index])
        track_scenarios_text.append(track_scenario_text_single)
        
    return track_intentions_text, track_affordances_text, track_scenarios_text
    

def insert_all_info_into_result(embedding, scenario_id_list, objects_type_list, track_index_to_predict, 
                                track_intentions, track_affordances, track_scenarios):
    num_of_agent = embedding.shape[0]
    for agent_index in range(num_of_agent):
        agent_type = objects_type_list[agent_index]
        embedding_dict[agent_type]["scenario_id"].append(scenario_id_list[agent_index])
        embedding_dict[agent_type]["track_index"].append(track_index_to_predict[agent_index].item())
        embedding_dict[agent_type]["embedding"].append(embedding[agent_index].tolist())
        embedding_dict[agent_type]["intention"].append(track_intentions[agent_index])
        embedding_dict[agent_type]["affordance"].append(track_affordances[agent_index])
        embedding_dict[agent_type]["scenario"].append(track_scenarios[agent_index])
        

def save_batch_embedding(batch_dict):
    # Here we generate the embedding for each batch
    # ==================Needed data===================
    # batch_dict['center_objects_feature'] shape (#agent, 256)
    # batch_dict['map_feature'] shape (#agent, #polyline, 256)
    # batch_dict['input_dict']['center_objects_type'] shape (#agent)
    
    # batch_dict['input_dict']['track_intentions'] shape (#agent, len(type_of_intention)) <= this is one-hot vector
    # batch_dict['input_dict']['track_affordances'] shape (#agent, len(type_of_affordance)) <= this is one-hot vector
    # batch_dict['input_dict']['track_scenarios'] shape (#agent, len(type_of_scenario)) <= this is one-hot vector
    
    # bacth_dict['map_pos'] shape (#agent, #polyline, 3)
    # batch_dict['map_mask'] shape (#agent, #polyline)
    scenario_id_list = batch_dict['input_dict']['scenario_id']
    track_index_to_predict = batch_dict['input_dict']['track_index_to_predict_origin']
    center_objects_feature = batch_dict['center_objects_feature'] #shape (#agent, 256)
    map_feature =  batch_dict['map_feature'] # shape (#agent, #polyline, 256)
    map_pos = batch_dict['map_pos']
    map_mask = batch_dict['map_mask']
    center_objects_type = batch_dict['input_dict']['center_objects_type'] # shape (#agent)
    # context infomation
    track_intentions = batch_dict['input_dict']['track_intentions'] # shape (#agent, #track_class) <= this is one-hot vector
    track_affordances = batch_dict['input_dict']['track_affordances']
    track_scenarios = batch_dict['input_dict']['track_scenarios']
    
    k_nearset_map_feature = find_k_nearest_map_embedding(32, map_feature, map_pos, map_mask)
    
    final_embedding = torch.cat([center_objects_feature, k_nearset_map_feature], dim=1) # type: ignore
    
    # decode one hot to text
    track_intentions_text, track_affordances_text, track_scenarios_text = covert_onehot_context_into_text(track_intentions, track_affordances, track_scenarios) # type: ignore
    
    # insert embedding into embedding file
    insert_all_info_into_result(final_embedding, scenario_id_list, center_objects_type, track_index_to_predict, 
                                track_intentions_text, track_affordances_text, track_scenarios_text)
    

def generate_feature_vector(cfg, model, dataloader, epoch_id, logger, dist_test=False, save_to_file=False, result_dir=None, logger_iter_interval=50, extra_tag=""):
    cur_dir_name = os.path.dirname(os.path.abspath(__file__))
    tools_dir_name = os.path.dirname(cur_dir_name)
    llm_integrate_dir_name = os.path.dirname(tools_dir_name)
    embedding_file_dir = os.path.join(llm_integrate_dir_name, "LLM_output", "context_file")

    logger.info('*************** EPOCH %s GENERATE FEATURE VECTOR *****************' % epoch_id)
    if dist_test:
        if not isinstance(model, torch.nn.parallel.DistributedDataParallel):
            num_gpus = torch.cuda.device_count()
            local_rank = cfg.LOCAL_RANK % num_gpus
            model = torch.nn.parallel.DistributedDataParallel(
                    model,
                    device_ids=[local_rank],
                    broadcast_buffers=False
            )
    model.eval()

    if cfg.LOCAL_RANK == 0:
        progress_bar = tqdm.tqdm(total=len(dataloader), leave=True, desc='gen fv', dynamic_ncols=True)
    start_time = time.time()

    for i, batch_dict in enumerate(dataloader):
        with torch.no_grad():
            encode_batch_dict = model(batch_dict)
            save_batch_embedding(encode_batch_dict)
            dist.barrier()

        disp_dict = {}

        if cfg.LOCAL_RANK == 0 and (i % logger_iter_interval == 0 or i == 0 or i + 1== len(dataloader)):
            past_time = progress_bar.format_dict['elapsed']
            second_each_iter = past_time / max(i, 1.0)
            remaining_time = second_each_iter * (len(dataloader) - i)
            disp_str = ', '.join([f'{key}={val:.3f}' for key, val in disp_dict.items() if key != 'lr'])
            batch_size = batch_dict.get('batch_size', None)
            logger.info(f'generate embedding: batch_iter={i}/{len(dataloader)}, batch_size={batch_size}, iter_cost={second_each_iter:.2f}s, '
                        f'time_cost: {progress_bar.format_interval(past_time)}/{progress_bar.format_interval(remaining_time)}, '
                        f'{disp_str}')

    if dist_test:
        logger.info(f'Number of different ego car before merging from multiple GPUs: VEHICLE({len(embedding_dict["TYPE_VEHICLE"]["embedding"])}), PEDESTRIAN({len(embedding_dict["TYPE_PEDESTRIAN"]["embedding"])}), CYCLIST({len(embedding_dict["TYPE_CYCLIST"]["embedding"])})')
        merged_embedding_dict = common_utils.merge_embedding_dict_dist(embedding_dict, tmpdir=os.path.join(embedding_file_dir, "tmpdir"))

    if cfg.LOCAL_RANK == 0:
        logger.info(f'Number of different ego car after merging from multiple GPUs: VEHICLE({len(merged_embedding_dict["TYPE_VEHICLE"]["embedding"])}), PEDESTRIAN({len(merged_embedding_dict["TYPE_PEDESTRIAN"]["embedding"])}), CYCLIST({len(merged_embedding_dict["TYPE_CYCLIST"]["embedding"])})') # type: ignore
        save_feature_vector(merged_embedding_dict, embedding_file_dir, extra_tag)
        logger.info('****************GENERATE FEATURE VECTOR FINISHED.*****************')
        
    if cfg.LOCAL_RANK == 0:
        progress_bar.close()
    


if __name__ == '__main__':
    pass
