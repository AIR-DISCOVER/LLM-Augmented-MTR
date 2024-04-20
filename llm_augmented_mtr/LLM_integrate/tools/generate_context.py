# in this script, we retrieve context data for each agent in [ training | validation | testing ] set, 
# and store them for [ training | validation | testing ] processes

import torch.distributed
import _init_path # insert the project path into system path, so that we can use `import mtr.xxx`
import argparse
import datetime
import glob
import os
import re
import time
from pathlib import Path

import numpy as np
import torch
from tensorboardX import SummaryWriter
import pickle
from tqdm import tqdm

from LLM_integrate.tools.generate_feature_vector_utils import generate_feature_vector_utils
from mtr.config import cfg, cfg_from_list, cfg_from_yaml_file, log_config_to_file
from mtr.datasets import build_dataloader
from mtr.models import model as model_utils
from mtr.utils import common_utils

def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default=None, help='specify the config file')

    parser.add_argument('--batch_size', type=int, default=None, required=False, help='batch size for training')
    parser.add_argument('--workers', type=int, default=4, help='number of workers for dataloader')
    parser.add_argument('--extra_tag', type=str, default='default', help='extra tag for this experiment')
    parser.add_argument('--ckpt', required=True, type=str, default=None, help='checkpoint to start from')
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm'], default='none')
    parser.add_argument('--tcp_port', type=int, default=18888, help='tcp port for distrbuted training')
    parser.add_argument('--local_rank', type=int, default=0, help='local rank for distributed training')
    parser.add_argument('--fix_random_seed', action='store_true', default=False, help='')
    parser.add_argument('--set', dest='set_cfgs', default=None, nargs=argparse.REMAINDER,
                        help='set extra config keys if needed')

    parser.add_argument('--max_waiting_mins', type=int, default=30, help='max waiting minutes')
    parser.add_argument('--start_epoch', type=int, default=0, help='')
    parser.add_argument('--ckpt_dir', type=str, default=None, help='specify a ckpt directory to be evaluated if needed')
    parser.add_argument('--save_to_file', action='store_true', default=False, help='')
    # new add
    parser.add_argument('--retrieval_database', required=True, type=str, help='path of generated llm_out_put_with_embedding')
    parser.add_argument('--retrieval_window_size', required=False, type=int, default=4, help='the retrieval window size')
    parser.add_argument('--dataset_type', choices=["train", "valid", "test"], required=True, type=str, default="", help='generate context data of which part of dataset')

    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)
    cfg.TAG = Path(args.cfg_file).stem
    cfg.EXP_GROUP_PATH = '/'.join(args.cfg_file.split('/')[1:-1])  # remove 'cfgs' and 'xxxx.yaml'

    np.random.seed(1024)

    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs, cfg)

    return args, cfg

def save_context_data(merged_embedding_dict, save_context_dir, args):
    save_context_path = os.path.join(save_context_dir, f"context_data_{args.extra_tag}.pkl")
    pickle.dump(merged_embedding_dict, open(save_context_path, 'wb'))
    
def generate_batch_context(batch_dict, context_dict, retrieval_database_embedding_dict, retrieval_database, retrieval_window_size=4):
    # 1. calc embeddings for each agent in current batch
    scenario_id_list = batch_dict['input_dict']['scenario_id'] # shape (#agent)
    track_index_to_predict = batch_dict['input_dict']['track_index_to_predict_origin'] # shape (#agent)
    center_objects_feature = batch_dict['center_objects_feature'] #shape (#agent, 256)
    map_feature =  batch_dict['map_feature'] # shape (#agent, #polyline, 256)
    map_pos = batch_dict['map_pos'] # shape (#agent, #polyline, 3)
    map_mask = batch_dict['map_mask'] # shape (#agent, #polyline)
    center_objects_type = batch_dict['input_dict']['center_objects_type'] # shape (#agent)
    
    # if no HD-Map num_of_polyline = 1 in default
    # in this case, we only use agent embedding to do retrieval
    num_of_polylines = map_feature.shape[1]
    if num_of_polylines < 32:
        print("empty HD Map or num_of_polyline is less than 32, skip map embedding, retrieval via agent embedding")
        # shape (#agent, 256)
        batch_embedding = center_objects_feature
        # shape (#agent, num_of_retrieval_sample, 256)
        batch_retrieval_embedding_database = torch.stack([retrieval_database_embedding_dict[agent_type][:, :256] for agent_type in center_objects_type], dim=0)
        # (#agent, num_of_retrieval_sample, 256) => (#agent, num_of_retrieval_sample)
        num_of_agent, _, dim_of_embedding = batch_retrieval_embedding_database.shape
        batch_dist = (batch_embedding.view(num_of_agent, 1, dim_of_embedding) - batch_retrieval_embedding_database).norm(dim=-1)
        # get the index of cloest sample in database for current query
        # #window index will be obtained
        # shape (#agent, #window)
        topk_dists, topk_idxes = batch_dist.topk(k=retrieval_window_size, dim=-1, largest=False)
    else:
        # shape (#agent, 32*256)
        k_nearset_map_feature = generate_feature_vector_utils.find_k_nearest_map_embedding(32, map_feature, map_pos, map_mask)
        # shape (#agent, 33*256)
        batch_embedding = torch.cat([center_objects_feature, k_nearset_map_feature], dim=-1)
        # 2. retrieval corresponding context info for each agent
        # shape (#agent, num_of_retrieval_sample, 33*256)
        batch_retrieval_embedding_database = torch.stack([retrieval_database_embedding_dict[agent_type] for agent_type in center_objects_type], dim=0)
        
        num_of_agent, _, dim_of_embedding = batch_retrieval_embedding_database.shape
        # shape (#agent, num_of_retrieval_sample)
        batch_dist = (batch_embedding.view(num_of_agent, 1, dim_of_embedding) - batch_retrieval_embedding_database).norm(dim=-1)
        # get the index of cloest sample in database for current query
        # #window index will be obtained
        # shape (#agent, #window)
        topk_dists, topk_idxes = batch_dist.topk(k=retrieval_window_size, dim=-1, largest=False)

    # save these context data into a dict
    for agent_index in range(num_of_agent):
        scenario_id = scenario_id_list[agent_index]
        track_index = track_index_to_predict[agent_index].item()
        agent_type = center_objects_type[agent_index]
        track_intentions = [retrieval_database[agent_type]["intention"][sample_index] for sample_index in topk_idxes[agent_index]]
        track_affordances = [retrieval_database[agent_type]["affordance"][sample_index] for sample_index in topk_idxes[agent_index]]
        track_scenarios = [retrieval_database[agent_type]["scenario"][sample_index] for sample_index in topk_idxes[agent_index]]
        
        if scenario_id in context_dict.keys():
            context_dict[scenario_id]["track_indexes"].append(track_index)
            context_dict[scenario_id]["track_intentions"].append(track_intentions)
            context_dict[scenario_id]["track_affordances"].append(track_affordances)
            context_dict[scenario_id]["track_scenarios"].append(track_scenarios)
        else:
            context_dict.update({
                scenario_id: {
                    "track_indexes": [track_index],
                    "track_intentions": [track_intentions],
                    "track_affordances": [track_affordances],
                    "track_scenarios": [track_scenarios]
                }
            })
    
    
def unify_retrieval_database_by_padding_1st_dim(vehicle_database, pedestrian_database, cyclist_database):
    database_list = [vehicle_database, pedestrian_database, cyclist_database]
    max_agent_num = max([len(database) for database in database_list])
    feature_num = len(vehicle_database[0])
    
    padding_embedding = [1000000 for _ in range(feature_num)]
    
    for database in database_list:
        for _ in range(max_agent_num - len(database)):
            database.append(padding_embedding)
            

def obtain_context_data(model, dataloader, args, log_dir, logger, epoch_id, dist_test=False):
    # we first load the context data
    if args.ckpt is not None:
        it, epoch = model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=dist_test)
    else:
        return
    # send all params to GPU
    model.cuda()
    logger.info(f'*************** LOAD MODEL (epoch={epoch}, iter={it}) for CONTEXT GENERATE *****************')
    # set retrieval database path and save path
    retrieval_database_path = args.retrieval_database
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    save_context_dir = os.path.join("/".join(cur_dir.split("/")[:-1]), "context_data", args.dataset_type)
    os.makedirs(save_context_dir, exist_ok=True)
    
    # load retrieval database
    with open(retrieval_database_path, 'rb') as retrieval_database_file:
        retrieval_database = pickle.load(retrieval_database_file)
    # load the retrieval database into GPU
    retrieval_database_embedding_vehicle = retrieval_database["TYPE_VEHICLE"]["embedding"]
    retrieval_database_embedding_pedestrian = retrieval_database["TYPE_PEDESTRIAN"]["embedding"]
    retrieval_database_embedding_cyclist = retrieval_database["TYPE_CYCLIST"]["embedding"]
    # shape (N_v, 33*256), (N_p, 33*256), (N_c, 33*256)
    unify_retrieval_database_by_padding_1st_dim(retrieval_database_embedding_vehicle, retrieval_database_embedding_pedestrian, retrieval_database_embedding_cyclist)
       
    cur_device = f"cuda:{os.environ['LOCAL_RANK']}"
    
    retrieval_database_embedding_vehicle = torch.tensor(retrieval_database_embedding_vehicle, dtype=torch.float32, device=cur_device)
    retrieval_database_embedding_pedestrian = torch.tensor(retrieval_database_embedding_pedestrian, dtype=torch.float32, device=cur_device)
    retrieval_database_embedding_cyclist = torch.tensor(retrieval_database_embedding_cyclist, dtype=torch.float32, device=cur_device)

    assert retrieval_database_embedding_vehicle.shape == retrieval_database_embedding_pedestrian.shape and retrieval_database_embedding_pedestrian.shape == retrieval_database_embedding_cyclist.shape, "shape of embedding database is not equal"
    
    retrieval_database_embedding_dict = {
        "TYPE_VEHICLE": retrieval_database_embedding_vehicle,
        "TYPE_PEDESTRIAN": retrieval_database_embedding_pedestrian,
        "TYPE_CYCLIST": retrieval_database_embedding_cyclist
    }
    
    print(f'^.^ Load embedding database to CUDA={os.environ["LOCAL_RANK"]} successfully ^.^')
    
    # iterate and retrieve context info for each agent
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
        progress_bar = tqdm(total=len(dataloader), leave=True, desc='gen fv', dynamic_ncols=True)

    context_dict = {} # key: scenario_id, value: dict(key: track_indexes, track_intentions, track_affordances, track_scenarios)
    for i, batch_dict in enumerate(dataloader):
        with torch.no_grad():
            encode_batch_dict = model(batch_dict)
            # context_dict will change after this function finished
            generate_batch_context(encode_batch_dict, context_dict, retrieval_database_embedding_dict, retrieval_database, retrieval_window_size=args.retrieval_window_size)
            torch.distributed.barrier()

        disp_dict = {}

        if cfg.LOCAL_RANK == 0 and (i % 50 == 0 or i == 0 or i + 1== len(dataloader)):
            past_time = progress_bar.format_dict['elapsed']
            second_each_iter = past_time / max(i, 1.0)
            remaining_time = second_each_iter * (len(dataloader) - i)
            disp_str = ', '.join([f'{key}={val:.3f}' for key, val in disp_dict.items() if key != 'lr'])
            batch_size = batch_dict.get('batch_size', None)
            logger.info(f'generate context: batch_iter={i}/{len(dataloader)}, batch_size={batch_size}, iter_cost={second_each_iter:.2f}s, '
                        f'time_cost: {progress_bar.format_interval(past_time)}/{progress_bar.format_interval(remaining_time)}, '
                        f'{disp_str}')
            
        
    if dist_test:
        logger.info(f'Number of scenarios in context_dict from multiple GPUs: {len(context_dict)}')
        merged_embedding_dict = common_utils.merge_context_dict_dist(context_dict, tmpdir=os.path.join(save_context_dir, "tmpdir"))

    if cfg.LOCAL_RANK == 0:
        logger.info(f'Number of scenarios in context_dict from multiple GPUs: {len(merged_embedding_dict)}')
        save_context_data(merged_embedding_dict, save_context_dir, args)
        logger.info('****************GENERATE CONTEXT FINISHED.*****************')
        
    if cfg.LOCAL_RANK == 0:
        progress_bar.close()
    

def main():
    args, cfg = parse_config()
    
    if args.launcher == 'none':
        dist_test = False
        total_gpus = 1
    else:
        total_gpus, cfg.LOCAL_RANK = getattr(common_utils, 'init_dist_%s' % args.launcher)(
            args.tcp_port, args.local_rank, backend='nccl'
        )
        dist_test = True
        
    if args.batch_size is None:
        args.batch_size = cfg.OPTIMIZATION.BATCH_SIZE_PER_GPU # type: ignore
    else:
        assert args.batch_size % total_gpus == 0, 'Batch size should match the number of gpus'
        args.batch_size = args.batch_size // total_gpus
        
    output_dir = cfg.ROOT_DIR / 'output' / cfg.EXP_GROUP_PATH / cfg.TAG / args.extra_tag
    output_dir.mkdir(parents=True, exist_ok=True)

    log_dir = output_dir / 'log'

    epoch_id = None

    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / ('log_obtain_context_data_%s.txt' % datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
    logger = common_utils.create_logger(log_file, rank=cfg.LOCAL_RANK)
    
    # log to file
    logger.info('**********************Start logging**********************')
    gpu_list = os.environ['CUDA_VISIBLE_DEVICES'] if 'CUDA_VISIBLE_DEVICES' in os.environ.keys() else 'ALL'
    logger.info('CUDA_VISIBLE_DEVICES=%s' % gpu_list)

    if dist_test:
        logger.info('total_batch_size: %d' % (total_gpus * args.batch_size))
    for key, val in vars(args).items():
        logger.info('{:16} {}'.format(key, val))
    log_config_to_file(cfg, logger=logger)

    if args.fix_random_seed:
        common_utils.set_random_seed(666)
    
    assert args.ckpt is not None, "ckpt_dir is none"
    
    dataset, dataloader, sampler = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        batch_size=args.batch_size,
        dist=dist_test, workers=args.workers, logger=logger, training=False, dataset_type=args.dataset_type
    )

    cfg.MODEL.GENERATE_EMBEDDING = cfg.DATA_CONFIG.GENERATE_EMBEDDING
    model = model_utils.MotionTransformer(config=cfg.MODEL)
    with torch.no_grad():
        obtain_context_data(model, dataloader, args, log_dir, logger, epoch_id, dist_test=dist_test)


if __name__ == "__main__":
    main()