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

from LLM_integrate.tools.generate_feature_vector_utils import generate_feature_vector_utils # type: ignore
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

    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)
    cfg.TAG = Path(args.cfg_file).stem
    cfg.EXP_GROUP_PATH = '/'.join(args.cfg_file.split('/')[1:-1])  # remove 'cfgs' and 'xxxx.yaml'

    np.random.seed(1024)

    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs, cfg)

    return args, cfg


def generate_feature_vector_by_ckpt_encoder(model, test_loader, args, log_dir, logger, epoch_id, dist_test=False):
    # load checkpoint
    if args.ckpt is not None: 
        it, epoch = model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=dist_test)
    else:
        it, epoch = -1, -1
    model.cuda()

    logger.info(f'*************** LOAD MODEL (epoch={epoch}, iter={it}) for EVALUATION *****************')
    # start iterate and generate feature vector
    generate_feature_vector_utils.generate_feature_vector(
        cfg, model, test_loader, epoch_id, logger, dist_test=dist_test,
        result_dir=log_dir, save_to_file=args.save_to_file, extra_tag = args.extra_tag
    )

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
    log_file = log_dir / ('log_feature_vector_%s.txt' % datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
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

    test_set, test_loader, sampler = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        batch_size=args.batch_size,
        dist=dist_test, workers=args.workers, logger=logger, training=False
    )
    # before we inistialize MTR, we need tell him there is no need to init decoder and use it
    cfg.MODEL.GENERATE_EMBEDDING = cfg.DATA_CONFIG.GENERATE_EMBEDDING
    model = model_utils.MotionTransformer(config=cfg.MODEL)
    with torch.no_grad():
        generate_feature_vector_by_ckpt_encoder(model, test_loader, args, log_dir, logger, epoch_id, dist_test=dist_test)


if __name__ == '__main__':
    main()