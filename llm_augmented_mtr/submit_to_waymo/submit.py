# LLM-Augemnted-MTR
# Write by Hongrui Zhur
# All Right Reserved

import math
import os
import pickle
import random
from functools import partial

import numpy as np
import tensorflow as tf
import pandas as pd
from tqdm import tqdm

# =============================== Usage ==================================== #
# 1. set the path of your result.pkl (the result file of test set)
data = pd.read_pickle('INPUT YOUR result.pkl PATH')
# 2. set the name of the file generated later
file_name = 'my_submit'
# 3. input some information below
info_dict = {
    'account_name': 'Input Your Account Name Here', # Attention! make sure this account name consist of your email adress that used to register WOMD
    'unique_method_name': 'Input Your Method Name',
    'authors': ['Author_1', "Author_2"],
    'affiliation': 'Input Your Affiliation',
    'method_link': 'https://input.your.method.link',
    'uses_lidar_data': False # if your method use Lidar data, you may set this field to True
} 


data_list = np.array(data)

num_of_agent = 0
for scenario in data_list:
    num_of_agent += len(scenario)
    
process_bar = tqdm(total=num_of_agent)

from waymo_open_dataset.protos import motion_submission_pb2

output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
os.makedirs(output_dir, exist_ok=True)
other_params = ['out_submit']

def generate_protobuf(output_dir, file_name, other_params, list, process_bar, info_dict):
    submission = motion_submission_pb2.MotionChallengeSubmission()
    submission.account_name = info_dict['account_name']
    submission.unique_method_name = info_dict['unique_method_name']
    submission.authors.extend(info_dict['authors'])
    submission.affiliation = info_dict['affiliation']
    submission.method_link = info_dict['method_link']
    submission.uses_lidar_data = info_dict['uses_lidar_data']
    MOTION_PREDICTION = True
    scenario_id = []
    object_id = []
    prediction_trajectory = []
    prediction_score = []

    for i in range(list.shape[0]):
        scenario_id = data_list[i][0]['scenario_id']
        scenario_prediction = submission.scenario_predictions.add()
        scenario_prediction.scenario_id = scenario_id
        prediction_set = scenario_prediction.single_predictions
        for data_obj in data_list[i]:
            object_id = data_obj['object_id']
            prediction_trajectory = data_obj['pred_trajs']
            prediction_score = data_obj['pred_scores']
        
            if isinstance(prediction_trajectory, np.ndarray):
                prediction_trajectory = tf.convert_to_tensor(prediction_trajectory)
            if isinstance(prediction_score, np.ndarray):
                prediction_score = tf.convert_to_tensor(prediction_score)

            predict_num = len(prediction_trajectory)

            if len(prediction_trajectory.shape) == 5:
                MOTION_PREDICTION = False
                joint_prediction = scenario_prediction.joint_prediction
                assert predict_num == 1
                for i in range(predict_num):
                    for k in range(6):
                        ScoredJointTrajectory = joint_prediction.joint_trajectories.add()
                        ScoredJointTrajectory.confidence = prediction_score[i, k]
                        assert prediction_trajectory.shape[2] == 2
                        for c in range(2):
                            ObjectTrajectory = ScoredJointTrajectory.trajectories.add()
                            ObjectTrajectory.object_id = object_id[c]
                            Trajectory = ObjectTrajectory.trajectory

                            interval = 5
                            traj = prediction_trajectory[i, k, c, (interval - 1)::interval, :]
                            Trajectory.center_x[:] = traj[:, 0].numpy().tolist()
                            Trajectory.center_y[:] = traj[:, 1].numpy().tolist()
                        print(Trajectory)
                pass
            else:
                # SingleObjectPrediction
                prediction = prediction_set.predictions.add()
                prediction.object_id = object_id
                # obj = structs.MultiScoredTrajectory(prediction_score[i, :].numpy(), prediction_trajectory[i, :, :, :].numpy())
                # waymo_pred[(scenario_id, object_id.numpy()[i])] = obj

                for k in range(6):
                    # ScoredTrajectory
                    scored_trajectory = prediction.trajectories.add()
                    scored_trajectory.confidence = prediction_score[k]
                    trajectory = scored_trajectory.trajectory

                    if prediction_trajectory.shape[1] == 16:
                        traj = prediction_trajectory[k, :, :]
                    else:
                        # 如果是预测了80帧，则进行处理，每5帧取一次结果
                        assert prediction_trajectory.shape[1] == 80, prediction_trajectory.shape
                        interval = 5
                        traj = prediction_trajectory[k, (interval - 1)::interval, :]

                    trajectory.center_x[:] = traj[:, 0].numpy().tolist()
                    trajectory.center_y[:] = traj[:, 1].numpy().tolist()
                    
            process_bar.update(1)


    if MOTION_PREDICTION:
        submission.submission_type = motion_submission_pb2.MotionChallengeSubmission.SubmissionType.MOTION_PREDICTION
    else:
        submission.submission_type = motion_submission_pb2.MotionChallengeSubmission.SubmissionType.INTERACTION_PREDICTION


    # embed()
    if 'out_submit' in other_params:
        path = os.path.join(output_dir, file_name)
        with open(path, "wb") as f:
            f.write(submission.SerializeToString())

        os.system(f'tar -zcvf {path}.tar.gz {path}')
        os.system(f'rm {path}')


generate_protobuf(output_dir=output_dir,file_name=file_name,other_params=other_params,list=data_list,process_bar=process_bar, info_dict=info_dict)