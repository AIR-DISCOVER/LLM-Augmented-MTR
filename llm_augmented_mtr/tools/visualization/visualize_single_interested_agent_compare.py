import math
from visualization_variables import colorTable, vis_scenario_id, vis_dict

from tqdm import tqdm
import numpy as np
import os
import matplotlib.transforms as transforms
from matplotlib.patches import Rectangle
from matplotlib.colors import Normalize
import pickle
import matplotlib.pyplot as plt
import argparse
import json
import copy
from IPython import embed
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
import json

center_x, center_y = (0, 0)
interested_agent_color = '#ff0000'
other_agent_color = "#48A6DB"
interested_agent_traj_width = 30
other_agent_traj_width = 10
maker_star_size = 3200
maker_circle_size = 4000

# their ground truth order
other_agent_trajs_zorder = 20
other_agent_zorder = 25
# interested agent's gt trajs and pred trajs
interested_agent_trajs_zorder = 30
interested_agent_zorder = 35


pred_trajs_color_list = ['red', '#ff4d4d', '#ff6666', '#ff8080', '#ff9999','#ffb3b3']
context_map_range_dict = {
    "TYPE_VEHICLE": 60,
    "TYPE_CYCLIST": 40,
    "TYPE_PEDESTRIAN": 30
}

def GetColorViaAgentType(agentType):
    if(agentType == "TYPE_VEHICLE"):
        return "blue"
    elif(agentType == "TYPE_PEDESTRAIN"):
        return "purple"
    elif(agentType == "TYPE_CYCLIST"):
        return "orange"
    else:
        return "blue"
    
def StoreAgentsMotionInformation(output_path, scenario_id, ori_obj_types, ori_obj_ids, ori_obj_trajs_full):
    agentsInfo = {"scenario_id": scenario_id}
    agentsData = []

    st, ed = (0, 11)
    for i in range(ori_obj_trajs_full.shape[0]):
        # [cx, cy, cz, dx, dy, dz, heading, vel_x, vel_y, valid]
        obj_traj = ori_obj_trajs_full[i][st:ed]
        obj_valid = np.bool_(ori_obj_trajs_full[i][st:ed, -1])
        obj_traj_new = obj_traj
        obj_type = ori_obj_types[i]
        obj_id = ori_obj_ids[i]
        position_x = obj_traj_new[:, 0]
        position_y = obj_traj_new[:, 1]
        bbox_yaw = obj_traj_new[:, 6]
        vel_x = obj_traj_new[:, 7]
        vel_y = obj_traj_new[:, 8]

        tempAgent = {}
        tempAgent['Agent_ID'] = obj_id
        tempAgent['Agent_Type'] = obj_type
        tempAgent['Agent_Position_X'] = position_x.tolist()
        tempAgent['Agent_Position_Y'] = position_y.tolist()
        tempAgent['Agent_Velocity_X'] = vel_x.tolist()
        tempAgent['Agent_Velocity_Y'] = vel_y.tolist()
        tempAgent['Agent_Heading_Angle'] = bbox_yaw.tolist()
        tempAgent['Agent_Data_Is_Vaild'] = obj_valid.tolist()

        agentsData.append(tempAgent)
    
    agentsInfo['agentsData'] = agentsData
    # Writing to a JSON file
    with open(output_path + '/' + scenario_id +'.json', 'w') as json_file:
        json.dump(agentsInfo, json_file, indent=4)
        
def generate_batch_polylines_from_map(polylines, point_sampled_interval=1, vector_break_dist_thresh=1.0, num_points_each_polyline=20):
    """
    Args:
        polylines (num_points, 7): [x, y, z, dir_x, dir_y, dir_z, global_type]

    Returns:
        ret_polylines: (num_polylines, num_points_each_polyline, 7)
        ret_polylines_mask: (num_polylines, num_points_each_polyline)
    """
    point_dim = polylines.shape[-1]

    sampled_points = polylines[::point_sampled_interval]
    sampled_points_shift = np.roll(sampled_points, shift=1, axis=0)
    buffer_points = np.concatenate((sampled_points[:, 0:2], sampled_points_shift[:, 0:2]), axis=-1)  # [ed_x, ed_y, st_x, st_y]
    buffer_points[0, 2:4] = buffer_points[0, 0:2]

    break_idxs = (np.linalg.norm(buffer_points[:, 0:2] - buffer_points[:, 2:4], axis=-1) > vector_break_dist_thresh).nonzero()[0]
    polyline_list = np.array_split(sampled_points, break_idxs, axis=0)
    ret_polylines = []
    ret_polylines_mask = []

    def append_single_polyline(new_polyline):
        cur_polyline = np.zeros((num_points_each_polyline, point_dim), dtype=np.float32)
        cur_valid_mask = np.zeros((num_points_each_polyline), dtype=np.int32)
        cur_polyline[: len(new_polyline)] = new_polyline
        cur_valid_mask[: len(new_polyline)] = 1
        ret_polylines.append(cur_polyline)
        ret_polylines_mask.append(cur_valid_mask)

    for k in range(len(polyline_list)):
        if polyline_list[k].__len__() <= 0:
            continue
        for idx in range(0, len(polyline_list[k]), num_points_each_polyline):
            append_single_polyline(polyline_list[k][idx : idx + num_points_each_polyline])

    ret_polylines = np.stack(ret_polylines, axis=0)
    ret_polylines_mask = np.stack(ret_polylines_mask, axis=0)

    return ret_polylines, ret_polylines_mask

def plt_road_edges(road_edges, polylines, ax):
    for edge_idx in road_edges:
        edge_sta = edge_idx['polyline_index'][0]
        edge_end = edge_idx['polyline_index'][1]
        if edge_end - edge_sta == 1:
            continue
        edge_polylines = polylines[edge_sta:edge_end, :2]
        ax.plot(edge_polylines[:, 0], edge_polylines[:, 1], color='black', alpha=1, zorder=2, linewidth=8)

def area_of_irregular_quadrilateral(points):
    """Calculate the area of an irregular quadrilateral given four points."""
    if len(points) != 4:
        return 0

    # Function to calculate the cross product of two vectors
    def cross_product(p1, p2, p3):
        return (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0])

    # Split the quadrilateral into two triangles and calculate the area of each
    area1 = abs(cross_product(points[0], points[1], points[2])) / 2.0
    area2 = abs(cross_product(points[2], points[3], points[0])) / 2.0

    # Sum the areas of the two triangles
    return area1 + area2

def plt_crosswalks(crosswalks, polylines, ax):
    for crosswalk in crosswalks:
        cross_sta = crosswalk['polyline_index'][0]
        cross_end = crosswalk['polyline_index'][1]
        if cross_end - cross_sta == 1:
            continue
        crosswalk_area = area_of_irregular_quadrilateral(polylines[cross_sta:cross_end, :2])
        if crosswalk_area > 500:
            continue
        cross_polylines = np.concatenate((polylines[cross_sta:cross_end, :2], polylines[cross_sta:cross_sta+1, :2]))
        ax.fill(cross_polylines[:, 0], cross_polylines[:, 1], color='#E0D9D8', hatch='//', edgecolor='#494949', alpha=0.5, zorder=3) # #ABABAB

def plt_lanes(lanes, polylines, ax):
    lane_head_width=1
    lane_head_height=1.4
    closest_lane_length = 0
    selected_lane_ids = []
    closest_exit_lanes = []
    for lane in lanes:
        lane_sta = lane['polyline_index'][0]
        lane_end = lane['polyline_index'][1]
        if lane_end - lane_sta == 1:
            continue
        lane_type = lane['type']
        lane_polylines = polylines[lane_sta:lane_end]

        if lane_type != 'TYPE_UNDEFINED':
            ax.plot(lane_polylines[:, 0], lane_polylines[:, 1], color='black', alpha=0.8, zorder=2, linewidth=4)
        else:
            ax.plot(lane_polylines[:, 0], lane_polylines[:, 1], color='gray', alpha=0.6, zorder=2, linewidth=3)
        
        poly_incre_x = lane_polylines[-1, 0] - lane_polylines[-2, 0]
        poly_incre_y = lane_polylines[-1, 1] - lane_polylines[-2, 1]
        ax.arrow(lane_polylines[-1, 0] - poly_incre_x, lane_polylines[-1, 1] - poly_incre_y, poly_incre_x, poly_incre_y, head_width=lane_head_width, head_length=lane_head_height, edgecolor='black', facecolor='white', alpha=1, zorder=6)
    return closest_lane_length, selected_lane_ids, closest_exit_lanes

def plt_road_lines(road_lines, polylines, ax):
    for line in road_lines:
        line_sta = line['polyline_index'][0]
        line_end = line['polyline_index'][1]
        if line_end - line_sta == 1:
            continue
        line_type = line['type']
        line_polylines = polylines[line_sta:line_end]
        if line_type == 'TYPE_SOLID_SINGLE_YELLOW' or line_type == 'TYPE_SOLID_DOUBLE_YELLOW':
            ax.plot(line_polylines[:, 0], line_polylines[:, 1], color='black', alpha=1, zorder=2, linewidth=8)
        # elif line_type == 'TYPE_BROKEN_SINGLE_YELLOW' or line_type == 'TYPE_BROKEN_DOUBLE_YELLOW':
        #     ax.plot(line_polylines[:, 0], line_polylines[:, 1], color='yellow', alpha=1, zorder=2, linewidth=4, linestyle='dashed')
        # elif line_type == 'TYPE_SOLID_SINGLE_YELLOW' or line_type == 'TYPE_SOLID_DOUBLE_YELLOW':
        #     ax.plot(line_polylines[:, 0], line_polylines[:, 1], color='yellow', alpha=1, zorder=2, linewidth=4)
        # else:  # TYPE_UNKNOWN, TYPE_BROKEN_SINGLE_WHITE, TYPE_PASSING_DOUBLE_YELLOW
        #     ax.plot(line_polylines[:, 0], line_polylines[:, 1], color='white', alpha=1, zorder=2, linewidth=4, linestyle='dashed')

def DrawMap(polylines, ori_map_infos, ax):
    road_edges = ori_map_infos['road_edge']
    crosswalks = ori_map_infos['crosswalk']
    lanes = ori_map_infos['lane']
    road_lines = ori_map_infos['road_line']
    stop_signs = ori_map_infos['stop_sign']
    
    plt_road_edges(road_edges, polylines, ax)
    plt_crosswalks(crosswalks, polylines, ax)
    plt_lanes(lanes, polylines, ax)
    plt_road_lines(road_lines, polylines, ax)

def GetBoundary(timestamp):
    if timestamp == 3:
        st, ed = (0, 41)
    elif timestamp == 5:
        st, ed = (0, 61)
    else:
        st, ed = (0, 91)
    return st, ed

def ComputeThreshold(vel_x, vel_y, t):
    vel = math.hypot(vel_x, vel_y)
    if vel <= 1.4:
        a = 0.5
    elif vel >= 11:
        a = 1
    else:
        a = 0.5 + 0.5 * (vel - 1.4) / (11 - 1.4)

    if t == 3:
        threshold_lat, threshold_lon = (1, 2)
    elif t == 5:
        threshold_lat, threshold_lon = (1.8, 3.6)
    else:
        threshold_lat, threshold_lon = (3, 6)

    return threshold_lat * a, threshold_lon * a

def DrawCar(scenario_id, ori_obj_trajs_full, ori_sdc_track_index, track_index_to_predict, ori_obj_ids, ori_obj_types, ax):
    st, ed = (0, 11)
    for i in range(ori_obj_trajs_full.shape[0]):
        # [cx, cy, cz, dx, dy, dz, heading, vel_x, vel_y, valid]
        obj_traj = ori_obj_trajs_full[i][st:ed]
        obj_valid = np.bool_(ori_obj_trajs_full[i][st:ed, -1])
        obj_traj_new = obj_traj[obj_valid]
        if(not obj_valid[-1]):
            continue

        position_x = obj_traj_new[:, 0]
        position_y = obj_traj_new[:, 1]
        width = obj_traj_new[:, 3]
        length = obj_traj_new[:, 4]
        bbox_yaw = obj_traj_new[:, 6]
        vel_x = obj_traj_new[:, 7]
        vel_y = obj_traj_new[:, 8]

        color_bbox = "black"
        color = other_agent_color
        # color = GetColorViaAgentType(ori_obj_types[i])
        # if i == ori_sdc_track_index:
        #     color = "green"

        zorder = other_agent_zorder
        if i == track_index_to_predict:
            color = interested_agent_color
            zorder = interested_agent_zorder
            #TODO add numbers to each agent
        # ax.text(position_x[-1], position_y[-1], i, color='black', zorder=20, fontweight="bold")

        w = width[-1]
        h = length[-1]
        theta = bbox_yaw[-1]
        x1, y1 = (position_x + w / 2 * np.cos(theta) + h / 2 * np.sin(theta), position_y + w / 2 * np.sin(theta) - h / 2 * np.cos(theta))
        x2, y2 = (position_x + w / 2 * np.cos(theta) - h / 2 * np.sin(theta), position_y + w / 2 * np.sin(theta) + h / 2 * np.cos(theta))
        x3, y3 = (position_x - w / 2 * np.cos(theta) - h / 2 * np.sin(theta), position_y - w / 2 * np.sin(theta) + h / 2 * np.cos(theta))
        x4, y4 = (position_x - w / 2 * np.cos(theta) + h / 2 * np.sin(theta), position_y - w / 2 * np.sin(theta) - h / 2 * np.cos(theta))

        # x5, y5 = (position_x + (w / 2 - w / 4) * np.cos(theta) + (h / 2) * np.sin(theta), position_y + (w / 2 - w / 4) * np.sin(theta) - (h / 2) * np.cos(theta))
        # x6, y6 = (position_x + (w / 2 - w / 4) * np.cos(theta) - (h / 2) * np.sin(theta), position_y + (w / 2 - w / 4) * np.sin(theta) + (h / 2) * np.cos(theta))

        ax.plot([x1[-1], x2[-1], x3[-1], x4[-1], x1[-1]], [y1[-1], y2[-1], y3[-1], y4[-1], y1[-1]], color=color_bbox, zorder=zorder, alpha=0.7, linewidth=3)
        ax.fill([x1[-1], x2[-1], x3[-1], x4[-1]], [y1[-1], y2[-1], y3[-1], y4[-1]], color=color, zorder=zorder, alpha=0.7)

        # ax.fill([(x1[-1] + x2[-1]) / 2, x6[-1], x5[-1],], [(y1[-1] + y2[-1]) / 2, y6[-1], y5[-1],], color="black", zorder=10)

def DrawThresholdBox(obj_traj, obj_valid, t, color, ax):
    # idx = 11 + t * 10 - 1
    idx = -1

    if obj_valid[idx]:
        vel_x = obj_traj[idx][7]
        vel_y = obj_traj[idx][8]

        endpoint_x = obj_traj[idx][0]
        endpoint_y = obj_traj[idx][1]

        theta = obj_traj[idx][6]

        threshold_lon, threshold_lat = ComputeThreshold(vel_x, vel_y, t)

        bbox_x_1, bbox_y_1 = (endpoint_x + threshold_lat / 2 * np.cos(theta) + threshold_lon / 2 * np.sin(theta), endpoint_y + threshold_lat / 2 * np.sin(theta) - threshold_lon / 2 * np.cos(theta))
        bbox_x_2, bbox_y_2 = (endpoint_x + threshold_lat / 2 * np.cos(theta) - threshold_lon / 2 * np.sin(theta), endpoint_y + threshold_lat / 2 * np.sin(theta) + threshold_lon / 2 * np.cos(theta))
        bbox_x_3, bbox_y_3 = (endpoint_x - threshold_lat / 2 * np.cos(theta) - threshold_lon / 2 * np.sin(theta), endpoint_y - threshold_lat / 2 * np.sin(theta) + threshold_lon / 2 * np.cos(theta))
        bbox_x_4, bbox_y_4 = (endpoint_x - threshold_lat / 2 * np.cos(theta) + threshold_lon / 2 * np.sin(theta), endpoint_y - threshold_lat / 2 * np.sin(theta) - threshold_lon / 2 * np.cos(theta))

        ax.plot([bbox_x_1, bbox_x_2, bbox_x_3, bbox_x_4, bbox_x_1], [bbox_y_1, bbox_y_2, bbox_y_3, bbox_y_4, bbox_y_1], color=color, zorder=100, alpha=1)

def DrawGroundTruth(ori_obj_trajs_full, timeStamp, ori_sdc_track_index, track_index_to_predict, ori_obj_ids, ax, is_ground_truth):
    st, ed = GetBoundary(timeStamp)
    for i in range(ori_obj_trajs_full.shape[0]):
        # [cx, cy, cz, dx, dy, dz, heading, vel_x, vel_y, valid]
        obj_traj = ori_obj_trajs_full[i][st:ed]
        obj_valid = np.bool_(ori_obj_trajs_full[i][st:ed, -1])
        # keep align with DrawCar
        if not np.bool_(ori_obj_trajs_full[i][0:11, -1])[-1]:
            continue
        obj_traj_new = obj_traj[obj_valid]

        position_x = obj_traj_new[:, 0]
        position_y = obj_traj_new[:, 1]
        width = obj_traj_new[:, 3]
        length = obj_traj_new[:, 4]
        bbox_yaw = obj_traj_new[:, 6]
        vel_x = obj_traj_new[:, 7]
        vel_y = obj_traj_new[:, 8]

        color = other_agent_color

        # if i == ori_sdc_track_index:
        #     color = "green"

        if i == track_index_to_predict:
            color = interested_agent_color
            if is_ground_truth:
                ax.plot(position_x, position_y, color=color, linewidth=interested_agent_traj_width, zorder=interested_agent_trajs_zorder)
            ax.scatter(position_x[-1], position_y[-1], color=color, marker="*", s=maker_star_size, zorder=150, alpha=0.5, linewidth=5, edgecolors='black')
        #     color = list(colorTable.values())[ori_obj_ids[i] % len(colorTable)]
        #     DrawThresholdBox(obj_traj, obj_valid, timeStamp, color)
        else:
            ax.plot(position_x, position_y, color=color, linewidth=other_agent_traj_width, zorder=other_agent_trajs_zorder)
        # ax.scatter(position_x[-1], position_y[-1], color=color, marker="*", s=200, zorder=11)

        # if(timeStamp == 8 and obj_valid[-1] == False and i in ori_tracks_to_predict_track_index):
        #     ax.scatter(position_x[-1], position_y[-1], color=color, marker="*", s=200, zorder=11)

def DrawPredictTrajectories(predicted_traj, timeStamp, ax):
    st, ed = GetBoundary(timeStamp)
    pred_trajs = predicted_traj["pred_trajs"]
    pred_scores = predicted_traj["pred_scores"]
    object_id = predicted_traj["object_id"]
    object_type = predicted_traj["object_type"]
    gt_trajs = predicted_traj["gt_trajs"]
    track_index_to_predict = predicted_traj["track_index_to_predict"]

    traj_valid = np.bool_(gt_trajs[:, -1])
    color_list = pred_trajs_color_list
    # for pred_traj in pred_trajs:
    for i in range(len(color_list) - 1, -1, -1):
        color = color_list[i]
        pred_score = pred_scores[i]
        pred_traj = pred_trajs[i]

        pred_traj_x = gt_trajs[:, 0][:11]
        pred_traj_x = np.concatenate((pred_traj_x, pred_traj[:, 0]), axis=0)
        # pred_traj_x.extend(list(pred_traj[:, 0]))

        pred_traj_y = gt_trajs[:, 1][:11]
        pred_traj_y = np.concatenate((pred_traj_y, pred_traj[:, 1]), axis=0)
        # pred_traj_y.extend(list(pred_traj[:, 1]))

        pred_traj_x = list(pred_traj_x[st:ed][traj_valid[st:ed]])
        pred_traj_y = list(pred_traj_y[st:ed][traj_valid[st:ed]])

        ax.plot(pred_traj_x, pred_traj_y, color=color, linewidth=interested_agent_traj_width, alpha=0.8, zorder=interested_agent_trajs_zorder)

        ax.scatter(pred_traj_x[-1], pred_traj_y[-1], color=color, marker=".", s=maker_circle_size, zorder=interested_agent_trajs_zorder, alpha=0.9)

def DrawPictures(ori_obj_trajs_full, predicted_traj, timeStamp, ori_sdc_track_index, track_index_to_predict, ori_obj_ids, ax):
    is_ground_truth = predicted_traj is None
    # for ground truth: draw as default
    # for model results: only draw the final point as a star (the GT trajs will overlap with pred trajs, so here we do not draw it)
    DrawGroundTruth(ori_obj_trajs_full, timeStamp, ori_sdc_track_index, track_index_to_predict, ori_obj_ids, ax, is_ground_truth)
    if not is_ground_truth:
        DrawPredictTrajectories(predicted_traj, timeStamp, ax)

def rotate_whole_scene_by_track_index(polylines, ori_obj_trajs_full, mtr_predicted_trajs, llm_augmented_mtr_predicted_trajs, track_index_to_predict):
    # convert golbal coordinate to local coordinate
    local_obj_point = ori_obj_trajs_full[track_index_to_predict, 10].copy()
    assert local_obj_point[-1] == 1, "the interested agent is invalid at frame 11"
    local_x = local_obj_point[0]
    local_y = local_obj_point[1]
    local_angle = local_obj_point[6]
    rotate_angle = math.pi / 2 - local_angle
    cos_theta = math.cos(rotate_angle)
    sin_theta = math.sin(rotate_angle)
    
    polylines -= local_obj_point[:2]
    polylines = np.stack((polylines[:,0] * cos_theta - polylines[:,1] * sin_theta, polylines[:,0] * sin_theta + polylines[:,1] *cos_theta), axis=1)
    
    ori_obj_trajs_full[:, :, :2] -= local_obj_point[:2]
    ori_obj_trajs_full[:, :, :2] = np.stack((ori_obj_trajs_full[:, :, 0] *cos_theta - ori_obj_trajs_full[:, :, 1] * sin_theta, ori_obj_trajs_full[:, :, 0] * sin_theta + ori_obj_trajs_full[:, :, 1] *cos_theta), axis=2)
    ori_obj_trajs_full[:, :, 6] += rotate_angle
    
    # rotate all predicted trajs for mtr
    mtr_predicted_trajs["pred_trajs"][:, :, :2] -= local_obj_point[:2]
    mtr_predicted_trajs["gt_trajs"][:, :2] -= local_obj_point[:2]
    
    
    mtr_predicted_trajs["pred_trajs"] = np.stack((mtr_predicted_trajs["pred_trajs"][:, :, 0] *cos_theta - mtr_predicted_trajs["pred_trajs"][:, :, 1] * sin_theta, mtr_predicted_trajs["pred_trajs"][:, :, 0] * sin_theta + mtr_predicted_trajs["pred_trajs"][:, :, 1] *cos_theta), axis=2)
    mtr_predicted_trajs["gt_trajs"][:, :2] = np.stack((mtr_predicted_trajs["gt_trajs"][:, 0] *cos_theta - mtr_predicted_trajs["gt_trajs"][:, 1] * sin_theta, mtr_predicted_trajs["gt_trajs"][:, 0] * sin_theta + mtr_predicted_trajs["gt_trajs"][:, 1] *cos_theta), axis=1)
    mtr_predicted_trajs["gt_trajs"][:, 6] -= rotate_angle
        
    # rotate all predicted trajs for llm_augmented_mtr
    llm_augmented_mtr_predicted_trajs["pred_trajs"][:, :, :2] -= local_obj_point[:2]
    llm_augmented_mtr_predicted_trajs["gt_trajs"][:, :2] -= local_obj_point[:2]
    
    
    llm_augmented_mtr_predicted_trajs["pred_trajs"] = np.stack((llm_augmented_mtr_predicted_trajs["pred_trajs"][:, :, 0] *cos_theta - llm_augmented_mtr_predicted_trajs["pred_trajs"][:, :, 1] * sin_theta, llm_augmented_mtr_predicted_trajs["pred_trajs"][:, :, 0] * sin_theta + llm_augmented_mtr_predicted_trajs["pred_trajs"][:, :, 1] *cos_theta), axis=2)
    llm_augmented_mtr_predicted_trajs["gt_trajs"][:, :2] = np.stack((llm_augmented_mtr_predicted_trajs["gt_trajs"][:, 0] *cos_theta - llm_augmented_mtr_predicted_trajs["gt_trajs"][:, 1] * sin_theta, llm_augmented_mtr_predicted_trajs["gt_trajs"][:, 0] * sin_theta + llm_augmented_mtr_predicted_trajs["gt_trajs"][:, 1] *cos_theta), axis=1)
    llm_augmented_mtr_predicted_trajs["gt_trajs"][:, 6] -= rotate_angle
    
    return polylines, ori_obj_trajs_full, mtr_predicted_trajs, llm_augmented_mtr_predicted_trajs

def vis_frame(all_mtr_predicted_trajs, all_llm_augmented_mtr_predicted_trajs, ori_data_path, output_path, scenario_context):
    scenario_id = all_mtr_predicted_trajs[0]["scenario_id"]
    ori_data_path = os.path.join(ori_data_path, "sample_" + scenario_id + ".pkl")
    with open(ori_data_path, "rb") as f:
        ori_data = pickle.load(f)

    ori_track_infos = ori_data["track_infos"]
    ori_obj_types = ori_track_infos["object_type"]
    ori_obj_ids = ori_track_infos["object_id"]
    ori_obj_trajs_full = ori_track_infos["trajs"]

    ori_dynamic_map_infos = ori_data["dynamic_map_infos"]
    ori_map_infos = ori_data["map_infos"]
    ori_scenario_id = ori_data["scenario_id"]
    ori_timestamps_seconds = ori_data["timestamps_seconds"]
    ori_current_time_index = ori_data["current_time_index"]
    ori_objects_of_interest = ori_data["objects_of_interest"]
    ori_tracks_to_predict = ori_data["tracks_to_predict"]

    ori_sdc_track_index = ori_data["sdc_track_index"]
    ori_tracks_to_predict_track_index = ori_tracks_to_predict["track_index"]

    # get all the map info
    # set polylines(n, 7)   7: x, y, z, dir_x, dir_y, dir_z, global_type
    polylines = ori_map_infos["all_polylines"][:, :2]

    for plot_index, track_index_to_predict in enumerate(ori_tracks_to_predict_track_index):
        assert scenario_context['track_indexes'][plot_index] == track_index_to_predict, "context_data's track_index is inconsistent with ori_track_to_predict_track_index"
        
        interested_agent_context = {
            "track_intentions": scenario_context["track_intentions"][plot_index],
            "track_affordances": scenario_context["track_affordances"][plot_index],
            "track_scenarios": scenario_context["track_scenarios"][plot_index]
        }
        
        with open(f"{output_path}/{scenario_id}_{track_index_to_predict}_context.json", 'w') as file_pointer:
            json.dump(interested_agent_context, file_pointer, indent=4)
        
        interested_agent_type = ori_obj_types[track_index_to_predict]
        context_map_range = context_map_range_dict[interested_agent_type]
        
        mtr_predicted_trajs = all_mtr_predicted_trajs[plot_index]
        llm_augmented_mtr_predicted_trajs = all_llm_augmented_mtr_predicted_trajs[plot_index]
        
        polylines_rotated, ori_obj_trajs_full_rotated, mtr_predicted_trajs_rotated, llm_augmented_mtr_predicted_trajs_rotated = rotate_whole_scene_by_track_index(copy.deepcopy(polylines), copy.deepcopy(ori_obj_trajs_full), copy.deepcopy(mtr_predicted_trajs), copy.deepcopy(llm_augmented_mtr_predicted_trajs), track_index_to_predict)

        fig = plt.figure(figsize=(120, 32), dpi=200, facecolor="white")

        plt.rc('font', size=60)          # controls default text sizes
        plt.rc('axes', titlesize=70)     # fontsize of the axes title
        plt.rc('axes', labelsize=60)     # fontsize of the x and y labels
        plt.rc('xtick', labelsize=50)    # fontsize of the tick labels
        plt.rc('ytick', labelsize=50)    # fontsize of the tick labels
        plt.rc('legend', fontsize=60)    # legend fontsize
        plt.rc('figure', titlesize=80)   # fontsize of the figure title
        
        # four subfigure, from left to right: Ground Truth, MTR, LLM-Augmented-MTR, colorbar
        gs = gridspec.GridSpec(1, 4, width_ratios=[1, 1, 1, 0.05])
        
        ax0 = fig.add_subplot(gs[0, 0])
        ax0.set_title('Ground Truth')
        DrawMap(polylines_rotated, ori_map_infos, ax0)
        DrawCar(scenario_id, ori_obj_trajs_full_rotated, ori_sdc_track_index, track_index_to_predict, ori_obj_ids, ori_obj_types, ax0)
        DrawPictures(ori_obj_trajs_full_rotated, None, 8, ori_sdc_track_index, track_index_to_predict, ori_obj_ids, ax0)
        
        # ax1.axis('off')
        ax0.set_xlim([-context_map_range, context_map_range])
        ax0.set_ylim([-context_map_range, context_map_range])
        # Set the linewidth of the spines (borders) for the second subplot in one line
        for spine in ax0.spines.values():
            spine.set_linewidth(3)
        
        ax1 = fig.add_subplot(gs[0, 1])
        ax1.set_title('MTR')
        DrawMap(polylines_rotated, ori_map_infos, ax1)
        DrawCar(scenario_id, ori_obj_trajs_full_rotated, ori_sdc_track_index, track_index_to_predict, ori_obj_ids, ori_obj_types, ax1)
        DrawPictures(ori_obj_trajs_full_rotated, mtr_predicted_trajs_rotated, 8, ori_sdc_track_index, track_index_to_predict, ori_obj_ids, ax1)
        
        # ax1.axis('off')
        ax1.set_xlim([-context_map_range, context_map_range])
        ax1.set_ylim([-context_map_range, context_map_range])
        # Set the linewidth of the spines (borders) for the second subplot in one line
        for spine in ax1.spines.values():
            spine.set_linewidth(3)
        
        
        ax2 = fig.add_subplot(gs[0, 2])
        ax2.set_title('LLM-Augmented-MTR')
        DrawMap(polylines_rotated, ori_map_infos, ax2)
        DrawCar(scenario_id, ori_obj_trajs_full_rotated, ori_sdc_track_index, track_index_to_predict, ori_obj_ids, ori_obj_types, ax2)
        DrawPictures(ori_obj_trajs_full_rotated, llm_augmented_mtr_predicted_trajs_rotated, 8, ori_sdc_track_index, track_index_to_predict, ori_obj_ids, ax2)
        
        # if(box_range != -1):
        # ax2.axis('off')
        ax2.set_xlim([-context_map_range, context_map_range])
        ax2.set_ylim([-context_map_range, context_map_range])
        # Set the linewidth of the spines (borders) for the second subplot in one line
        for spine in ax2.spines.values():
            spine.set_linewidth(3)
        
        # Create a custom colormap from white to red
        cmap = mcolors.LinearSegmentedColormap.from_list("", ["white", "red"])

        # Create a dummy scalar mappable for the color bar
        norm = mcolors.Normalize(vmin=0, vmax=1)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])

        # Add the color bar to the figure
        cbar_ax = fig.add_subplot(gs[0, 3])
        cbar = fig.colorbar(sm, cax=cbar_ax, orientation='vertical')
        cbar.set_label('Possibility')
        
        # Remove the black border of the colorbar
        cbar.outline.set_edgecolor('none')
        
        # Adjust layout to prevent overlap (no need for tight_layout)
        plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05, wspace=0.3)
        # fig.subplots_adjust()
        # fig.savefig(f"{output_path}/{scenario_id}_{track_index_to_predict}_multi_agent.png", bbox_inches="tight")
        fig.savefig(f"{output_path}/{scenario_id}_{track_index_to_predict}_single_agent_compare.png")
        plt.close()


    print(f"scenario id is {scenario_id}")

def main():
    # input your validation set directory
    ori_data_path = "./data/waymo/mtr_processed/processed_scenarios_validation"
    # input the result.pkl of your MTR model on validation set
    mtr_eval_result_path = "./output/waymo/mtr+100_percent_data/valid_mtr+100_percent/eval/epoch_29/default/result.pkl"
    # input the result.pkl of your LLM-Augmented-MTR model on validation set
    llm_augmented_mtr_eval_result_path = "./output/waymo/mtr+100_percent_data_llm_augmented/valid_mtr+100_percent_llm_augmented_LCI_W8_L4/eval/epoch_26/default/result.pkl"
    # set your output directory
    output_path = "./MTR_Vistualization/VisStaticPic"
    # input the generated context data of validation set, this script will save it for each interested agent
    context_file_path = './LLM_integrate/context_data/valid/context_data_encoder_100.pkl'
    # open context file
    with open(context_file_path, 'rb') as context_file_pointer:
        context_data = pickle.load(context_file_pointer)

    os.makedirs(output_path, exist_ok=True)

    # open mter's result
    with open(mtr_eval_result_path, "rb") as f:
        mtr_data = pickle.load(f)
    mtr_result_scenario_id_list = []
    for item in mtr_data:
        for sample in item:
            scenario_id = sample["scenario_id"]
            mtr_result_scenario_id_list.append(int(scenario_id, 16))
            break
    # open llm-augmented-mtr's result
    with open(llm_augmented_mtr_eval_result_path, "rb") as f:
        llm_augmented_mtr_data = pickle.load(f)
    llm_augmented_mtr_scenario_id_list = []
    for item in llm_augmented_mtr_data:
        for sample in item:
            scenario_id = sample["scenario_id"]
            llm_augmented_mtr_scenario_id_list.append(int(scenario_id, 16))
            break

    for s_id in tqdm(vis_dict.keys()):
        # find the scenario in mtr's result file
        mtr_index_list = np.where(np.array(mtr_result_scenario_id_list) == int(s_id, 16))[0]
        all_mtr_predicted_trajs = mtr_data[mtr_index_list[0]]
        # find the scenario in llm-augmented-mtr's result file
        llm_augmented_mtr_index_list = np.where(np.array(llm_augmented_mtr_scenario_id_list) == int(s_id, 16))[0]
        all_llm_augmented_mtr_predicted_trajs = llm_augmented_mtr_data[llm_augmented_mtr_index_list[0]]        
        
        vis_frame(all_mtr_predicted_trajs, all_llm_augmented_mtr_predicted_trajs, ori_data_path, output_path, context_data[s_id])

if __name__ == "__main__":
    main()
