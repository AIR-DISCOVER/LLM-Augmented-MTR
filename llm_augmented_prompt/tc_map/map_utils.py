import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import matplotlib.cm as cm
import math
from copy import deepcopy


object_type_map = {
    'TYPE_VEHICLE': 'V',
    'TYPE_CYCLIST': 'C',
    'TYPE_PEDESTRIAN': 'P',
}

kMaxSpeedForStationary = 2.0
kMaxDisplacementForStationary = 5.0
kMaxLateralDisplacementForStraight = 5
kMinLongitudinalDisplacementForUTurn = 0  
kMaxAbsHeadingDiffForStraight = math.pi / 6

kMaxDistanceForSameLine = 2.5
RateCoOPPDirection = 0.5
RatePositionLeftRight = 0.6
RateSubDeltaXYFordirection = 0.5
SubLen = 10


def get_dis_point(a, b):
    return np.sqrt(a ** 2 + b ** 2)


def calculate_angle_with_y_axis(point1, point2):
    x1, y1 = point1
    x2, y2 = point2

    distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    sin_theta = abs(x2 - x1) / distance
    angle_radians = math.asin(sin_theta)
    angle_degrees = math.degrees(angle_radians)

    if y2 < y1:
        angle_degrees = 180 - angle_degrees

    return angle_degrees


def area_of_irregular_quadrilateral(points):
    """Calculate the area of an irregular quadrilateral given four points."""
    if len(points) != 4:
        return 0

    def cross_product(p1, p2, p3):
        return (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0])

    area1 = abs(cross_product(points[0], points[1], points[2])) / 2.0
    area2 = abs(cross_product(points[2], points[3], points[0])) / 2.0

    return area1 + area2


def GetStandardAngle(angle):
    while angle < 0 or angle >= 2 * math.pi:
        if angle < 0:
            angle += 2 * math.pi
        else:
            angle -= 2 * math.pi
    return angle


def GetStandardHeadingDiff(angle):
    if abs(angle) > math.pi:
        if angle < 0:
            return 2 * math.pi - abs(angle)
        else:
            return -(2 * math.pi - abs(angle))
    else:
        return angle


def classify_track(trajectory, n=None):
    start_state = 0
    end_state = -1

    ## The difference in the horizontal coordinates between the starting
    ## point and the ending point of a vehicle's trajectory.
    x_delta = trajectory[end_state, 0] - trajectory[start_state, 0]
    y_delta = trajectory[end_state, 1] - trajectory[start_state, 1]
    
    final_displacement = np.sqrt((trajectory[1:, 0] - trajectory[:-1, 0]) ** 2 + \
        (trajectory[1:, 1] - trajectory[:-1, 1]) ** 2).sum()
    
    heading_diff = GetStandardAngle(trajectory[end_state, 2]) - GetStandardAngle(trajectory[start_state, 2])  # heading
    heading_diff = GetStandardHeadingDiff(heading_diff)

    # Normalized deltas
    cos_ = math.cos(-trajectory[start_state, 2])
    sin_ = math.sin(-trajectory[start_state, 2])
    dx, dy = x_delta * cos_ - y_delta * sin_, x_delta * sin_ + y_delta * cos_

    # Speed calculations
    start_speed = math.hypot(trajectory[start_state, 3], trajectory[start_state, 4])  # vel_x, vel_y
    end_speed = math.hypot(trajectory[end_state, 3], trajectory[end_state, 4])  # vel_x, vel_y
    max_speed = max(start_speed, end_speed)
    
    # Trajectory type classification
    if max_speed < kMaxSpeedForStationary and final_displacement < kMaxDisplacementForStationary:
        return "STATIONARY"
    if abs(heading_diff) < kMaxAbsHeadingDiffForStraight:
        if abs(dy) < kMaxLateralDisplacementForStraight:
            return "STRAIGHT"
        return "STRAIGHT-RIGHT" if dy < 0 else "STRAIGHT-LEFT"
    if heading_diff < -kMaxAbsHeadingDiffForStraight and dy < 0:
        return "RIGHT-U-TURN" if dx < kMinLongitudinalDisplacementForUTurn else "RIGHT-TURN"
    if dx < kMinLongitudinalDisplacementForUTurn:
        return "LEFT-U-TURN"

    return "LEFT-TURN"



def rotate(x, y, angle):
    res_x = x * math.cos(angle) - y * math.sin(angle)
    res_y = x * math.sin(angle) + y * math.cos(angle)
    return res_x, res_y


class Normalizer:
    """Agent direction normalization"""
    def __init__(self, x, y, yaw):
        self.x = x
        self.y = y
        self.yaw = yaw
        self.origin = rotate(0.0 - x, 0.0 - y, yaw)

    def __call__(self, points, reverse=False):
        points = np.array(points)
        if points.shape == (2,):
            points.shape = (1, 2)
        assert len(points.shape) <= 3
        if len(points.shape) == 3:
            for each in points:
                each[:] = self.__call__(each, reverse)
        else:
            assert len(points.shape) == 2
            for point in points:
                if reverse:
                    point[0], point[1] = rotate(point[0] - self.origin[0],
                                                point[1] - self.origin[1], -self.yaw)
                else:
                    point[0], point[1] = rotate(point[0] - self.x,
                                                point[1] - self.y, self.yaw)
        return points


def _get_normalized(polygons, x, y, angle):
    cos_ = math.cos(angle)
    sin_ = math.sin(angle)
    n = polygons.shape[1]
    
    new_polygons = np.zeros((polygons.shape[0], n, 2), dtype=np.float32)
    polygons = deepcopy(polygons)[:, :, :2]
    polygons[:, :, 0] -= x
    polygons[:, :, 1] -= y
    new_polygons[:, :, 0] = polygons[:, :, 0] * cos_ - polygons[:, :, 1] * sin_
    new_polygons[:, :, 1] = polygons[:, :, 0] * sin_ + polygons[:, :, 1] * cos_
    
    return new_polygons


def get_normalized(trajectorys, normalizer, reverse=False):
    # trajectorys (n, point_num, 2)   a[None, :, :]
    if isinstance(trajectorys, np.float32):
        trajectorys = trajectorys.astype(np.float32)

    if reverse:
        return _get_normalized(trajectorys, normalizer.origin[0], normalizer.origin[1], -normalizer.yaw)
    
    return _get_normalized(trajectorys, normalizer.x, normalizer.y, normalizer.yaw)



def plt_road_edges(road_edges, polylines):
    for edge_idx in road_edges:
        edge_sta = edge_idx['polyline_index'][0]
        edge_end = edge_idx['polyline_index'][1]
        if edge_end - edge_sta == 1:
            continue
        edge_polylines = polylines[edge_sta:edge_end, :2]
        plt.plot(edge_polylines[:, 0], edge_polylines[:, 1], color='black', alpha=1, zorder=2, linewidth=8)


def plt_crosswalks(crosswalks, polylines):
    for crosswalk in crosswalks:
        cross_sta = crosswalk['polyline_index'][0]
        cross_end = crosswalk['polyline_index'][1]
        if cross_end - cross_sta == 1:
            continue
        crosswalk_area = area_of_irregular_quadrilateral(polylines[cross_sta:cross_end, :2])
        if crosswalk_area > 500:
            continue
        cross_polylines = np.concatenate((polylines[cross_sta:cross_end, :2], polylines[cross_sta:cross_sta+1, :2]))
        plt.fill(cross_polylines[:, 0], cross_polylines[:, 1], color='white', hatch='//', edgecolor='#494949', alpha=0.6, zorder=3) # #ABABAB


def select_lane_index_from_map_points(lanes, inter_past_end, ori_map_infos, polylines):
    # Calculate the points closest to the ego-agent (global_type belonging to lane: 1, 2, 3)
    first_bool = 1
    min_lanes_distance = 2   # Filter points whose distance from the ego-agent is less than 2
    less_2_lane_distances_indices = []
    map_points_distances = np.linalg.norm(inter_past_end[:2] - ori_map_infos['all_polylines'][:, :2], axis=1)
    map_points_distances_indices = np.argsort(map_points_distances)
    sorted_map_points = ori_map_infos['all_polylines'][map_points_distances_indices].tolist()
    for map_idx, sorted_map_point in enumerate(sorted_map_points):
        if sorted_map_point[6] in [1, 2, 3]:
            if first_bool:
                first_lane_index = map_idx
                first_bool = 0
            if map_points_distances[map_points_distances_indices][map_idx] < min_lanes_distance:
                less_2_lane_distances_indices.append(map_points_distances_indices[map_idx])
    agent_lane_distance = map_points_distances[map_points_distances_indices][first_lane_index]
    
    # Calculate the angle between lanes at points <2m (taking the direction of the ego-agent as the initial angle)
    lanes_angle = []
    lanes_angle_arr = np.array([])
    for lane_idx, lane_distances_indice in enumerate(less_2_lane_distances_indices):
        for lane3 in lanes:
            lane3_sta = lane3['polyline_index'][0]
            lane3_end = lane3['polyline_index'][1]
            if lane3_end - lane3_sta == 1:
                continue
            if lane3_sta <= lane_distances_indice < lane3_end:
                if(lane3_end - lane_distances_indice >= 5):
                    cur_lane_angle = calculate_angle_with_y_axis(polylines[lane_distances_indice], polylines[lane_distances_indice + 5])
                    if cur_lane_angle < 90:
                        lanes_angle.append([lane_distances_indice, cur_lane_angle])
                else:
                    cur_lane_angle = calculate_angle_with_y_axis(polylines[lane_distances_indice - 5], polylines[lane_distances_indice])
                    if cur_lane_angle < 90:
                        lanes_angle.append([lane_distances_indice, cur_lane_angle])
                
    # Sort the filtered lanes according to their angles
    if len(lanes_angle) > 0:
        lanes_angle_arr = np.array(lanes_angle)
        lanes_angle_arr_indices = np.argsort(lanes_angle_arr[:, 1])
        sorted_lanes_angle = lanes_angle_arr[lanes_angle_arr_indices]
        selected_lanes_index = sorted_lanes_angle[:, 0]
    else:
        selected_lanes_index = []
    # If the angle difference between the surrounding lanes is very small (less than 2 degrees), only select one according to the distance.
    if len(lanes_angle) >= 2:
        closest_lane_angle = sorted_lanes_angle[0][1]
        fina_lane_angle = sorted_lanes_angle[-1][1]
        if abs(fina_lane_angle - closest_lane_angle) < 2:
            selected_lanes_index = lanes_angle_arr[:1, 0].tolist()
    return selected_lanes_index, agent_lane_distance


def plt_lanes(lanes, polylines, selected_lanes_index, lane_head_width, lane_head_height):
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
        for selected_lane_index in selected_lanes_index:
            if lane_sta <= selected_lane_index < lane_end:
                closest_lane_id = lane['id']
                selected_lane_ids.append(closest_lane_id)
                closest_exit_lanes = lane['exit_lanes']
                closest_lane_points = polylines[int(selected_lane_index):lane_end]
                for i in range(lane_end - int(selected_lane_index) - 1):
                    closest_lane_length += math.sqrt((closest_lane_points[i+1, 0] - closest_lane_points[i, 0])**2 + (closest_lane_points[i+1, 1] - closest_lane_points[i, 1])**2)
                break   

        if lane_type != 'TYPE_UNDEFINED':
            plt.plot(lane_polylines[:, 0], lane_polylines[:, 1], color='black', alpha=0.8, zorder=2, linewidth=4)
        else:
            plt.plot(lane_polylines[:, 0], lane_polylines[:, 1], color='gray', alpha=0.6, zorder=2, linewidth=3)
        
        poly_incre_x = lane_polylines[-1, 0] - lane_polylines[-2, 0]
        poly_incre_y = lane_polylines[-1, 1] - lane_polylines[-2, 1]
        plt.arrow(lane_polylines[-1, 0] - poly_incre_x, lane_polylines[-1, 1] - poly_incre_y, poly_incre_x, poly_incre_y, head_width=lane_head_width, head_length=lane_head_height, edgecolor='black', facecolor='white', alpha=1, zorder=6)
    return closest_lane_length, selected_lane_ids, closest_exit_lanes


def lanes_classify(main_agent_object, agent_lane_distance, closest_lane_length, selected_lane_ids, lanes, polylines, closest_exit_lanes):
    agent_lane_type = ''
    lane_directions = []
    if main_agent_object == 'P':
        agent_lane_type = 'UNSURE'
    else:
        if main_agent_object == 'V':
            max_drive_distance = 70
        else:
            max_drive_distance = 30

        if agent_lane_distance > 6:
            agent_lane_type = 'UNSURE'

        # max_distance_V = 100m, max_distance_C = 40m
        total_lanes_length = closest_lane_length
        # Ouput final closest_exit_lanes
        if len(selected_lane_ids) == 1:
            if total_lanes_length > max_drive_distance:
                closest_exit_lanes = selected_lane_ids
            while_flag = False
            while total_lanes_length < max_drive_distance:
                for idx, lane2 in enumerate(lanes):
                    lane2 = lanes[idx]
                    lane2_sta = lane2['polyline_index'][0]
                    lane2_end = lane2['polyline_index'][1]

                    if len(closest_exit_lanes) > 1:
                        # The ego-agent is located on the straight road at an intersection/increased road with high probability, judge directly 
                        while_flag = True
                        break
                    elif len(closest_exit_lanes) == 0:
                        closest_exit_lanes = selected_lane_ids
                        while_flag = True
                        break
                    else:
                        if lane2['id'] in closest_exit_lanes:
                            # Calculate the length of the first preceding lane
                            fore_lane_points = polylines[lane2_sta:lane2_end]
                            for i in range(lane2_end - lane2_sta - 1):
                                fore_lane_length = math.sqrt((fore_lane_points[i+1, 0] - fore_lane_points[i, 0])**2 + (fore_lane_points[i+1, 1] - fore_lane_points[i, 1])**2)
                                total_lanes_length += fore_lane_length
                            # No more searching when the next predecessor is empty
                            if len(lane2['exit_lanes']) == 0 or total_lanes_length > max_drive_distance:
                                while_flag = True
                            else:
                                closest_exit_lanes = lane2['exit_lanes']                
                if while_flag:
                    break      
        elif len(selected_lane_ids) > 1:
            closest_exit_lanes = selected_lane_ids
        else:
            closest_exit_lanes = selected_lane_ids
            agent_lane_type = 'UNSURE'

        # Determine the direction of the final predecessor lanes
        if len(closest_exit_lanes) > 0:
            for lane3 in lanes:
                lane3_sta = lane3['polyline_index'][0]
                lane3_end = lane3['polyline_index'][1]
                if lane3_end - lane3_sta == 1:
                    continue
                if lane3['id'] in closest_exit_lanes:
                    agent_lane_angle = calculate_angle_with_y_axis(polylines[lane3_end - 5], polylines[lane3_end - 1])
                    if agent_lane_angle < 30:
                        lane_directions.append('FORWARD')
                    else:
                        if polylines[lane3_end - 1][0] < 0:
                            lane_directions.append('LEFT-TURN')
                        else:
                            lane_directions.append('RIGHT-TURN')
    return agent_lane_type, lane_directions


def judge_lane_type_and_get_lane_caption(agent_lane_type, lane_directions, caption_type):
    # Judge lane type
    final_lane_type = 'UNSURE'
    if agent_lane_type == 'UNSURE':
        final_lane_type = 'UNSURE'
    else:
        if 'FORWARD' in lane_directions:
            if 'LEFT-TURN' in lane_directions:
                if 'RIGHT-TURN' in lane_directions:
                    final_lane_type = 'UNSURE'
                else:
                    final_lane_type = 'FORWARD-LEFT-TURN'
            elif 'RIGHT-TURN' in lane_directions:
                final_lane_type = 'FORWARD-RIGHT-TURN'
            else:
                final_lane_type = 'FORWARD'
        else:
            if 'LEFT-TURN' in lane_directions:
                if 'RIGHT-TURN' in lane_directions:
                    final_lane_type = 'UNSURE'
                else:
                    final_lane_type = 'LEFT-TURN'
            else:
                if 'RIGHT-TURN' in lane_directions:
                    final_lane_type = 'RIGHT-TURN'
                else:
                    final_lane_type = 'UNSURE'

    # Get lane caption
    lane_caption = ""
    if final_lane_type == 'LEFT-TURN':
        lane_caption = f"The ego_red_{caption_type} is driving in LEFT-TURN lane, it may go STRAIGHT, go STRAIGHT-LEFT, make a LEFT-TURN or LEFT-U-TURN in the next 8s."
    elif final_lane_type == 'RIGHT-TURN':
        lane_caption = f"The ego_red_{caption_type} is driving in the RIGHT-TURN lane, it may go STRAIGHT, go STRAIGHT-RIGHT or make a RIGHT-TURN in the next 8s."
    elif final_lane_type == 'FORWARD':
        lane_caption = f"The ego_red_{caption_type} is driving in FORWARD lane, it may go STRAIGHT, go STRAIGHT-LEFT or go STRAIGHT-RIGHT in the next 8s. If there is a parking lot nearby, ego_red_vehicle may also make a LEFT-TURN or RIGHT-TURN in the next 8s."
    elif final_lane_type == 'FORWARD-LEFT-TURN':
        lane_caption = f"The ego_red_{caption_type} is driving in FORWARD-LEFT-TURN lane, it may go STRAIGHT, go STRAIGHT-LEFT, go STRAIGHT-RIGHT or make a LEFT-TURN in the next 8s."
    elif final_lane_type == 'FORWARD-RIGHT-TURN':
        lane_caption = f"The ego_red_{caption_type} is driving in FORWARD-RIGHT-TURN lane, it may go STRAIGHT, go STRAIGHT-RIGHT, go STRAIGHT-LEFT or make a RIGHT-TURN in the next 8s."
    else:
        lane_caption = f"The ego_red_{caption_type} is driving in an unsure lane(this lane supports left turns, right turns and straight travel within the next 8s.), or a non-road scene such as a parking lot or open space, or the ego_red_vehicle is a pedestrian, a complete analysis of seven actions from ACTION LIBRARY is required."
    return lane_caption


def plt_road_lines(road_lines, polylines):
    for line in road_lines:
        line_sta = line['polyline_index'][0]
        line_end = line['polyline_index'][1]
        if line_end - line_sta == 1:
            continue
        line_type = line['type']
        line_polylines = polylines[line_sta:line_end]
        # Only plot two types
        if line_type == 'TYPE_SOLID_SINGLE_YELLOW' or line_type == 'TYPE_SOLID_DOUBLE_YELLOW':
            plt.plot(line_polylines[:, 0], line_polylines[:, 1], color='black', alpha=1, zorder=2, linewidth=8)


def design_caption_for_agent(caption_type, neighbor_indices, neighbor_and_inter_vel, neighbor_and_inter_position, neighbor_and_inter_colors, neighbot2inter_distances):
    nearest_str = ['closest', 'second closest', 'third closest', 'fourth closest', 'fifth closest', 'sixth closest', 'seventh closest', 'eighth closest', 'ninth closest', 'tenth closest']
    agent_caption = f"""The ego_red_{caption_type}'s speed is {neighbor_and_inter_vel[-1]}m/s. """

    for n_idx in range(len(neighbor_indices[0:3])):
        x = neighbor_and_inter_position[n_idx][0]
        y = neighbor_and_inter_position[n_idx][1]
        neighbor_agent_angle = calculate_angle_with_y_axis([0, 0], neighbor_and_inter_position[n_idx])
        if neighbor_agent_angle < 15:
            relative_position = "in the upper"
        elif 15 <= neighbor_agent_angle < 75:
            if x < 0:
                relative_position = "in the upper left"
            else:
                relative_position = "in the upper right"
        elif 75 <= neighbor_agent_angle < 105:
            if x < 0:
                relative_position = "on the left"
            else:
                relative_position = "on the right"
        elif 105 <= neighbor_agent_angle < 165:
            if x < 0:
                relative_position = "in the lower left"
            else:
                relative_position = "in the lower right"
        else:
            relative_position = "in the lower"

        agent_caption += f"""The {neighbor_and_inter_colors[n_idx]}_vehicle is the {nearest_str[n_idx]} to ego_red_{caption_type}, its speed is {neighbor_and_inter_vel[n_idx]}m/s, and it is {relative_position} of the ego_red_{caption_type} about {str(round(neighbot2inter_distances[n_idx], 3))}m away. """
    
    agent_caption += "The black map icon in the upper right corner of the 'Motion Prediction Map' identify the right, lower, left, and upper directions of the 'Motion Prediction Map', the default driving direction of ego_red_" + caption_type + " is 'upper'. White striped areas represent crosswalks. Determine the style of the image before answering the question. Please answer according to the FORMAT format based on the 'Motion Prediction Map' provided and the above information."
    return agent_caption


def vis_frame(scenario_info, output_dir, neighbor_num=10, closest_num=3):
    scenario_id             = scenario_info['scenario_id']
    object_types_full       = scenario_info['track_infos']['object_type']
    ori_track_infos         = scenario_info['track_infos']
    ori_obj_trajs_full      = ori_track_infos['trajs']  # (num_agents, 91, 10) [x, y, z, l, w, h, t, vel_x, vel_y, valid]
    ori_map_infos           = scenario_info['map_infos']
    all_polylines           = ori_map_infos['all_polylines']  # (point.x, point.y, point.z, global_type, dir_x, dir_y, dir_z)
    road_edges              = ori_map_infos['road_edge']
    crosswalks              = ori_map_infos['crosswalk']
    lanes                   = ori_map_infos['lane']
    road_lines              = ori_map_infos['road_line']

    # Get all agents past 11 frames trajectories and valid values
    ori_past_trajs = ori_obj_trajs_full[:, :11]
    ori_past_valid = ori_past_trajs[:, :, -1]

    ori_future_trajs = ori_obj_trajs_full[:, 11:]
    ori_future_valid = ori_future_trajs[:, :, -1]
    tracks_to_predict_index = scenario_info['tracks_to_predict']['track_index']

    # Get the total number of agents
    num_agents = ori_obj_trajs_full.shape[0]
    
    # Draw each TC-Map based on the number of agents
    for t_idx, track_id in enumerate(tracks_to_predict_index):
        plt.figure(figsize=(32, 32), dpi=100, facecolor='#C0C0C0') 

        # Get the type of the ego-agent（vehicle->V, pedestrian->P, cyclist->C）
        main_agent_object = object_type_map[object_types_full[track_id]]
        if main_agent_object == "V":
            # Take the scene picture of a certain range (x, y) with the ego-agent as the center
            main_agent_range_x = 60
            main_agent_range_y = 60

            # Determine the size of the lane arrow based on the type of ego-agent
            lane_head_width = 1.4
            lane_head_height = 1.8

            caption_type = 'vehicle'
        elif main_agent_object == "C":
            main_agent_range_x = 40
            main_agent_range_y = 40
            lane_head_width = 1.6
            lane_head_height = 2.0
            caption_type = 'cyclist'
        else:
            main_agent_range_x = 30
            main_agent_range_y = 30
            lane_head_width = 1
            lane_head_height = 1.4
            caption_type = 'pedestrian'

        # Based on track_id, obtain the trajectory and effective value of the ego-agent in the future
        inter_future_trajs = ori_future_trajs[track_id]
        inter_future_valid = np.bool_(ori_future_valid[track_id])
        inter_future_trajs_valid_x = inter_future_trajs[inter_future_valid, 0]
        inter_future_trajs_valid_y = inter_future_trajs[inter_future_valid, 1]
        inter_future_trajs_valid_t = inter_future_trajs[inter_future_valid, 6]
        inter_future_trajs_valid_vx = inter_future_trajs[inter_future_valid, 7]
        inter_future_trajs_valid_vy = inter_future_trajs[inter_future_valid, 8] 
        inter_future_trajs_valid = np.stack([inter_future_trajs_valid_x, \
                                             inter_future_trajs_valid_y, 
                                             inter_future_trajs_valid_t,
                                             inter_future_trajs_valid_vx,
                                             inter_future_trajs_valid_vy], axis=1)
        
        inter_past_valid = ori_past_valid[track_id]  # (11, 10)
        inter_past_valid_end = inter_past_valid.nonzero()[0].max()
        agent_in_valid_end_indices = ori_past_valid[:, inter_past_valid_end]
        agent_traj_past_valid_end = ori_past_trajs[:, inter_past_valid_end, :]  # (num_agents, 10)   
        inter_past_end = agent_traj_past_valid_end[track_id]  # [x, y, z, l, w, h, t, vel_x, vel_y, valid]

        # Rotate
        trajectorys = agent_traj_past_valid_end[None, :, :2]
        ori_past_trajs_normalizer = Normalizer(inter_past_end[0], inter_past_end[1], -inter_past_end[6] + math.radians(90))
        trajectorys_new = get_normalized(deepcopy(trajectorys), ori_past_trajs_normalizer).squeeze()
        trajectorys_new_t = agent_traj_past_valid_end[:, 6] - inter_past_end[6] + math.radians(90)
        trajectorys_new = np.concatenate([trajectorys_new, trajectorys_new_t[:, np.newaxis]], axis=1)

        # Map info
        polylines = all_polylines[:, :2]  # (x, y)
        polylines = get_normalized(deepcopy(polylines[np.newaxis, :, :]), ori_past_trajs_normalizer).squeeze()

        # Get direction of ego-agent
        direction = classify_track(inter_future_trajs_valid)

        # Plot road_edges
        plt_road_edges(road_edges, polylines)  

        # Plot crosswalks
        plt_crosswalks(crosswalks, polylines)

        # Get the index of points in the map with distance <2m and angle <90, the closest distance to the lane line
        selected_lanes_index, agent_lane_distance = select_lane_index_from_map_points(lanes, inter_past_end, ori_map_infos, polylines)
     
        # Plot lanes
        closest_lane_length, selected_lane_ids, closest_exit_lanes = plt_lanes(lanes, polylines, selected_lanes_index, lane_head_width, lane_head_height)

        # LEFT-TURN, RIGHT-TURN, FORWARD, FORWARD-LEFT-TURN, FORWARD-RIGHT-TURN, UNSURE
        # P:UNSURE,  Distance from lane line >6m (parking lot scene):UNSURE
        agent_lane_type, lane_directions = lanes_classify(main_agent_object, agent_lane_distance, closest_lane_length, selected_lane_ids, lanes, polylines, closest_exit_lanes)
        
        # Get lane caption
        lane_caption = judge_lane_type_and_get_lane_caption(agent_lane_type, lane_directions, caption_type)

        # Plot road_line
        plt_road_lines(road_lines, polylines)

        
        # Calculate the distance from the ego-agent to all agents
        distances = np.linalg.norm(inter_past_end[:2] - agent_traj_past_valid_end[:, :2], axis=1)
        assert distances[track_id] == 0
        distances[track_id] = np.inf
        
        # Get the index of cars whose distance is less than 30m
        indices_where_range_min_10 = np.where(distances < 30)[0]
        top_indices = np.argsort(distances[indices_where_range_min_10])[:neighbor_num]
        neighbor_indices = indices_where_range_min_10[top_indices].tolist()    
        neighbor_and_inter_indices = neighbor_indices + [track_id]
        neighbor_and_inter_colors = ['orange','green', 'blue', 'gray' , 'gray', 'gray', 'gray', 'gray', 'gray', 'gray'][:len(neighbor_and_inter_indices)]
        neighbor_and_inter_position = [[round(num, 3) for num in d] for d in trajectorys_new[neighbor_and_inter_indices, :2]]
        neighbor_and_inter_vel = [round(d, 3) for d in np.linalg.norm(agent_traj_past_valid_end[neighbor_and_inter_indices, 7:9], axis=1).tolist()]
        neighbot2inter_distances = [round(d, 3) for d in distances[neighbor_and_inter_indices]]
        
        # Get caption for ego-agent
        agent_caption = design_caption_for_agent(caption_type, neighbor_indices, neighbor_and_inter_vel, neighbor_and_inter_position, neighbor_and_inter_colors, neighbot2inter_distances)
        caption = lane_caption + agent_caption
        output_path = output_dir + "/" + scenario_id
        file_path = output_path + "_" + str(track_id) +"_" + main_agent_object + "_" + direction + ".txt"
        open(file_path, "w").close()
        with open(file_path, "w") as file:
            file.write(caption)

        for cnt in range(num_agents):
            # If a agent is invalid at the last valid frame of the main car past, it will not be drawn
            if not agent_in_valid_end_indices[cnt]:
                continue
            
            # Get the type of agent
            object_type = object_type_map[object_types_full[cnt]]            
            # Determine the size of the agent arrow based on the type of agent
            if object_type == "V":
                agent_head_width = 1.8
                agent_head_height = 2.2
            elif object_type == "P":
                agent_head_width = 1
                agent_head_height = 1.5
            else:
                agent_head_width = 0.6
                agent_head_height = 0.8

            # Non ego-agent and adjacent agents
            agent_color = '#494949'
            head_color = '#494949'
            alpha_tmp = 0.9
            zorder_tmp = 101
            agent_text = ""  

            # Adjacent agents
            neighbor_and_inter_texts = ["1", "2", "3"]
            if cnt in neighbor_indices[0:closest_num]:
                nei_related_idx = neighbor_indices.index(cnt)
                agent_color = neighbor_and_inter_colors[nei_related_idx]
                head_color = neighbor_and_inter_colors[nei_related_idx]
                alpha_tmp = 0.95
                zorder_tmp = 101
                agent_text = neighbor_and_inter_texts[nei_related_idx]
                zorder_tmp = 103
            
            # Ego-agent
            if cnt == track_id:
                agent_color = 'red'
                head_color = 'red'
                alpha_tmp = 1
                zorder_tmp = 105
                agent_text = "0"
                img = Image.open("/DATA_EDS2/yanzj/WLX/code/MTR/tools/agent_icons/map_direction.jpg")
                resized_img = img.resize((300, 300))
                plt.figimage(resized_img, xo=2150, yo=2200, zorder=111)

                # Plot the historical trajectory
                trajs_color_map = cm.get_cmap('gray')
                past_trajs_now = ori_past_trajs[track_id][np.bool_(ori_past_valid)[track_id]]
                normalize = np.linspace(0, 0.1, len(past_trajs_now))
                trajs_color_now = trajs_color_map(normalize)
                future_trajs_now = ori_future_trajs[track_id][np.bool_(ori_future_valid)[track_id]]
                past_trajs_now = get_normalized(deepcopy(past_trajs_now[np.newaxis, :, :2]), ori_past_trajs_normalizer).squeeze()
                future_trajs_now = get_normalized(deepcopy(future_trajs_now[np.newaxis, :, :2]), ori_past_trajs_normalizer).squeeze()
                if len(past_trajs_now.shape) == 1:
                    past_trajs_now = past_trajs_now[np.newaxis, :]
                plt.scatter(past_trajs_now[:, 0], past_trajs_now[:, 1], color=trajs_color_now[:], s=100, alpha=1, zorder=110)

            # Plot agent
            x = trajectorys_new[cnt, 0]
            y = trajectorys_new[cnt, 1]
            l = agent_traj_past_valid_end[cnt, 3]
            w = agent_traj_past_valid_end[cnt, 4]
            t = trajectorys_new[cnt, 2]  
            x1, y1 = (x + l / 2 * np.cos(t) + w / 2 * np.sin(t), y + l / 2 * np.sin(t) - w / 2 * np.cos(t))
            x2, y2 = (x + l / 2 * np.cos(t) - w / 2 * np.sin(t), y + l / 2 * np.sin(t) + w / 2 * np.cos(t))
            x3, y3 = (x - l / 2 * np.cos(t) - w / 2 * np.sin(t), y - l / 2 * np.sin(t) + w / 2 * np.cos(t))
            x4, y4 = (x - l / 2 * np.cos(t) + w / 2 * np.sin(t), y - l / 2 * np.sin(t) - w / 2 * np.cos(t))
            if cnt in neighbor_indices or cnt == track_id:
                plt.plot([x1, x2, x3, x4, x1], [y1, y2, y3, y4, y1], color='black', zorder=zorder_tmp, alpha=alpha_tmp)
                plt.plot([x1, x2], [y1, y2], color=agent_color, zorder=zorder_tmp+1, alpha=alpha_tmp, linewidth=4)
                plt.fill([x1, x2, x3, x4], [y1, y2, y3, y4], color=head_color, zorder=zorder_tmp, alpha=alpha_tmp)

                # Plot head of agent
                incre_x = x - (x3 + x4) / 2
                incre_y = y - (y3 + y4) / 2
                plt.arrow(x, y, incre_x, incre_y, head_width=agent_head_width, head_length=agent_head_height, edgecolor=agent_color, facecolor=agent_color, alpha=alpha_tmp, zorder=zorder_tmp+1)

                # Plot text
                text = plt.text(x, y3, s=f'{agent_text}', fontsize=32, color='white', zorder=zorder_tmp+4)
                text.set_bbox({"facecolor": "black", "alpha": 0.9})

        # Scenario crop
        plt.xlim(-main_agent_range_x, main_agent_range_x)
        plt.ylim(-main_agent_range_y, main_agent_range_y)
        plt.text(-main_agent_range_x, main_agent_range_y, s="Motion Prediction Map", fontsize=64, fontweight='bold', color='black', zorder=110)

        plt.xticks([])
        plt.yticks([])
        plt.axis('off')
        plt.savefig(output_path + '_' + str(track_id) + '_' + main_agent_object + '_' + direction +'.png', bbox_inches='tight')
        print(output_path + '_' + str(track_id) + '_' + main_agent_object + '_' + direction +'.png')
        plt.close()
    return 1


if  __name__ == "__main__":
    pass