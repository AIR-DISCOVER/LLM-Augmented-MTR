embedding_dict = {
    "TYPE_VEHICLE": {
        "scenario_id": [],
        "track_index": [],
        "embedding": [],
        "intention": [],
        "affordance": [],
        "scenario": []
    },
    "TYPE_PEDESTRIAN": {
        "scenario_id": [],
        "track_index": [],
        "embedding": [],
        "intention": [],
        "affordance": [],
        "scenario": []
    },
    "TYPE_CYCLIST": {
        "scenario_id": [],
        "track_index": [],
        "embedding": [],
        "intention": [],
        "affordance": [],
        "scenario": []
    }
}

intention_type_decode_dict = {
    0: "STATIONARY",
    1: "STRAIGHT",
    2: "STRAIGHT_LEFT",
    3: "STRAIGHT_RIGHT",
    4: "LEFT_TURN",
    5: "RIGHT_TURN",
    6: "LEFT_U_TURN",
    7: "RIGHT_U_TURN"
}

affordance_type_decode_dict = {
    0: "SLOW_ALLOW",
    1: "ACCELERATE_ALLOW",
    2: "LEFT_ALLOW",
    3: "RIGHT_ALLOW"
}

scenario_type_decode_dict = {
    0: "INTERSECTION",
    1: "ON_STRAIGHT_ROAD",
    2: "PARKING_LOT",
    3: "ON_ROADSIDE",
    4: "UNSURE"
}