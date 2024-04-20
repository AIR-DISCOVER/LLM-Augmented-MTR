intention_types = {
    "STATIONARY": 0,
    "STRAIGHT": 1,
    "STRAIGHT_LEFT": 2,
    "STRAIGHT_RIGHT": 3,
    "LEFT_TURN": 4,
    "RIGHT_TURN": 5,
    "LEFT_U_TURN": 6,
    "RIGHT_U_TURN": 7
}

affordance_types = {
    "SLOW_ALLOW": 0,
    "ACCELERATE_ALLOW": 1,
    "LEFT_ALLOW": 2,
    "RIGHT_ALLOW": 3
}

scenario_types = {
    "INTERSECTION": 0,
    "ON_STRAIGHT_ROAD": 1,
    "PARKING_LOT": 2,
    "ON_ROADSIDE": 3,
    "UNSURE": 4
}

context_types_dict = {
    "intention": intention_types,
    "affordance": affordance_types,
    "scenario": scenario_types
}