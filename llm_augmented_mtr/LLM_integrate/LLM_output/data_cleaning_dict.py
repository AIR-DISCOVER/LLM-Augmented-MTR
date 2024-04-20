intention_clean_dict = {
    #========standard convert==========#
    "STATIONARY": "STATIONARY",
    "STRAIGHT": "STRAIGHT",
    "STRAIGHT-LEFT": "STRAIGHT_LEFT",
    "STRAIGHT-RIGHT": "STRAIGHT_RIGHT",
    "LEFT-TURN": "LEFT_TURN",
    "RIGHT-TURN": "RIGHT_TURN",
    "LEFT-U-TURN": "LEFT_U_TURN",
    "RIGHT-U-TURN": "RIGHT_U_TURN",
    #========below are corner cases (llm sometimes make mistakes)=========#
    "STATIONAR": "STATIONARY",
    "STAIGHT": "STRAIGHT",
    "SAFE-LEFT": "STRAIGHT_LEFT",
    "SAFE-RIGHT": "STRAIGHT_RIGHT",
    "STOP": "STATIONARY",
    "SLOW": "STATIONARY",
    "SLOW-ALLOW": "STATIONARY",
    "SLOW DOWN": "STATIONARY",
    "SLOW-DOWN": "STATIONARY",
    "LEFT-ALLOW": "LEFT_TURN",
    "ACCELERATE": "STRAIGHT",
    "ACCELERATE-ALLOW": "STRAIGHT",
    "RIGHT-ALLOW": "RIGHT_TURN"
}

affordance_clean_dict = {
    #========standard convert==========#
    "ACCELERATE-ALLOW": "ACCELERATE_ALLOW",
    "SLOW-ALLOW": "SLOW_ALLOW",
    "LEFT-ALLOW": "LEFT_ALLOW",
    "RIGHT-ALLOW": "RIGHT_ALLOW",
    #========below are corner cases (llm sometimes make mistakes)=========#    
    "STRAIGHT-ALLOW": "ACCELERATE_ALLOW",
    "STRAIGHT-RIGHT": "RIGHT_ALLOW",
    "STRAIGHT-LEFT": "LEFT_ALLOW",
    "STRAIGHT-RIGHT-ALLOW": "RIGHT_ALLOW",
    "LEFT-U-TURN-ALLOW": "LEFT_ALLOW",
    "LEFT-TURN": "LEFT_ALLOW",
    "RIGHT-TURN": "RIGHT_ALLOW",
    "STOP-ALLOW": "SLOW_ALLOW"
}

scenario_clean_dict = {
    #========standard convert==========#
    "INTERSECTION": "INTERSECTION",
    "ON-STRAIGHT-ROAD": "ON_STRAIGHT_ROAD",
    "PARKING-LOT": "PARKING_LOT",
    "ON-ROADSIDE": "ON_ROADSIDE",
    "UNSURE": "UNSURE"
}