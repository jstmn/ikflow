import numpy as np

#
""" Tags for dataset. Currently the only tags specify whether a dataset should be sampled with increase/ decreases
to the joint range of a given robot

-pi_on_<x>: Reduce the joint range.

Example for: -pi_on_8
  (initially) joint_1_lower, joint_1_upper = -1, 1
  (after)     joint_1_lower, joint_1_upper = -1 + pi/8, 1 - pi/8

+pi_on_<x>: Expand the joint range.

Example for: +pi_on_8
  (initially) joint_1_lower, joint_1_upper = -2, 2
  (after)     joint_1_lower, joint_1_upper = -2 - pi/8, 2 + pi/8

"""
_2point5_mil = int(2.5 * 1e6)
_50_mil = int(50 * 1e6)
DATASET_TAGS = {
    # (Oct23) The following tags are legacy. They were created when tags were first added. Their actual size is unknown
    "minus_pi_on_8": {"joint_limit_delta": -np.pi / 8, "tr_set_size": _2point5_mil},
    "minus_2pi_on_8": {"joint_limit_delta": -2 * np.pi / 8, "tr_set_size": _2point5_mil},
    "minus_3pi_on_8": {"joint_limit_delta": -3 * np.pi / 8, "tr_set_size": _2point5_mil},
    "minus_4pi_on_8": {"joint_limit_delta": -4 * np.pi / 8, "tr_set_size": _2point5_mil},
    "minus_5pi_on_8": {"joint_limit_delta": -5 * np.pi / 8, "tr_set_size": _2point5_mil},
    "minus_6pi_on_8": {"joint_limit_delta": -6 * np.pi / 8, "tr_set_size": _2point5_mil},
    "minus_7pi_on_8": {"joint_limit_delta": -7 * np.pi / 8, "tr_set_size": _2point5_mil},
    # Nov3: `minus_8pi_on_8` has (-7 * np.pi / 8) instead of (-8 * np.pi / 8)
    "minus_8pi_on_8": {"joint_limit_delta": -8 * np.pi / 8, "tr_set_size": _2point5_mil},
    # "minus_8pi_on_8": {"joint_limit_delta": -7 * np.pi / 8, "tr_set_size": _2point5_mil},
    "plus_pi_on_8": {"joint_limit_delta": np.pi / 8, "tr_set_size": _2point5_mil},
    "plus_2pi_on_8": {"joint_limit_delta": 2 * np.pi / 8, "tr_set_size": _2point5_mil},
    "plus_3pi_on_8": {"joint_limit_delta": 3 * np.pi / 8, "tr_set_size": _2point5_mil},
    "plus_4pi_on_8": {"joint_limit_delta": 4 * np.pi / 8, "tr_set_size": _2point5_mil},
    "plus_5pi_on_8": {"joint_limit_delta": 5 * np.pi / 8, "tr_set_size": _2point5_mil},
    "plus_6pi_on_8": {"joint_limit_delta": 6 * np.pi / 8, "tr_set_size": _2point5_mil},
    "plus_7pi_on_8": {"joint_limit_delta": 7 * np.pi / 8, "tr_set_size": _2point5_mil},
    "plus_8pi_on_8": {"joint_limit_delta": 8 * np.pi / 8, "tr_set_size": _2point5_mil},
    #  Use the following tags
    "minus_pi_on_8_MD_SIZE": {"joint_limit_delta": -1 * np.pi / 8, "tr_set_size": _50_mil},
    "minus_2pi_on_8_MD_SIZE": {"joint_limit_delta": -2 * np.pi / 8, "tr_set_size": _50_mil},
    "minus_3pi_on_8_MD_SIZE": {"joint_limit_delta": -3 * np.pi / 8, "tr_set_size": _50_mil},
    "minus_4pi_on_8_MD_SIZE": {"joint_limit_delta": -4 * np.pi / 8, "tr_set_size": _50_mil},
    "minus_5pi_on_8_MD_SIZE": {"joint_limit_delta": -5 * np.pi / 8, "tr_set_size": _50_mil},
    "minus_6pi_on_8_MD_SIZE": {"joint_limit_delta": -6 * np.pi / 8, "tr_set_size": _50_mil},
    "minus_7pi_on_8_MD_SIZE": {"joint_limit_delta": -7 * np.pi / 8, "tr_set_size": _50_mil},
    "minus_8pi_on_8_MD_SIZE": {"joint_limit_delta": -8 * np.pi / 8, "tr_set_size": _50_mil},
    # Nov3: `minus_8pi_on_8_MD_SIZE` has (-7 * np.pi / 8) instead of (-8 * np.pi / 8)
    # "minus_8pi_on_8_MD_SIZE": {"joint_limit_delta": -7 * np.pi / 8, "tr_set_size": _50_mil},
    "plus_pi_on_8_MD_SIZE": {"joint_limit_delta": 1 * np.pi / 8, "tr_set_size": _50_mil},
    "plus_2pi_on_8_MD_SIZE": {"joint_limit_delta": 2 * np.pi / 8, "tr_set_size": _50_mil},
    "plus_3pi_on_8_MD_SIZE": {"joint_limit_delta": 3 * np.pi / 8, "tr_set_size": _50_mil},
    "plus_4pi_on_8_MD_SIZE": {"joint_limit_delta": 4 * np.pi / 8, "tr_set_size": _50_mil},
    "plus_5pi_on_8_MD_SIZE": {"joint_limit_delta": 5 * np.pi / 8, "tr_set_size": _50_mil},
    "plus_6pi_on_8_MD_SIZE": {"joint_limit_delta": 6 * np.pi / 8, "tr_set_size": _50_mil},
    "plus_7pi_on_8_MD_SIZE": {"joint_limit_delta": 7 * np.pi / 8, "tr_set_size": _50_mil},
    "plus_8pi_on_8_MD_SIZE": {"joint_limit_delta": 8 * np.pi / 8, "tr_set_size": _50_mil},
    # Follow up tags
    "minus_1pi_on_32_MD_SIZE": {"joint_limit_delta": -1 * np.pi / 32, "tr_set_size": _50_mil},
    "minus_2pi_on_32_MD_SIZE": {"joint_limit_delta": -2 * np.pi / 32, "tr_set_size": _50_mil},
    "minus_3pi_on_32_MD_SIZE": {"joint_limit_delta": -3 * np.pi / 32, "tr_set_size": _50_mil},
    "minus_4pi_on_32_MD_SIZE": {"joint_limit_delta": -4 * np.pi / 32, "tr_set_size": _50_mil},
    "minus_5pi_on_32_MD_SIZE": {"joint_limit_delta": -5 * np.pi / 32, "tr_set_size": _50_mil},
    "minus_6pi_on_32_MD_SIZE": {"joint_limit_delta": -6 * np.pi / 32, "tr_set_size": _50_mil},
    "minus_7pi_on_32_MD_SIZE": {"joint_limit_delta": -7 * np.pi / 32, "tr_set_size": _50_mil},
    "minus_8pi_on_32_MD_SIZE": {"joint_limit_delta": -8 * np.pi / 32, "tr_set_size": _50_mil},
    "plus_1pi_on_32_MD_SIZE": {"joint_limit_delta": 1 * np.pi / 32, "tr_set_size": _50_mil},
    "plus_2pi_on_32_MD_SIZE": {"joint_limit_delta": 2 * np.pi / 32, "tr_set_size": _50_mil},
    "plus_3pi_on_32_MD_SIZE": {"joint_limit_delta": 3 * np.pi / 32, "tr_set_size": _50_mil},
    "plus_4pi_on_32_MD_SIZE": {"joint_limit_delta": 4 * np.pi / 32, "tr_set_size": _50_mil},
    "plus_5pi_on_32_MD_SIZE": {"joint_limit_delta": 5 * np.pi / 32, "tr_set_size": _50_mil},
    "plus_6pi_on_32_MD_SIZE": {"joint_limit_delta": 6 * np.pi / 32, "tr_set_size": _50_mil},
    "plus_7pi_on_32_MD_SIZE": {"joint_limit_delta": 7 * np.pi / 32, "tr_set_size": _50_mil},
    "plus_8pi_on_32_MD_SIZE": {"joint_limit_delta": 8 * np.pi / 32, "tr_set_size": _50_mil},
    "TESTTAG": {"joint_limit_delta": 0.0, "tr_set_size": _2point5_mil},
    "LARGE25": {"joint_limit_delta": 0.0, "tr_set_size": int(25 * 1e6)},
    "LARGE50": {"joint_limit_delta": 0.0, "tr_set_size": _50_mil},
    "LARGE100": {"joint_limit_delta": 0.0, "tr_set_size": int(100 * 1e6)},
    "LARGE250": {"joint_limit_delta": 0.0, "tr_set_size": int(250 * 1e6)},
    #     7 total
    "includes_l2_loss_pts": {"joint_limit_delta": 0.0, "tr_set_size": int(5 * 1e6)},
    # Don't use this for l2-multi - two points doen't uniquely define a pose
    #     "includes_l2_loss_pts1pt": {"joint_limit_delta": 0.0, "tr_set_size": int(5 * 1e6)},
    #     3 total
    "includes_l2_loss_pts2pt": {"joint_limit_delta": 0.0, "tr_set_size": int(5 * 1e6)},
}

SJR_EXP = [
    "LARGE50",
    "plus_pi_on_8_MD_SIZE",
    "plus_2pi_on_8_MD_SIZE",
    "plus_3pi_on_8_MD_SIZE",
    "plus_4pi_on_8_MD_SIZE",
    "plus_5pi_on_8_MD_SIZE",
    "plus_6pi_on_8_MD_SIZE",
    "plus_7pi_on_8_MD_SIZE",
    "plus_8pi_on_8_MD_SIZE",
]

# This experiment is a follow up to the original SJR experiment, but uses smaller increases to the joint limits (k*pi/32)
SJR_EXP_2 = [
    "plus_1pi_on_32_MD_SIZE",
    "plus_2pi_on_32_MD_SIZE",
    "plus_3pi_on_32_MD_SIZE",
    "plus_4pi_on_32_MD_SIZE",
    "plus_5pi_on_32_MD_SIZE",
    "plus_6pi_on_32_MD_SIZE",
    "plus_7pi_on_32_MD_SIZE",
    "plus_8pi_on_32_MD_SIZE",
]
