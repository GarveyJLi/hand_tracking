import mediapipe as mp
import numpy as np
import pandas as pd


"""
Calculates euclidian distance in 3D space
"""
def euc_dist(x1, y1, z1, x2, y2, z2):
    return np.sqrt((x2-x1)**2 + (y2-y1)**2 + (z2-z1) ** 2)

"""
Calculates the distance of joints for each finger with repect to the wrist
Probably care about joins [2, 3, 4, 6, 7, 8, 10, 11, 12, 14, 15, 16, 18, 19, 20]
"""
def joints_dists(results, base=0, joints=[4]):
    # Initialize dictionary to store data
    # Have to figure out which one is left and which one is right for when
    # camera is reversed or something

    # MAYBE CAN CALCULATE DISTANCE BETWEEN WRISTS FOR BOTH OR COMPARE THEIR 
    # XYZ COORDINATES FOR POS OR NEG OR SOMETHING TO DETERMINE LEFT AND RIGHT

    right_joint_dict = {}
    left_joint_dict = {}

    # Initialize right and left hand markers and see if they exist
    right_hand = results.right_hand_landmarks
    left_hand = results.left_hand_landmarks
    if right_hand:
        # If right hand is in view, initialize base(wrist)
        right_wrist = right_hand.landmark[base]
        # Calculate distance for joints from wrist
        for joint in joints:
            joint_all = right_hand.landmark[joint]
            dist = euc_dist(joint_all.x, joint_all.y, joint_all.z, \
                right_wrist.x, right_wrist.y, right_wrist.z)
            if joint not in right_joint_dict:
                right_joint_dict[joint] = [dist]
            else:
                right_joint_dict[joint].append(dist)
    if left_hand:
        left_wrist = left_hand.landmark[base]
        for joint in joints:
            joint_all = left_hand.landmark[joint]
            dist = euc_dist(joint_all.x, joint_all.y, joint_all.z, \
                left_wrist.x, left_wrist.y, left_wrist.z)
            if joint not in right_joint_dict:
                left_joint_dict[joint] = [dist]
            else:
                #Something wrong here. crashes
                left_joint_dict[joint].append(dist)
    return right_joint_dict, left_joint_dict
