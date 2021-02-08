
"""
Keypoints in AIChallenger dataset:
1: Right shoulder
2: Right elbow
3: Right wrist

4: Left shoulder
5: Left elbow
6: Left wrist

7: Right hip
8: Right knee
9: Right ankle

10: Left hip
11: Left knee
12: Left ankle

13: head top
14: neck

"""

# Keypoint connection of bones in AIChallenger dataset:
aic_bones = [
    [1, 2], 
    [2, 3],  

    [4, 5],  
    [5, 6],

    [14, 1],
    [14, 4],

    [1, 7], 
    [4, 10], 

    [7, 8],
    [8, 9],

    [10, 11],
    [11, 12],

    [13, 14]] 

aic_bone_pairs = (
    ([14, 1], [1, 2]),  
    ([1, 2], [2, 3]), 
    ([14, 4], [4, 5]),
    ([4, 5], [5, 6]),

    ([7, 8], [8, 9]),
    ([10, 11], [11, 12]),
)