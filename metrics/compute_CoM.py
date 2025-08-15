"""
Compute the Center of mass of a body from ViTPose 2D keypoints (COCO17 format)
Return the coordinates of the CoM (x, y)
"""
import numpy as np

class CoM():
    def __init__(self, sex:str, N_keypoints:int):
        if N_keypoints == 17:
            # (proximal, distal) joint indices for each bodypart
            # bodyparts : handR, handL, forearmR, forearmL, upperarmR, upperarmL, footR, footL, shankR, shankL, thighR, thighL, trunk, head
            bodyparts = [(9,9), (10,10), (7,9), (8,10), (5,7), (6,8), (15,15), (16,16), (13,15), (14,16), (11,13), (12,14), (12,5), (0, 0)]
        
        elif N_keypoints == 13:
            # (proximal, distal) joint indices for each bodypart
            # bodyparts : handR, handL, forearmR, forearmL, upperarmR, upperarmL, footR, footL, shankR, shankL, thighR, thighL, trunk, head
            bodyparts = [(3,3), (6,6), (3,5), (4,6), (1,3), (2,4), (11,11), (12,12), (9,11), (10,12), (7,9), (8,10), (8, 1), (0, 0)]

        else:
            print("Class only made for 17 or 13 keypoints.")

        # --- MEN ---
        if sex == 'men':
            # proximal proportions of the cm for each bodypart
            cm_distances = [0, 0, .43, .43, .436, .436, 0, 0, .434, .434, .433, .433, .63, 0]
            # mass proportion of the bodymass for each bodypart
            masses = [.0065, .0065, .0187, .0187, .0325, .0325, .0143, .0143, .0475, .0475, .105, .105, .4682, .0826]

        # --- WOMEN ---
        elif sex == 'women':
            cm_distances = [0, 0, .434, .434, .458, .458, 0, 0, .419, .419, .428, .428, .569, 0]
            masses = [.005, .005, .0157, .0157, .029, .029, .0133, .0133, .0535, .0535, .1175, .1175, .4522, .082]

        else:
            print("Please specify a valid sex: 'men' OR 'women'.")

        self.bodyparts = bodyparts
        self.cm_distances = cm_distances
        self.masses = np.array(masses).reshape((-1, 1))

    def get_cm_bodypart(self, x, y, cm_dist):
        xmin, xmax = x
        ymin, ymax = y
        cm = np.array((xmax-xmin, ymax-ymin)) * cm_dist + np.array((xmin, ymin))
        return cm

    def compute_global_cm(self, keypts:np.ndarray):
        keypts = keypts.reshape((-1, 2))
        cm_bodyparts = []
        for bp, cm_dist in zip(self.bodyparts, self.cm_distances):
            x = keypts[bp, 0]
            y = keypts[bp, 1]
            cm = self.get_cm_bodypart(x, y, cm_dist)
            cm_bodyparts.append(cm)

        cm_bodyparts = np.array(cm_bodyparts)
        global_cm = cm = np.sum(np.multiply(cm_bodyparts, self.masses), axis=0) / self.masses.sum()

        return global_cm