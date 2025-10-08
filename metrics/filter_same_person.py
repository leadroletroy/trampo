"""
Determine if both detections are associated to the same person
according to the Center of Mass and the overlap of the bounding boxes
Return True or False
"""
import numpy as np
from compute_CoM import CoM

def compute_bbox(keypts: np.ndarray):
    n = max(keypts.shape)
    keypts = keypts.reshape((n, -1))
    
    xmin = keypts[:, 0].min()
    xmax = keypts[:, 0].max()
    ymin = keypts[:, 1].min()
    ymax = keypts[:, 1].max()

    d = keypts.shape[1]
    if d == 2:
        return (xmin, ymin, xmax, ymax)

    elif d == 3:
        zmin = keypts[:, 2].min()
        zmax = keypts[:, 2].max()
        return (xmin, ymin, zmin, xmax, ymax, zmax)
    
    return None

def compute_iou(bbox1, bbox2):
    x1_min, y1_min, x1_max, y1_max = bbox1
    x2_min, y2_min, x2_max, y2_max = bbox2

    # Intersection
    x_left = max(x1_min, x2_min)
    y_top = max(y1_min, y2_min)
    x_right = min(x1_max, x2_max)
    y_bottom = min(y1_max, y2_max)

    if x_right <= x_left or y_bottom <= y_top:
        return 0.0  # pas de recouvrement

    inter_area = (x_right - x_left) * (y_bottom - y_top)

    area1 = (x1_max - x1_min) * (y1_max - y1_min)
    area2 = (x2_max - x2_min) * (y2_max - y2_min)

    union_area = area1 + area2 - inter_area
    return inter_area / union_area


def same_person(keypts1, keypts2, thresh_CoM, thresh_IoU=0.5):
    com = CoM('women', 13)
    CoM1 = com.compute_global_cm(keypts1)
    CoM2 = com.compute_global_cm(keypts2)
    distance_CoM = np.linalg.norm(CoM1 - CoM2)

    bbox1 = compute_bbox(keypts1)
    bbox2 = compute_bbox(keypts2)
    iou = compute_iou(bbox1, bbox2)

    return distance_CoM <= thresh_CoM and iou >= thresh_IoU