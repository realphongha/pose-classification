import numpy as np 
import math


def unit_vector(vector):
    """ 
    Returns the unit vector of the vector  
    """
    return vector / np.linalg.norm(vector)


def angle_between(v1, v2):
    """ 
    Returns the angle in radians between vectors 'v1' and 'v2'
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

def rotate_kps(kps, degrees=None, alpha=None, anchor=None):
    # kps: array with shape (num_points, 2)
    assert degrees is None or alpha is None
    assert not (degrees is None and alpha is None)
    kps_arr = np.array(kps, dtype=np.float32)
    if degrees:
        alpha = -degrees * math.pi / 180
    if anchor:
        x_anchor, y_anchor = anchor
    else:  # center as anchor
        x_anchor = np.mean(kps_arr[:, 0])
        y_anchor = np.mean(kps_arr[:, 1])
    kps_arr[:, 0] -= x_anchor
    kps_arr[:, 1] -= y_anchor
    kps_apu = np.zeros(kps_arr.shape, dtype=np.float32)
    kps_apu[:, 0] = math.cos(alpha)*kps_arr[:, 0] - math.sin(alpha)*kps_arr[:, 1] + x_anchor
    kps_apu[:, 1] = math.sin(alpha)*kps_arr[:, 0] + math.cos(alpha)*kps_arr[:, 1] + y_anchor
    return kps_apu