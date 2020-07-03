import numpy as np
import scipy


def gaussian_filter_density(gt):
    """generate ground truth density map
    Args:
        gt: (height, width), object center is 1.0, otherwise 0.0
    Returns:
        density map
    """
    density = np.zeros(gt.shape, dtype=np.float32)
    gt_count = np.count_nonzero(gt)
    if gt_count == 0:
        return density
    pts = np.array(list(zip(np.nonzero(gt)[1], np.nonzero(gt)[0])))  # (x,y)
    leaf_size = 2048
    # build kd tree
    tree = scipy.spatial.KDTree(pts.copy(), leafsize=leaf_size)
    # query kd tree
    distances, locations = tree.query(pts, k=4)

    for i, pt in enumerate(pts):
        pt2d = np.zeros(gt.shape, dtype=np.float32)
        pt2d[pt[1], pt[0]] = 1.
        if gt_count > 1:
            sigma = (distances[i][1] + distances[i][2] + distances[i][3]) * 0.085
            sigma = min(sigma, 999)  # avoid inf
        else:
            raise NotImplementedError('should not be here!!')
        density += scipy.ndimage.filters.gaussian_filter(pt2d, sigma, mode='constant')
    return density
