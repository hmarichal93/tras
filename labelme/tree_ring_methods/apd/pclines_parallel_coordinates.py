import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from lib.image import Drawing



def pclines_straight_all(l, d = 1 ):
    x1 = l[:, 0]
    y1 = l[:, 1]
    x2 = l[:, 2]
    y2 = l[:, 3]

    dy = y2 - y1
    dx = x2 - x1

    m = dy / dx
    b = (y1 * x2 - y2 * x1) / dx

    PCline = np.array([np.ones(b.shape[0]) * d, b, 1 - m]).T #homogeneous coordinates

    u = PCline[:,0] / PCline[:,2]
    v = PCline[:, 1] / PCline[:,2]

    return u,v


def pclines_twisted_all(l, d = 1):
    x1 = l[:, 0]
    y1 = l[:, 1]
    x2 = l[:, 2]
    y2 = l[:, 3]

    dy = y2 - y1
    dx = x2 - x1

    m = dy / dx
    b = (y1 * x2 - y2 * x1) / dx

    PCline = np.array([-np.ones(b.shape[0])*d, -b, 1 + m]).T #homogeneous coordinates

    u = PCline[:,0] / PCline[:,2]
    v = PCline[:, 1] / PCline[:,2]

    return u,v

def ts_space( img, lines, output_dir, d=1, debug=True):
    H, W, _ = img.shape
    v_maximum = np.maximum(W/2, H/2)
    l_lo = lines.reshape(-1, 4)
    u, v = pclines_straight_all(l_lo / np.asarray([W,H,W,H]), d=d)
    points_straight = np.vstack((u,v)).T
    u, v = pclines_twisted_all(l_lo / np.array([W,H,W,H]), d=d)
    points_twisted = np.vstack((u,v)).T

    #Impose boundaries of pclines space
    ## Straight space boundaries (u,v) in ([0,d],[-v_maximum, v_maximum])
    mask_idx_straight_half = np.where( 0 <= points_straight[:,0], 1, 0) & np.where( d >= points_straight[:,0], 1, 0) & \
                        np.where(-v_maximum <= points_straight[:,1], 1, 0) & np.where(v_maximum >= points_straight[:,1], 1, 0)

    idx_straight_half = np.where(mask_idx_straight_half>0)[0]
    points_straight = points_straight[idx_straight_half]
    lines_straight = l_lo[idx_straight_half]

    ## Twisted space boundaries (u,v) in ([-d,0],[-v_maximum, v_maximum])
    mask_idx_twisted_half = np.where( -d <= points_twisted[:,0], 1, 0) & np.where( 0 >= points_twisted[:,0], 1, 0) & \
                        np.where(-v_maximum <= points_twisted[:,1], 1, 0) & np.where(v_maximum >= points_twisted[:,1], 1, 0)

    idx_twisted_half = np.where(mask_idx_twisted_half>0)[0]
    points_twisted = points_twisted[idx_twisted_half]
    lines_twisted = l_lo[idx_twisted_half]



    if debug:

        #generate two subplots. One for points stragiht and one for points twisted. Use small dot size
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(7,7))
        axes[1].scatter(points_straight[:,0], points_straight[:,1], s=0.1)
        axes[0].scatter(points_twisted[:,0], points_twisted[:,1], s=0.1)
        axes[1].set_title('Straight Space')
        axes[0].set_title('Twisted Space')
        axes[1].set_xlabel('u')
        axes[1].set_ylabel('v')
        axes[0].set_xlabel('u')
        axes[0].set_ylabel('v')
        axes[1].grid(True)
        axes[0].grid(True)
        plt.savefig(f"{output_dir}/pc_m_b_representation_ts_space_subplots.png")
        plt.close()


    return points_straight, points_twisted, lines_straight, lines_twisted


def robust_line_estimation( X, y, residual_threshold = 0.03):
    from sklearn.linear_model import RANSACRegressor

    # Create the RANSAC regressor
    min_samples = np.round(X.shape[0]*0.05).astype(int)
    min_samples = np.maximum(100,min_samples)
    min_samples = np.minimum(X.shape[0], min_samples)
    ransac = RANSACRegressor( min_samples=min_samples, max_trials=1000, residual_threshold = residual_threshold,
                              random_state=42)
    # Fit the regressor to the data
    try:
        ransac.fit(X.reshape(-1, 1), y.reshape(-1, 1))
    except ValueError:
        """ransac not convergs"""
        return [0,0], np.ones(X.shape[0])*np.inf

    # Get the model coefficient and bias
    coefficient = ransac.estimator_.coef_
    bias = ransac.estimator_.intercept_
    line = np.array([coefficient.ravel(), bias.ravel()]).ravel()
    residual = np.abs(line[0] * X + line[1] - y)

    return line, residual

def find_detections( points, output_path, debug=False, outliers_threshold=0.1, alineation_threshold=0.45):

    v = points[:,1]
    u = points[:,0]
    if v.ravel().shape[0] <2 :
        return np.array([],dtype=int), None, None, np.array([]), False

    line, residual = robust_line_estimation(u.reshape(-1, 1), v.reshape(-1, 1),  residual_threshold=outliers_threshold)

    inliers = np.where(residual <= outliers_threshold)[0]
    inliers = inliers.astype(int)
    #percentile 99 residual
    p90 = np.percentile(residual, 99)
    mean = np.mean(residual)
    median = np.median(residual)

    if debug:
        plt.figure()
        plt.scatter(u, v, s=0.1, c='b', label='data')
        plt.scatter(u[inliers], v[inliers], s=0.1, c='g', label='inliers')
        plt.plot(u, line[0] * u + line[1], color='red', label='Line estimation')
        plt.legend(loc='lower right')
        #plt.title(f" Mean: {mean:.2f}  Percentile 90: {p90:.2f} \n median: {median:.2f}")
        plt.xlabel('u')
        plt.ylabel('v')
        plt.savefig(output_path)
        plt.close()


    return inliers, line[0], line[1], residual[inliers], p90 < alineation_threshold

def get_duplicated_elements_in_array(l:np.array):
    unique, counts = np.unique(l, return_counts=True, axis=0)
    duplicated_row = unique[counts > 1]
    return duplicated_row

def get_indexes_relative_to_src_list_if_there_is_more_than_one(src_array, dst_array):
    """Get indexes of src_array in dst_array
    params:
        dst_array: 2D array to search in src_array
        src_array: 2D array where dst_array is searched
    """
    idx = []
    for i in range(src_array.shape[0]):
        indexes = np.where((dst_array == src_array[i]).all(axis=1))[0][:-1].tolist()
        idx += indexes
    return idx


def get_converging_lines_pc(img, m_lsd, coherence, output_dir, outlier_th = 0.05, debug=False):
    if output_dir is not None:
        vp_output_dir = Path(output_dir)
    else:
        vp_output_dir = Path(".")

    if debug:
        vp_output_dir.mkdir(parents=True, exist_ok=True)

    # 1.0 convert to TS-Space
    m_pc_straight, m_pc_twisted, m_img_straight, m_img_twisted = ts_space(img, m_lsd,
                                                                              output_dir=f"{vp_output_dir}", debug=debug)

    # 2.0 Get converging m_lsd on each space
    idx_inliers_straight, m1, b1, residual_straight, alineation_st = find_detections(m_pc_straight,
                                                                        output_path=f"{vp_output_dir}/ts_straight.png",
                                                                        outliers_threshold=outlier_th, debug=debug)

    idx_inliers_twisted, m2, b2, residual_twisted, alineation_tw = find_detections(m_pc_twisted,
                                                                      output_path=f"{vp_output_dir}/ts_twisted.png",
                                                                      outliers_threshold=outlier_th, debug=debug)
    if debug:
        Drawing.draw_lsd_lines(m_lsd, img,
                          output_path=f"{vp_output_dir}/lsd.png", lines_all=m_lsd)

        Drawing.draw_lsd_lines(m_img_straight[idx_inliers_straight], img,
                          output_path=f"{vp_output_dir}/straight_lines_in_image.png", lines_all=m_lsd)

        Drawing.draw_lsd_lines(m_img_twisted[idx_inliers_twisted], img, output_path=f"{vp_output_dir}/twisted_lines_in_image.png",
                          lines_all=m_lsd)

    # 3.0 Remove duplicated m_lsd
    converging_lines = np.vstack((m_img_straight[idx_inliers_straight], m_img_twisted[idx_inliers_twisted]))
    residual = np.vstack((residual_straight.reshape((-1,1)), residual_twisted.reshape((-1,1))))
    if coherence is not None:
        coherence = np.hstack((coherence[idx_inliers_straight], coherence[idx_inliers_twisted]))
    idx_lines = np.hstack((idx_inliers_straight, idx_inliers_twisted))

    duplicated_lines = get_duplicated_elements_in_array(converging_lines)
    # get row index for duplicated m_lsd
    idx_duplicated_lines = get_indexes_relative_to_src_list_if_there_is_more_than_one(duplicated_lines, converging_lines)
    # remove duplicated elements
    converging_lines = np.delete(converging_lines, idx_duplicated_lines, axis=0)
    residual = np.delete(residual, idx_duplicated_lines, axis=0)
    if coherence is not None:
        coherence = np.delete(coherence, idx_duplicated_lines, axis=0)
    idx_lines = np.delete(idx_lines, idx_duplicated_lines, axis=0)

    if debug:
        Drawing.draw_lsd_lines(converging_lines, img, output_path=f"{vp_output_dir}/converging_segment_in_image.png",
                          lines_all=m_lsd)

        Drawing.draw_lines(converging_lines, img, output_path=f"{vp_output_dir}/converging_lo_in_image.png")

    #idx_lines = get_indexes_relative_to_src_list(m_lsd, converging_lines)
    return converging_lines, idx_lines, coherence, alineation_st and alineation_tw



def rotate_lines(L, degrees=90):
    X1, Y1, X2, Y2 = L[:, 0], L[:, 1], L[:, 2], L[:, 3]
    # 1.0 compute the center of the line
    C = (np.array([X1 + X2, Y1 + Y2]) * 0.5 ).T
    Cx = C[:, 0]
    Cy = C[:, 1]
    # rotate
    angle = np.deg2rad(degrees)
    X1r = np.cos(angle) * (X1 - Cx) - np.sin(angle) * (Y1 - Cy) + Cx
    Y1r = np.sin(angle) * (X1 - Cx) + np.cos(angle) * (Y1 - Cy) + Cy
    X2r = np.cos(angle) * (X2 - Cx) - np.sin(angle) * (Y2 - Cy) + Cx
    Y2r = np.sin(angle) * (X2 - Cx) + np.cos(angle) * (Y2 - Cy) + Cy
    L_rotated = np.array([X1r, Y1r, X2r, Y2r]).T
    return L_rotated

def new_remove_segmented_that_are_selected_twice(sub_1, idx_1, coh_1, sub_2, idx_2, coh2):
    sub_1_c = sub_1.copy()
    sub_2_c = sub_2.copy()

    idx_1_rm = []
    idx_2_rm = []
    idx_2 = idx_2.tolist()
    idx_1 = idx_1.tolist()
    # for idx, ele1 in enumerate(idx_1):
    #     if ele1 in idx_2:
    #         idx_1_rm.append(idx)
    #         idx_2_rm.append(idx_2.index(ele1))

    set_idx_2 = set(idx_2)
    idx_1_rm = [idx for idx, ele1 in enumerate(idx_1) if ele1 in set_idx_2]
    idx_2_rm = [idx_2.index(ele1) for ele1 in idx_1 if ele1 in set_idx_2]

    sub_1_c = np.delete(sub_1_c, idx_1_rm, axis=0)
    coh_1_c = np.delete(coh_1, idx_1_rm, axis=0) if coh_1 is not None else None
    sub_2_c = np.delete(sub_2_c, idx_2_rm, axis=0)
    coh_2_c = np.delete(coh2, idx_2_rm, axis=0) if coh2 is not None else None

    return sub_1_c, coh_1_c, sub_2_c, coh_2_c

def pclines_local_orientation_filtering(img_in, m_lsd, coherence=None, lo_dir=None, outlier_th=0.03, debug=True):
    m_lsd_radial, idx_lsd_radial, coherence_radial,_ = get_converging_lines_pc(img_in, m_lsd, coherence,
                                                                             lo_dir / "lsd_converging_lines" if lo_dir is not None else None,
                                                                             outlier_th=outlier_th, debug=debug)
    # 2.1
    l_rotated_lsd_lines = rotate_lines(m_lsd)
    # outlier_th = 0.1
    sub_2, idx_2, coherence_2, radial_alineation = get_converging_lines_pc(img_in, l_rotated_lsd_lines, coherence,
                                                        lo_dir / "lsd_rotated_converging_lines" if lo_dir is not None else None,
                                                        outlier_th=outlier_th, debug=debug)
    # if not radial_alineation:
    #     return m_lsd, coherence, radial_alineation

    # 2.2 Remove segments that where selected twice
    m_lsd_radial, coherence_radial, sub_2, coherence_2 = new_remove_segmented_that_are_selected_twice(m_lsd_radial,
                                                                                                      idx_lsd_radial,
                                                                                                      coherence_radial,
                                                                                                      sub_2, idx_2,
                                                                                                      coherence_2)

    # 3.0 get rotated and radial intersecting segments
    converging_lines = np.vstack((m_lsd_radial, sub_2))
    converging_coherence = np.hstack((coherence_radial, coherence_2)) if coherence is not None else None
    m_lsd_intersecting, _, coherence_intersecting,_ = get_converging_lines_pc(img_in, converging_lines,
                                                                            converging_coherence,
                                                                            lo_dir / "both_subset_convering_lines" if lo_dir is not None else None,
                                                                            outlier_th=outlier_th, debug=debug)

    return m_lsd_intersecting, coherence_intersecting, True
