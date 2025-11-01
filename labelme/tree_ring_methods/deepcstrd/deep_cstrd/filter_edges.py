import numpy as np
import cv2

from cross_section_tree_ring_detection.filter_edges import (change_reference_axis, normalized_row_matrix,
                                                            compute_angle_between_gradient_and_edges,
                                                            convert_masked_pixels_to_curves, get_border_curve,
                                                            get_gradient_vector_for_each_edge_pixel, contour_to_curve)


def filter_edges_by_threshold(m_ch_e, theta, alpha_low=30):
    alpha_high = 180 - alpha_low
    X_edges_filtered = m_ch_e.copy()
    mask = (theta >= alpha_low) & (theta <= alpha_high) | np.isnan(theta)
    X_edges_filtered[mask] = -1
    return X_edges_filtered

def filter_edges(m_ch_e, cy, cx, Gx, Gy, alpha, im_pre):
    """
    Edge detector find three types of edges: early wood transitions, latewood transitions and radial edges produced by
    cracks and fungi. Only early wood edges are the ones that forms the rings. In other to filter the other ones
    collineary with the ray direction is computed and filter depending on threshold (alpha).  Implements Algorithm 4 in
    the supplementary material
    @param m_ch_e: devernay curves in matrix format
    @param cy: pith y's coordinate
    @param cx: pith x's coordinate
    @param Gx: Gradient over x direction
    @param Gy: Gradient over y direction
    @param alpha: threshold filter
    @param im_pre: input image
    @return:
    - l_ch_f: filtered devernay curves
    """

    # Line 1 change reference axis
    Xb = change_reference_axis(m_ch_e, cy, cx)
    # Line 2 get normalized gradient at each edge
    G = get_gradient_vector_for_each_edge_pixel(m_ch_e, Gx, Gy)
    #G = np.vstack((Gx.flatten(), Gy.flatten())).T
    # Line 3 and 4 Normalize gradient and rays
    Xb_normalized = normalized_row_matrix(Xb.T)
    G_normalized = normalized_row_matrix(G)
    # Line 5 Compute angle between gradient and edges
    theta = compute_angle_between_gradient_and_edges(Xb_normalized, G_normalized)
    # Line 6 filter pixels by threshold
    X_edges_filtered = filter_edges_by_threshold(m_ch_e, theta, alpha)
    # Line 7 Convert masked pixel to object curve
    l_ch_f = convert_masked_pixels_to_curves(X_edges_filtered)
    # Line 8  Border disk is added as a curve
    border_curve = get_border_curve(im_pre, l_ch_f)
    # Line 9
    l_ch_f.append(border_curve)
    return l_ch_f

