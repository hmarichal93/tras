import time
import os
from pathlib import Path
from argparse import Namespace

from cross_section_tree_ring_detection.preprocessing import preprocessing
from cross_section_tree_ring_detection.canny_devernay_edge_detector import canny_devernay_edge_detector
from cross_section_tree_ring_detection.filter_edges import filter_edges
from cross_section_tree_ring_detection.sampling import sampling_edges
from cross_section_tree_ring_detection.merge_chains import merge_chains
from cross_section_tree_ring_detection.postprocessing import postprocessing
from cross_section_tree_ring_detection.utils import (chain_2_labelme_json, save_config,
                                                     saving_results)
from cross_section_tree_ring_detection.io import load_json

def TreeRingDetection(im_in, cy, cx, sigma, th_low, th_high, height, width, alpha, nr, mc,
                      debug= False, debug_image_input_path=None, debug_output_dir=None):
    """
    Method for delineating tree ring over pine cross sections images. Implements Algorithm 1 from the paper.
    @param im_in: segmented input image. Background must be white (255,255,255).
    @param cy: pith y's coordinate
    @param cx: pith x's coordinate
    @param sigma: Canny edge detector gausssian kernel parameter
    @param th_low: Low threshold on the module of the gradient. Canny edge detector parameter.
    @param th_high: High threshold on the module of the gradient. Canny edge detector parameter.
    @param height: img_height of the image after the resize step
    @param width: width of the image after the resize step
    @param alpha: Edge filtering parameter. Collinearity threshold
    @param nr: rays number
    @param mc: min ch_i length
    @param debug: boolean, debug parameter
    @param debug_image_input_path: Debug parameter. Path to input image. Used to write labelme json.
    @param debug_output_dir: Debug parameter. Output directory. Debug results are saved here.
    @return:
     - l_rings: Final results. Json file with rings coordinates.
     - im_pre: Debug Output. Preprocessing image results
     - m_ch_e: Debug Output. Intermediate results. Devernay curves in matrix format
     - l_ch_f: Debug Output. Intermediate results. Filtered Devernay curves
     - l_ch_s: Debug Output. Intermediate results. Sampled devernay curves as Chain objects
     - l_ch_s: Debug Output. Intermediate results. Chain lists after connect stage.
     - l_ch_p: Debug Output. Intermediate results. Chain lists after posprocessing stage.
    """
    to = time.time()
    args = Namespace(cy=cy, cx=cx, sigma=sigma, th_low=th_low, th_high=th_high,
                hsize=height, wsize=width, alpha=alpha, nr=nr,
                min_chain_length=mc, debug=debug, output_dir=debug_output_dir)

    save_config(args,  debug_output_dir)

    # Line 1 Preprocessing image. Algorithm 1 in the supplementary material. Image is  resized, converted to gray
    # scale and equalized
    im_pre, cy, cx = preprocessing(im_in, height, width, cy, cx)
    # Line 2 Edge detector module. Algorithm: A Sub-Pixel Edge Detector: an Implementation of the Canny/Devernay Algorithm,
    m_ch_e, gx, gy = canny_devernay_edge_detector(im_pre, sigma, th_low, th_high)
    # Line 3 Edge filtering module. Algorithm 4 in the supplementary material.
    l_ch_f = filter_edges(m_ch_e, cy, cx, gx, gy, alpha, im_pre)
    # Line 4 Sampling edges. Algorithm 6 in the supplementary material.
    l_ch_s = sampling_edges(l_ch_f, cy, cx, im_pre, mc, nr, debug=debug)
    # Line 5 Connect chains. Algorithm 7 in the supplementary material. Im_pre is used for debug purposes
    l_ch_c = merge_chains(l_ch_s, cy, cx, nr, debug, im_pre, debug_output_dir)
    # Line 6 Postprocessing chains. Algorithm 19 in the paper. Im_pre is used for debug purposes
    l_ch_p = postprocessing(l_ch_c, debug, debug_output_dir, im_pre)
    # Line 7
    debug_execution_time = time.time() - to
    l_rings = chain_2_labelme_json(l_ch_p, height, width, cy, cx, im_in, debug_image_input_path, debug_execution_time)

    # Line 8
    return im_in, im_pre, m_ch_e, l_ch_f, l_ch_s, l_ch_c, l_ch_p, l_rings
