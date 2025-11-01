import time

import cv2

from pathlib import Path
from cross_section_tree_ring_detection.cross_section_tree_ring_detection import (merge_chains,
                                                                                 postprocessing, chain_2_labelme_json)

from deep_cstrd.preprocessing import preprocessing
from deep_cstrd.model import deep_contour_detector


def DeepTreeRingDetection(im_in, cy, cx, height, width, alpha, nr, mc, weights_path, total_rotations,
                      debug= False, debug_image_input_path=None, debug_output_dir=None, tile_size=0,
                      prediction_map_threshold=0.2, batch_size=1, encoder='resnet18'):
    """
    Method for delineating tree ring in wood cross-section images.
    @param im_in: segmented input image. Background must be white (255,255,255).
    @param cy: pith y's coordinate
    @param cx: pith x's coordinate
    @param height: img_height of the image after the resize step
    @param width: width of the image after the resize step
    @param alpha: Edge filtering parameter. Collinearity threshold
    @param nr: rays number
    @param mc: min ch_i length
    @param total_rotations: inference image rotations number
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

    # Line 1 Preprocessing image.
    im_pre, cy, cx = preprocessing(im_in, height, width, cy, cx)
    # Line 2 Edge detector module.
    l_ch_s = deep_contour_detector(im_pre, weights_path=weights_path, output_dir=Path(debug_output_dir),
                                           cy=cy, cx=cx, total_rotations=total_rotations, debug=debug,
                                           tile_size=tile_size,
                                           prediction_map_threshold = prediction_map_threshold,
                                           alpha=alpha, mc=mc, nr=nr, batch_size=batch_size, encoder=encoder
                                   )
    #conver im_pre to gray scale
    im_in = cv2.cvtColor(im_in, cv2.COLOR_RGB2BGR)
    im_pre = cv2.cvtColor(im_pre, cv2.COLOR_BGR2GRAY)

    # Line 5 Connect chains. Algorithm 7 in the supplementary material. Im_pre is used for debug purposes
    l_ch_c = merge_chains(l_ch_s, cy, cx, nr, False, im_pre, debug_output_dir)
    # Line 6 Postprocessing chains. Algorithm 19 in the paper. Im_pre is used for debug purposes
    l_ch_p = postprocessing(l_ch_c, False, debug_output_dir, im_pre)
    # Line 7
    debug_execution_time = time.time() - to
    l_rings = chain_2_labelme_json(l_ch_p, im_pre.shape[0], im_pre.shape[1], cy, cx, im_in, debug_image_input_path, debug_execution_time)

    return im_in, im_pre, [], [], l_ch_s, l_ch_c, l_ch_p, l_rings




