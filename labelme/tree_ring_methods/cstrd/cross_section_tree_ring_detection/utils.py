#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Copyright (c) 2023 Author(s) Henry Marichal (hmarichal93@gmail.com

This program is free software: you can redistribute it and/or modify it under the terms of the GNU Affero General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License along with this program. If not, see <http://www.gnu.org/licenses/>.
"""

from pathlib import Path
from typing import List
import pickle

from cross_section_tree_ring_detection.io import write_json, load_config
import cross_section_tree_ring_detection.chain as ch
from cross_section_tree_ring_detection.preprocessing import resize



def save_config(args, output_dir, repo_path="./"):

    config = load_config()
    config['result_path'] = str(output_dir)

    if args.nr:
        config['nr'] = args.nr
    if args.hsize and args.wsize:
        if args.hsize > 0 and args.wsize > 0:
            config['resize'] = [args.hsize, args.wsize]
    if args.min_chain_length:
        config["min_chain_length"] = args.min_chain_length
    if args.alpha:
        config["alpha"] = args.alpha
    if args.sigma:
        config['sigma'] = args.sigma
    if args.th_high:
        config['th_high'] = args.th_high
    if args.th_low:
        config['th_low'] = args.th_low
    if args.debug:
        config['debug'] = True

    config['devernay_path'] = str(Path(repo_path) / "externas/devernay_1.0")
    write_json(config, Path(repo_path) / 'config/general.json')
    return 0
def saving_results( res, output_dir, save_imgs=True):
    im_seg, im_pre, ch_e, ch_f, ch_s, ch_c, ch_p, rings = res
    write_json(rings, f"{output_dir}/labelme.json")
    if not save_imgs:
        return
    M,N,_ = im_seg.shape
    M_n, N_n = im_pre.shape
    if M != M_n:
        im_seg,_,_ = resize(im_seg, M_n, N_n)
    ch.visualize_chains_over_image(img=im_seg, filename=f"{output_dir}/segmentation.png")
    ch.visualize_chains_over_image(img=im_pre, filename=f"{output_dir}/preprocessing.png")
    ch.visualize_chains_over_image(img=im_pre, filename=f"{output_dir}/edges.png", devernay=ch_e)
    ch.visualize_chains_over_image(img=im_pre, filename=f"{output_dir}/filter.png", filter=ch_f)
    ch.visualize_chains_over_image(chain_list=ch_s, img=im_seg, filename=f"{output_dir}/chains.png")
    ch.visualize_chains_over_image(chain_list=ch_c, img=im_seg, filename=f"{output_dir}/connect.png")
    ch.visualize_chains_over_image(chain_list=ch_p, img=im_seg, filename=f"{output_dir}/postprocessing.png")
    ch.visualize_chains_over_image(chain_list=[chain for chain in ch_p if chain.is_closed() and chain.type not in
                                               [ ch.TypeChains.center, ch.TypeChains.border]], img=im_seg, filename=f"{output_dir}/output.png")


    return


def chain_2_labelme_json(chain_list: List[ch.Chain], image_height, image_width, cy, cx, img_orig, image_path,
                         exec_time):
    """
    Converting ch_i list object to labelme format. This format is used to store the coordinates of the rings at the image
    original resolution
    @param chain_list: ch_i list
    @param image_path: image input path
    @param image_height: image hegith
    @param image_width: image width_output
    @param img_orig: input image
    @param exec_time: method execution time
    @param cy: pith y's coordinate
    @param cx: pith x's coordinate
    @return:
    - labelme_json: json in labelme format. Ring coordinates are stored here.
    """
    init_height, init_width, _ = img_orig.shape
    completed_chains = [chain for chain in chain_list if chain.is_closed() and chain.type not in [ch.TypeChains.center, ch.TypeChains.border]]



    width_cte = init_width / image_width if image_width is not 0 else 1
    height_cte = init_height / image_height if image_height is not 0 else 1
    labelme_json = {"imagePath":image_path, "imageHeight":None,
                    "imageWidth":None, "version":"5.0.1",
                    "flags":{},"shapes":[],"imageData": None, 'exec_time(s)':exec_time,'center':[cy*height_cte, cx*width_cte]}
    for idx, chain in enumerate(completed_chains):
        ring = {"label":str(idx+1)}
        ring["points"] = [[node.x*width_cte,node.y*height_cte] for node in chain.l_nodes]
        ring["shape_type"]="polygon"
        ring["flags"]={}
        labelme_json["shapes"].append(ring)

    return labelme_json

def load_pickle(file_path):
    """
    Load pickle file
    @param file_path: pickle file path
    @return: object stored in the pickle file
    """
    with open(file_path, 'rb') as f:
        return pickle.load(f)


def write_pickle(file_path, obj):
    """
    Write object to pickle file
    @param file_path: pickle file path
    @param obj: object to store
    """
    with open(file_path, 'wb') as f:
        pickle.dump(obj, f)



