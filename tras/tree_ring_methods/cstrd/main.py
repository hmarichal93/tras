#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Copyright (c) 2023 Author(s) Henry Marichal (hmarichal93@gmail.com)

This program is free software: you can redistribute it and/or modify it under the terms of the GNU Affero General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License along with this program. If not, see <http://www.gnu.org/licenses/>.
"""
from pathlib import Path


from cross_section_tree_ring_detection.cross_section_tree_ring_detection import TreeRingDetection
from cross_section_tree_ring_detection.utils import save_config, saving_results
from cross_section_tree_ring_detection.io import load_image


def main(args):
    save_config(args, args.output_dir)
    if not Path(args.input).exists():
        raise FileNotFoundError(f"Input image {args.input} not found")

    im_in = load_image(args.input)
    Path(args.output_dir).mkdir(exist_ok=True, parents=True)

    res = TreeRingDetection(im_in, args.cy, args.cx, args.sigma, args.th_low, args.th_high, args.hsize, args.wsize,
                            args.alpha, args.nr, args.min_chain_length, args.debug, args.input, args.output_dir)
    saving_results(res, args.output_dir, args.save_imgs)


    return 0

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--cy", type=int, required=True)
    parser.add_argument("--cx", type=int, required=True)
    parser.add_argument("--root", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--save_imgs", type=str, required=False)
    parser.add_argument("--sigma", type=float, required=False,default=3)
    parser.add_argument("--nr", type=int, required=False,default=360)
    parser.add_argument("--hsize", type=int, required=False, default=0)
    parser.add_argument("--wsize", type=int, required=False, default=0)
    parser.add_argument("--alpha", type=int, required=False, default=30)
    parser.add_argument("--th_high", type=float, required=False, default=20)
    parser.add_argument("--th_low", type=float, required=False, default=5)
    parser.add_argument("--min_chain_length", type=int, required=False, default=2)
    parser.add_argument("--debug", type=int, required=False)

    args = parser.parse_args()

    main(args)








