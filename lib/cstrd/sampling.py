"""
Copyright (c) 2023 Author(s) Henry Marichal (hmarichal93@gmail.com

This program is free software: you can redistribute it and/or modify it under the terms of the GNU Affero General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License along with this program. If not, see <http://www.gnu.org/licenses/>.
"""
import numpy as np
import time
from shapely.geometry import Point
from shapely.geometry.linestring import LineString
from lib.drawing import Drawing
import cv2

from lib.chain import Node, euclidean_distance, get_node_from_list_by_angle, Chain, TypeChains
from lib.io import load_json

class Ray(LineString):
    def __init__(self, direction, center, M, N):
        self.direction = direction
        self.border = self._image_border_radii_intersection(direction, center, M, N)
        super().__init__([center, self.border])

    @staticmethod
    def _image_border_radii_intersection(theta, origin, M, N):
        degree_to_radians = np.pi / 180
        theta = theta % 360
        yc, xc = origin
        if 0 <= theta < 45:
            ye = M - 1
            xe = np.tan(theta * degree_to_radians) * (M - 1 - yc) + xc

        elif 45 <= theta < 90:
            xe = N - 1
            ye = np.tan((90 - theta) * degree_to_radians) * (N - 1 - xc) + yc

        elif 90 <= theta < 135:
            xe = N - 1
            ye = yc - np.tan((theta - 90) * degree_to_radians) * (xe - xc)

        elif 135 <= theta < 180:
            ye = 0
            xe = np.tan((180 - theta) * degree_to_radians) * (yc) + xc

        elif 180 <= theta < 225:
            ye = 0
            xe = xc - np.tan((theta - 180) * degree_to_radians) * (yc)

        elif 225 <= theta < 270:
            xe = 0
            ye = yc - np.tan((270 - theta) * degree_to_radians) * (xc)

        elif 270 <= theta < 315:
            xe = 0
            ye = np.tan((theta - 270) * degree_to_radians) * (xc) + yc

        elif 315 <= theta < 360:
            ye = M - 1
            xe = xc - np.tan((360 - theta) * degree_to_radians) * (ye - yc)

        else:
            raise 'Error'

        return (ye, xe)


def build_rays(Nr, M, N, center):
    """

    @param Nr: total rays
    @param N: widht image
    @param M: height_output image
    @param center: (y,x)
    @return: list_position rays
    """
    angles_range = np.arange(0, 360, 360 / Nr)
    radii_list = [Ray(direction, center, M, N) for direction in angles_range]
    return radii_list


def get_coordinates_from_intersection(inter):
    """Shapely intersection formating"""
    if 'MULTI' in inter.wkt:
        inter = inter[0]

    if type(inter) == Point:
        y, x = inter.xy
        y,x = y[0], x[0]

    elif 'LINESTRING' in inter.wkt:
        y, x = inter.xy
        y,x = y[1], x[1]

    elif 'STRING' in inter.wkt:
        y, x = inter.coords.xy
        y,x = y[0], x[0]

    else:
        raise

    return y, x


def compute_intersection(l_rays, curve, chain_id, center):
    """
    Compute intersection between rays and devernay curve
    @param l_rays: rays list
    @param curve: devernay curve
    @param chain_id: chain id
    @param center: disk image center
    @return: nodes list
    """
    l_curve_nodes = []
    for radii in l_rays:
        inter = radii.intersection(curve)
        if not inter.is_empty:
            try:
                y, x = get_coordinates_from_intersection(inter)
            except NotImplementedError:
                continue
            i, j = np.array(y), np.array(x)
            params = {'y': i, 'x': j, 'angle': int(radii.direction), 'radial_distance':
                euclidean_distance([i, j], center), 'chain_id': chain_id}

            dot = Node(**params)
            if dot not in l_curve_nodes and get_node_from_list_by_angle(l_curve_nodes, radii.direction) is None:
                l_curve_nodes.append(dot)

    return l_curve_nodes


def intersections_between_rays_and_devernay_curves(center, l_rays, l_curves, min_chain_length, nr, height, width):
    """
    Compute chains sampling devernay curves.  Sampling is made finding the intersection
    between rays and devernay curves. A chain is a list of nodes. A node is a point in the image with the following
    attributes: x, y, angle, radial_distance, chain_id. The chain_id is the id of the chain to which the node belongs.
    @param center: pith center
    @param l_rays: ray list
    @param l_curves: devernay curves list
    @param min_chain_length: minimum length of chain
    @param nr: number of rays
    @param height: image height
    @param width: image widht
    @return: nodes list and chain list
    """
    l_chain, l_nodes = [], []
    for idx, curve in enumerate(l_curves):
        chain_id = len(l_chain)
        l_curve_nodes = compute_intersection(l_rays, curve, chain_id, center)

        if len(l_curve_nodes) < min_chain_length:
            continue

        l_nodes += l_curve_nodes
        chain = Chain(chain_id, nr, center=center, img_height=height, img_width=width)
        chain.add_nodes_list(l_curve_nodes)
        l_chain.append(chain)

    # Devernay border curve is the last element of the list l_curves.
    l_chain[-1].type = TypeChains.border

    return l_nodes, l_chain


def generate_virtual_center_chain(cy, cx, nr, chains_list, nodes_list, height, width):
    """
    Generate virtual center chain. This chain is used to connect the other chains.
    :param cy: y's center coordinate in pixel.
    :param cx: x's center coordinate in pixel
    :param nr: number of rays
    :param chains_list: chain list
    :param nodes_list: node list
    :param height: image height
    :param width: image width
    :return: chain is added to the chain list and nodes are added to the nodes list
    """
    chain_id = len(chains_list) - 1
    center_list = [Node(**{'x': cx, 'y': cy, 'angle': angle, 'radial_distance': 0,
                           'chain_id': chain_id}) for angle in np.arange(0, 360, 360 / nr)]
    nodes_list += center_list

    chain = Chain(chain_id, nr, center=chains_list[0].center, img_height=height, img_width=width,
                  type=TypeChains.center)
    chain.add_nodes_list(center_list)

    chains_list.append(chain)

    # set border ch_i as the last element of the list
    chains_list[-2].change_id(len(chains_list) - 1)

    return 0


def draw_ray_curve_and_intersections(dots_lists, rays_list, curves_list, img_draw, filename):

    for ray in rays_list:
        img_draw = Drawing.radii(ray, img_draw)

    for curve in curves_list:
        img_draw = Drawing.curve(curve, img_draw)

    for dot in dots_lists:
        img_draw = Drawing.intersection(dot, img_draw)

    cv2.imwrite(filename, img_draw)

def add_gt_rings_as_chain(chains_list, nodes_list, gt_ring_json, height, width, cy, cx, include_them_in_output=False):
    """
    Add gt rings as a chain
    @param chains_list: chain list
    @param nodes_list: node list
    @param gt_ring_json: gt rings json
    @param height: image height
    @param width: image width
    @param cy: pith y's coordinate
    @param cx: pith x's coordinate
    @param include_them_in_output: boolean, include them in the output json file
    @return:
    """
    if gt_ring_json is None:
        return

    #load json
    gt_ring_labels = load_json(gt_ring_json)
    chain_id = len(chains_list)
    gt_rings = gt_ring_labels['shapes']
    for idx, ring in enumerate(gt_rings):

        nodes = [Node(**{'x': pix[0], 'y': pix[1], 'angle': idx, 'radial_distance': euclidean_distance([pix[1], pix[0]], [cy,cx]),
                         'chain_id': chain_id}) for idx, pix in enumerate(ring['points'])]
        #last node is the first node
        nodes.pop(-1)
        # add to list
        nodes_list += nodes
        chain = Chain(chain_id, chains_list[0].Nr, center=[cy, cx], img_height=height, img_width=width,
                      type=TypeChains.gt_ring if not include_them_in_output else TypeChains.normal)

        chain.add_nodes_list(nodes)
        chains_list.append(chain)
        chain_id += 1

    return

def sampling_edges(l_ch_f, cy, cx, im_pre, min_chain_length, nr, debug=False, gt_ring_json = None,
                   include_gt_rings_in_output=False):
    """
    Devernay curves are sampled using the rays directions. Implements Algoritm 7 in the paper.
    @param l_ch_f:  edges devernay curves
    @param cy: pith y's coordinate
    @param cx: pith x's coordinate
    @param im_pre: input image
    @param nr: total ray number
    @param min_chain_length:  minumim chain length
    @param debug: debugging flag
    @return:
    - l_ch_s: sampled edges curves. List of chain objects
    - l_nodes_s: nodes list.
    """
    # Line 1
    height, width = im_pre.shape
    # Line 2
    l_rays = build_rays(nr, height, width, [cy, cx])
    # Line 3
    l_nodes_s, l_ch_s = intersections_between_rays_and_devernay_curves([cy, cx], l_rays, l_ch_f, min_chain_length, nr,
                                                                       height, width)

    # Add gt rings as a chain
    add_gt_rings_as_chain(l_ch_s, l_nodes_s, gt_ring_json, height, width, cy, cx,
                          include_them_in_output=include_gt_rings_in_output)

    # Line 4
    generate_virtual_center_chain(cy, cx, nr, l_ch_s, l_nodes_s, height, width)

    # Debug purposes, not illustrated in the paper
    if debug:
        img_draw = np.zeros((im_pre.shape[0], im_pre.shape[1], 3))
        draw_ray_curve_and_intersections(l_nodes_s, l_rays, l_ch_f, img_draw, './dots_curve_and_rays.png')

    # Line 5
    return l_ch_s, l_nodes_s
