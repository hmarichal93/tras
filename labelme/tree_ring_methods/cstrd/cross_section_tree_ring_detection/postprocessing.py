"""
Copyright (c) 2023 Author(s) Henry Marichal (hmarichal93@gmail.com

This program is free software: you can redistribute it and/or modify it under the terms of the GNU Affero General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License along with this program. If not, see <http://www.gnu.org/licenses/>.
"""
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import LineString, Point, Polygon
import cv2
from typing import List


import cross_section_tree_ring_detection.chain as ch
from cross_section_tree_ring_detection.drawing import Drawing, Color
from cross_section_tree_ring_detection.basic_properties import similarity_conditions
from cross_section_tree_ring_detection.merge_chains import (SystemStatus, select_closest_chain, merge_two_chains, close_chain)


def build_boundary_poly(outward_ring, inward_ring):
    """
    Convert shaeply poly to region poly
    @rtype: object
    @param outward_ring: outward shapely polygon
    @param inward_ring: inward shapely poylgon
    @return: poly region
    """
    if outward_ring is None and inward_ring is None:
        return None
    if outward_ring is not None and inward_ring is not None:
        x, y = outward_ring.exterior.coords.xy
        pts_ext = [[j, i] for i, j in zip(y, x)]
        x, y = inward_ring.exterior.coords.xy
        pts_int = [[j, i] for i, j in zip(y, x)]
        poly = Polygon(pts_ext, [pts_int])

    else:
        if outward_ring is None:
            x, y = inward_ring.exterior.coords.xy
        else:
            x, y = outward_ring.exterior.coords.xy

        pts_ext = [[j, i] for i, j in zip(y, x)]
        poly = Polygon(pts_ext)

    return poly


def search_for_polygons_within_region(shapley_incomplete_chain, outward_ring, inward_ring):
    """
    Search for shapely polygon inside region delimitad for outward and inward rings
    @param shapley_incomplete_chain: shapely polygon chains not closed. Not nr nodes
    @param outward_ring: shapely polygon chain closed. Nr nodes
    @param inward_ring: shapely polygon chain closed. Nr nodes
    @return:
    """
    poly = build_boundary_poly(outward_ring, inward_ring)
    if poly is None:
        return []
    contains = np.vectorize(lambda p: poly.contains(Point(p)), signature='(n)->()')
    shapely_inward_chains_subset = []
    for cadena in shapley_incomplete_chain:
        x, y = cadena.xy
        pts = [[i, j] for i, j in zip(y, x)]
        if len(pts) == 0:
            continue
        try:
            vector = contains(np.array(pts))
        except Exception as e:
            continue
        if outward_ring is not None:
            if vector.sum() == vector.shape[0]:
                shapely_inward_chains_subset.append(cadena)
        else:
            if vector.sum() == 0:
                shapely_inward_chains_subset.append(cadena)

    return shapely_inward_chains_subset


def from_shapely_to_chain(uncompleted_shapely_chain, uncomplete_chain, shapely_inward_chains_subset):
    inward_chain_subset = [uncomplete_chain[uncompleted_shapely_chain.index(cad_shapely)]
                           for cad_shapely in shapely_inward_chains_subset]
    inward_chain_subset.sort(key=lambda x: x.size, reverse=True)
    return inward_chain_subset


class Ring(Polygon):
    def __init__(self, chain: ch.Chain, id: int):
        lista_pts = [[node.x, node.y] for node in chain.sort_dots()]
        self.id = id
        super(self.__class__, self).__init__(lista_pts)

    def __str__(self):
        return f'Ring {self.id}'

    def __repr__(self):
        return f'Ring {self.id}'

    def draw(self, image):
        x, y = self.exterior.coords.xy
        lista_pts = [[i, j] for i, j in zip(y, x)]
        pts = np.array(lista_pts,
                       np.int32)

        pts = pts.reshape((-1, 1, 2))
        is_closed = True
        # Blue color in BGR
        color = (255, 0, 0)
        # Line thickness of 2 px
        thickness = 1
        # Using cv2.polylines() method
        # Draw a Blue polygon with
        # thickness of 1 px
        image = cv2.polylines(image, [pts],
                              is_closed, color, thickness)

        image = Drawing.put_text(f'{self.id}', image, (int(y[0]), int(x[0])))

        return image


class ChainsBag:
    def __init__(self, inward_chain_set):
        """
        Iterate over chains in a region.
        @param inward_chain_set: set of inward chains inside region sorted by size
        """
        self.chain_set = inward_chain_set
        self.chains_id_already_selected = []

    def update_chain_list(self, ch_k):
        if ch_k is not None and ch_k.parent is not None and ch_k.parent in self.chain_set:
            #ch_k was splitted originally
            # remove parent chain
            self.chain_set.remove(ch_k.parent)
            sub_1, sub_2 = split_chain(ch_k.parent, ch_k.split_node)
            other_chain = sub_1 if ch_k == sub_2 else sub_2
            # add not connected subchain
            if other_chain not in self.chain_set and other_chain is not None:
                other_chain.change_id(ch_k.parent.id)
                other_chain.parent = None
                self.chain_set.append(other_chain)

        if ch_k.parent is None:
            #ch_k was not splitted
            self.chain_set.remove(ch_k)

        nodes_list = ch.get_nodes_from_chain_list(self.chain_set)
        return nodes_list, self.chain_set




    def get_next_chain(self, ch_j=None, ch_k=None):
        if ch_k is not None:
            if ch_j.size < ch_j.Nr:
                #ch_k was merge with ch_j during this iteration
                return ch_j
            else:
                #ch_k was merge with ch_j during this iteration. However a chain was completed. Need to exit loop
                return None

        next = None
        for chain in self.chain_set:
            if chain.id not in self.chains_id_already_selected:
                next = chain
                self.chains_id_already_selected.append(next.id)
                break

        return next


def select_support_chain(outward_ring, inward_ring, endpoint):
    """
    Select the closest ring to the ch_j endpoint. Over endpoint ray direction, the support chain with the smallest
    distance between nodes is selected
    @param outward_ring: outward support chain
    @param inward_ring:  inward support chain
    @param endpoint: ch_j endpoint
    @return: closest support chain
    """
    chains_in_radial_direction = []
    if outward_ring is not None:
        chains_in_radial_direction.append(outward_ring)

    if inward_ring is not None:
        chains_in_radial_direction.append(inward_ring)

    dot_list_in_radial_direction = ch.get_closest_dots_to_angle_on_radial_direction_sorted_by_ascending_distance_to_center(
        chains_in_radial_direction, endpoint.angle)

    distance = [ch.euclidean_distance_between_nodes(endpoint, completed_chain_node) for completed_chain_node in
                dot_list_in_radial_direction]

    if len(distance) < 2:
        support_chain = outward_ring if outward_ring is not None else inward_ring

    else:
        support_chain = ch.get_chain_from_list_by_id(chains_in_radial_direction,
                                                     dot_list_in_radial_direction[np.argmin(distance)].chain_id)

    return support_chain


def exist_angular_intersection_with_src_chain(chain: ch.Chain, src_chain_angular_domain: List[int]):
    domain = [node.angle for node in chain.l_nodes]
    if len(np.intersect1d(domain, src_chain_angular_domain)) == 0:
        return False
    return True


def angular_domain_intersection_higher_than_threshold(src_chain_angular_domain: List[int], inter_chain: ch.Chain,
                                                      intersection_threshold: int = 45):
    """
    Check if intersecting angular domain between two chains is higher than a threshold
    @param src_chain_angular_domain: src chain
    @param inter_chain: another chain
    @param intersection_threshold: intersection threshold
    @return: boolean value
    """
    inter_domain = [node.angle for node in inter_chain.l_nodes]
    inter = np.intersect1d(inter_domain, src_chain_angular_domain)
    rays_length = len(inter)
    angle_length = rays_length * 360 / inter_chain.Nr
    if (len(inter) >= len(src_chain_angular_domain)) or (angle_length > intersection_threshold):
        return True
    else:
        return False


def split_chain(chain: ch.Chain, node: ch.Node, id_new=10000000):
    """
    Split a chain in two chains
    @param chain: Parent chain. Chain to be split
    @param node: node element where the chain will be split
    @param id_new: new id
    @return: tuple of child chains
    """
    node_list = chain.sort_dots()
    idx_split = node_list.index(node)
    ch1_node_list = [ch.copy_node(node) for node in node_list[:idx_split]]
    if idx_split < len(node_list) - 1:
        ch2_node_list = [ch.copy_node(node) for node in node_list[idx_split + 1:]]
    else:
        ch2_node_list = []

    if len(ch1_node_list) > 1:
        ch1_sub = ch.Chain(id_new, chain.Nr, center=chain.center, img_height=chain.img_height,
                           img_width=chain.img_width, parent=chain, split_node=node)
        for node_ch in ch1_node_list:
            node_ch.chain_id = ch1_sub.id

        ch1_sub.add_nodes_list(ch1_node_list)
    else:
        ch1_sub = None

    if len(ch2_node_list) > 1:
        ch2_sub = ch.Chain(id_new, chain.Nr, center=chain.center, img_height=chain.img_height,
                           img_width=chain.img_width, parent=chain, split_node=node)
        for node_ch in ch2_node_list:
            node_ch.chain_id = ch2_sub.id
        ch2_sub.add_nodes_list(ch2_node_list)
    else:
        ch2_sub = None

    return (ch1_sub, ch2_sub)


def select_no_intersection_chain_at_endpoint(ch1_sub: ch.Chain, ch2_sub: ch.Chain, src_chain: ch.Chain,
                                             ray_direction: int, total_nodes=10):
    """
    Select the chain that does not intersect with the source chain at endpoint
    @param ch1_sub: child chain 1
    @param ch2_sub:  child chain 2
    @param src_chain: source chain, ch_i
    @param ray_direction: ray direction source chain
    @param total_nodes:
    @return: chain that does not intersect with the ch1_sub at endpoint
    """
    endpoint = ch.EndPoints.A if ray_direction == src_chain.extA.angle else ch.EndPoints.B
    direction = ch.ClockDirection.clockwise if endpoint == ch.EndPoints.B else ch.ClockDirection.anti_clockwise
    nodes_neighbourhood = src_chain.sort_dots(direction=direction)[:total_nodes]
    src_nodes = ch.get_nodes_angles_from_list_nodes(nodes_neighbourhood)
    if ch1_sub is None and ch2_sub is None:
        return None
    if ch1_sub is None and ch2_sub is not None:
        domain2 = ch.get_nodes_angles_from_list_nodes(ch2_sub.l_nodes) if ch2_sub.size > 0 else src_nodes
        if np.intersect1d(domain2, src_nodes).shape[0] == 0:
            return ch2_sub
        else:
            return None
    if ch2_sub is None and ch1_sub is not None:
        domain1 = ch.get_nodes_angles_from_list_nodes(ch1_sub.l_nodes) if ch1_sub.size > 0 else src_nodes
        if np.intersect1d(domain1, src_nodes).shape[0] == 0:
            return ch1_sub
        else:
            return None

    domain2 = ch.get_nodes_angles_from_list_nodes(ch2_sub.l_nodes) if ch2_sub.size > 0 else src_nodes
    domain1 = ch.get_nodes_angles_from_list_nodes(ch1_sub.l_nodes) if ch1_sub.size > 0 else src_nodes
    if np.intersect1d(domain1, src_nodes).shape[0] == 0:
        return ch1_sub
    elif np.intersect1d(domain2, src_nodes).shape[0] == 0:
        return ch2_sub
    else:
        return None


def split_intersecting_chains(direction, l_filtered_chains, ch_j, debug_params = None, save_path = None):
    """
    Split intersecting chains.
    @param direction: endpoint direction for split chains
    @param l_filtered_chains: list of chains to be split
    @param ch_j: source chain
    @return: split chains list
    """
    img, iteration, debug = debug_params if debug_params is not None else (None, [0], False)

    l_search_chains = []
    for inter_chain in l_filtered_chains:
        # Line 3
        split_node = inter_chain.get_node_by_angle(direction)
        if split_node is None:
            # It is not possible to split the chain due to split_node is None. Continue to next chain
            continue

        if debug:
            ch.visualize_selected_ch_and_chains_over_image_(
                [inter_chain, ch_j], [], img,
                f'{save_path}/{iteration[0]}_split_intersecting_chains_{ch_j.label_id}_{inter_chain.label_id}_1.png')
            iteration[0] += 1

        # Line 4
        sub_ch1, sub_ch2 = split_chain(inter_chain, split_node)
        if debug:
            l_sub_chain = []
            if sub_ch1 is not None:
                l_sub_chain.append(sub_ch1)
            if sub_ch2 is not None:
                l_sub_chain.append(sub_ch2)
            ch.visualize_selected_ch_and_chains_over_image_(
                [ch_j] + l_sub_chain, [inter_chain], img,
                f'{save_path}/{iteration[0]}_split_intersecting_chains_{ch_j.label_id}_{inter_chain.label_id}_2.png')
            iteration[0] += 1
        # Line 5 Found what ch_i intersect the longest one
        ch_k = select_no_intersection_chain_at_endpoint(sub_ch1, sub_ch2, ch_j, direction)
        if ch_k is None:
            # There is not chain that does not intersect with ch_j at endpoint. Continue to next chain
            continue

        if debug:
            ch.visualize_selected_ch_and_chains_over_image_(
                [ch_j, ch_k], [inter_chain], img,
                f'{save_path}/{iteration[0]}_split_intersecting_chains_{ch_j.label_id}_{inter_chain.label_id}_3.png')
            iteration[0] += 1


        # Line 11
        ch_k.change_id(inter_chain.id)
        ch_k.label_id = inter_chain.label_id

        # Line 12
        l_search_chains.append(ch_k)

    # Line 12
    return l_search_chains



def split_intersecting_chain_in_other_endpoint(endpoint, ch_j, l_within_chains, l_within_nodes, l_candidates):
    """
    Split intersecting chain in other endpoint
    @param endpoint:
    @param ch_j: source chain
    @param l_within_chains: chains within the region
    @param l_within_nodes: nodes within the region
    @param l_candidates: chains to be split
    @return:
    """
    # Line 1 Get the chains that intersect in the other endpoint within chains_in_region
    node_other_endpoint = ch_j.extB if endpoint == ch.EndPoints.A else ch_j.extA
    direction = node_other_endpoint.angle
    node_direction = [node for node in l_within_nodes if
                      ((node.angle == direction) and not (node.chain_id == ch_j.id))]
    direction_cad_id = set([node.chain_id for node in node_direction])
    intersect_chain_id = [cad.id for cad in l_within_chains if
                          cad.id in direction_cad_id]

    intersecting_chains_in_other_endpoint = [chain for chain in l_candidates if
                                             chain.id in intersect_chain_id]
    l_candidates = [chain for chain in l_candidates if
                        chain not in intersecting_chains_in_other_endpoint]

    # Line 3 Split intersecting chains in other endpoint
    chain_search_set_in_other_endpoint = split_intersecting_chains(direction, intersecting_chains_in_other_endpoint,
                                                                   ch_j)

    # Line 4 add the chains that are not intersected with the ch_j and are not far from the endpoint
    l_candidates += chain_search_set_in_other_endpoint

    return l_candidates


def filter_no_intersected_chain_far(no_intersecting_chains, src_chain, endpoint, neighbourhood_size=45):
    """
    Filter the chains that are not intersected with the ch_j and are far from the endpoint
    @param no_intersecting_chains: list of no intersecting chain with src chain
    @param src_chain: source chain
    @param endpoint: endpoint of source chain
    @param neighbourhood_size: angular neighbourhood size in degrees
    @return: list of chains that are not intersected with the ch_j and are not far from the endpoint
    """
    closest_chains_set = []
    for chain in no_intersecting_chains:
        distance = ch.angular_distance_between_chains(src_chain, chain, endpoint)
        if distance < neighbourhood_size:
            closest_chains_set.append((distance, chain))

    # sort by proximity to endpoint and return
    closest_chains_set.sort(key=lambda x: x[0])
    no_intersecting_chain_set = [chain for distance, chain in
                                 closest_chains_set]

    return no_intersecting_chain_set


def add_chains_that_intersect_in_other_endpoint(within_chain_set, no_intersections_chain, search_chain_set, src_chain,
                                                neighbourhood_size, endpoint):
    """
    Add chains that intersect in other endpoint
    @param within_chain_set: chains in region
    @param no_intersections_chain: chains that do not intersect with ch_j.  ch_i that can be connected by this
    endpoint is added
    @param search_chain_set: candidate chains to be connected
    @param src_chain: source chan
    @param neighbourhood_size: neighbourhood size in degrees
    @param endpoint: source chain endpoint
    @return:
    """
    for in_chain in within_chain_set:
        if in_chain in no_intersections_chain + search_chain_set:
            continue
        if ch.angular_distance_between_chains(src_chain, in_chain, endpoint) < neighbourhood_size:
            endpoint_in_chain = in_chain.extA if ch.EndPoints.A == endpoint else in_chain.extB
            exist_intersection_in_other_endpoint = src_chain.get_node_by_angle(endpoint_in_chain.angle) is not None
            if exist_intersection_in_other_endpoint:
                # Check that there no intersection between both src endpoints
                sorted_order = ch.ClockDirection.clockwise if endpoint == ch.EndPoints.A else ch.ClockDirection.anti_clockwise
                in_chain_neighbourhood_nodes = in_chain.sort_dots(sorted_order)[:neighbourhood_size]
                if src_chain.get_node_by_angle(in_chain_neighbourhood_nodes[0].angle) is None:
                    search_chain_set.append(in_chain)
    return search_chain_set


def get_chains_that_satisfy_similarity_conditions(state, support_chain, src_chain, search_chain_set,
                                                  endpoint):
    """
    Get chains that satisfy similarity conditions
    @param state: debugging variable
    @param support_chain: support chain
    @param src_chain: source chain
    @param search_chain_set: list of candidate chains
    @param endpoint: source chain endpoint
    @return: list of chain that satisfy similarity conditions
    """
    candidate_chains = []
    radial_distance_candidate_chains = []
    candidate_chain_euclidean_distance = []
    candidate_chain_idx = 0
    while True:
        if len(search_chain_set) <= candidate_chain_idx:
            break

        candidate_chain = search_chain_set[candidate_chain_idx]
        candidate_chain_idx += 1

        check_pass, distribution_distance = similarity_conditions(state=state, th_radial_tolerance=0.2,
                                                                  th_distribution_size=3, th_regular_derivative=2,
                                                                  derivative_from_center=False, ch_i=support_chain,
                                                                  ch_j=src_chain, candidate_chain=candidate_chain,
                                                                  endpoint=endpoint, check_overlapping=False)
        if check_pass:
            candidate_chains.append(candidate_chain)
            radial_distance_candidate_chains.append(distribution_distance)

            endpoint_src = src_chain.extA if endpoint == ch.EndPoints.A else src_chain.extB
            endpoint_candidate_chain = candidate_chain.extB if endpoint == ch.EndPoints.A else candidate_chain.extA
            endpoint_distance = ch.euclidean_distance_between_nodes(endpoint_src, endpoint_candidate_chain)
            candidate_chain_euclidean_distance.append(endpoint_distance)

    return candidate_chain_euclidean_distance, radial_distance_candidate_chains, candidate_chains


def select_closest_candidate_chain(l_candidate_chains, l_candidate_chain_euclidean_distance,
                                   l_radial_distance_candidate_chains, l_within_chains, aux_chain):
    """
    Select closest chain by euclidean distance to ch_j chain.
    @param l_candidate_chains: list of chain candidate
    @param l_candidate_chain_euclidean_distance: euclidean distance of list of candidate chains
    @param l_radial_distance_candidate_chains: radial distance of list of candidate chains
    @param l_within_chains: full list of chain within region
    @param aux_chain: check if the candidate chain is the same as aux_chain
    @return: closest candidate chain by euclidean distance and radial distance to ch_j chain.
    """
    candidate_chain = None
    diff = -1
    if len(l_candidate_chains) > 0:
        candidate_chain = l_candidate_chains[np.argmin(l_candidate_chain_euclidean_distance)]
        diff = np.min(l_radial_distance_candidate_chains)

    if aux_chain is not None and candidate_chain == aux_chain:
        # if candidate_chain in chains_in_region:
        #     chains_in_region.remove(aux_chain)
        candidate_chain = None
        diff = -1

    return candidate_chain, diff


def select_nodes_within_region_over_ray(src_chain, endpoint_node, within_node_list):
    return [node for node in within_node_list if
            ((node.angle == endpoint_node.angle) and not (node.chain_id == src_chain.id))]


def extract_chains_ids_from_nodes(nodes_ray):
    return set([node.chain_id for node in nodes_ray])


def get_chains_from_ids(within_chains_set, chain_id_ray):
    return [chain for chain in within_chains_set if chain.id in chain_id_ray]


def get_chains_that_no_intersect_src_chain(src_chain, src_chain_angle_domain, within_chains_set,
                                           endpoint_chain_intersections):
    return [cad for cad in within_chains_set if
            cad not in endpoint_chain_intersections and cad != src_chain and
            not exist_angular_intersection_with_src_chain(cad, src_chain_angle_domain)]


def remove_chains_with_higher_intersection_threshold(src_chain_angle_domain, endpoint_chain_intersections,
                                                     neighbourhood_size):
    return [inter_chain for inter_chain in endpoint_chain_intersections if
            not angular_domain_intersection_higher_than_threshold(src_chain_angle_domain,
                                                                  inter_chain, intersection_threshold=neighbourhood_size)]


def remove_none_elements_from_list(list):
    return [element for element in list if element is not None]

def find_candidate_chains(outward_ring, inward_ring, ch_j, l_within_nodes, l_within_chains, endpoint,
                          neighbourhood_size, debug_params, save_path) -> List[ch.Chain]:

    img, iteration, debug = debug_params
    # Line 1 Get angle domain for source ch_j
    ch_j_angle_domain = ch.get_nodes_angles_from_list_nodes(ch_j.l_nodes)

    # Line 2 Get endpoint node
    ch_j_node = ch_j.extA if endpoint == ch.EndPoints.A else ch_j.extB

    # Line 3 Select ch_j  support chain over endpoint
    ch_i = select_support_chain(outward_ring, inward_ring, ch_j_node)

    # Line 4 Select within nodes over endpoint ray
    l_nodes_ray = select_nodes_within_region_over_ray(ch_j, ch_j_node, l_within_nodes)

    # Line 5 Select within chains id over endpoint ray
    l_chain_id_ray = extract_chains_ids_from_nodes(l_nodes_ray)

    # Line 6 Select within chains over endpoint ray by chain id
    l_endpoint_chains = get_chains_from_ids(l_within_chains, l_chain_id_ray)

    # Line 7 filter out chains that intersect with an intersection threshold higher than 45 degrees.
    # If intersecting threshold is so big, it is not a good candidate to connect
    l_filtered_chains = remove_chains_with_higher_intersection_threshold(ch_j_angle_domain, l_endpoint_chains,
                                                                         neighbourhood_size)

    if debug:
        boundary_ring_list = remove_none_elements_from_list([outward_ring, inward_ring])
        ch.visualize_selected_ch_and_chains_over_image_(
            [ch_i, ch_j] + l_filtered_chains + boundary_ring_list, l_within_chains
            , img,
            f'{save_path}/{iteration[0]}_split_chains_{ch_j.label_id}_2_1_{endpoint}_{ch_i.label_id}.png')
        iteration[0] += 1
    return l_filtered_chains, ch_j_node, ch_i, ch_j_angle_domain, l_endpoint_chains








def split_and_connect_neighbouring_chains(l_within_nodes: List[ch.Node], l_within_chains, ch_j: ch.Chain,
                                          endpoint: int, outward_ring, inward_ring, neighbourhood_size,
                                          debug_params, save_path, aux_chain=None) -> ch.Chain:
    """
    Logic for split and connect chains within region.
    @param l_within_nodes: nodes within region
    @param l_within_chains: chains within region
    @param ch_j: source chain. The one that is being to connect if condition are met.
    @param endpoint: endpoint of chain ch_j to find candidate chains to connect.
    @param outward_ring: outward support chain ring
    @param inward_ring: inward support chain ring
    @param neighbourhood_size: angular neighbourhood size in degrees to search for candidate chains
    @param debug_params: debug param
    @param save_path: debug param. Path to save debug images
    @param aux_chain: chain candidate to be connected by other endpoint. It is used to check that
     it is not connected by this endpoint
    @return: candidate chain to connect, radial distance to ch_j and support chain.
    """
    img, iteration, debug = debug_params
    l_filtered_chains, ch_j_node, ch_i, ch_j_angle_domain, l_endpoint_chains = (
                    find_candidate_chains(outward_ring, inward_ring, ch_j, l_within_nodes,
                                                               l_within_chains, endpoint, neighbourhood_size,
                                                               debug_params, save_path))

    l_candidates = split_intersecting_chains(ch_j_node.angle, l_filtered_chains, ch_j, debug_params, save_path)
    if debug:
        boundary_ring_list = remove_none_elements_from_list([outward_ring, inward_ring])
        ch.visualize_selected_ch_and_chains_over_image_(
            [ch_i, ch_j] + l_candidates + boundary_ring_list, l_within_chains
            , img,
            f'{save_path}/{iteration[0]}_split_chains_{ch_j.label_id}_2_2_{endpoint}.png')
        iteration[0] += 1

    l_no_intersection_j = get_chains_that_no_intersect_src_chain(ch_j, ch_j_angle_domain, l_within_chains,
                                                                 l_endpoint_chains)
    if aux_chain is not None:
        # If aux_chain is candidate to connect by the other endpoint, add it to the list of chains that do not intersect
        l_no_intersection_j += [aux_chain]

    l_candidates = add_chains_that_intersect_in_other_endpoint(l_within_chains, l_no_intersection_j, l_candidates, ch_j,
                                                neighbourhood_size, endpoint)
    if debug:
        boundary_ring_list = remove_none_elements_from_list([outward_ring, inward_ring])
        ch.visualize_selected_ch_and_chains_over_image_(
            [ch_i, ch_j] + l_candidates + boundary_ring_list, l_within_chains
            , img,
            f'{save_path}/{iteration[0]}_split_chains_{ch_j.label_id}_2_3_{endpoint}.png')
        iteration[0] += 1

    l_candidates = split_intersecting_chain_in_other_endpoint(endpoint, ch_j, l_within_chains,
                                                              l_within_nodes,
                                                              l_candidates)
    if debug:
        boundary_ring_list = remove_none_elements_from_list([outward_ring, inward_ring])
        ch.visualize_selected_ch_and_chains_over_image_(
            [ch_i, ch_j] + l_candidates + boundary_ring_list, l_within_chains
            , img,
            f'{save_path}/{iteration[0]}_split_chains_{ch_j.label_id}_2_4_{endpoint}.png')
        iteration[0] += 1

    l_candidates += filter_no_intersected_chain_far(l_no_intersection_j, ch_j, endpoint, neighbourhood_size)
    if debug:
        boundary_ring_list = remove_none_elements_from_list([outward_ring, inward_ring])
        ch.visualize_selected_ch_and_chains_over_image_(
            [ch_i, ch_j] + l_candidates + boundary_ring_list, l_within_chains
            , img,
            f'{save_path}/{iteration[0]}_split_chains_{ch_j.label_id}_2_5_{endpoint}.png')
        iteration[0] += 1
        counter_init = iteration[0]
        state = SystemStatus([ch_j], [ch_j.l_nodes], np.zeros((2, 2)), ch_j.center[0], ch_j.center[1],
                             debug=debug,  counter=iteration[0], save=f"{save_path}", img=img)

    else:
        state = None

    l_ch_k_euclidean_distance, l_ch_k_radial_distance, l_ch_k = \
        get_chains_that_satisfy_similarity_conditions(state, ch_i, ch_j, l_candidates, endpoint)

    ch_k, diff = select_closest_candidate_chain(l_ch_k, l_ch_k_euclidean_distance, l_ch_k_radial_distance,
                                                l_within_chains, aux_chain)
    if debug:
        iteration[0] += state.counter - counter_init
        if ch_k is not None:
            boundary_ring_list = remove_none_elements_from_list([outward_ring, inward_ring])
            ch.visualize_selected_ch_and_chains_over_image_(
                [ch_i, ch_j] + [ch_k] + boundary_ring_list, l_within_chains
                , img,
                f'{save_path}/{iteration[0]}_split_chains_{ch_j.label_id}_2_6_2_{endpoint}.png')
            iteration[0] += 1

    return ch_k, diff, ch_i


def debugging_postprocessing(debug, l_ch, img, l_within_chain_subset, filename, iteration):
    if debug:
        ch.visualize_selected_ch_and_chains_over_image_(l_ch, l_within_chain_subset
                                                        , img,
                                                        filename)
        iteration[0] += 1

def split_and_merge_chains_in_region(chains_in_region: List[ch.Chain], inward_ring: ch.Chain, outward_ring: ch.Chain,
                                     neighbourhood_size=45, debug=False, img=None, save_path=None, iteration=None,
                                     threshold = 0.9):
    """
    Line 5 to 16 in Algorithm 7 in the paper.
    Split and merge chains in region. It is used to split and connect chains in regions.
    @param chains_in_region: chains in region
    @param inward_ring: inward ring of region
    @param outward_ring: outward ring of region
    @param neighbourhood_size: angular neighbourhood size in degrees to search for candidate chains

    """
    # Initialization step. Sort chains.
    chains_in_region.sort(key=lambda x: x.size, reverse=True)
    debug_params = img, iteration, debug

    # Get inward nodes
    l_within_nodes = ch.get_nodes_from_chain_list(chains_in_region)

    # Line 5 Main loop to split chains that intersect over endpoints. Generator is defined to get next chain.
    generator = ChainsBag(chains_in_region)
    ch_j = generator.get_next_chain()
    # Line 6 While loop
    while ch_j:
        debugging_postprocessing(debug = debug, l_ch = [ch_j, inward_ring, outward_ring],
                                 l_within_chain_subset = chains_in_region,
                                 img = img,
                                 filename = f'{save_path}/{iteration[0]}_split_chains_{ch_j.label_id}_init.png',
                                 iteration = iteration)

        debugging_postprocessing(debug = debug, l_ch = chains_in_region,
                                 l_within_chain_subset = [],
                                 img = img,
                                 filename = f'{save_path}/{iteration[0]}_split_chains_{ch_j.label_id}_within_chains_init.png',
                                 iteration = [iteration[0]-1])


        # Line 7 to 10
        endpoint = ch.EndPoints.A
        ch_k_a, _, _ = split_and_connect_neighbouring_chains(l_within_nodes, chains_in_region, ch_j,
                                                                       endpoint, outward_ring, inward_ring,
                                                                       neighbourhood_size, debug_params,
                                                                       save_path=save_path)
        endpoint = ch.EndPoints.B
        ch_k_b, _, _ = split_and_connect_neighbouring_chains(l_within_nodes, chains_in_region, ch_j,
                                                                       endpoint, outward_ring, inward_ring,
                                                                       neighbourhood_size, debug_params,
                                                                       save_path=save_path, aux_chain=ch_k_a)

        ch_k, endpoint = select_closest_chain(ch_j, ch_k_a, ch_k_b)

        # debug
        if debug:
            candidates_set = [ch_k] if ch_k is not None else []
            debugging_postprocessing(debug = debug, l_ch = [ch_j] + candidates_set, l_within_chain_subset= chains_in_region,
                                     img  = img,
                                     filename = f'{save_path}/{iteration[0]}_split_chains_{ch_j.label_id}_candidate.png',
                                     iteration = iteration)
            iteration[0] += 1

        # endpoint == None means that connectivity goodness condition is not met. ch_j == ch_k means there is
        # no chain to connect with ch_j.
        if not (endpoint is None or ch_j == ch_k):
            # Line 12.
            merge_two_chains(ch_j, ch_k, endpoint, inward_ring, support2 = outward_ring)
            # Line 13
            l_within_nodes, chains_in_region = generator.update_chain_list( ch_k)

        # Line 14
        if ch_j.Nr > ch_j.size >= threshold * ch_j.Nr:
            # Line 15
            close_chain(chain=ch_j, ch_i=inward_ring, support2=outward_ring)

        # Line 16. Go to next chain
        ch_j = generator.get_next_chain(ch_j, ch_k)

        debugging_postprocessing(debug = debug, l_ch = [inward_ring, outward_ring] + [ch_j] if ch_j is not None else [] ,
                                 l_within_chain_subset = chains_in_region,
                                 img = img,
                                 filename = f'{save_path}/{iteration[0]}_split_chains_end.png',
                                 iteration = iteration)


    return







class Region:
    "Region. It has an outer ring, an inner ring and a list of chains in the region."
    def __init__(self, outer_ring, inner_ring, chains, area=None):
        self.outer_ring = outer_ring
        self.inner_ring = inner_ring
        self.chains = chains
        self.area = area

def control_check(l_ch_c):
    """
    Check if all the nodes in each chain belong to the chain
    :param l_ch_c:
    :return:
    """
    for chain in l_ch_c:
        for node in chain.l_nodes:
            if node.chain_id != chain.id:
                print(f"Node {node} does not belong to chain {chain.id}")
                raise "Node does not belong to chain."
    return True

class Regions:
    """
    Class to manage regions. It is used to split and connect chains in regions.
    """

    def __init__(self, l_ch_c,  save_path=None):
        self.l_ch_c = l_ch_c
        self.l_nodes_c = ch.get_nodes_from_chain_list(l_ch_c)
        self.current_idx = -1
        self.region_list = self._build_region_list(l_ch_c)
        self.save_path = save_path



    def _build_region_list(self, l_ch_c):
        self.completed_chains = [cad for cad in l_ch_c if cad.size >= cad.Nr]
        self.completed_chains, self.poly_completed_chains = self._from_completed_chain_to_poly(
            self.completed_chains)

        self.uncompleted_chains = [cad for cad in l_ch_c if cad.size < cad.Nr]
        self.uncompleted_polygons = self._from_uncompleted_chains_to_poly(self.uncompleted_chains)

        region_list = []
        for i in range(len(self.poly_completed_chains)-1):
            inner_ring = self.poly_completed_chains[i]
            outer_ring = self.poly_completed_chains[i+1]
            l_incomplete_polygons_within_region = search_for_polygons_within_region(self.uncompleted_polygons,
                                                                                    outer_ring,
                                                                                    inner_ring)

            chains_in_region = from_shapely_to_chain(self.uncompleted_polygons, self.uncompleted_chains,
                                                     l_incomplete_polygons_within_region)

            inner_chain,  outer_chain = self._from_shapely_ring_to_chain(inner_ring, outer_ring)

            region = Region(outer_chain, inner_chain, chains_in_region, area = outer_ring.area)
            region_list.append(region)

        return region_list


    def _from_shapely_ring_to_chain(self, poly_ring_inward, poly_ring_outward):
        inward_chain_ring = None
        outward_chain_ring = None
        if poly_ring_inward is not None:
            inward_chain_ring = self.completed_chains[self.poly_completed_chains.index(
                poly_ring_inward)]

        if poly_ring_outward is not None:
            outward_chain_ring = self.completed_chains[
                self.poly_completed_chains.index(poly_ring_outward)]
        return inward_chain_ring, outward_chain_ring

    def sort_list_by_index_array(self, indexes, list_position):
        Z = [list_position[position] for position in indexes]
        return Z

    def sort_shapely_list_and_chain_list(self, cadena_list, shapely_list):
        idx_sort = [i[0] for i in sorted(enumerate(shapely_list), key=lambda x: x[1].area)]
        shapely_list = self.sort_list_by_index_array(idx_sort, shapely_list)
        cadena_list = self.sort_list_by_index_array(idx_sort, cadena_list)
        return cadena_list, shapely_list

    def _from_completed_chain_to_poly(self, completed_chain):
        poly_completed_chains = []
        for chain in completed_chain:
            ring = Ring(chain, id=chain.id)
            poly_completed_chains.append(ring)

        completed_chain, poly_completed_chains = self.sort_shapely_list_and_chain_list(completed_chain,
                                                                                       poly_completed_chains)

        return completed_chain, poly_completed_chains

    def _from_uncompleted_chains_to_poly(self, uncompleted_chain):
        uncompleted_chain_shapely = []
        for chain in uncompleted_chain:
            lista_pts = [Point(punto.y, punto.x) for punto in chain.sort_dots()]
            uncompleted_chain_shapely.append(LineString(lista_pts))

        return uncompleted_chain_shapely

    def get_next_region(self, current = None, debug=False, iteration=None):
        if current is not None:
            #1. Check if there are completed chains in the region
            chains_in_region = current.chains

            completed_chains = [chain for chain in chains_in_region if chain.size >= chain.Nr]
            #1.1 If there are completed chains in the region split in subregion
            if len(completed_chains) > 0:
                completed_chains += [current.outer_ring, current.inner_ring]
                # if debug:
                #     image = np.zeros((completed_chains[0].img_height, completed_chains[0].img_width, 3)) + 255
                #     ch.visualize_selected_ch_and_chains_over_image_(completed_chains, [],
                #                                                     img=image, filename=f'completed_chains.png')
                    # 1.2 Add sub-region to the list
                subregion_list = self._build_region_list(chains_in_region + [current.outer_ring, current.inner_ring])
                self.region_list += subregion_list
                self.region_list.remove(current)
                self.region_list.sort(key=lambda x: x.area, reverse=False)
                # 1.3 Next region is the innermost region
                self.current_idx = self.region_list.index(subregion_list[0]) - 1

            #2. If there are not completed region, next region is the next region in the list



        self.current_idx += 1
        if self.current_idx >= len(self.region_list):
            return None
        region = self.region_list[self.current_idx]
        if debug:
            self.draw_regions(iteration)
        return region

    def draw_regions(self, iteration):
        region = self.region_list[0]
        chain = region.outer_ring
        image = np.zeros((chain.img_height, chain.img_width, 3)) + 255
        for idx, region in enumerate(self.region_list):
            image = Drawing.chain(region.outer_ring, image, Color.red)
            image = Drawing.chain(region.inner_ring, image, Color.blue)
            if self.current_idx == idx:
                #fill region
                image = Drawing.fill_region(region.outer_ring, region.inner_ring, image, Color.green)

        cv2.imwrite(f'{self.save_path}/{iteration[0]}_regions.png', image)
        iteration[0]+=1



    def get_chains(self):
        l_ch_p = []
        for region in self.region_list:
            l_ch_p += region.chains

            if region.outer_ring not in l_ch_p:
                l_ch_p.append(region.outer_ring)

            if region.inner_ring not in l_ch_p:
                l_ch_p.append(region.inner_ring)

        return l_ch_p


def postprocessing(l_ch_c, debug, save_path, debug_img_pre):
    """
    Postprocessing step. It is used to split and connect chains in regions. Algorithm 7 in the paper.
    :param l_ch_c: chain list
    :param debug: debug flag
    :param save_path: path to save debug images
    :param debug_img_pre: debug image
    :return: chain list
    """
    # Line 1.
    iteration = [0]
    regions = Regions([ch.copy_chain(chain) for chain in l_ch_c], save_path=save_path)
    # Line 2. Get the first region
    current = regions.get_next_region()
    # Line 3. While Loop.
    while current:
        # Line 4. Get the inner ring, chains in region and outer ring.
        ring1, chains_in_region, ring2 = current.inner_ring, current.chains, current.outer_ring

        # Line 5 to 16. Split and merge chains in region.
        split_and_merge_chains_in_region(chains_in_region, ring1, ring2, neighbourhood_size=45, debug=debug,
                                         img=debug_img_pre, save_path=save_path, iteration = iteration)

        # Line 17 to 20 Close chains in region if size higher than Nr * 0.5
        threshold = 0.5
        chains_in_region.sort(key=lambda x: x.size, reverse=True)
        for chain in chains_in_region:
            if chain.Nr > chain.size >= threshold * chain.Nr:
                close_chain(chain=chain, ch_i=ring1, support2=ring2)

            if chain.size >= chain.Nr:
                #a chain was closed within the region. Region must be divided.
                break

        # Line 21. Get the next region
        current = regions.get_next_region(current, debug=debug, iteration=iteration)

    return regions.get_chains()








