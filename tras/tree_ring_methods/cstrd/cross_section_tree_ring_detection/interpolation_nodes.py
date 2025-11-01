"""
Copyright (c) 2023 Author(s) Henry Marichal (hmarichal93@gmail.com

This program is free software: you can redistribute it and/or modify it under the terms of the GNU Affero General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License along with this program. If not, see <http://www.gnu.org/licenses/>.
"""
import numpy as np
from typing import List

import cross_section_tree_ring_detection.chain as ch


def compute_interpolation_domain(endpoint: str, endpoint_cad1: ch.Node, endpoint_cad2: ch.Node, Nr):
    interpolation_domain = []
    step = 360 / Nr if endpoint == ch.EndPoints.B else - 360 / Nr
    current_angle = endpoint_cad1.angle
    while current_angle % 360 != endpoint_cad2.angle:
        current_angle += step
        current_angle = current_angle % 360
        interpolation_domain.append(current_angle)

    return interpolation_domain[:-1]


def from_polar_to_cartesian(r, angulo, centro):
    y = centro[0] + r * np.cos(angulo * np.pi / 180)
    x = centro[1] + r * np.sin(angulo * np.pi / 180)
    return (y, x)


def generate_node_list_between_two_support_chains_and_two_radial_distances(r2_ratio, r1_ratio, total_nodes,
                                                                           interpolation_angle_domain, center,
                                                                           inward_chain, outward_chain, chain):
    cad_id = chain.id
    generated_node_list = []
    m = (r2_ratio - r1_ratio) / total_nodes
    n = r1_ratio
    for idx_current_node, angle in enumerate(interpolation_angle_domain):
        dot_list_in_radial_direction = ch.get_closest_dots_to_angle_on_radial_direction_sorted_by_ascending_distance_to_center(
            [inward_chain], angle % 360)
        support_node = dot_list_in_radial_direction[0]
        radio_init = support_node.radial_distance
        dot_list_in_radial_direction = ch.get_closest_dots_to_angle_on_radial_direction_sorted_by_ascending_distance_to_center(
            [outward_chain], angle % 360)
        support_node = dot_list_in_radial_direction[0]
        radio_superior = support_node.radial_distance
        radial_distance_between_chains = radio_superior - radio_init

        radio_inter = (m * (idx_current_node) + n) * radial_distance_between_chains + radio_init
        i, j = from_polar_to_cartesian(radio_inter, angle % 360, center)
        i = i if i < chain.img_height else chain.img_height - 1
        j = j if j < chain.img_width else chain.img_width - 1

        # radios.append(radio_inter)
        params = {
            "x": j,
            "y": i,
            "angle": angle % 360,
            "radial_distance": radio_inter,
            "chain_id": cad_id
        }

        node = ch.Node(**params)
        generated_node_list.append(node)

    return generated_node_list


def generate_nodes_list_between_two_radial_distances(r2, r1, total_nodes, interpolation_angular_domain, center, sign,
                                                     ch_i, ch_j):
    """
    Generate a list of nodes between two radial distances.
    @param r2: radial distance of the last node.
    @param r1: radial distance of the first node.
    @param total_nodes: total nodes to generate.
    @param interpolation_angular_domain: radii angle of the nodes to generate.
    @param center: center node of the disk.
    @param sign: Indicates if nodes are generated inward or outward from support chain, ch_i
    @param ch_i: support chain
    @param ch_j: source chain
    @return: Generated nodes list
    """
    cad_id = ch_j.id
    l_generated_node = []
    m = (r2 - r1) / (total_nodes - 0)
    n = r1 - m * 0
    for current_idx_node, angle in enumerate(interpolation_angular_domain):
        if ch_i is not None:
            dot_list_in_radial_direction = ch.get_closest_dots_to_angle_on_radial_direction_sorted_by_ascending_distance_to_center(
                [ch_i], angle % 360)

            support_node = dot_list_in_radial_direction[0]
            radio_init = support_node.radial_distance
            # radio_init = r1
        else:
            radio_init = 0

        radio_inter = sign * (m * (current_idx_node) + n) + radio_init
        i, j = from_polar_to_cartesian(radio_inter, angle % 360, center)
        i = i if i < ch_j.img_height else ch_j.img_height - 1
        j = j if j < ch_j.img_width else ch_j.img_width - 1

        l_generated_node.append(ch.Node(**{'x': j, 'y': i, 'angle': angle % 360, 'radial_distance': radio_inter,
                                              'chain_id': cad_id}))

    return l_generated_node


def get_radial_distance_to_chain(chain, dot):
    dot_list_in_radial_direction = ch.get_closest_dots_to_angle_on_radial_direction_sorted_by_ascending_distance_to_center(
        [chain], dot.angle)
    soporte_pto1 = dot_list_in_radial_direction[0]
    rii = ch.euclidean_distance_between_nodes(soporte_pto1, dot)
    return rii


def compute_radial_ratio(cadena_inferior, cadena_superior, dot):
    r1_inferior = get_radial_distance_to_chain(cadena_inferior, dot)
    r1_superior = get_radial_distance_to_chain(cadena_superior, dot)
    return r1_inferior / (r1_superior + r1_inferior)


def interpolate_nodes_two_chains(inward_support_chain, outward_support_chain, ch1_endpoint, ch2_endpoint,
                                 endpoint, ch1, node_list):
    """
    Interpolate between ch_i endpoints via two support chains.
    @param inward_support_chain:
    @param outward_support_chain:
    @param ch1_endpoint: source endpoint
    @param ch2_endpoint: destination endpoint
    @param endpoint: source endpoint type
    @param ch1: source ch_i
    @param node_list: node list to be updated
    @return: None. Generated nodes are returned via node_list
    """
    # 1. Domain angle interpolation
    domain_angle_interpolation = compute_interpolation_domain(endpoint, ch1_endpoint, ch2_endpoint, ch1.Nr)
    center = ch1.center

    # 2. Compute radial ratio
    r1_ratio = compute_radial_ratio(inward_support_chain, outward_support_chain, ch1_endpoint)
    r2_ratio = compute_radial_ratio(inward_support_chain, outward_support_chain, ch2_endpoint)

    # 3. Generate nodes
    total_nodes = len(domain_angle_interpolation)
    if total_nodes == 0:
        return

    generated_nodes = generate_node_list_between_two_support_chains_and_two_radial_distances(r2_ratio, r1_ratio,
                                                                                             total_nodes,
                                                                                             domain_angle_interpolation,
                                                                                             center,
                                                                                             inward_support_chain,
                                                                                             outward_support_chain, ch1)

    node_list += generated_nodes

    return



def interpolate_nodes_given_chains(ch_i, ch_j_endpoint, ch_k_endpoint, endpoint, ch_j, support2=None):
    """
    Interpolate between two chains using a support chain or two support chains. Logic used in the Algorithms 4 and 5 in
    the paper.
    :param ch_i: support chain
    :param ch_j_endpoint: source chain endpoint
    :param ch_k_endpoint: destination chain endpoint
    :param endpoint: endpoint type (A or B)
    :param ch_j: source chain
    :param support2: optional second support chain
    :return: interpolated nodes
    """
    interpolated = []
    if support2:
        interpolate_nodes_two_chains(ch_i, support2, ch_j_endpoint, ch_k_endpoint, endpoint,
                                     ch_j, interpolated)

    else:
        interpolate_nodes(ch_i, ch_j_endpoint, ch_k_endpoint, endpoint, ch_j, interpolated)

    return interpolated


def interpolate_nodes(ch_i: ch.Chain, ch_j_endpoint: ch.Node, ch_k_endpoint: ch.Node, endpoint: int, ch_j: ch.Chain,
                      l_nodes: List[ch.Node]):
    """
    Interpolate between endpoint ch_j_endpoint and ch_k_endpoint using ch_i as support ch_j. Ch_j is the source ch_j to
    be connected
    @param ch_i: support ch_j
    @param ch_j_endpoint:  source ch_j node endpoint
    @param ch_k_endpoint: destination ch_j node endpoint
    @param endpoint: integer indicating the endpoint type (A, B)
    @param ch_j: source ch_j
    @param l_nodes: list of nodes to be updated
    @return: Void. Generated nodes are added to list l_nodes.
    """
    domain_angles = compute_interpolation_domain(endpoint, ch_j_endpoint, ch_k_endpoint, ch_j.Nr)
    center = ch_j.center

    if ch_i is not None:
        dot_list_in_radial_direction = ch.get_closest_dots_to_angle_on_radial_direction_sorted_by_ascending_distance_to_center(
            [ch_i], ch_j_endpoint.angle)
        node1_support = dot_list_in_radial_direction[0]
        r1 = ch.euclidean_distance_between_nodes(node1_support, ch_j_endpoint)
        dot_list_in_radial_direction = ch.get_closest_dots_to_angle_on_radial_direction_sorted_by_ascending_distance_to_center(
            [ch_i], ch_k_endpoint.angle)
        node2_support = dot_list_in_radial_direction[0]
        sign = -1 if node2_support.radial_distance > ch_k_endpoint.radial_distance else +1
        r2 = ch.euclidean_distance_between_nodes(node2_support, ch_k_endpoint)
    else:
        r1 = ch_j_endpoint.radial_distance
        r2 = ch_k_endpoint.radial_distance
        sign = 1

    total_nodes = len(domain_angles)
    if total_nodes == 0:
        return

    l_generated_nodes = generate_nodes_list_between_two_radial_distances(r2, r1, total_nodes, domain_angles, center, sign,
                                                                       ch_i, ch_j)

    l_nodes += l_generated_nodes

    return


def complete_chain_using_2_support_ring(inward_chain, outward_chain, chain):
    """
    Complete ch_i using two support rings
    @param inward_chain: inward support ch_i
    @param outward_chain: outward support ch_i
    @param chain: ch_i to be completed
    @return: boolean indicating if the border of the ch_i has changed
    """
    ch1_endpoint = chain.extB
    ch2_endpoint = chain.extA
    endpoint = ch.EndPoints.B
    generated_nodes = []
    interpolate_nodes_two_chains(inward_chain, outward_chain, ch1_endpoint, ch2_endpoint, endpoint, chain,
                                 generated_nodes)

    change_border = chain.add_nodes_list(generated_nodes)

    return change_border


def complete_chain_using_support_ring(support_chain, ch1):
    ch1_endpoint = ch1.extB
    ch2_endpoint = ch1.extA
    endpoint = ch.EndPoints.B
    generated_list_nodes = []
    interpolate_nodes(support_chain, ch1_endpoint, ch2_endpoint, endpoint, ch1, generated_list_nodes)
    change_border = ch1.add_nodes_list(generated_list_nodes)

    return change_border


def connect_2_chain_via_inward_and_outward_ring(outward_ring, inward_ring, ch_j, candidate_chain, l_nodes_c, endpoint):
    """
    Connect 2 ch_i via inward and outward ring
    @param outward_ring: outward ch_i
    @param inward_ring: inward ch_i
    @param ch_j: chain1 to connect
    @param candidate_chain: chain2 to connect
    @param l_nodes_c: full node list
    @param endpoint: endpoint to connect
    @param add: add generated node list to ch_i 1
    @return: generated node list and boolean value indicating if border has changed
    """
    ch1_endpoint = ch_j.extA if endpoint == ch.EndPoints.A else ch_j.extB
    ch2_endpoint = candidate_chain.extB if endpoint == ch.EndPoints.A else candidate_chain.extA

    # 1.0
    generated_node_list = []
    interpolate_nodes_two_chains(inward_ring, outward_ring, ch1_endpoint, ch2_endpoint, endpoint,
                                 ch_j, generated_node_list)
    l_nodes_c += generated_node_list

    # 2.0
    chain_2_nodes = []
    chain_2_nodes += candidate_chain.l_nodes
    for node in chain_2_nodes:
        node.chain_id = ch_j.id

    change_border = ch_j.add_nodes_list(chain_2_nodes + generated_node_list)


    return generated_node_list, change_border
