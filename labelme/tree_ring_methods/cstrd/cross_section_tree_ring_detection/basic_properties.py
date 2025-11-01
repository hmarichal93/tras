"""
Copyright (c) 2023 Author(s) Henry Marichal (hmarichal93@gmail.com)

This program is free software: you can redistribute it and/or modify it under the terms of the GNU Affero General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License along with this program. If not, see <http://www.gnu.org/licenses/>.
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2
from typing import List


from cross_section_tree_ring_detection.drawing import Color, Drawing
from cross_section_tree_ring_detection.interpolation_nodes import generate_nodes_list_between_two_radial_distances
import cross_section_tree_ring_detection.chain as ch
from cross_section_tree_ring_detection.interpolation_nodes import interpolate_nodes

def draw_segment_between_nodes(pto1, pto2, img, color=(0, 255, 0), thickness=2):
    pts = np.array([[pto1.y, pto1.x], [pto2.y, pto2.x]], dtype=int)
    isClosed = False
    img = cv2.polylines(img, [pts],
                        isClosed, color, thickness)

    return img


class InfoVirtualBand:
    DOWN = 0
    INSIDE = 1
    UP = 2

    def __init__(self, l_nodes, ch_j, ch_k, endpoint, ch_i=None, band_width=None, debug=False, inf_band=None,
                 sup_band=None, domain=None):
        """
        Class for generating a virtual band between two chains. It also incorporates a method to check if chain is
        inside the band
        @param l_nodes: list of interpolated nodes between the two chains plus the two endpoints
        @param ch_j: chain ch_j to compute the band
        @param ch_k: chain ch_j to compute the band
        @param endpoint: ch_j endpoint
        @param ch_i: support chain of ch_j and ch_k
        @param band_width: band with in percentage of the distance between ch_j and support chain (ch_i)
        """
        if band_width is None:
            band_width = 0.05 if ch_i.type == ch.TypeChains.center else 0.1

        self.l_nodes = l_nodes
        self.ch_j = ch_j
        self.ch_k = ch_k
        self.endpoint = endpoint
        self.ch_i = ch_i
        params = {'y': self.ch_j.center[1], 'x': self.ch_j.center[0], 'angle': 0, 'radial_distance': 0,
                  'chain_id': -1}
        self.center = ch.Node(**params)


        ext1 = self.ch_j.extB if endpoint == ch.EndPoints.B else self.ch_j.extA
        ext1_support = self.ch_i.get_node_by_angle(ext1.angle) if self.ch_i is not None else self.center
        ext2 = self.ch_k.extB if endpoint == ch.EndPoints.A else self.ch_k.extA
        ext2_support = self.ch_i.get_node_by_angle(ext2.angle) if self.ch_i is not None else self.center
        delta_r1 = ch.euclidean_distance_between_nodes(ext1, ext1_support)
        delta_r2 = ch.euclidean_distance_between_nodes(ext2, ext2_support)
        self.inf_cand = delta_r2 * (1 - band_width)
        self.sup_cand = delta_r2 * (1 + band_width)
        self.inf_orig = delta_r1 * (1 - band_width)
        self.sup_orig = delta_r1 * (1 + band_width)

        if not debug:
            self.generate_band()
        else:
            self.inf_band = inf_band
            self.sup_band = sup_band
            self.interpolation_domain = domain

    def generate_band_limit(self, r2, r1, total_nodes):
        interpolation_domain = [node.angle for node in self.l_nodes]
        endpoint_cad2 = self.l_nodes[-1]
        support_node2 = self.ch_i.get_node_by_angle(endpoint_cad2.angle) if self.ch_i is not None else self.center
        sign = -1 if support_node2.radial_distance > endpoint_cad2.radial_distance else +1
        generated_dots = generate_nodes_list_between_two_radial_distances(r2, r1, total_nodes, interpolation_domain,
                                                                          self.ch_k.center, sign,
                                                                          self.ch_i, self.ch_k)
        self.interpolation_domain = interpolation_domain
        return generated_dots

    def generate_band(self):
        total_nodes = len(self.l_nodes)
        r1 = self.sup_orig
        r2 = self.sup_cand
        self.sup_band = self.generate_band_limit(r2, r1, total_nodes)
        r1 = self.inf_orig
        r2 = self.inf_cand
        self.inf_band = self.generate_band_limit(r2, r1, total_nodes)

        return

    @staticmethod
    def mean_radial_in_node_list(node_list: List[ch.Node]):
        return np.mean([node.radial_distance for node in node_list])

    def is_dot_in_band(self, node: ch.Node):
        inf_mean_radii = self.mean_radial_in_node_list(self.inf_band)
        sup_mean_radii = self.mean_radial_in_node_list(self.sup_band)
        inner_band = self.inf_band if inf_mean_radii < sup_mean_radii else self.sup_band
        outter_band = self.sup_band if inf_mean_radii < sup_mean_radii else self.inf_band
        lowest = [inf for inf in inner_band if inf.angle == node.angle][0]
        highest = [inf for inf in outter_band if inf.angle == node.angle][0]

        if node.radial_distance <= lowest.radial_distance:
            relative_position = InfoVirtualBand.DOWN

        elif highest.radial_distance >= node.radial_distance >= lowest.radial_distance:
            relative_position = InfoVirtualBand.INSIDE

        else:
            relative_position = InfoVirtualBand.UP

        return relative_position

    def is_chain_in_band(self, chain: ch.Chain):
        """
        Check if a chain is inside the band
        @param chain: chain to check if belong to band
        @return: boolean
        """
        node_chain_in_interval = [node for node in chain.l_nodes if
                                  node.angle in self.interpolation_domain]
        res = False
        prev_status = None
        for node in node_chain_in_interval:
            res = self.is_dot_in_band(node)
            if res == InfoVirtualBand.INSIDE:
                res = True
                break

            if prev_status is not None  and prev_status != res:
                res = True
                break

            prev_status = res
            res = False

        return res

    def generate_chain_from_node_list(self, l_node: List[ch.Node]):
        chain = ch.Chain(chain_id = l_node[0].chain_id, center = self.ch_j.center, img_height=self.ch_j.img_height,
                         img_width= self.ch_j.img_width, Nr = self.ch_j.Nr)
        chain.add_nodes_list(l_node)

        return chain

    def draw_band(self, img, overlapping_chain: List[ch.Chain]):
        img = Drawing.chain(self.generate_chain_from_node_list(self.inf_band), img, color=Color.orange)
        img = Drawing.chain(self.generate_chain_from_node_list(self.sup_band), img, color=Color.maroon)
        img = Drawing.chain(self.ch_j, img, color=Color.blue)
        img = Drawing.chain(self.ch_k, img, color=Color.yellow)
        if self.ch_i is not None:
            img = Drawing.chain(self.ch_i, img, color=Color.red)

        for chain in overlapping_chain:
            img = Drawing.chain(chain, img, color=Color.purple)

        return img



def vector_derivative(f, Nr):
    """center derivative of a vector f"""
    return np.gradient(f)



def regularity_of_the_derivative_condition(state, Nr, ch_jk_nodes, ch_j_nodes, ch_k_nodes, endpoint_j, th_deriv=1):
    """
    Compute radial derivative of the serie formed by src nodes + virtual nodes + dst nodes
    @param Nr: number of rays
    @param ch_jk_nodes: all nodes involved. src nodes + virtual nodes + dst nodes
    @param ch_j_nodes: src nodes
    @param ch_k_nodes: dst nodes
    @param th_deriv: threshold of the derivative
    @param endpoint_j: endpoint of the src ch_j
    @return: boolean indicative of the regularity of the derivative
    """
    ch_jk_radials = [node.radial_distance for node in ch_jk_nodes]
    nodes_radial_distance_src_chain = [node.radial_distance for node in ch_j_nodes]
    nodes_radial_distance_dst_chain = [node.radial_distance for node in ch_k_nodes]

    abs_der_1 = np.abs(vector_derivative(nodes_radial_distance_src_chain, Nr))
    abs_der_2 = np.abs(vector_derivative(nodes_radial_distance_dst_chain, Nr))
    abs_der_3 = np.abs(vector_derivative(ch_jk_radials, Nr))
    maximum_derivative_chains = np.maximum(abs_der_1.max(), abs_der_2.max())

    max_derivative_end = np.max(abs_der_3)
    RegularDeriv = max_derivative_end <= maximum_derivative_chains * th_deriv

    if state is not None and state.debug:
        f, (ax2, ax1) = plt.subplots(2, 1)
        ax1.plot(abs_der_3)
        if endpoint_j == ch.EndPoints.A :
            ax1.plot(np.arange(0,len(abs_der_2)), abs_der_2[::-1])
            ax1.plot(np.arange(len(ch_jk_radials)-len(nodes_radial_distance_src_chain), len(ch_jk_radials)), abs_der_1)
        else:
            ax1.plot(np.arange(0, len(abs_der_1)), abs_der_1[::-1])
            ax1.plot(np.arange(len(ch_jk_radials) - len(nodes_radial_distance_dst_chain), len(ch_jk_radials)), abs_der_2)

        ax1.hlines(y=max_derivative_end, xmin=0, xmax= np.maximum(len(nodes_radial_distance_src_chain), len(nodes_radial_distance_dst_chain)), label='Salto')
        ax1.hlines(y=th_deriv * maximum_derivative_chains, xmin=0, xmax=np.maximum(len(nodes_radial_distance_dst_chain), len(nodes_radial_distance_src_chain)), colors='r', label='umbral')
        ax1.legend()

        ax2.plot(ch_jk_radials)
        if endpoint_j == ch.EndPoints.A :
            ax2.plot( np.arange( 0, len(abs_der_2)), nodes_radial_distance_dst_chain[::-1] , 'r')
            ax2.plot(np.arange(len(ch_jk_radials)-len(nodes_radial_distance_src_chain), len(ch_jk_radials)),
                     nodes_radial_distance_src_chain)
        else:
            ax2.plot(np.arange(0, len(abs_der_1)), nodes_radial_distance_src_chain[::-1], 'r')
            ax2.plot(np.arange(len(ch_jk_radials) - len(nodes_radial_distance_dst_chain), len(ch_jk_radials)),
                     nodes_radial_distance_dst_chain)

        plt.title(f'{RegularDeriv}')
        plt.savefig(f'{str(state.path)}/{state.counter}_derivada_{RegularDeriv}.png')
        plt.close()
        state.counter += 1

    return RegularDeriv


def generate_virtual_nodes_without_support_chain(ch_1, ch_2, endpoint):
    ch1_border = ch_1.extA if endpoint == ch.EndPoints.A else ch_1.extB
    ch2_border = ch_2.extB if endpoint == ch.EndPoints.A else ch_2.extA

    virtual_nodes = []
    support_chain = None
    interpolate_nodes(support_chain, ch1_border, ch2_border, endpoint, ch_1, virtual_nodes)
    return virtual_nodes

def regularity_of_the_derivative(state, ch_j, ch_k, endpoint_j, node_list, ch_j_nodes, ch_k_nodes, th_deriv=1,
                                 derivative_from_center=False):
    """
    Regularity of the derivative for the virtual nodes generated between the two chains.
    @param state: at this moment is used only for debug
    @param ch_j: source chain ch_j to be connected
    @param ch_k: destination chain ch_j to be connected
    @param endpoint_j: endpoint of ch_j to be connected
    @param node_list: all the nodes involved in the connection, including the virtual nodes
    @param ch_j_nodes: nodes of ch_j
    @param ch_k_nodes: nodes of ch_k
    @param th_deriv: derivative threshold
    @param derivative_from_center: bool for regenerating the virtual nodes interpolating from the center of the ch_i.
    @return: boolean indicative of the regularity of the derivative
    """
    if derivative_from_center:
        new_list = []
        virtual_nodes = generate_virtual_nodes_without_support_chain(ch_j, ch_k, endpoint_j)
        angles = [n.angle for n in virtual_nodes]
        for node in node_list:
            if node.angle not in angles:
                new_list.append(node)
            else:
                new_list.append(ch.get_node_from_list_by_angle(virtual_nodes, node.angle))

        node_list = new_list

    RegularDeriv = regularity_of_the_derivative_condition(state, ch_j.Nr, node_list, ch_j_nodes, ch_k_nodes, endpoint_j,
                                                          th_deriv=th_deriv)

    return RegularDeriv



def generate_virtual_nodes_between_two_chains(src_chain, dst_chain, support_chain, endpoint):
    virtual_nodes = []

    cad1_endpoint = src_chain.extA if endpoint == ch.EndPoints.A else src_chain.extB
    cad2_endpoint = dst_chain.extB if endpoint == ch.EndPoints.A else dst_chain.extA

    interpolate_nodes(support_chain, cad1_endpoint, cad2_endpoint, endpoint, src_chain, virtual_nodes)

    return virtual_nodes


def radial_tolerance_for_connecting_chains(state, th_radial_tolerance, endpoints_radial):
    """
    Check maximum radial distance allowed to connect chains
    @param state: state of the algorithm. At this point, it is used to debug.
    @param th_radial_tolerance: radial tolerance threshold
    @param endpoints_radial: radial distance between endpoints and support chain
    @return: bool indicating if radial distance is within tolerance
    """
    delta_ri = endpoints_radial[0]
    delta_ri_plus_i = endpoints_radial[1]
    inf_delta_ri = delta_ri * (1 - th_radial_tolerance)
    sup_delta_ri = delta_ri * (1 + th_radial_tolerance)
    RadialTol = inf_delta_ri <= delta_ri_plus_i <= sup_delta_ri

    #debug info
    if state is not None and state.debug:
        plt.figure()
        plt.axvline(x=delta_ri, color='b', label=f'delta_ri')
        plt.axvline(x=delta_ri_plus_i, color='r', label=f'delta_ri_plus_i')
        plt.axvline(x=inf_delta_ri, color='k', label=f'inf radial')
        plt.axvline(x=sup_delta_ri, color='k', label=f'sup radial')
        plt.title(f"{RadialTol}: Th {state.th_radial_tolerance}.")
        plt.savefig(f'{str(state.path)}/{state.counter}_max_radial_condition.png')
        plt.close()
        state.counter += 1

    return RadialTol


class Neighbourhood:
    """
    Class to compute and store the total_nodes of a chain and the candidate chains to connect to it.
    It generates the virtual nodes to compute the similarity condition.
    """
    def __init__(self, src_chain, dst_chain, support_chain, endpoint, n_nodes=20):
        """

        @param src_chain: src chain to be connected
        @param dst_chain: dst chain to be connected
        @param support_chain: support chain to be used to generate the virtual nodes
        @param endpoint: src chain endpoint to be connected
        @param n_nodes: number of nodes to be considered in the total_nodes
        """
        self.virtual_nodes = generate_virtual_nodes_between_two_chains(src_chain, dst_chain, support_chain, endpoint)
        self.src_endpoint = src_chain.extB if endpoint == ch.EndPoints.B else src_chain.extA
        self.dst_endpoint = dst_chain.extA if endpoint == ch.EndPoints.B else dst_chain.extB
        self.endpoint_and_virtual_nodes = [self.src_endpoint] + self.virtual_nodes + [self.dst_endpoint]
        self.radial_distance_endpoints_to_support = radial_distance_between_nodes_belonging_to_same_ray(
            [self.src_endpoint, self.dst_endpoint],
            support_chain)
        self.radial_distance_virtual_nodes_to_support = radial_distance_between_nodes_belonging_to_same_ray(
            self.virtual_nodes, support_chain)

        self.src_chain_nodes = src_chain.sort_dots(direction=ch.ClockDirection.anti_clockwise)[
                               :n_nodes] if endpoint == ch.EndPoints.A \
            else src_chain.sort_dots(direction=ch.ClockDirection.clockwise)[:n_nodes]
        self.dst_chain_nodes = dst_chain.sort_dots(direction=ch.ClockDirection.clockwise)[
                               :n_nodes] if endpoint == ch.EndPoints.A \
            else dst_chain.sort_dots(direction=ch.ClockDirection.anti_clockwise)[:n_nodes]
        self.set_i = radial_distance_between_nodes_belonging_to_same_ray(self.src_chain_nodes,
                                                                         support_chain)
        self.set_k = radial_distance_between_nodes_belonging_to_same_ray(
            self.dst_chain_nodes, support_chain)

        # sort all nodes, src + virtuales + dst
        if endpoint == ch.EndPoints.B:
            self.neighbourhood_nodes = self.src_chain_nodes[::-1] + self.virtual_nodes + self.dst_chain_nodes
        else:
            self.neighbourhood_nodes = self.dst_chain_nodes[::-1] + self.virtual_nodes[::-1] + self.src_chain_nodes

    def draw_neighbourhood(self, name):
        r = [node.radial_distance for node in self.neighbourhood_nodes]
        plt.figure()
        plt.plot(r, 'b')
        plt.savefig(name)
        plt.close()


def similar_radial_distances_of_nodes_in_both_chains(state, distribution_th, set_j, set_k):
    """
    Check if the radial distances of the nodes in both chains are similar via distribution of the radial distances
    @param state: state of the algorithm. At this point, it is used to debug.
    @param distribution_th: size of the distributions to check if they are similar
    @param set_j: radial distance between nodes of the first chain and the support chain
    @param set_k: radial distances between nodes of the second ch_i and the support ch_i
    @return:
    + bool indicating if the radial distances are similar
    + distance between the mean of the distributions
    """
    mean_j = np.mean(set_j)
    sigma_j = np.std(set_j)
    inf_range_j = mean_j - distribution_th * sigma_j
    sup_range_j = mean_j + distribution_th * sigma_j

    mean_k = np.mean(set_k)
    std_k = np.std(set_k)
    inf_range_k = mean_k - distribution_th * std_k
    sup_range_k = mean_k + distribution_th * std_k
    # check intersection between intervals
    SimilarRadialDist = inf_range_k <= sup_range_j and inf_range_j <= sup_range_k

    if state is not None and state.debug:
        plt.figure()
        plt.hist(set_j, bins=10, alpha=0.3, color='r', label='src radial')
        plt.hist(set_k, bins=10, alpha=0.3, color='b', label='dst radial')
        plt.axvline(x=inf_range_j, color='r', label=f'inf src')
        plt.axvline(x=sup_range_j, color='r', label=f'sup src')
        plt.axvline(x=inf_range_k, color='b', label=f'inf dst')
        plt.axvline(x=sup_range_k, color='b', label=f'sup dst')
        plt.legend()
        plt.title(f"{SimilarRadialDist}: Th {distribution_th}.")
        plt.savefig(f'{str(state.path)}/{state.counter}_distribution_condition.png')
        plt.close()
        state.counter += 1

    return SimilarRadialDist, np.abs(mean_j - mean_k)


def similarity_conditions(state, th_radial_tolerance, th_distribution_size, th_regular_derivative,
                          derivative_from_center, ch_i, ch_j, candidate_chain, endpoint, check_overlapping=True,
                          chain_list=None):
    """
    Similarity conditions defined in equation 6 in the paper
    @param state: state of the algorithm. At this point is used to debug.
    @param th_radial_tolerance:  Described at Table1 in the paper
    @param th_distribution_size:  Described at Table1 in the paper
    @param th_regular_derivative:  Described at Table1 in the paper
    @param derivative_from_center: Described at Table1 in the paper
    @param ch_i: support chain
    @param ch_j: chain j to connect
    @param candidate_chain: candidate chain to connect
    @param endpoint: endpoint of the source chain
    @param check_overlapping: check if the chains are overlapping
    @param chain_list: list of chains to check if the chains are overlapping
    @return: return a bool indicating if the similarity conditions are satisfied and the radial distance between
    the chains radial distributions
    """
    neighbourhood = Neighbourhood(ch_j, candidate_chain, ch_i, endpoint)
    if state is not None and state.debug:
        ch.visualize_selected_ch_and_chains_over_image_([ch_i, ch_j, candidate_chain], state.l_ch_s,
                                                        img=state.img, filename=f'{state.path}/{state.counter}_radial_conditions_{endpoint}.png')
        state.counter += 1
        neighbourhood.draw_neighbourhood(f'{state.path}/{state.counter}_radials_conditions_neighbourdhood_{ch_j.label_id}_{candidate_chain.label_id}.png')
        state.counter += 1

    if len(neighbourhood.set_i) <= 1 or len(
            neighbourhood.set_k) <= 1:
        return False, -1

    # 1. Radial tolerance for connecting chains
    RadialTol = radial_tolerance_for_connecting_chains(state, th_radial_tolerance,
                                                                   neighbourhood.radial_distance_endpoints_to_support)



    # 2. Similar radial distances of nodes in both chains
    SimilarRadialDist, distribution_distance = similar_radial_distances_of_nodes_in_both_chains(state,
                                                                                                th_distribution_size,
                                                                                                neighbourhood.set_i,
                                                                                                neighbourhood.set_k)

    check_pass = RadialTol or SimilarRadialDist
    if not check_pass:
        return (False, distribution_distance)

    # 3. Derivative condition
    RegularDeriv = regularity_of_the_derivative(state, ch_j, candidate_chain, endpoint,
                                                neighbourhood.neighbourhood_nodes,
                                                ch_j_nodes=neighbourhood.src_chain_nodes,
                                                ch_k_nodes=neighbourhood.dst_chain_nodes,
                                                th_deriv=th_regular_derivative,
                                                derivative_from_center=derivative_from_center)

    if not RegularDeriv:
        return (False, distribution_distance)

    # 4.0 Check there is not chains in region
    if check_overlapping:
        exist_chain = exist_chain_overlapping(state.l_ch_s if chain_list is None else chain_list,
                                              neighbourhood.endpoint_and_virtual_nodes, ch_j, candidate_chain, endpoint,
                                              ch_i)

        if exist_chain:
            return (False, distribution_distance)

    return (True, distribution_distance)


def radial_distance_between_nodes_belonging_to_same_ray(node_list, support_chain):
    """
    Compute radial distance between nodes belonging to same ray
    @param node_list: list of nodes
    @param support_chain: support chain
    @return: list of radial distances between list nodes and support chain nodes
    """
    radial_distances = []
    for node in node_list:
        support_node = support_chain.get_node_by_angle(node.angle)
        if support_node is None:
            break
        radial_distances.append(ch.euclidean_distance_between_nodes(support_node, node))

    return radial_distances


def exist_chain_in_band_logic(chain_list:List[ch.Chain], band_info:InfoVirtualBand)-> list:
    chain_of_interest = [band_info.ch_k, band_info.ch_j]
    if band_info.ch_i is not None:
        chain_of_interest.append(band_info.ch_i)

    try:
        fist_chain_in_region = next(chain for chain in chain_list if
                                           chain not in chain_of_interest and band_info.is_chain_in_band(chain))
    except StopIteration:
        fist_chain_in_region = None

    return [fist_chain_in_region] if fist_chain_in_region is not None else []


def exist_chain_overlapping(l_ch_s, l_nodes, ch_j, ch_k, endpoint_type, ch_i):
    """
    Algorithm 6 in the paper. Check if there is a chain in the area within the band
    @param l_ch_s: full chains list
    @param l_nodes: both endpoints and virtual nodes
    @param ch_j: chain j
    @param ch_k: chain k
    @param endpoint_type: ch_j endpoint type
    @param ch_i: support chain
    @return: boolean indicating if exist chain in band
    """
    # Line 1
    info_band = InfoVirtualBand(l_nodes, ch_j, ch_k, endpoint_type,
                                ch_i)
    # Line 2
    l_chains_in_band = exist_chain_in_band_logic(l_ch_s, info_band)

    # Line 3
    exist_chain = len(l_chains_in_band) > 0

    # Line 4
    return exist_chain
