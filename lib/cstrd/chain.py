"""
Copyright (c) 2023 Author(s) Henry Marichal (hmarichal93@gmail.com

This program is free software: you can redistribute it and/or modify it under the terms of the GNU Affero General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License along with this program. If not, see <http://www.gnu.org/licenses/>.
"""
import numpy as np
import matplotlib.pyplot as plt
from typing import List
from shapely.geometry import Polygon

from lib.canny_devernay_edge_detector import write_curves_to_image
from lib.filter_edges import write_filter_curves_to_image


def euclidean_distance(pix1, pix2):
    return np.sqrt((pix1[0] - pix2[0]) ** 2 + (pix1[1] - pix2[1]) ** 2)


def get_node_from_list_by_angle(dot_list, angle):
    try:
        dot = next(dot for dot in dot_list if (dot.angle == angle))
    except StopIteration as e:
        dot = None
    return dot


def get_chain_from_list_by_id(chain_list, chain_id):
    try:
        chain_in_list = next(chain for chain in chain_list if (chain.id == chain_id))

    except StopIteration:
        chain_in_list = None
    return chain_in_list


########################################################################################################################
# Class Node
########################################################################################################################
class Node:
    def __init__(self, x, y, chain_id, radial_distance, angle):
        self.x = x
        self.y = y
        self.chain_id = chain_id
        self.radial_distance = radial_distance
        self.angle = angle

    def __repr__(self):
        return (f'({self.x},{self.y}) ang:{self.angle} radio:{self.radial_distance:0.2f} cad.id {self.chain_id}\n')

    def __str__(self):
        return (f'({self.x},{self.y}) ang:{self.angle} radio:{self.radial_distance:0.2f} id {self.chain_id}')

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y and self.angle == other.angle


def euclidean_distance_between_nodes(d1: Node, d2: Node):
    v1 = np.array([d1.x, d1.y], dtype=float)
    v2 = np.array([d2.x, d2.y], dtype=float)
    return euclidean_distance(v1, v2)


def copy_node(node: Node):
    return Node(**{'y': node.y, 'x': node.x, 'angle': node.angle, 'radial_distance':
        node.radial_distance, 'chain_id': node.chain_id})


########################################################################################################################
# Class Chain
########################################################################################################################
class TypeChains:
    center = 0
    normal = 1
    border = 2
    gt_ring = 3


class ClockDirection:
    clockwise = 0
    anti_clockwise = 1


class Chain:
    def __init__(self, chain_id: int, Nr: int, center, img_height: int, img_width: int,
                 type: TypeChains = TypeChains.normal, A_outward=None, A_inward=None, B_outward=None, B_inward=None):
        self.l_nodes = []
        self.id = chain_id
        self.size = 0
        self.Nr = Nr
        self.extA = None
        self.extB = None
        self.type = type
        self.A_outward = A_outward
        self.A_inward = A_inward
        self.B_outward = B_outward
        self.B_inward = B_inward

        #Less important attributes
        self.center = center # center of the disk (pith pixel location)
        self.img_height = img_height # image height
        self.img_width = img_width # image width
        self.label_id = chain_id # debug purpose
    def __eq__(self, other):
        if other is None:
            return False
        return self.id == other.id and self.size == other.size

    def is_closed(self, threshold=0.95):
        if len(self.l_nodes) >= threshold * self.Nr:
            return True
        else:
            return False

    def sort_dots(self, direction=ClockDirection.clockwise):
        return self.clockwise_sorted_dots if direction == ClockDirection.clockwise else self.clockwise_sorted_dots[::-1]

    def _sort_dots(self, direction=ClockDirection.clockwise):
        clock_wise_sorted_dots = []
        step = 360 / self.Nr
        angle_k = self.extB.angle if direction == ClockDirection.clockwise else self.extA.angle
        while len(clock_wise_sorted_dots) < self.size:
            dot = self.get_node_by_angle(angle_k)
            if dot is None:
                continue
            clock_wise_sorted_dots.append(dot)
            angle_k = (angle_k - step) % 360 if direction == ClockDirection.clockwise else (angle_k + step) % 360

        return clock_wise_sorted_dots

    def __repr__(self):
        return (f'(id_l:{self.label_id},id:{self.id}, size {self.size}')

    def __find_endpoints(self):
        diff = np.zeros(self.size)
        extA_init = self.extA if self.extA is not None else None
        extB_init = self.extB if self.extB is not None else None
        self.l_nodes.sort(key=lambda x: x.angle, reverse=False)
        diff[0] = (self.l_nodes[0].angle + 360 - self.l_nodes[-1].angle) % 360

        for i in range(1, self.size):
            diff[i] = (self.l_nodes[i].angle - self.l_nodes[i - 1].angle)

        border1 = diff.argmax()
        if border1 == 0:
            border2 = diff.shape[0] - 1
        else:
            border2 = border1 - 1

        self.extAind = border1
        self.extBind = border2

        change_border = True if (extA_init is None or extB_init is None) or \
                                (extA_init != self.l_nodes[border1] or extB_init != self.l_nodes[
                                    border2]) else False
        self.extA = self.l_nodes[border1]
        self.extB = self.l_nodes[border2]

        return change_border

    def add_nodes_list(self, l_nodes):
        self.l_nodes += l_nodes
        change_border = self.update()
        return change_border

    def update(self):
        self.size = len(self.l_nodes)
        if self.size > 1:
            change_border = self.__find_endpoints()
            self.clockwise_sorted_dots = self._sort_dots()
        else:
            raise

        return change_border

    def get_nodes_coordinates(self):
        x = [dot.x for dot in self.l_nodes]
        y = [dot.y for dot in self.l_nodes]
        x_rot = np.roll(x, -self.extAind)
        y_rot = np.roll(y, -self.extAind)
        return x_rot, y_rot

    def get_dot_angle_values(self):
        return [dot.angle for dot in self.l_nodes]

    def get_node_by_angle(self, angle):
        return get_node_from_list_by_angle(self.l_nodes, angle)

    def change_id(self, index):
        for dot in self.l_nodes:
            dot.chain_id = index
        self.id = index
        return 0

    def to_array(self):
        """
        Return nodes coordinates in a numpy array. Return as well the endpoint coordinates
        @return:
        """
        x1, y1 = self.get_nodes_coordinates()
        nodes = np.vstack((x1, y1)).T

        c1a = np.array([self.extA.x, self.extA.y], dtype=float)
        c1b = np.array([self.extB.x, self.extB.y], dtype=float)
        return nodes.astype(float), c1a, c1b

    def get_area(self):
        """
        Compute chain area
        @return: chain area
        """
        x, y = self.get_nodes_coordinates()
        #using contour area
        points = np.array([x, y]).T
        #convert points to shapely polygon
        polygon = Polygon(points)
        #compute area
        return polygon.area



def copy_chain(chain: Chain):
    aux_chain = Chain(chain.id, chain.Nr, chain.center, chain.img_height, chain.img_width, type=chain.type)
    aux_chain_node_list = [copy_node(node)
                           for node in chain.l_nodes]
    aux_chain.add_nodes_list(aux_chain_node_list)

    return aux_chain


class EndPoints:
    A = 0
    B = 1


class ChainLocation:
    inwards = 0
    outwards = 1


def angular_distance_between_endpoints(endpoint_j: Node, endpoint_k: Node) -> float:
    """
    Compute angular distance between endpoints
    @param endpoint_j: node endpoint j of chain ch_j
    @param endpoint_k: node endpoint k of chain ch_k
    @return: angular distance in degrees
    """
    cte_degrees_in_a_circle = 360
    angular_distance = (endpoint_j.angle - endpoint_k.angle + cte_degrees_in_a_circle) % cte_degrees_in_a_circle
    return angular_distance


def angular_distance_between_chains(ch_j, ch_k, endpoint_j_type):
    """
    Compute angular distance between chains endpoints. If endpoint_j == A then compute distance between chain.extA and
    cad2.extB. In the other case, compute distance between chain.extB and cad2.extA
    @param ch_j: chain j
    @param ch_k: chain k
    @param endpoint_j_type: type endpoint of chain j (A or B)
    @return: angular distance between endpoints in degrees
    """

    endpoint_k = ch_k.extB if endpoint_j_type == EndPoints.A else ch_k.extA
    endpoint_j = ch_j.extA if endpoint_j_type == EndPoints.A else ch_j.extB

    angular_distance = angular_distance_between_endpoints(endpoint_k, endpoint_j) if endpoint_j_type == EndPoints.B \
        else angular_distance_between_endpoints(endpoint_j, endpoint_k)

    return angular_distance


def minimum_euclidean_distance_between_vector_and_matrix(vector, matrix):
    """
    Compute euclidean distance between vector ext and each row in matrix. Return minium distance
    @param vector: numpy vector
    @param matrix: numpy matrix
    @return: scalar value
    """
    distances = np.sqrt(np.sum((matrix - vector) ** 2, axis=1))
    return np.min(distances)


def minimum_euclidean_distance_between_chains_endpoints(ch_j: Chain, ch_k: Chain):
    """
    Compute minimum euclidean distance between ch_j and ch_k endpoints.
    @param ch_j: chain j
    @param ch_k: chain k
    @return:
    """
    nodes1, c1a, c1b = ch_j.to_array()
    nodes2, c2a, c2b = ch_k.to_array()
    c2a_min = minimum_euclidean_distance_between_vector_and_matrix(nodes1, c2a)
    c2b_min = minimum_euclidean_distance_between_vector_and_matrix(nodes1, c2b)
    c1a_min = minimum_euclidean_distance_between_vector_and_matrix(nodes2, c1a)
    c1b_min = minimum_euclidean_distance_between_vector_and_matrix(nodes2, c1b)
    return np.min([c2a_min, c2b_min, c1a_min, c1b_min])


def get_chains_within_angle(angle: int, chain_list: List[Chain]):
    chains_list = []
    for chain in chain_list:
        A = chain.extA.angle
        B = chain.extB.angle
        if ((A <= B and A <= angle <= B) or
                (A > B and (A <= angle or angle <= B))):
            chains_list.append(chain)

    return chains_list


def get_closest_chain_border_to_angle(chain: Chain, angle: int):
    B = chain.extB.angle
    A = chain.extA.angle
    if B < A:
        dist_to_b = 360 - angle + B if angle > B else B - angle
        dist_to_a = angle - A if angle > B else 360 - A + angle

    else:
        dist_to_a = A - angle
        dist_to_b = angle - B
    # assert dist_to_a > 0 and dist_to_b > 0
    dot = chain.extB if dist_to_b < dist_to_a else chain.extA
    return dot


def get_closest_dots_to_angle_on_radial_direction_sorted_by_ascending_distance_to_center(chains_list: List[Chain],
                                                                                         angle: int):
    """
    get nodes of all chains that are over the ray defined by angle and sort them by ascending distance to center
    @param chains_list: full ch_i list
    @param angle: ray angle direction
    @return: nodes list sorted by ascending distance to center over ray direction angle.
    """
    node_list_over_ray = []
    for chain in chains_list:
        try:
            node = [node for node in chain.l_nodes if node.angle == angle][0]
        except IndexError:
            node = get_closest_chain_border_to_angle(chain, angle)
            pass

        if node not in node_list_over_ray:
            node_list_over_ray.append(node)

    if len(node_list_over_ray) > 0:
        node_list_over_ray = sorted(node_list_over_ray, key=lambda x: x.radial_distance, reverse=False)

    return node_list_over_ray


def get_nodes_from_chain_list(chain_list: List[Chain]):
    inner_nodes = []
    for chain in chain_list:
        inner_nodes += chain.l_nodes
    return inner_nodes


def get_nodes_angles_from_list_nodes(node_list: List[Node]):
    return [node.angle for node in node_list]


###########################################
##Display
###########################################

def visualize_chains_over_image(chain_list=[], img=None, filename=None, devernay=None, filter=None):
    if devernay is not None:
        img = write_curves_to_image(devernay, img)
    elif filter is not None:
        img = write_filter_curves_to_image(filter, img)

    figsize = (10, 10)
    plt.figure(figsize=figsize)
    plt.imshow(img, cmap='gray')
    for chain in chain_list:
        x, y = chain.get_nodes_coordinates()
        if chain.type == TypeChains.normal:
            if chain.is_closed():
                x = x.tolist() + [x[0]]
                y = y.tolist() + [y[0]]
                plt.plot(x, y, 'b', linewidth=1)
            else:
                plt.plot(x, y, 'r', linewidth=1)
        elif chain.type == TypeChains.gt_ring:
            x = x.tolist() + [x[0]]
            y = y.tolist() + [y[0]]
            plt.plot(x, y, 'g', linewidth=1)

        elif chain.type == TypeChains.border:
            plt.plot(x, y, 'k', linewidth=1)

        else:
            plt.scatter(int(x[0]), int(y[0]), c='k')

    plt.tight_layout()
    plt.axis('off')
    plt.savefig(filename)
    plt.close()


def visualize_selected_ch_and_chains_over_image_(selected_ch=[], chain_list=[], img=None, filename=None, devernay=None,
                                                 filter=None):
    if devernay is not None:
        img = write_curves_to_image(devernay, img)
    elif filter is not None:
        img = write_filter_curves_to_image(filter, img)

    figsize = (10, 10)
    plt.figure(figsize=figsize)
    plt.imshow(img, cmap='gray')
    for chain in chain_list:
        x, y = chain.get_nodes_coordinates()
        plt.plot(x, y, 'w', linewidth=3)

    # draw selected ch_i
    for ch in selected_ch:
        x, y = ch.get_nodes_coordinates()
        plt.plot(x, y, linewidth=3)
        plt.annotate(str(ch.label_id), (x[0], y[0]), c='b')

    plt.tight_layout()
    plt.axis('off')
    plt.savefig(filename)
    plt.close()
