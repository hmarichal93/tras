"""
Copyright (c) 2023 Author(s) Henry Marichal (hmarichal93@gmail.com

This program is free software: you can redistribute it and/or modify it under the terms of the GNU Affero General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License along with this program. If not, see <http://www.gnu.org/licenses/>.
"""
import numpy as np
import cv2
import warnings

warnings.filterwarnings("ignore")
from shapely.geometry.linestring import LineString

from cross_section_tree_ring_detection.drawing import Color

DELIMITE_CURVE_ROW = np.array([-1, -1])


class Curve(LineString):
    def __init__(self, pixels_list, name):
        self.id = name
        super().__init__(np.array(pixels_list)[:, [1, 0]].tolist())

    def __setattr__(self, name, value) -> None:
        object.__setattr__(self, name, value)

    def draw(self, img, thickness=1):
        y, x = self.xy
        y = np.array(y).astype(int)
        x = np.array(x).astype(int)
        pts = np.vstack((x, y)).T
        isClosed = False
        img = cv2.polylines(img, [pts],
                            isClosed, Color.black, thickness)
        return img


def normalized_row_matrix(matrix):
    sqrt = np.sqrt((matrix ** 2).sum(axis=1))
    normalized_array = matrix / sqrt[:, np.newaxis]
    return normalized_array


def erosion(erosion_size, src):
    erosion_shape = cv2.MORPH_CROSS

    element = cv2.getStructuringElement(erosion_shape, (2 * erosion_size + 1, 2 * erosion_size + 1),
                                        (erosion_size, erosion_size))
    erosion_dst = cv2.erode(src, element)
    return erosion_dst


def dilatation(dilatation_size, src):
    dilation_shape = cv2.MORPH_RECT
    element = cv2.getStructuringElement(dilation_shape, (2 * dilatation_size + 1, 2 * dilatation_size + 1),
                                        (dilatation_size, dilatation_size))
    dilatation_dst = cv2.dilate(src, element)
    return dilatation_dst


def mask_background(img):
    mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
    mask[img[:, :] == Color.gray_white] = Color.gray_white
    return mask


def blur(img, blur_size=11):
    return cv2.GaussianBlur(img, (blur_size, blur_size), 0)


def thresholding(mask, threshold=0):
    #mask = np.where(mask > threshold, Color.gray_white, 0).astype(np.uint8)
    ret, img_th = cv2.threshold(mask, threshold, 255, 0)
    return img_th


def padding_mask(mask, pad = 3):
    mask = np.pad(mask, ((pad, pad), (pad, pad)), mode='constant', constant_values=255).astype(np.uint8)
    return mask


def find_border_contour(img):
    """
    Find the contour of the border of the disk.
    @param img: disk image
    @return: disk border contour
    """
    contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    # 1.0 Perimeter of the image. Used to discard contours that are similar in size to the image perimeter.
    perimeter_image = 2 * img.shape[0] + 2 * img.shape[1]
    # 1.1 Error threshold. Used to discard contours that are similar in size to the image perimeter. For us, similar
    # is lower than 10% of the perimeter of the image.
    error_threshold = 0.1 * perimeter_image

    # 2.0 Approximate area of the wood cross-section disk.
    area_image = img.shape[0] * img.shape[1]
    approximate_disk_area = area_image / 2

    # 3.0 Find the contour that is closest in area to the approximate wood disk area (%50 of the image area)
    max_cont = None
    area_difference_min = np.inf
    for idx, c in enumerate(contours):
        contour_area = cv2.contourArea(c)
        contour_perimeter = cv2.arcLength(c, True)
        if np.abs(contour_perimeter - perimeter_image) < error_threshold:
            # Contour similar to image perimeter. Discard it.
            continue

        area_difference = np.abs(contour_area - approximate_disk_area)
        if area_difference < area_difference_min:
            max_cont = c.reshape((-1, 2))
            area_difference_min = area_difference

    return max_cont


def contour_to_curve(contour, name):
    curve = Curve(contour, name)
    return curve

def convex_hull(contour):
    from shapely.geometry import Polygon
    border_contour_poly = Polygon(contour)
    #convert to convex hull
    border_contour = np.array(border_contour_poly.convex_hull.exterior.coords)
    return border_contour

def get_border_curve(img, l_ch_f):
    """
    Get disk border border_curve of the image.
    @param img: segmented gray image
    @param l_ch_f: list of curves
    @return: border object border_curve

    """
    mask = mask_background(img)
    mask = blur(mask)
    mask = thresholding(mask, 127)
    mask = padding_mask(mask)
    border_contour = find_border_contour(mask)
    if border_contour is None:
        return None
    border_contour = convex_hull(border_contour)
    border_curve = contour_to_curve(border_contour, len(l_ch_f))
    return border_curve


def change_reference_axis(ch_e_matrix, cy, cx):
    center = [cx, cy]
    curve_border_index = np.where(ch_e_matrix == DELIMITE_CURVE_ROW)[0]
    X = ch_e_matrix.copy()

    # change reference axis
    Xb = np.array([[1, 0], [0, 1]]).dot(X.T) + (np.array([-1, -1]) * np.array(center, dtype=float)).reshape(
        (-1, 1))

    # mask delimiting edge row by -1
    Xb[:, curve_border_index] = -1
    return Xb


def convert_masked_pixels_to_curves(X_edges_filtered):
    curve_border_index = np.unique(np.where(X_edges_filtered == DELIMITE_CURVE_ROW)[0])
    start = -1
    ch_f = []
    for end in curve_border_index:
        if end - start > 2:
            pixel_list = X_edges_filtered[start + 1:end].tolist()
            curve = Curve(pixel_list, len(ch_f))
            ch_f.append(curve)
        start = end

    return ch_f


def get_gradient_vector_for_each_edge_pixel(ch_e, Gx, Gy):
    G = np.vstack(
        (Gx[ch_e[:, 1].astype(int), ch_e[:, 0].astype(int)], Gy[ch_e[:, 1].astype(int), ch_e[:, 0].astype(int)])).T
    return G


def compute_angle_between_gradient_and_edges(Xb_normed, gradient_normed):
    theta = np.arccos(np.clip((gradient_normed * Xb_normed).sum(axis=1), -1.0, 1.0)) * 180 / np.pi
    return theta


def filter_edges_by_threshold(m_ch_e, theta, alpha):
    X_edges_filtered = m_ch_e.copy()
    X_edges_filtered[theta >= alpha] = -1
    return X_edges_filtered


def filter_edges(m_ch_e, cy, cx, Gx, Gy, alpha, im_pre):
    """
    Edge detector find three types of edges: early wood transitions, latewood transitions and radial edges produced by
    cracks and fungi. Only early wood edges are the ones that forms the rings. In other to filter the other ones
    collineary with the ray direction is computed and filter depending on threshold (alpha).
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
    Xb = change_reference_axis(m_ch_e, cy, cx)
    G = get_gradient_vector_for_each_edge_pixel(m_ch_e, Gx, Gy)
    Xb_normalized = normalized_row_matrix(Xb.T)
    G_normalized = normalized_row_matrix(G)
    theta = compute_angle_between_gradient_and_edges(Xb_normalized, G_normalized)
    X_edges_filtered = filter_edges_by_threshold(m_ch_e, theta, alpha)
    l_ch_f = convert_masked_pixels_to_curves(X_edges_filtered)
    border_curve = get_border_curve(im_pre, l_ch_f)
    if border_curve is not None:
        l_ch_f.append(border_curve)

    return l_ch_f


def write_filter_curves_to_image(curves, img):
    img = np.zeros_like(img) + 255
    for c in curves:
        img = c.draw(img, thickness=3)
    return img
