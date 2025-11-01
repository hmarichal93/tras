import numpy as np
import cv2





def from_lines_to_matrix(l_lo):
    L = np.zeros((len(l_lo), 4))
    for idx, li in enumerate(l_lo):
        x1, y1 = li.p1.ravel()
        x2, y2 = li.p2.ravel()
        L[idx] = [x1, y1, x2, y2]

    return L
def euclidean_distance(p1, p2):
    """
    Compute euclidean distance between two points
    :param p1: point 1
    :param p2: point 2
    :return: euclidean distance between p1 and p2
    """
    return np.linalg.norm(p1 - p2)

class Line:
    def __init__(self, p1_rel=None, p2_rel=None, p1=None, p2=None, certainty=1, a=None, b=None, c=None):
        """
        Line of symmetry. Line equation: a*x+b*y+c=0
        :param p1_rel: point 1 relative to the block. Pixel coordinates
        :param p2_rel: point 2 relative to the block. Pixel coordinates
        :param p1: point 1 absolute coordinates. Pixel
        :param p2: point 2 absolute coordinates. Pixel.
        :param certainty: certainty of the line of symmetry estimation
        """
        self.p1_rel = np.array(p1_rel) if p1_rel is not None else None
        self.p2_rel = np.array(p2_rel) if p2_rel is not None else None
        self.p1 = np.array(p1) if p1 is not None else None
        self.p2 = np.array(p2) if p2 is not None else None
        self.certainty = certainty
        self.block_pointer = None
        self.a = a
        self.b = b
        self.c = c
        self.length = euclidean_distance(self.p1, self.p2) if self.p1 is not None and self.p2 is not None else None

    def get_slope_bias(self, H, W):
        """
        Compute slope and bias of the line.
        :return: slope and bias
        """
        if self.p1 is None or self.p2 is None:
            return None, None
        y1, x1 = self.p1.ravel()
        y2, x2 = self.p2.ravel()

        # normalize coordinate
        x1/= W
        y1/= H
        x2/= W
        y2/= H

        # compute slope and bias
        if x2 == x1:
            # Vertical line: x = c
            return None, x1
        elif y1 == y2:
            # Horizontal line: y = c
            return 0, y1
        else:
            slope = (y2 - y1) / (x2 - x1)
            bias = y1 - slope * x1
            return slope, bias

    def set_block_pointer(self, idx: int):
        """
        Set block to whom line is within. Block identifier is an index
        :param idx:  block identifier refer to original block list
        :return: None
        """
        self.block_pointer = idx
        return

    @staticmethod
    def compute_line_coefficients(p1, p2):
        """
        Given two points, compute the coefficients of the line equation
        a*x+b*y+c=0
        """
        x1, y1 = p1.ravel()
        x2, y2 = p2.ravel()

        if x2 == x1:
            # Vertical line: x = c
            a = 1
            b = 0
            c = -x1
        elif y1 == y2:
            # Horizontal line: y = c
            a = 0
            b = 1
            c = -y1
        else:
            a = y1 - y2
            b = x2 - x1
            c = x2 * y1 - x1 * y2

        return a, b, c

    def compute_intersection_with_line(self, line):
        """
        Compute the intersection between two m_lsd
        :param line:
        :return: (x,y) coordinates of the intersection
        """
        a1, b1, c1 = self.compute_line_coefficients(self.p1, self.p2)
        a2, b2, c2 = line.compute_line_coefficients(line.p1, line.p2)
        # Compute the intersection between two m_lsd
        determinant = a1 * b2 - a2 * b1
        if determinant == 0:
            # No intersection
            return None, None
        x = (b2 * c1 - b1 * c2) / determinant
        y = (a1 * c2 - a2 * c1) / determinant
        return x, y

    @staticmethod
    def get_line_coordinates(a, b, c, x=None, y=None):
        if x is None and y is None:
            return None, None

        if a == 0 and b == 0:
            return None, None

        if a == 0 and b != 0:
            y = c / b
            return 0, y

        if b == 0 and a != 0:
            x = c / a
            return x, 0

        if x is not None:
            y = (c - a * x) / b
            return x, y

        if y is not None:
            x = (c - b * y) / a
            return x, y

        return None, None

    def compute_intersection_with_block_boundaries(self, p1, p2, img):
        # Get the image dimensions
        if img.ndim == 3:
            height, width, _ = img.shape
        else:
            height, width = img.shape
        a, b, c = self.compute_line_coefficients(p1, p2) if self.a is None else (self.a, self.b, self.c)
        if b == 0 and np.abs(a) > 0:
            # Vertical line
            x = int(p2[0])
            x1, y1 = x, 0
            x2, y2 = x, height - 1
            return x1, y1, x2, y2, None, None, None, None
        if a == 0 and np.abs(b) > 0:
            # Horizontal line
            y = int(p2[1])
            x1, y1 = 0, y
            x2, y2 = width - 1, y
            return x1, y1, x2, y2, None, None, None, None
        x1, y1 = 0, None
        x2, y2 = width - 1, None
        x3, y3 = None, 0
        x4, y4 = None, height - 1
        x1, y1 = self.get_line_coordinates(a, b, c, x1, y1)
        x2, y2 = self.get_line_coordinates(a, b, c, x2, y2)
        x3, y3 = self.get_line_coordinates(a, b, c, x3, y3)
        x4, y4 = self.get_line_coordinates(a, b, c, x4, y4)

        return x1, y1, x2, y2, x3, y3, x4, y4

    @staticmethod
    def draw_line(x1, y1, x2, y2, block, thickness, color):
        # Draw the line
        integer = lambda x: int(np.round(x))
        return cv2.line(block.copy(), (integer(x1), integer(y1)), (integer(x2), integer(y2)), color, thickness)

    def block_draw_line(self, block, color= (0,0,255), thickness=2, extended=True):

        # convert binary block array to rgb
        if len(block.shape) == 2:
            block = cv2.cvtColor(block.astype(np.uint8), cv2.COLOR_GRAY2RGB).astype(np.uint8)

        if not extended:
            return block.copy()

        # Calculate the intersections with the image boundaries
        x1, y1, x2, y2, x3, y3, x4, y4 = self.compute_intersection_with_block_boundaries(self.p1_rel, self.p2_rel,
                                                                                         block)
        # Draw the line on the image
        extended_line = self.draw_line(x1, y1, x2, y2, block, thickness, color)

        return extended_line

    def img_draw_line(self, img, color=(0,0,255), thickness=2):
        # Calculate the intersections with the image boundaries
        x1, y1, x2, y2, x3, y3, x4, y4 = self.compute_intersection_with_block_boundaries(
            self.p1.ravel(), self.p2.ravel(), img)
        # Draw the line on the image
        extended_line = self.draw_line(x1, y1, x2, y2, img, thickness, color)

        return extended_line

    def img_draw_segment(self, img, color=(0,0,255), thickness=2):
        # Draw the line on the image
        extended_line = self.draw_line(self.p1[0], self.p1[1], self.p2[0], self.p2[1], img, thickness, color)

        return extended_line

    def distance_to_point(self, point):
        """Minimum distance between the line and the point"""
        a, b, c = self.compute_line_coefficients(self.p1, self.p2)
        x, y = point
        return np.abs(a*x+b*y+c)/np.sqrt(a**2+b**2)