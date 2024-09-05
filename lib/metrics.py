"""
Module for computing important metrics
"""
import numpy as np
import pandas as pd

from shapely.geometry import Polygon, LineString



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

class Ring:
    def __init__(self, inner_curve, outer_curve, year = None,
                 early_wood_curve = None):

        self.inner_edge = inner_curve
        self.outer_edge = outer_curve
        self.polygon = Polygon([inner_curve, outer_curve])
        self.year = year
        self.early_wood_curve = early_wood_curve


    def area(self):
        self.area_in_pixels = self.polygon.area
        return self.area_in_pixels

    @staticmethod
    def ring_width(outer_edge):
        "equivalent radii"
        outer_polygon = Polygon([outer_edge])
        circle_equivalent_area = outer_polygon.area # area of the circle with the same area as the ring (outer edge)
        ring_width = np.sqrt( circle_equivalent_area / np.pi )
        return ring_width, circle_equivalent_area

    def annual_ring_width(self):
        annual_ring_width, circle_equivalent_area = self.ring_width(self.outer_edge)
        return annual_ring_width

    @staticmethod
    def ring_width_estimated(outer_ring, inner_ring, radii_list):
        """
        Estimate the ring width.
        :param outer_ring:
        :param inner_ring:
        :param radii_list:
        :return:
        """
        poly_outer_ring = Polygon([outer_ring])
        poly_inner_ring = Polygon([inner_ring])
        ring_width_per_direction = []
        for ray in radii_list:
            intersection = poly_outer_ring.intersection(ray)
            if intersection.is_empty:
                continue

            intersection = intersection.intersection(poly_inner_ring)
            if intersection.is_empty:
                continue

            ring_width_per_direction.append(intersection.length)

        mean_ring_width = np.mean(ring_width_per_direction)
        std_ring_width = np.std(ring_width_per_direction)

        return mean_ring_width, std_ring_width


    def early_wood_area(self):
        if self.early_wood_curve is None:
            return 0
        early_wood_polygon = Polygon([self.inner_edge, self.early_wood_curve])
        early_wood_area = early_wood_polygon.area
        return early_wood_area

    def late_wood_area(self):
        if self.early_wood_curve is None:
            return 0
        late_wood_area = self.area() - self.early_wood_area()
        return late_wood_area

    def late_wood_width(self):
        late_wood_width, late_wood_area = self.ring_width(self.outer_edge)
        return late_wood_width





class Metrics:
    def __init__(self, early_wood_path = None, late_wood_path = None):
        self.early_wood_path = early_wood_path
        self.late_wood_path = late_wood_path




def main():


    return


if __name__=="__main__":
    main()