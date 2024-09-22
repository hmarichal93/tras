"""
Module for computing important metrics
"""
import numpy as np
import pandas as pd

from shapely.geometry import Polygon, LineString

import numpy as np
import cv2
import datetime

from pathlib import Path

import pandas as pd
from shapely.geometry import Polygon, Point

from lib.image import Color, Drawing

from backend.labelme_layer import AL_AnnualRings

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




def export_results(labelme_latewood_path : str, labelme_earlywood_path : str, image_path : str, metadata: dict,
                   output_path="output/measures.csv", draw=True):
    #metadata
    year = metadata["year"]
    year = datetime.datetime(year, 1, 1)

    pixels_millimeter_relation = float(metadata["pixels_millimeter_relation"])

    image = cv2.imread(image_path)
    al_annual_rings = AL_AnnualRings(late_wood_path=Path(labelme_latewood_path),
                                     early_wood_path=Path(labelme_earlywood_path))
    annual_rings_list = al_annual_rings.read()

    df = pd.DataFrame(columns=["Annual Ring (main_label)", "Annual Ring (secondary label)", "Year",
                               "Area [mm2]", "Area EW [mm2]", "Area LW [mm2]", "Area LW/EW (%)",
                               "Width Annual Ring [mm]", "Width EW [mm]", "Width LW [mm]", "Width LW/EW (%)",
                               "Eccentricity Module [mm]", "Eccentricity Phase [°]", "Ring Similarity Factor [0-1]"])

    pith = Point(0, 0)
    image_full = image.copy()
    for idx, ring in enumerate(annual_rings_list):
        #area
        area = ring.area
        latewood_area = ring.late_wood.area if ring.late_wood is not None else 0
        earlywood_area = ring.early_wood.area if ring.early_wood is not None else 0
        area_latewood_earlywood = latewood_area / earlywood_area if latewood_area > 0 and earlywood_area > 0 else 0

        #width
        width_annual_ring = ring.equivalent_radii()
        width_latewood = ring.late_wood.equivalent_radii() if ring.late_wood is not None else 0
        width_earlywood = ring.early_wood.equivalent_radii() if ring.early_wood is not None else 0
        width_latewood_earlywood = width_latewood / width_earlywood if width_latewood > 0 and width_earlywood > 0 else 0

        #eccentricity
        if idx == 0:
            pith = ring.centroid
        eccentricity_module = ring.centroid.distance(pith)
        if eccentricity_module == 0:
            eccentricity_phase = 0
        else:
            x, y = (ring.centroid - pith).coords.xy
            eccentricity_phase = np.arctan2(y, x)[0]* 180 / np.pi if x != 0 else 0

        ring_similarity_factor = ring.similarity_factor()

        #save results
        df.loc[idx] = [
            f"{ring.main_label}", f"{ring.secondary_label}", year.year,
            area * (pixels_millimeter_relation**2), earlywood_area * (pixels_millimeter_relation**2),
            latewood_area * (pixels_millimeter_relation**2) , area_latewood_earlywood * (pixels_millimeter_relation**2),
            width_annual_ring * pixels_millimeter_relation, width_earlywood * pixels_millimeter_relation ,
            width_latewood * pixels_millimeter_relation , width_latewood_earlywood * pixels_millimeter_relation,
            eccentricity_module * pixels_millimeter_relation, eccentricity_phase, ring_similarity_factor
        ]
        image_full = ring.draw_rings(image_full, thickness=3)
        #increase year by 1 year more
        year = year + datetime.timedelta(days=365)
        if draw:
            thickness= 3
            image_debug = ring.draw(image.copy(), full_details=True, opacity=0.1)
            image_debug = Drawing.curve(ring.exterior.coords, image_debug, Color.black, thickness)
            inner_points = np.array([list(interior.coords) for interior in ring.interiors]).squeeze()
            if len(inner_points) > 0:
                aux_poly = Polygon(inner_points)
                image_debug = Drawing.curve(aux_poly.exterior.coords, image_debug, Color.black, thickness)
            output_name = f"output/{ring.main_label}.png"
            cv2.imwrite(output_name, image_debug)

    cv2.imwrite("output/rings.png", image_full)
    #format df %.02f
    df = df.round(2)
    df.to_csv(output_path, index=False)
    return



def main():
    root = "./input/C14/"
    image_path = f"{root}image.jpg"
    labelme_latewood_path = f"{root}latewood.json"
    labelme_earlywood_path = f"{root}earlywood.json"
    metadata = {
        "year": 1993,
        "pixels_millimeter_relation": 10 / 52
    }
    export_results(labelme_latewood_path, labelme_earlywood_path, image_path, metadata)




if __name__ == "__main__":
    main()