from abc import ABC, abstractmethod

import cv2
import numpy as np
from shapely.geometry import Polygon, Point

from lib.image import Color, Drawing

"""
Generic Cross Section Tree structure
"""
class DiskWoodStructure(Polygon):
    def __init__(self, exterior = None, hole = None):
        self.external_points = exterior
        self.internal_points = hole
        if hole is not None:
            super().__init__(exterior, [hole])
        else:
            super().__init__(exterior)
    @abstractmethod
    def draw(self, image:np.array, color: Color) -> np.array:
        """
        Draw the disk wood structure on the image
        :param image: image to draw on
        :param color: color to draw the disk wood structure
        :return:
        """
        pass


"""
Specific Cross Section Tree structures
"""

class AnnualRing(DiskWoodStructure):
    """
    Annual ring structure. In conifers, it is composed of late and early wood structures. That is why the parameter
    late_early_wood_boundary is used to define the boundary between them ,and it is optional.
    """
    def __init__(self, exterior = None, hole = None, late_early_wood_boundary = None, main_label = None,
                 secondary_label = None):
        if exterior is None:
            raise ValueError("Exterior points must be provided")
        if hole is None:
            #pith
            super().__init__(exterior)
        else:
            #check if the hole is inside the exterior
            if not Point(hole[0]).within(Polygon(exterior)):
                raise ValueError("Hole points must be inside the exterior points")
            super().__init__(exterior, hole)
        self.main_label = main_label
        self.secondary_label = secondary_label
        self.late_wood = None
        self.early_wood = None
        if late_early_wood_boundary is not None:
            self.late_wood = AnnualRing(exterior, late_early_wood_boundary, main_label=f"{self.main_label}_late_wood",
                                        secondary_label=secondary_label)
            self.early_wood = AnnualRing(late_early_wood_boundary, hole, main_label=f"{self.main_label}_early_wood",
                                        secondary_label=secondary_label)
        else:
            self.late_wood = self



    def compute_non_intersecting_area(self, other_shapes):
        """
        Compute the area of the disk wood structure that does not intersect with other shapes
        :param other_shapes: list of other shapes to compute the area
        :return: area of the disk wood structure that does not intersect with other shapes
        """
        if len(other_shapes) == 0:
            return self.area
        else:
            #compute the intersection area
            intersection_area = 0
            #sort the shapes by area
            other_shapes = sorted(other_shapes, key=lambda x: x.area, reverse=True)
            #remove the shapes that intersect each other
            for i in range(len(other_shapes)):
                for j in range(i+1, len(other_shapes)):
                    if other_shapes[i].intersects(other_shapes[j]):
                        other_shapes[i] = other_shapes[i].difference(other_shapes[j])

            #compute the intersection area
            for shape in other_shapes:
                intersection_area += self.intersection(shape).area
            return self.area - intersection_area


    def get_centroid(self):
        outter_boundary = Polygon(self.exterior)
        return outter_boundary.centroid

    def similarity_factor(self):
        perimeter = self.exterior.length
        area = self.area
        radii_perfect_circle = np.sqrt(area / np.pi)
        ring_similarity_factor = 1 - (perimeter - 2*np.pi*radii_perfect_circle) / perimeter
        return ring_similarity_factor

    def draw(self, image:np.array, color: Color = Color.red, thickness: int = 1, opacity: float = 0.3, full_details=True) -> np.array:
        """
        Draw the disk wood structure on the image
        :param image: image to draw on
        :param color: color to draw the disk wood structure
        :param full_details: if True, draw the late and early wood
        :return:
        """
        if full_details:
            #draw late wood
            if self.late_wood is not None:
                image = self.late_wood.draw(image, Color.blue, thickness, opacity, False)

            #draw early wood
            if self.early_wood is not None:
                image = self.early_wood.draw(image, Color.green, thickness, opacity, False)

        else:
            mask_exterior = np.zeros_like(image)
            mask_exterior = Drawing.fill(self.exterior.coords, mask_exterior, Color.white, opacity=1)
            ######
            mask_interiors = np.zeros_like(image)
            inner_points = np.array([list(interior.coords) for interior in self.interiors]).squeeze()
            if len(inner_points)>0:
                aux_poly = Polygon(inner_points)
                mask_interiors = Drawing.fill(aux_poly.exterior.coords, mask_interiors,  Color.white, opacity=1)

            mask = mask_exterior - mask_interiors

            y,x = np.where(mask[:,:,0]>0)
            mask[y,x] = color
            cv2.addWeighted(mask, opacity, image, 1 - opacity, 0, image)


        return image

    def draw_rings(self, image:np.array, thickness: int = 1) -> np.array:
        """
        Draw the disk wood structure on the image
        :param image: image to draw on
        :param color: color to draw the disk wood structure
        :param full_details: if True, draw the late and early wood
        :return:
        """
        image = Drawing.curve(self.exterior.coords, image, Color.blue, thickness)

        if self.early_wood is not None:
            image = Drawing.curve(self.early_wood.exterior.coords, image, Color.red, thickness)


        return image

class KnotWood(DiskWoodStructure):

    def __init__(self, exterior = None, label = None):
        super().__init__(exterior)
        self.label = label

    def draw(self, image:np.array, color: Color) -> np.array:
        """
        Draw the disk wood structure on the image
        :param image: image to draw on
        :param color: color to draw the disk wood structure
        :return:
        """
        pass

    def __str__(self):
        return f"KnotWood_{self.label}"

class PithWood(DiskWoodStructure):
    def __init__(self, exterior = None):
        super().__init__(exterior)

    def __str__(self):
        return "Pith"

    def draw(self, image:np.array, color: Color) -> np.array:
        """
        Draw the disk wood structure on the image
        :param image: image to draw on
        :param color: color to draw the disk wood structure
        :return:
        """
        pass

class CompressionWood(DiskWoodStructure):
    def __init__(self, exterior = None, label= None):
        super().__init__(exterior)
        self.label = label

    def __str__(self):
        return f"CompressionWood_{self.label}"

    def draw(self, image:np.array, color: Color) -> np.array:
        """
        Draw the disk wood structure on the image
        :param image: image to draw on
        :param color: color to draw the disk wood structure
        :return:
        """
        pass

class Bark(DiskWoodStructure):
    def __init__(self, exterior = None, hole=None, label = None):
        super().__init__(exterior, hole)
        self.label = label

    def __str__(self):
        return f"Bark_{self.label}"

    def draw(self, image:np.array, color: Color) -> np.array:
        """
        Draw the disk wood structure on the image
        :param image: image to draw on
        :param color: color to draw the disk wood structure
        :return:
        """
        pass


