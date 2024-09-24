import numpy as np

from abc import ABC, abstractmethod



class CrossSectionTree(ABC):
    def __init__(self, image_path, pith_path=None, knots_path=None, compression_wood_path=None, annual_ring_path=None,
                 early_late_path = None, bark_path=None):

        self.image = None

        #wood structure
        self.pith = None
        self.knots = []
        self.compression_wood = []
        self.annual_rings = []
        self.bark = None


    @abstractmethod
    def draw(self, image:np.array, color: Color) -> np.array:
        """
        Draw the disk wood structure on the image
        :param image: image to draw on
        :param color: color to draw the disk wood structure
        :return:
        """
        pass

    def __str__(self):
        return f"CrossSectionTree_{self.disk_wood_structures}"