

class CrossSectionTree(ABC):
    def __init__(self, image_path, pith_path=None, knots_path=None, compression_wood_path=None, annual_ring_path=None,
                 early_late_path = None, bark_path=None):

        self.image = load_image(image_path)
        self.pith = labelme_generator(pith_path, shape_type = ShapeType.PITH)
        self.knots = labelme_generator(knots_path,shape_type = ShapeType.KNOT_WOOD)
        self.compression_wood = labelme_generator(compression_wood_path, shape_type = ShapeType.COMPRESSION_WOOD)
        self.annual_rings = labelme_generator(annual_ring_path, early_late_path, shape_type = ShapeType.ANNUAL_RING)
        self.bark = labelme_generator(bark_path, ShapeType.BARK)

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