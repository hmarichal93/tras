import numpy as np
import cv2
import os

from abc import abstractmethod
from pathlib import Path
from typing import List

from shapely.geometry import Polygon, Point
from lib.io import load_json, write_json
from lib.image import Color, Drawing, load_image
from backend.abstraction_layer import UserInterface
from backend.disk_wood_structure import AnnualRing

class LabelmeShapeType:
    polygon = "polygon"
    point = "point"
    linestrip = "linestrip"
    line = "line"


class LabelmeShape:
    def __init__(self, shape):
        self.label = shape['label']
        self.points = np.array(shape['points'])[:,[1,0]]
        self.shape_type = shape['shape_type']
        self.flags = shape['flags']

    def to_dict(self):
        return dict(
            label = self.label,
            points = self.points.tolist(),
            shape_type = self.shape_type,
            flags = self.flags
        )

    def __str__(self):
        return f"(main_label={self.label}, shape_type={self.shape_type}, size = {self.points.shape})"

    def __repr__(self):
        return f"(main_label={self.label}, shape_type={self.shape_type}, size = {self.points.shape})"

    def area(self):
        poly  = Polygon(self.points)
        return poly.area

class LabelmeObject:
    def __init__(self, json_labelme_path = None):
        if json_labelme_path is not None:
            labelme_json = load_json(json_labelme_path)
            self.parser(labelme_json)

    def from_memory(self, version : str = "5.0", flags : dict = None, shapes : List[LabelmeShape] = None,
                    imagePath : str = "", imageData : str = "", imageHeight : str = "",
                    imageWidth : str = ""):
        if shapes is None:
            shapes = []
        self.version = version
        self.flags = flags
        self.shapes = shapes
        self.imagePath = imagePath
        self.imageData = imageData
        self.imageHeight = imageHeight
        self.imageWidth = imageWidth

    def to_dict(self):
        return dict(
            version = self.version,
            flags = self.flags,
            shapes = [ s.to_dict() for s in self.shapes ],
            imagePath = self.imagePath,
            imageData = self.imageData,
            imageHeight = self.imageHeight,
            imageWidth = self.imageWidth
        )
    def parser(self, labelme_json):
        self.version = labelme_json["version"]
        self.flags = labelme_json["flags"]
        self.shapes = [LabelmeShape(shape) for shape in  labelme_json["shapes"]]
        if len(self.shapes)> 0 and self.shapes[0].shape_type == LabelmeShapeType.polygon:
            self.shapes.sort(key=lambda x: x.area())
        self.imagePath = labelme_json["imagePath"]
        self.imageData = labelme_json["imageData"]
        self.imageHeight = labelme_json["imageHeight"]
        self.imageWidth = labelme_json["imageWidth"]



class LabelmeInterface(UserInterface):

    def __init__(self, version = "4.5.6", read_file_path = None, write_file_path = None, edit=False):
        self.version = version
        self.read_file_path = read_file_path
        self.write_file_path = write_file_path
        self.edit = edit

    def write(self, args):
        imagePath = args.get("imagePath")
        imageHeight = args.get("imageHeight")
        imageWidth = args.get("imageWidth")
        structure_list = args.get("shapes")
        shapes = self.from_structure_to_labelme_shape(structure_list)

        labelme_dict = dict(
            version = self.version,
            flags = {},
            shapes = shapes,
            imagePath = str(imagePath),
            imageData = None,
            imageHeight = imageHeight,
            imageWidth = imageWidth
        )

        write_json(labelme_dict, self.write_file_path)

    def read(self):
        labelme_parser = LabelmeObject(self.read_file_path)
        structure_list = [self.from_labelme_shape_to_structure(shape) for shape in labelme_parser.shapes]
        structure_list.sort(key=lambda x: x.area())
        return structure_list

    @abstractmethod
    def from_structure_to_labelme_shape(self, structure_list):
        pass

    @abstractmethod
    def from_labelme_shape_to_structure(self, shape: LabelmeShape):
        pass

    def interface(self):
        if self.edit:
            command = f"labelme {self.write_file_path}"

        else:
            command = f"labelme {self.read_file_path} -O {self.write_file_path}  --nodata "

        print(command)
        os.system(command)

    @abstractmethod
    def parse_output(self):
        pass

    @staticmethod
    def load_shapes(output_path):
        try:
            json_content = load_json(output_path)
            l_rings = []
            for ring in json_content['shapes']:
                l_rings.append(Polygon(np.array(ring['points'])[:, [1, 0]].tolist()))

        except FileNotFoundError:
            l_rings = []

        return l_rings








class AL_LateWood_EarlyWood(LabelmeInterface):

    def __init__(self, json_labelme_path, write_file_path, image_path = None):
        super().__init__(read_file_path = json_labelme_path, write_file_path = write_file_path)
        self.image_path = image_path

    def from_structure_to_labelme_shape(self, structure_list):
        return structure_list

    def from_labelme_shape_to_structure(self, shape: LabelmeShape):
        return shape

    def write_list_of_points_to_labelme_json(self, shapes: List[List[List[int]]]):
        shapes = [LabelmeShape(dict(points=s, shape_type=LabelmeShapeType.polygon, flags={}, label=str(idx)))
                  for idx, s in enumerate(shapes)]
        object = LabelmeObject()
        object.from_memory(shapes=shapes, imagePath=str(self.image_path))
        json_content = object.to_dict()
        self.write(json_content)

        return

    def parse_output(self):
        pass

def resize_annotations( image_orig_path, image_resized_path, annotations_orig_path):
    """
    Resize the annotations to the new image size
    :param image_orig_path: path to the original image file
    :param image_resized_path: path to the resized image path
    :param annotations_orig_path: annotations made in the original resolution in labelme format
    :return: new annotation file path
    """
    image_orig = load_image(image_orig_path)
    H, W = image_orig.shape[:2]
    image_r = load_image(image_resized_path)
    h, w = image_r.shape[:2]
    gt_path_resized = str(annotations_orig_path).replace(".json", "resized.json")
    al = AL_LateWood_EarlyWood(annotations_orig_path,
                               gt_path_resized,
                               image_path=str(image_resized_path)
                               )
    shapes = al.read()
    shapes = [(np.array(s.points) * [h / H, w / W]).tolist() for s in shapes]
    al.write_list_of_points_to_labelme_json(shapes)

    return gt_path_resized




class AL_AnnualRings:

    def __init__(self, early_wood_path: Path = None, late_wood_path : Path = None):
        self.al_earlywood = None
        if early_wood_path is not None:
            early_read_path, early_write_path = self._generate_read_and_write_paths(early_wood_path)
            self.al_earlywood = AL_LateWood_EarlyWood(early_read_path, early_write_path)

        if late_wood_path is None:
            raise ValueError("late_wood_path is None")

        late_read_path, late_write_path = self._generate_read_and_write_paths(late_wood_path)
        self.al_latewood = AL_LateWood_EarlyWood(late_read_path, late_write_path)


    def _generate_read_and_write_paths(self, file_path : Path):
        parent_dir = file_path.parent
        file_name = file_path.stem
        read_file_path = parent_dir / f"{file_name}_read.json"
        write_file_path = parent_dir / f"{file_name}_write.json"
        return read_file_path, write_file_path

    def get_early_ring_within_late_rings(self, late_ring: LabelmeShape, previous_ring: LabelmeShape ,
                                         early_rings_list: List[LabelmeShape],  image: np.array):
        if previous_ring is None:
            late_ring_poly = Polygon(late_ring.points)
            return None
        else:
            late_ring_poly = Polygon(late_ring.points.tolist(),[previous_ring.points.tolist()])
        ###
        #early_rings_within_late_ring = [early for early in early_rings_list if Polygon(early.points).within(late_ring_poly)]
        outter_poly = Polygon(late_ring.points)
        inner_poly = Polygon(previous_ring.points)
        image = draw_circular_region(image, outter_poly, inner_poly, Color.red, 0.3)
        region = Polygon(outter_poly.exterior.coords, [inner_poly.exterior.coords])
        early = None
        for early in early_rings_list:
            poly_early = Polygon(early.points)
            image = Drawing.curve(poly_early.exterior, image, Color.blue, 1)
            does_intersect = poly_early.intersects(region)
            if does_intersect:
                break

        return early

    def read(self):
        late_structures = self.al_latewood.read()

        if self.al_earlywood is not None:
            early_structures = self.al_earlywood.read()

        else:
            early_structures = None

        idx = 0
        previous = None
        list_annual_rings = []
        height, width =1006, 900
        image = np.zeros((height, width, 3), dtype=np.uint8)
        for late in late_structures:
            try:
                early = self.get_early_ring_within_late_rings(late, previous, early_structures, image.copy()) if early_structures is not None else None


                if idx == 0:
                    ring = AnnualRing(exterior=late.points, hole=None, late_early_wood_boundary=early.points
                           if early is not None else None, main_label=late.label,
                                      secondary_label= early.label if early is not None else None)
                else:
                    ring = AnnualRing(exterior=late.points, hole=previous.points, late_early_wood_boundary=early.points
                           if early is not None else None, main_label=late.label,
                                      secondary_label= early.label if early is not None else None)

            except ValueError as e:
                print(f"Error: {e}")
                continue
            list_annual_rings.append(ring)
            previous = late
            idx += 1

        return list_annual_rings


    def write(self, shape: LabelmeShape):
        pass

def draw_circular_region(image, poly_outter, poly_inner, color, opacity):
    mask_exterior = np.zeros_like(image)
    mask_exterior = Drawing.fill(poly_outter.exterior.coords, mask_exterior, Color.white, opacity=1)
    ######
    mask_interiors = np.zeros_like(image)
    mask_interiors = Drawing.fill(poly_inner.exterior.coords, mask_interiors, Color.white, opacity=1)

    mask = mask_exterior - mask_interiors

    y, x = np.where(mask[:, :, 0] > 0)
    mask[y, x] = color
    cv2.addWeighted(mask, opacity, image, 1 - opacity, 0, image)
    return image


