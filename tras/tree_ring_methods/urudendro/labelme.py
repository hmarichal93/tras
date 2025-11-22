import numpy as np
import cv2
import os

from pathlib import Path
from typing import List
from abc import ABC, abstractmethod
from shapely.geometry import Polygon, Point

from urudendro.io import load_json, write_json, load_image
from urudendro.drawing import Color, Drawing

class UserInterface(ABC):
    @abstractmethod
    def read(self):
        pass

    @abstractmethod
    def write(self, data):
        pass
#from backend.disk_wood_structure import AnnualRing

class PointLabelme(Point):
    def __init__(self, x, y, label):
        self.label = label
        super().__init__([x, y])

    def scale(self, new_width, new_height):
        x, y = self.coords.xy
        x = x[0] * new_width
        y = y[0] * new_height
        return PointLabelme(x, y, self.label)

class LabelmeShapeType:
    polygon = "polygon"
    point = "point"
    linestrip = "linestrip"
    line = "line"



def load_ring_shapes(json_file):
    al = AL_LateWood_EarlyWood(json_file)
    shapes = al.read()
    #tranpose points
    shapes_l = []
    for s in shapes:
        l = LabelmeShape(dict(
            label = s.label,
            points = s.points,
            shape_type = s.shape_type,
            flags = s.flags
        ))
        shapes_l.append(l)
    return shapes_l
class LabelmeShape:
    def __init__(self, shape):
        self.label = shape['label']
        self.points = np.array(shape['points'])[:,[1,0]]
        self.shape_type = shape['shape_type']
        self.flags = shape['flags']
        if self.shape_type == LabelmeShapeType.polygon:
            self.area = Polygon(self.points).area

    def to_dict(self):
        return dict(
            label = self.label,
            points = self.points.tolist(),
            shape_type = self.shape_type,
            flags = self.flags
        )

    def set_flag(self, value):
        self.flags = value

    def __str__(self):
        return f"(main_label={self.label}, shape_type={self.shape_type}, size = {self.points.shape}, flag = {self.flags })"

    def __repr__(self):
        return f"(main_label={self.label}, shape_type={self.shape_type}, size = {self.points.shape}, flag = {self.flags })"


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
        if len(self.shapes)> 0:
            check_if_all_shapes_are_polygon = (len([s for s in self.shapes if s.shape_type == LabelmeShapeType.polygon])
                                               ==len(self.shapes))
            if check_if_all_shapes_are_polygon:
                self.shapes.sort(key=lambda x: x.area)
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
        structure_list.sort(key=lambda x: x.area)
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

    def __init__(self, json_labelme_path = None, write_file_path = None, image_path = None):
        super().__init__(read_file_path = json_labelme_path, write_file_path = write_file_path)
        self.image_path = image_path

    def from_structure_to_labelme_shape(self, structure_list):
        return structure_list

    def from_labelme_shape_to_structure(self, shape: LabelmeShape):
        return shape

    def write_list_of_points_to_labelme_json(self, shapes: List[List[List[int]]], labels = None):
        if labels is None:
            shapes = [LabelmeShape(dict(points=s, shape_type=LabelmeShapeType.polygon, flags={}, label=str(idx)))
                      for idx, s in enumerate(shapes)]
        else:
            shapes = [LabelmeShape(dict(points=s, shape_type=LabelmeShapeType.polygon, flags={}, label=str(idx)))
                      for idx, s in zip(labels, shapes)]
        object = LabelmeObject()
        object.from_memory(shapes=shapes, imagePath=str(Path(self.image_path).name))
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



def ring_relabelling(image_path: str, json_path: str, harvest_date: int, output_path: str = None):
    """
    Relabel the rings according to the harvest date
    :param json_path: path to the json file
    :param harvest_date: harvest date
    :return:
    """
    output_path = json_path if output_path is None else output_path
    al = AL_LateWood_EarlyWood(json_path, output_path, image_path=image_path)
    shapes = al.read()
    shapes = shapes[::-1]
    labels = []
    for idx, shape in enumerate(shapes):
        date = harvest_date - (idx)
        labels.append(date)
    al.write_list_of_points_to_labelme_json([shape.points for shape in shapes], labels)

    return



def add_prefix_to_labels(json_path, image_path, prefix, output_path):
    al = AL_LateWood_EarlyWood(json_path, output_path, image_path=image_path)
    shapes = al.read()
    shapes = shapes[::-1]
    labels = []
    for idx, shape in enumerate(shapes):
        #if prefix  not in shape.label:
        labels.append(f"{prefix}_{shape.label}")
        #return
    al.write_list_of_points_to_labelme_json([shape.points for shape in shapes], labels)
    return

def write_ring_shapes(shapes, output_path_ann, image_path):
    al = AL_LateWood_EarlyWood(write_file_path=str(output_path_ann), image_path=str(image_path))
    object = LabelmeObject()
    object.from_memory(shapes=shapes, imagePath=str(Path(image_path)))
    json_content = object.to_dict()
    al.write(json_content)
    return


def draw_curves_over_image(
            json_path = "/data/maestria/datasets/Pinus_Taeda/PinusTaedaV2/annotations/oscuro_claro_tardio(importante)/postprocessed/veronica/B2C.json" ,
            image_path = "/data/maestria/datasets/Pinus_Taeda/PinusTaedaV2/images/original/B2C.jpg",
            output_path="output.jpg"):

    al = AL_LateWood_EarlyWood(json_path, output_path, image_path=image_path)
    shapes = al.read()
    image = load_image(image_path)
    #convert image to BGR
    canvas = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    for shape in shapes:
        shapely_curve = Polygon(shape.points)
        canvas = Drawing.curve(shapely_curve.exterior.coords, canvas, color=Color.blue, thickness=2)

    cv2.imwrite(output_path, canvas)
    return

if __name__ == "__main__":
    draw_curves_over_image()


