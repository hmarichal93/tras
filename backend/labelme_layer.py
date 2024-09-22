import numpy as np
import cv2

from abc import ABC, abstractmethod
from pathlib import Path
from typing import List

import pandas as pd
from shapely.geometry import Polygon, Point

from lib.io import load_json, write_json
from lib.image import Color, Drawing

from backend.abstraction_layer import UserInterface
from backend.disk_wood_structure import AnnualRing

class LabelmeShapeType:
    polygon = "polygon"


class LabelmeShape:
    def __init__(self, shape):
        self.label = shape['label']
        self.points = np.array(shape['points'])[:,[1,0]]
        self.shape_type = shape['shape_type']
        self.flags = shape['flags']

    def __str__(self):
        return f"(main_label={self.label}, shape_type={self.shape_type}, size = {self.points.shape})"

    def __repr__(self):
        return f"(main_label={self.label}, shape_type={self.shape_type}, size = {self.points.shape})"

    def area(self):
        poly  = Polygon(self.points)
        return poly.area

class LoadLabelmeObject:
    def __init__(self, json_labelme_path):
        labelme_json = load_json(json_labelme_path)
        self.parser(labelme_json)

    def parser(self, labelme_json):
        self.version = labelme_json["version"]
        self.flags = labelme_json["flags"]
        self.shapes = [LabelmeShape(shape) for shape in  labelme_json["shapes"]]
        self.imagePath = labelme_json["imagePath"]
        self.imageData = labelme_json["imageData"]
        self.imageHeight = labelme_json["imageHeight"]
        self.imageWidth = labelme_json["imageWidth"]



class LabelmeInterface(UserInterface):

    def __init__(self, version = "4.5.6", read_file_path = None, write_file_path = None):
        self.version = version
        self.read_file_path = read_file_path
        self.write_file_path = write_file_path

    def write(self, args):
        imagePath = args.get("image_path")
        imageHeight = args.get("imageHeight")
        imageWidth = args.get("imageWidth")
        structure_list = args.get("structure_list")
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
        labelme_parser = LoadLabelmeObject(self.read_file_path)
        structure_list = [self.from_labelme_shape_to_structure(shape) for shape in labelme_parser.shapes]
        structure_list.sort(key=lambda x: x.area())
        return structure_list

    @abstractmethod
    def from_structure_to_labelme_shape(self, structure_list):
        pass

    @abstractmethod
    def from_labelme_shape_to_structure(self, shape: LabelmeShape):
        pass


class AL_LateWood_EarlyWood(LabelmeInterface):

    def __init__(self, json_labelme_path, write_file_path):
        super().__init__(read_file_path = json_labelme_path, write_file_path = write_file_path)

    def from_structure_to_labelme_shape(self, structure_list):
        shapes = []
        for structure in structure_list:
            pass

        return shapes

    def from_labelme_shape_to_structure(self, shape: LabelmeShape):
        return shape




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




        cv2.imwrite("output/debug.png", image)
        return early


        #if len(early_rings_within_late_ring) == 0:
        #    return None
        #return early_rings_within_late_ring[0]

    def read(self):
        late_structures = self.al_latewood.read()

        if self.al_earlywood is not None:
            early_structures = self.al_earlywood.read()

        else:
            early_structures = [None]*len(late_structures)

        idx = 0
        previous = None
        list_annual_rings = []
        height, width =1006, 900
        image = np.zeros((height, width, 3), dtype=np.uint8)
        for late in late_structures:
            try:
                early = self.get_early_ring_within_late_rings(late, previous, early_structures, image.copy())


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


def export_results(labelme_latewood_path : str, labelme_earlywood_path : str, image_path : str, metadata: dict,
                   output_path="output/measures.csv", draw=True):
    #metadata
    year = metadata["year"]
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
            eccentricity_phase = np.arctan2(y, x)[0] if x != 0 else 0

        ring_similarity_factor = ring.similarity_factor()

        #save results
        df.loc[idx] = [
            f"{ring.main_label}", f"{ring.secondary_label}", year,
            area, earlywood_area, latewood_area, area_latewood_earlywood,
            width_annual_ring, width_earlywood, width_latewood, width_latewood_earlywood,
            eccentricity_module, eccentricity_phase, ring_similarity_factor
        ]
        if draw:
            image_debug = ring.draw(image.copy(), full_details=True, opacity=0.1)
            image_full = ring.draw_rings(image_full, thickness=3)

            cv2.imwrite(f"output/ring_{idx}.png", image_debug)

    cv2.imwrite("output/annual_rings.png", image_full)
    df.to_csv(output_path, index=False)
    return

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

def main():
    root = "./input/C14/"
    image_path = f"{root}image.jpg"
    labelme_latewood_path = f"{root}latewood.json"
    labelme_earlywood_path = f"{root}earlywood.json"
    metadata = {
        "year": 2000,
        "pixels_millimeter_relation": 10 / 52
    }
    export_results(labelme_latewood_path, labelme_earlywood_path, image_path, metadata)




if __name__ == "__main__":
    main()

