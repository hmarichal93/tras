import cv2
import numpy as np

from shapely.geometry import Polygon
from shapely.errors import TopologicalError
from typing import List
from pathlib import Path
from abc import abstractmethod

from automatic_methods.tree_ring_delineation.mlbrief_inbd.uruDendro.metric_influence_area import build_rays, \
    InfluenceArea
from backend.labelme_layer import (LabelmeInterface, LabelmeShapeType, AL_LateWood_EarlyWood, LabelmeShape,
                                   resize_annotations)
from lib.image import resize_image_using_pil_lib, load_image, write_image
from lib.io import get_python_path

class LabelmeWriter(LabelmeInterface):
    def __init__(self, write_file_path):
        super().__init__(write_file_path=write_file_path)

    def from_structure_to_labelme_shape(self, polygon_list):
        shapes = []
        for idx, poly in enumerate(polygon_list):

            shape = dict(
                label= str(idx),
                points= [[int(y), int(x)] for x, y in poly.exterior.coords],
                shape_type=LabelmeShapeType.polygon,
                flags={}
            )
            shapes.append(shape)
        return shapes

    def from_labelme_shape_to_structure(self, shapes):
        pass

    def parse_output(self):
        pass


class Model:
    def __init__(self, image_path, pith_mask_path, model_path, output_dir, Nr = 360, resize_factor = 1, background_path = None):
        self.image_path = image_path
        self.pith_mask = pith_mask_path
        self.model_path = model_path
        self.output_dir = output_dir
        self.python_path = get_python_path()
        self.Nr = Nr
        self.resize_factor = resize_factor
        self.background_path = background_path



    @abstractmethod
    def run(self):
        pass

    def _resize_image(self, image_path: Path, resize_factor: float, compute_shape: bool = False)-> Path:
        if not compute_shape:
            image = load_image(image_path)
            self.height, self.width, _ = image.shape
        image = resize_image_using_pil_lib(image, int(self.height / resize_factor), int(self.width / resize_factor))
        self.working_height, self.working_width, _ = image.shape
        resized_image_path = Path(image_path).parent / f"{Path(image_path).stem}_resized.png"
        write_image(str(resized_image_path), image)
        return resized_image_path

    def _sampling_polygons(self, dt_file, output_path, pith_mask_path):
        image = load_image(self.image_path)
        height, width, _ = image.shape

        pith_mask = load_image(pith_mask_path)
        pith_mask = cv2.cvtColor(pith_mask, cv2.COLOR_BGR2GRAY)
        #get the center of the pith mask to build the rays
        x, y = np.where(pith_mask == 255)
        center = (int(np.mean(x)), int(np.mean(y)))

        l_rays = build_rays(self.Nr, height, width, center)

        al_wood = AL_LateWood_EarlyWood(dt_file, None)
        dt_shapes = al_wood.read()
        dt_sampled_poly = self.sampling_rings(dt_shapes, l_rays, center)
        if self.resize_factor != 1:
            h_f = self.height / self.working_height
            w_f = self.width / self.working_width
            dt_updated_poly = []
            for poly in dt_sampled_poly:
                y, x = poly.exterior.coords.xy
                x = [x * w_f for x in x]
                y = [y * h_f for y in y]
                dt_updated_poly.append(Polygon(zip(y, x)))
        else:
            dt_updated_poly = dt_sampled_poly

        dt_updated_poly = self.rm_polygons_within_the_background(dt_updated_poly)

        writer = LabelmeWriter( write_file_path = output_path)
        args = dict(
            imagePath=str(self.image_path),
            imageHeight=height,
            imageWidth=width,
            shapes= dt_updated_poly
        )
        writer.write(args)
        return

    def rm_polygons_within_the_background(self, l_poly: List[Polygon]) -> List[Polygon]:
        if self.resize_factor != 1:
            image_resized_path = self._resize_image(self.image_path, self.resize_factor)
            new_annotations_path = resize_annotations(self.image_path, image_resized_path,
                                                      self.background_path)
            self.background_path = new_annotations_path
        background = AL_LateWood_EarlyWood(self.background_path, None).read()

        background_poly = Polygon(background[0].points)
        l_poly_processed = []
        for poly in l_poly:
            try:
                if poly.within(background_poly):
                    l_poly_processed.append(poly)
            except TopologicalError:
                continue

        return l_poly_processed


    @staticmethod
    def sampling_rings(l_shapes: List[LabelmeShape], l_rays, center):
        l_poly_samples = []
        cy, cx = center

        for idx, shape in enumerate(l_shapes):
            poly = Polygon(shape.points)
            sampled_poly = InfluenceArea._sampling_poly(poly, cy, cx, l_rays)
            if sampled_poly is None:
                continue

            l_poly_samples.append(sampled_poly)


        return l_poly_samples