import cv2
import numpy as np
import os
import sys

from shapely.geometry import Polygon
from shapely.errors import TopologicalError
from typing import List
from pathlib import Path

from automatic_methods.tree_ring_delineation.mlbrief_inbd.uruDendro.metric_influence_area import build_rays, \
    InfluenceArea
from backend.labelme_layer import LabelmeInterface, LabelmeShapeType, AL_LateWood_EarlyWood, LabelmeShape
from lib.image import resize_image_using_pil_lib

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


class INBD:
    def __init__(self, image_path, pith_mask_path, model_path, output_dir, Nr = 360, resize_factor = 1, background_path = None):
        self.image_path = image_path
        self.pith_mask = pith_mask_path
        self.model_path = model_path
        self.output_dir = output_dir
        self.python_path = self._get_python_path()
        self.Nr = Nr
        self.resize_factor = resize_factor
        self.background_path = background_path

    def _get_python_path(self):
        python_path = sys.executable
        return python_path

    def run(self):
        image_path = self._resize_image(self.image_path, self.resize_factor)
        pith_mask_path = self._resize_image(self.pith_mask, self.resize_factor, compute_shape=False)
        command = (f"{self.python_path} ./automatic_methods/tree_ring_delineation/mlbrief_inbd/INBD/main.py inference"
                   f" {self.model_path} {image_path} "
                   f"{pith_mask_path} --output {self.output_dir} ")

        os.system(command)
        root_inbd_results = self.output_dir / f"{self.model_path.parent.stem}_"
        command = (f"PYTHONPATH=\"./automatic_methods/tree_ring_delineation/mlbrief_inbd\" &&"
                   f" {self.python_path} ./automatic_methods/tree_ring_delineation/mlbrief_inbd/src/from_inbd_to_urudendro_labels.py "
                   f"--root_dataset {image_path} --root_inbd_results {root_inbd_results} --output_dir {self.output_dir} "
                   f" --center_mask_dir {pith_mask_path}")

        os.system(command)

        json_name = f"{image_path.stem}.json"
        results_dir = self.output_dir / f"inbd_urudendro_labels/{image_path.stem}"
        inbd_predictions_json_path = results_dir / json_name
        inbd_prediction_json_sampled_path = results_dir / f"{image_path.stem}_sampled.json"
        #sampling the polygons and resizing them if needed
        self._sampling_polygons(inbd_predictions_json_path, inbd_prediction_json_sampled_path, pith_mask_path)
        return inbd_prediction_json_sampled_path

    def _resize_image(self, image_path, resize_factor, compute_shape = False):
        if not compute_shape:
            image = cv2.imread(image_path)
            self.height, self.width, _ = image.shape
        image = resize_image_using_pil_lib(image, int(self.height / resize_factor), int(self.width / resize_factor))
        self.working_height, self.working_width, _ = image.shape
        resized_image_path = Path(image_path).parent / f"{Path(image_path).stem}_resized.png"
        cv2.imwrite(str(resized_image_path), image)
        return resized_image_path

    def _sampling_polygons(self, dt_file, output_path, pith_mask_path):
        image = cv2.imread(self.image_path)
        height, width, _ = image.shape

        pith_mask = cv2.imread(pith_mask_path)
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
            image_path=str(self.image_path),
            imageHeight=height,
            imageWidth=width,
            structure_list= dt_updated_poly
        )
        writer.write(args)
        return

    def rm_polygons_within_the_background(self, l_poly: List[Polygon]) -> List[Polygon]:
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