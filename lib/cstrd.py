import os
import numpy as np

from shapely.geometry import LineString, Polygon

from lib.models import Model
from lib.image import load_image
from lib.io import load_json
from lib.cstrd_lib.metric_influence_area import InfluenceArea
from backend.labelme_layer import resize_annotations, AL_LateWood_EarlyWood


class CSTRD(Model):
    def __init__(self, image_path, pith_mask_path, model_path, output_dir, Nr = 360, resize_factor = 1,
                 background_path = None, sigma=1.0, th_low=0.1, th_hight=0.3, gt_ring_json=None,
                 include_gt_rings_in_output=False):
        super().__init__(image_path, pith_mask_path, model_path, output_dir, Nr, resize_factor, background_path)
        self.sigma = sigma
        self.th_low = th_low
        self.th_hight = th_hight
        self.gt_ring_json = gt_ring_json
        self.include_gt_rings_in_output = include_gt_rings_in_output

    def run(self):
        image_path = self._resize_image(self.image_path, self.resize_factor)
        pith_mask_path = self._resize_image(self.pith_mask, self.resize_factor, compute_shape=False)
        pith_mask = load_image(pith_mask_path)
        #convert to gray scale
        pith_mask = pith_mask[:,:,0]
        y,x = np.where(pith_mask>0)
        cy = int(np.mean(y))
        cx = int(np.mean(x))
        command = (f"PYTHONPATH=\"./automatic_methods/tree_ring_delineation/cstrd_ipol\" && "
                   f"{self.python_path} ./automatic_methods/tree_ring_delineation/cstrd_ipol/main.py "
                   f"--input {image_path} --cy {cy} --cx {cx} "
                   f"--output_dir {self.output_dir} --root ./automatic_methods/tree_ring_delineation/cstrd_ipol/ "
                   f"--sigma {self.sigma} --th_low {self.th_low} --th_high {self.th_hight} --save_imgs 1 ")

        if self.gt_ring_json is not None:
            gt_path_resized = self.gt_ring_json if self.resize_factor == 1 else (
                resize_annotations(self.image_path, image_path, self.gt_ring_json))
            #####  annotation must have 360 points
            ia = InfluenceArea(gt_path_resized,dt_file=None,  img_filename = image_path , output_dir= self.output_dir,
                               threshold=0.6, cx = cx, cy= cy, Nr = 360)

            points_list = []
            for poly in ia.gt_poly:
                points_list.append([[int(x), int(y)] for x, y in poly.exterior.coords])

            al = AL_LateWood_EarlyWood(None,
                                       gt_path_resized,
                                       image_path=str(image_path)
                                       )

            al.write_list_of_points_to_labelme_json(points_list)

            #########################################
            command += f"--gt_ring_json {gt_path_resized} "

        #if self.include_gt_rings_in_output:
        #    command += f"--include_gt_rings_in_output 1 "

        print(command)
        os.system(command)

        json_name = f"labelme.json"
        predictions_json_path = self.output_dir / json_name
        prediction_json_sampled_path = self.output_dir / f"{image_path.stem}_sampled.json"
        #sampling the polygons and resizing them if needed
        self._sampling_polygons(predictions_json_path, prediction_json_sampled_path, pith_mask_path)
        image_draw_path = self.output_dir / "output.png" if self.gt_ring_json is None \
            else self.output_dir / "output_and_gt.png"
        os.system(f"cp {image_draw_path} {self.output_dir}/contours.png")
        return prediction_json_sampled_path

