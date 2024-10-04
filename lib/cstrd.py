import os
import numpy as np

from lib.models import Model
from lib.image import load_image


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
            command += f"--gt_ring_json {self.gt_ring_json} "
        os.system(command)

        json_name = f"labelme.json"
        predictions_json_path = self.output_dir / json_name
        prediction_json_sampled_path = self.output_dir / f"{image_path.stem}_sampled.json"
        #sampling the polygons and resizing them if needed
        self._sampling_polygons(predictions_json_path, prediction_json_sampled_path, pith_mask_path)
        os.system(f"cp {self.output_dir}/output.png {self.output_dir}/contours.png")
        return prediction_json_sampled_path

