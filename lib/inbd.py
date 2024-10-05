import os

from lib.models import Model



class INBD(Model):
    def __init__(self, image_path, pith_mask_path, model_path, output_dir, Nr = 360, resize_factor = 1,
                 background_path = None):
        super().__init__(image_path, pith_mask_path, model_path, output_dir, Nr, resize_factor, background_path)


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

