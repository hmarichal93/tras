import numpy as np
import os
import sys

class INBD:
    def __init__(self, image_path, pith_mask_path, model_path, output_dir):
        self.image_path = image_path
        self.pith_mask = pith_mask_path
        self.model_path = model_path
        self.output_dir = output_dir
        self.python_path = self._get_python_path()

    def _get_python_path(self):
        python_path = sys.executable
        return python_path

    def run(self):
        command = (f"{self.python_path} ./automatic_methods/tree_ring_delineation/mlbrief_inbd/INBD/main.py inference {self.model_path} {self.image_path} "
                   f"{self.pith_mask} --output {self.output_dir} ")
        print(command)
        os.system(command)
        root_inbd_results = self.output_dir / f"{self.model_path.parent.stem}_"
        command = (f"PYTHONPATH=\"./automatic_methods/tree_ring_delineation/mlbrief_inbd\" &&"
                   f" {self.python_path} ./automatic_methods/tree_ring_delineation/mlbrief_inbd/src/from_inbd_to_urudendro_labels.py "
                   f"--root_dataset {self.image_path} --root_inbd_results {root_inbd_results} --output_dir {self.output_dir} "
                   f" --center_mask_dir {self.pith_mask}")
        print(command)

        os.system(command)

        json_name = f"{self.image_path.stem}.json"
        results_dir = self.output_dir / "inbd_urudendro_labels/image"
        return results_dir / json_name