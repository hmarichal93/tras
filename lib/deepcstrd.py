import os
import numpy as np

from shapely.geometry import LineString, Polygon

from lib.models import Model
from lib.image import load_image
from lib.io import load_json
from lib.cstrd_lib.metric_influence_area import InfluenceArea
from backend.labelme_layer import resize_annotations, AL_LateWood_EarlyWood
from deep_cstrd.deep_tree_ring_detection import DeepTreeRingDetection
from cross_section_tree_ring_detection.cross_section_tree_ring_detection import saving_results

class DeepCSTRD_MODELS:
    pinus_v1 = "Pinus V1"
    pinus_v2 = "Pinus V2"
    gleditsia = "Gleditsia"
    salix = "Salix Glauca"
    all = "generic"

def get_model_path(model_id,tile_size=256):
    root_path = "automatic_methods/tree_ring_delineation/deepcstrd/models/deep_cstrd/"
    tile_size = 0 if tile_size not in [0, 256] else tile_size
    if model_id == DeepCSTRD_MODELS.pinus_v1:
        return os.path.join(root_path, f"{tile_size}_pinus_v1_1504.pth")
    elif model_id == DeepCSTRD_MODELS.pinus_v2:
        return os.path.join(root_path, f"{tile_size}_pinus_v2_1504.pth")
    elif model_id == DeepCSTRD_MODELS.gleditsia:
        return os.path.join(root_path, f"{tile_size}_gleditsia_1504.pth")
    elif model_id == DeepCSTRD_MODELS.salix:
        return os.path.join(root_path, f"{tile_size}_salix_1504.pth")

    elif model_id == DeepCSTRD_MODELS.all:
        return os.path.join(root_path, f"0_all_1504.pth")

    else:
        raise "models does not exist"
class DEEPCSTRD(Model):
    def __init__(self, image_path, pith_mask_path, model_path, output_dir, Nr = 360, resize_factor = 1,
                 background_path = None, sigma=1.0, alpha=45, weights_path=DeepCSTRD_MODELS.pinus_v1,
                 tile_size=0, prediction_map_threshold=0.5,
                 total_rotations=5):
        super().__init__(image_path, pith_mask_path, model_path, output_dir, Nr, resize_factor, background_path)
        self.sigma = sigma
        self.alpha = alpha
        self.tile_size = tile_size
        self.weights_path = weights_path
        self.prediction_map_threshold = prediction_map_threshold
        self.total_rotations = total_rotations
        self.output_dir = output_dir

    def run(self):
        image_path = self._resize_image(self.image_path, self.resize_factor)
        pith_mask_path = self._resize_image(self.pith_mask, self.resize_factor, compute_shape=False)
        pith_mask = load_image(pith_mask_path)
        #convert to gray scale
        pith_mask = pith_mask[:,:,0]
        y,x = np.where(pith_mask>0)
        cy = int(np.mean(y))
        cx = int(np.mean(x))

        img_in = load_image(image_path)
        res = DeepTreeRingDetection(im_in = img_in,
                                    cy=int(cy),
                                    cx=int(cx),
                                    height=0,
                                    width=0,
                                    alpha=self.alpha,
                                    nr=360,
                                    mc=2,
                                    weights_path=self.weights_path,
                                    total_rotations=self.total_rotations,
                                    debug_image_input_path= str(image_path),
                                    debug_output_dir=str(self.output_dir),
                                    tile_size=self.tile_size,
                                    prediction_map_threshold=self.prediction_map_threshold)

        saving_results(res, str(self.output_dir), True)
        json_name = f"labelme.json"
        predictions_json_path = self.output_dir / json_name
        prediction_json_sampled_path = self.output_dir / f"{image_path.stem}_sampled.json"
        #sampling the polygons and resizing them if needed
        self._sampling_polygons(predictions_json_path, prediction_json_sampled_path, pith_mask_path)
        image_draw_path = self.output_dir / "output.png"
        os.system(f"cp {image_draw_path} {self.output_dir}/contours.png")
        return prediction_json_sampled_path

