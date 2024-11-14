import numpy as np
import os
import sys
import pandas as pd

from urudendro.metric_influence_area import main as metric

from shapely.geometry import Polygon
from pathlib import Path
from typing import Tuple

from backend.labelme_layer import AL_LateWood_EarlyWood

urudendro_path = "./automatic_methods/tree_ring_delineation/mlbrief_inbd/uruDendro"
# Add the desired path to sys.path
root = Path(__file__).parent.parent
desired_path = Path(f"{root}/automatic_methods/tree_ring_delineation/mlbrief_inbd")
sys.path.append(str(desired_path))
class Metric:
    def __init__(self, ann_path, det_path, images_path, output_dir):
        self.ann_path = ann_path
        self.det_path = det_path
        self.images_path = images_path
        self.output_dir = output_dir
        self.script_path  = "metric_influence_area.py"
        self.python_path = sys.executable

    def __get_pith_polygon(self, ann_path: Path) -> Polygon:
        al = AL_LateWood_EarlyWood(ann_path, None)
        shapes = al.read()
        return Polygon(shapes[0].points)
    def _get_pith_center(self):
        pith_polygon = self.__get_pith_polygon(self.ann_path)
        return (int(pith_polygon.centroid.x), int(pith_polygon.centroid.y))

    def run(self):
        #get pith center
        cx, cy = self._get_pith_center()
        # command = (f"cd {urudendro_path} && {self.python_path} {self.script_path} "
        #            f"--dt_filename {self.det_path} --gt_filename {self.ann_path} --img_filename {self.images_path}"
        #            f" --output_dir {self.output_dir} --th 0.6 --cx {cx} --cy {cy}")
        # os.system(command)

        P, R, F, RMSE, TP, FP, TN, FN  = metric(self.det_path, self.ann_path, self.images_path, self.output_dir, 0.6, cx, cy)
        return (P, R, F, RMSE, TP, FP, TN, FN)


def run_metric(ann_path: Path, method, img_name, output_dir, det_dir, img_path):
    print(method)
    det_path = det_dir / f"{img_name}_lw_{method}.json"
    output_metric_dir = Path(f"{output_dir}/{img_name}/{method}")
    output_metric_dir.mkdir(parents=True, exist_ok=True)
    metric = Metric(ann_path, det_path, img_path, output_metric_dir)
    P, R, F, RMSE, TP, FP, TN, FN = metric.run()
    return (P, R, F, RMSE, TP, FP, TN, FN)


def main():
    dataset_dir = "/data/maestria/resultados/tras"
    ann_dir = Path(f"{dataset_dir}/annotations/late_wood")
    det_dir = Path(f"{dataset_dir}/results/latewood_comparison")
    images_dir = Path(f"{dataset_dir}/images")
    output_dir = Path(f"{dataset_dir}/results/latewood_comparison/metrics")
    cstrd_df = pd.DataFrame(columns=["sample", "P", "R", "F", "RMSE", "TP", "FP", "TN", "FN"])
    inbd_df = pd.DataFrame(columns=["sample", "P", "R", "F", "RMSE", "TP", "FP", "TN", "FN"])
    for img_path in images_dir.glob("*.jpg"):
        img_name = img_path.stem
        ann_path = ann_dir / f"{img_name}.json"

        method="CS-TRD"
        P, R, F, RMSE, TP, FP, TN, FN = run_metric(ann_path, method, img_name, output_dir, det_dir, img_path)
        #add to dataframe not append
        new_row = {"sample": img_name, "P": P, "R": R, "F": F, "RMSE": RMSE, "TP": TP, "FP": FP, "TN": TN, "FN": FN}
        cstrd_df.loc[len(cstrd_df)] = new_row

        method = "INBD"
        P, R, F, RMSE, TP, FP, TN, FN = run_metric(ann_path, method, img_name, output_dir, det_dir, img_path)
        new_row = {"sample": img_name, "P": P, "R": R, "F": F, "RMSE": RMSE, "TP": TP, "FP": FP, "TN": TN, "FN": FN}
        inbd_df.loc[len(inbd_df)] = new_row

    cstrd_df.to_csv(f"{output_dir}/cstrd.csv", index=False)
    inbd_df.to_csv(f"{output_dir}/inbd.csv", index=False)

    return

if __name__ == "__main__":
    main()