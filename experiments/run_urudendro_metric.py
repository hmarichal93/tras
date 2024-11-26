import numpy as np
import os
import sys
import pandas as pd

from urudendro.metric_influence_area import main as metric

from shapely.geometry import Polygon
from pathlib import Path
from typing import Tuple

from backend.labelme_layer import AL_LateWood_EarlyWood

class Metric:
    def __init__(self, ann_path, det_path, images_path, output_dir):
        self.ann_path = ann_path
        self.det_path = det_path
        self.images_path = images_path
        self.output_dir = output_dir
        self.script_path  = "metric_influence_area.py"
        self.python_path = sys.executable

    @staticmethod
    def __get_pith_polygon(ann_path: Path) -> Polygon:
        al = AL_LateWood_EarlyWood(ann_path, None)
        shapes = al.read()
        return Polygon(shapes[0].points)

    @staticmethod
    def _get_pith_center(ann_path):
        pith_polygon = Metric.__get_pith_polygon(ann_path)
        return (int(pith_polygon.centroid.x), int(pith_polygon.centroid.y))

    def run(self):
        cx, cy = Metric._get_pith_center(self.ann_path)

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


def extract_subfolders_path(dataset_dir):
    ann_dir = Path(f"{dataset_dir}/annotations/late_wood")
    det_dir = Path(f"{dataset_dir}/results/latewood_comparison")
    images_dir = Path(f"{dataset_dir}/images")
    output_dir = Path(f"{dataset_dir}/results/latewood_comparison/metrics")
    return ann_dir, det_dir, images_dir, output_dir

def main( dataset_dir = "/data/maestria/resultados/tras"):

    ann_dir, det_dir, images_dir, output_dir = extract_subfolders_path(dataset_dir)
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

def false_positive_bar_plot(inbd_path = "/data/maestria/resultados/tras/results/latewood_comparison/metrics/inbd.csv",
        cstrd_path = "/data/maestria/resultados/tras/results/latewood_comparison/metrics/cstrd.csv"):
    cstrd_df = pd.read_csv(cstrd_path)
    inbd_df = pd.read_csv(inbd_path)

    samples = cstrd_df["sample"].values[:-1]
    cstrd_fp = cstrd_df["FP"].values[:-1]
    inbd_fp = inbd_df["FP"].values[:-1]
    #plot overlapping bar plot
    import matplotlib.pyplot as plt
    x = np.arange(len(samples))
    plt.bar( x - 0.2, cstrd_fp, 0.4, label='cstrd')
    plt.bar( x + 0.2, inbd_fp, 0.4, label='inbd')

    plt.xticks(x,samples.tolist(),rotation=45)
    plt.yticks(np.arange(0, np.maximum(np.max(cstrd_fp), np.max(inbd_fp))+1, 1))
    plt.grid()
    plt.legend()
    plt.ylabel("Number of FP detections")
    #higher margins. X label is cut
    plt.savefig("fp_detections.png", bbox_inches='tight')
    plt.show()





def true_positive_bar_plot(inbd_path, cstrd_path):
    cstrd_df = pd.read_csv(cstrd_path)
    inbd_df = pd.read_csv(inbd_path)
    #columns year| tp_detections
    #plot overlapping bar plot
    import matplotlib.pyplot as plt
    year = cstrd_df["year"].values
    cstrd = cstrd_df["tp_detections"].values
    inbd = inbd_df["tp_detections"].values
    plt.bar(year - 0.1, cstrd, 0.2, label='cstrd')
    plt.bar(year + 0.1, inbd, 0.2, label='inbd')

    plt.xticks(rotation=45)
    plt.xticks(year)
    #y max value
    plt.ylim(0, 19)
    #yticks integer values
    plt.yticks(np.arange(0, 19, 1))
    #plot a tick line at 18
    plt.axhline(y=18, color='r', linestyle='--')
    plt.grid()
    plt.legend()
    #plt.xlabel("Year")
    plt.ylabel("Number of TP detections")
    #higher margins. X label is cut
    plt.savefig("tp_detections.png", bbox_inches='tight')
    plt.show()


def compute_plot_metrics(files_list):
    year = {}
    for f in files_list:
        shapes = AL_LateWood_EarlyWood(f, None).read()
        for shape in shapes:
            year[shape.label] = year[shape.label] + 1 if  shape.label in year.keys() else 1
    print(year)
    #sort descending by year key
    for y in np.arange(1999, 2023):
        y = y- (2022-1993)
        if str(y) not in year.keys():
            year[str(y)] = 0
    year = dict(sorted(year.items(), key=lambda item: item[0]))

    y,x = zip(*year.items())
    y  = np.array(y, dtype=int) + (2022-1993)
    #y = np.arange(1999, 2023)
    y = y.astype(str)
    #plot bar chart
    import matplotlib.pyplot as plt

    save_path_csv = "tp_detections_cstrd.csv"
    df = pd.DataFrame({"year": y, "tp_detections": x})
    df.to_csv(save_path_csv, index=False)

    plt.bar(y,x)
    plt.xticks(rotation=45)
    #y max value
    plt.ylim(0, len(files_list)+1)
    #yticks integer values
    plt.yticks(np.arange(0, len(files_list)+1, 1))
    #plt.xlabel("Year")
    plt.ylabel("Number of TP detections")
    #higher margins. X label is cut
    plt.savefig("tp_detections.png", bbox_inches='tight')
    plt.show()



def process_detection(dataset_dir):
    from urudendro.metric_influence_area import InfluenceArea
    ann_dir, det_dir, images_dir, output_dir = extract_subfolders_path(dataset_dir)
    output_dir = output_dir / "tp_annotations"
    output_dir.mkdir(parents=True, exist_ok=True)
    detections_files = []
    for img_path in images_dir.glob("*.jpg"):
        img_name = img_path.stem
        ann_path = ann_dir / f"{img_name}.json"
        method="CS-TRD"
        method="INBD"
        det_path = det_dir / f"{img_name}_lw_{method}.json"
        cx, cy = Metric._get_pith_center(ann_path)
        influence = InfluenceArea(ann_path, det_path, img_path, output_dir, 0.6, cx, cy)
        influence.compute_indicators()
        dt_and_gt_assignations = influence.dt_and_gt_assignation
        ####
        new_annotation_path = output_dir / det_path.name
        detections_files.append(new_annotation_path)
        al = AL_LateWood_EarlyWood(det_path, new_annotation_path, image_path=img_path)
        shapes = al.read()
        new_shapes = []
        labels = []
        for idx, shape in enumerate(shapes):
            is_dt_tp = dt_and_gt_assignations[idx] > -1
            if is_dt_tp:
                new_shapes.append(np.array(shape.points).tolist())
                labels.append(shape.label)

        ###
        al.write_list_of_points_to_labelme_json(new_shapes, labels=labels)

        ###








if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--compute_urudendro", type=bool, default=False)
    parser.add_argument("--process_detection", type=bool, default=False)
    parser.add_argument("--compute_plot_metrics", type=bool, default=True)
    parser.add_argument("--dataset_path", type=str, required=False,
                        default="/data/maestria/resultados/tras")
    args = parser.parse_args()
    if args.compute_urudendro:
        main(args.dataset_path)

    if args.process_detection:
        process_detection(args.dataset_path)

    if args.compute_plot_metrics:
        # method = "CS-TRD"
        # #method = "INBD"
        # dt_files = Path(args.dataset_path) / "results/latewood_comparison/metrics/tp_annotations"
        # files_list = [f for f in dt_files.glob("*.json")]
        # files_list = [f for f in files_list if method in f.name]
        # compute_plot_metrics(files_list)
        #true_positive_bar_plot("tp_detections_inbd.csv", "tp_detections_cstrd.csv")
        false_positive_bar_plot()