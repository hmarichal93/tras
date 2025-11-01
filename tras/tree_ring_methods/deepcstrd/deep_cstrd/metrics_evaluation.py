"""
In order to run metric please visit: https://github.com/hmarichal93/uruDendro and install urudendro package
"""
import matplotlib.cm as mplcm
import numpy as np
import pandas as pd
import PIL
import skimage

from pathlib import Path
from shapely.geometry import Polygon

from urudendro.labelme import AL_LateWood_EarlyWood
from urudendro.metric_influence_area import main as metric
from urudendro.image import load_image
from urudendro.io import load_json

from deep_cstrd.inbd_evaluation import evaluate_single_result_from_files_at_iou_levels, combine_metrics_at_iou_levels


def get_center_pixel(annotation_path):
    al = AL_LateWood_EarlyWood(annotation_path, None)
    shapes = al.read()
    points = shapes[0].points
    pith = Polygon(points).centroid
    cx, cy = pith.coords.xy
    return cy[0], cx[0]

def filter_labelmap(labelmap:np.ndarray, threshold=0.001) -> np.ndarray:
    N              = np.prod( labelmap.shape )
    labels, counts = np.unique( labelmap, return_counts=True )
    result         = labelmap.copy()
    for l,c in zip(labels, counts):
        if c/N < threshold:
            result[labelmap==l] = 0
    return result
def general_labelmap_from_labelme_file(labelme_file, outf, shape):
    al = AL_LateWood_EarlyWood(str(labelme_file), None)
    shapes = al.read()
    from PIL import ImageDraw, Image
    img = Image.new('L', shape, 0)
    draw = ImageDraw.Draw(img)
    for idx, s in enumerate(shapes[::-1], 1):
        points = s.points
        #fill polygon
        draw.polygon(points.ravel().tolist(), fill=idx)
    labelmap = np.array(img).T
    #print(np.unique(labelmap))
    labelmap = filter_labelmap(labelmap, threshold=0.001)
    #print(np.unique(labelmap))

    np.save(outf + '.labelmap.npy', labelmap)

    save_labelmap_image(labelmap, outf)
    return labelmap

def save_labelmap_image(labelmap, outf):
    labelmap_rgba = mplcm.gist_ncar(labelmap / labelmap.max())
    PIL.Image.fromarray((labelmap_rgba * 255).astype('uint8')).save(outf + '.labelmap.png')
def compute_metrics(root_database ="/data/maestria/resultados/deep_cstrd/pinus_v1/test",
                    results_path="/data/maestria/resultados/deep_cstrd_pinus_v1_test/inbd/inference/inbd_results/models_/inbd_urudendro_labels"):
    metadata_filename = Path(root_database).parent / 'dataset_ipol.csv'
    images_dir = Path(root_database).parent / "images/segmented"
    gt_dir = Path(root_database).parent / "annotations/labelme/images"
    results_path = Path(results_path)
    results_path.mkdir(exist_ok=True)

    metadata = pd.read_csv(metadata_filename)

    df = pd.DataFrame(columns=["Sample", "Precision", "Recall", "F1", "RMSE", "TP", "FP", "TN", "FN"])
    for idx in range(metadata.shape[0]):
        row = metadata.iloc[idx]
        sample = row.Imagen

        img_path = Path(f"{images_dir}/{sample}.png")
        if not img_path.exists():
            continue
        dt = results_path / f"{sample}/labelme.json"
        if not dt.exists():
            dt = results_path / f"{sample}/{sample}.json"
        print(dt)
        if not dt.exists():
            continue

        gt = Path(f"{gt_dir}/{sample}.json")
        cx = row.cx
        cy = row.cy

        output_sample_dir = results_path / sample
        output_sample_dir.mkdir(parents=True, exist_ok=True)
        if not gt.exists():
            P, R, F, RMSE, TP, FP, TN, FN = 0, 0, 0, 0, 0, 0, 0, 0
        else:
            P, R, F, RMSE, TP, FP, TN, FN = metric(str(dt), str(gt), str(img_path), str(output_sample_dir),0.6,  cy, cx)

        try:
            dt_json = load_json(str(dt))
            exec_time = dt_json["exec_time(s)"]
        except:
            exec_time = 0


        imgg = load_image(str(img_path))
        if not gt.exists():
            annotationfile =  str(Path(root_database) / "anotations_orig" / f"{sample}.tiff")
            labelmap_annotation = load_instanced_annotation(annotationfile, downscale=1)
            labelmap_gt = remove_boundary_class(labelmap_annotation)
            save_labelmap_image(labelmap_gt, str(output_sample_dir / "gt"))
        else:
            labelmap_gt = general_labelmap_from_labelme_file(gt, str(output_sample_dir / "gt"), imgg.shape[:2])
        labelmap_dt = general_labelmap_from_labelme_file(dt, str(output_sample_dir / "dt"), imgg.shape[:2])
        if labelmap_dt.shape != labelmap_gt.shape:
            labelmap_dt = skimage.transform.resize(labelmap_dt, labelmap_gt.shape, order=0)
        metrics_nbd = evaluate_single_result_from_files_at_iou_levels(labelmap_dt, labelmap_gt)
        inbd_metrics = combine_metrics_at_iou_levels([metrics_nbd])
        print(sample, inbd_metrics)


        ##save results
        df = pd.concat([df, pd.DataFrame([{"Sample": sample, "Precision": P, "Recall": R, "F1": F,
                                           "RMSE": RMSE, "TP": TP, "FP": FP, "TN": TN, "FN": FN,"ARAND":inbd_metrics['ARAND'], 'mAR': inbd_metrics['mAR'] ,"exec_time": exec_time}])],
                       ignore_index=True)
    df.to_csv(f"{results_path}/results.csv", index=False)

    return f"{results_path}/results.csv"

def load_instanced_annotation(annotationfile:str, *a, **kw) -> np.ndarray:
    if annotationfile.endswith('.tiff'):
        a = load_instanced_annotation_tiff(annotationfile, *a, **kw)

    return a

def load_instanced_annotation_tiff(file:str, downscale:float=1.0) -> np.ndarray:
    '''Load an annotation .tiff file with rings as integer labels'''
    assert file.endswith('.tiff')
    image          = PIL.Image.open(file)
    assert image.mode == 'I'
    image          = image.resize(
        [int(image.size[0]//downscale), int(image.size[1]//downscale)], 0
    ) #0:nearest
    labelmap       = np.array( image ).astype('int8')
    return labelmap

def remove_boundary_class(labelmap:np.ndarray, boundaryclass:int=0, bg_class:int=-1) -> np.ndarray:
    '''Remove the class boundaryclass from a labeled array, (so that the tree ring instances touch each other)'''
    import skimage
    boundarymask   = (labelmap==boundaryclass)
    backgroundmask = (labelmap==bg_class)
    result                 = labelmap.copy()
    result[boundarymask]   = 0
    result[backgroundmask] = 0
    while np.any( result[boundarymask]==0 ):
        result = skimage.segmentation.expand_labels(result, distance=100)
    result[backgroundmask] = bg_class
    return result


def compute_statics(results_path):

    df = pd.read_csv(results_path)
    #df_stats = pd.DataFrame(columns=["Model",  "Precision", "Recall", "F1", "RMSE", "TP", "FP",  "FN"])
    stats =df[["Precision", "Recall","F1", "RMSE", "TP", "FP", "FN", "ARAND", 'mAR', "exec_time"]].mean()
    df_stats = pd.DataFrame({"P": [stats["Precision"]], "R": [stats["Recall"]],"F1": [stats["F1"]], "RMSE": [stats["RMSE"]], "TP": [stats["TP"]], "FP": [stats["FP"]],
                           "FN": [stats["FN"]], "ARAND": [stats['ARAND']], 'mAR': stats['mAR'],"exec_time": [stats["exec_time"]]})
    #df_stats = pd.concat([df_stats, df_aux ])

    df_stats.to_csv(Path(results_path).parent / "results_stats.csv", index=False)
    return

def evaluate(args):
    res_path = compute_metrics(args.dataset_dir, args.results_path)
    compute_statics(res_path)

    return res_path

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, required=False, default="/data/maestria/resultados/deep_cstrd_datasets_train/pinus_v2_1504/test")
    parser.add_argument("--results_path", type=str, required=False, default="/data/maestria/resultados/deep_cstrd_inbd/pinus_v2_1504/inference/inbd_results/2025-01-30_19h41m28s_INBD_100e_a6.3__/inbd_urudendro_labels/")
    args = parser.parse_args()

    evaluate(args)

