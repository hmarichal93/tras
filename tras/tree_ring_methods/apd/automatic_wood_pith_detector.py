import numpy as np
import cv2
from pathlib import Path
from ultralytics import YOLO
import pandas as pd

from .structural_tensor import StructuralTensor, sampling_structural_tensor_matrix
from .optimization import Optimization, LeastSquaresSolution, filter_lo_around_c
from .pclines_parallel_coordinates import pclines_local_orientation_filtering


def local_orientation(img_in, st_sigma, st_window):

    gray_image = cv2.cvtColor(img_in, cv2.COLOR_RGB2GRAY).copy()
    STc, STo = StructuralTensor(gray_image, sigma=st_sigma, window_size=st_window)

    return STo, STc



def lo_sampling(STo, STc, lo_w, percent_lo, debug=False, img=None, output_folder=None):

    STc, STo, kernel_size = sampling_structural_tensor_matrix(STc, STo, lo_w)

    # get orientations with high coherence (above percent_lo)

    th = np.percentile(STc[STc > 0], 100 * (1 - percent_lo))
    y, x = np.where(STc > th)
    O = STo[y, x]

    # convert orientations to vector (x1,y1,x2,y2)
    V = np.array([np.sin(O), np.cos(O)]).T
    orientation_length = kernel_size / 2
    Pc = np.array([x, y], dtype=float).T
    P1 = Pc - V * orientation_length / 2
    P2 = Pc + V * orientation_length / 2
    L = np.hstack((P1, P2))


    if debug:
        img_s = img.copy()
        for x1, y1, x2, y2 in L:
            p1 = np.array((x1, y1), dtype=int)
            p2 = np.array((x2, y2), dtype=int)
            img_s = cv2.line(img_s, (p1[0], p1[1]), (p2[0], p2[1]), (0, 0, 255), 1)
            # draw rectangle
            top = p1
            bottom = p2
            img_s = cv2.rectangle(img_s, (top[0], top[1]), (bottom[0], bottom[1]), (255, 0, 0), 1)

        cv2.imwrite(str(output_folder / "img_end_s.png"), img_s)
    return L

def pclines_postprocessing(img_in, Lof, ransac_outlier_th=0.03, debug=False, output_folder=None):
    m_lsd, _, _ = pclines_local_orientation_filtering(img_in, Lof, outlier_th=ransac_outlier_th, debug=debug,
                                                      lo_dir=output_folder)
    return m_lsd


def optimization(img_in, m_lsd, ci=None):
    xo, yo = LeastSquaresSolution(m_lsd=m_lsd, img=img_in).run() if ci is None else ci

    peak = Optimization(m_lsd=m_lsd).run(xo, yo)

    peak = (peak[0], peak[1])
    #print(f"optimization peak {peak}")
    return np.array(peak)

def peak_is_not_in_rectangular_region(ci_plus_1, top_left, bottom_right):
    x, y = ci_plus_1
    res = x < top_left[0] or y < top_left[1] or \
                                        x > bottom_right[0] or y > bottom_right[1]
    return res

def apd(img_in, st_sigma, st_window, lo_w, percent_lo, max_iter, rf, epsilon, pclines = False, debug=False,
        output_dir=None):

    STo, STc = local_orientation(img_in, st_sigma=st_sigma, st_window = st_window)

    Lof = lo_sampling(STo, STc, lo_w, percent_lo, debug=debug, img=img_in, output_folder=output_dir)

    if pclines:
        Lof = pclines_postprocessing(img_in, Lof, debug=debug, output_folder=output_dir)

    Lor = Lof
    ci = None
    for i in range(max_iter):
        if i > 0:
            Lor, top_left, bottom_right = filter_lo_around_c(Lof, rf, ci, img_in)

        ci_plus_1 = optimization(img_in, Lor, ci)

        if i > 0:
            if np.linalg.norm(ci_plus_1 - ci) < epsilon:
                ci = ci_plus_1
                break

            if peak_is_not_in_rectangular_region(ci_plus_1, top_left, bottom_right):
                break

        ci = ci_plus_1

    return ci


def apd_pcl(img_in, st_sigma, st_window, lo_w, percent_lo, max_iter, rf, epsilon, debug=False, output_dir=None):
    peak = apd(img_in, st_sigma, st_window, lo_w, percent_lo, max_iter, rf, epsilon, pclines = True, debug=debug,
               output_dir=output_dir)
    return peak

def read_label(label_filename, img ):
    """
    Read label file.
    :param label_filename: label filename
    :return: label as dataframe
    """
    label = pd.read_csv(label_filename, sep=" ", header=None)
    if label.shape[0] > 1:
        label = label.iloc[0]
    cx, cy, w, h = int(label[1] * img.shape[1]), int(label[2] * img.shape[0]), int(label[3] * img.shape[1]), int(
                    label[4] * img.shape[0])
    return cx, cy, w, h

def apd_dl(img_in, output_dir, weights_path):
    if weights_path is None:
        raise ValueError("model is None")

    # Resolve the output directory path to handle symlinks (e.g., /var -> /private/var on macOS)
    output_dir = Path(output_dir).resolve()

    print(f"weights_path {weights_path}")
    model = YOLO(weights_path, task='detect')
    results = model(img_in, project=str(output_dir), save=True, save_txt=True, imgsz=640)
    
    # Find the actual output directory created by YOLO (it creates a 'predict' subdirectory)
    # YOLO might resolve paths differently, so we need to find where it actually saved the files
    predict_dir = output_dir / 'predict'
    labels_dir = predict_dir / 'labels'
    
    # Try to find the label file - YOLO might name it differently or save to a different location
    label_file = None
    if labels_dir.exists():
        # Look for label files in the labels directory
        label_files = list(labels_dir.glob('*.txt'))
        if label_files:
            label_file = label_files[0]  # Use the first label file found
        else:
            # Check if YOLO saved to a different location based on the results
            # YOLO might create subdirectories based on the run number
            for subdir in predict_dir.iterdir():
                if subdir.is_dir() and (subdir / 'labels').exists():
                    subdir_labels = list((subdir / 'labels').glob('*.txt'))
                    if subdir_labels:
                        label_file = subdir_labels[0]
                        break
    
    # If no label file found, try to extract coordinates from YOLO results directly
    if label_file is None or not label_file.exists():
        # Check results to see if any detections were made
        has_detections = False
        detection_box = None
        
        if results and len(results) > 0:
            result = results[0]
            # Check if any boxes were detected
            if hasattr(result, 'boxes') and result.boxes is not None:
                boxes = result.boxes
                if len(boxes) > 0:
                    has_detections = True
                    # Get the first (highest confidence) detection
                    # boxes.xyxy contains [x1, y1, x2, y2] coordinates
                    # boxes.xywh contains [x_center, y_center, width, height] coordinates
                    if hasattr(boxes, 'xywh') and len(boxes.xywh) > 0:
                        # Use center coordinates directly from YOLO
                        xywh = boxes.xywh[0].cpu().numpy() if hasattr(boxes.xywh[0], 'cpu') else boxes.xywh[0]
                        cx = int(xywh[0])
                        cy = int(xywh[1])
                        peak = np.array([cx, cy])
                        return peak
        
        if not has_detections:
            raise ValueError(
                "No pith detected by YOLO model. The image might not contain a visible pith, "
                "or the model confidence threshold might be too high. Please try manual pith selection."
            )
        else:
            # Detections exist but we couldn't extract coordinates - fallback to file reading
            # Try to find the label file by searching more thoroughly
            all_label_files = []
            if predict_dir.exists():
                all_label_files = list(predict_dir.rglob('*.txt'))
            
            if all_label_files:
                label_file = all_label_files[0]
            else:
                raise FileNotFoundError(
                    f"YOLO made detections but could not find label file or extract coordinates. "
                    f"Searched in: {predict_dir}"
                )
    
    # Read coordinates from label file
    cx, cy, _, _ = read_label(label_file, img_in)
    peak = np.array([cx, cy])

    return peak
