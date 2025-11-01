import numpy as np
import cv2
from ultralytics import YOLO
import pandas as pd

from lib.structural_tensor import StructuralTensor, sampling_structural_tensor_matrix
from lib.optimization import Optimization, LeastSquaresSolution, filter_lo_around_c
from lib.pclines_parallel_coordinates import pclines_local_orientation_filtering


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


    print(f"weights_path {weights_path}")
    model = YOLO(weights_path, task='detect')
    _ = model(img_in, project=output_dir, save=True, save_txt=True, imgsz=640)
    cx, cy, _, _ = read_label(output_dir / 'predict/labels/image0.txt', img_in)
    peak = np.array([cx, cy])

    return peak
