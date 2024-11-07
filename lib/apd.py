import sys
import cv2
import numpy as np

from pathlib import Path

# Add the desired path to sys.path
root = Path(__file__).parent.parent

desired_path = Path(f"{root}/automatic_methods/pith_detection/apd")
sys.path.append(str(desired_path))

from lib.automatic_wood_pith_detector import apd, apd_pcl, apd_dl
from lib.image import load_image, resize_image, write_image, Color
from backend.labelme_layer import LabelmeObject, AL_LateWood_EarlyWood
from ui.common import Pith
class APD:

    def __init__(self, filename, output_filename, percent_lo=0.7, st_w=3, method=0, resize_factor=5, lo_w=3,
                 st_sigma=1.2, weights_path="automatic_methods/pith_detection/apd/checkpoints/yolo/all_best_yolov8.pt",
                 output_dir=None):
        self.filename = filename
        self.output_filename = output_filename
        self.percent_lo = percent_lo
        self.st_w = st_w
        self.method = method
        self.resize_factor = resize_factor
        self.lo_w = lo_w
        self.st_sigma = st_sigma
        self.weigths_path = weights_path
        self.output_dir = Path(output_dir) if output_dir is not None else None

    def run(self):
        debug = True if self.output_dir is not None else False
        if debug:
            self.output_dir.mkdir(exist_ok=True, parents=True)
            import os
            os.system(f"rm -rf {self.output_dir}/*")


        img_in = load_image(self.filename)
        self.h_i, self.w_i = img_in.shape[:2]
        # 1.1 resize image
        if self.resize_factor != 1:
            image_resized_path,_,_ = resize_image(self.filename, self.resize_factor, output_path= f"{self.output_dir}/resized.png")
            img_in = load_image(image_resized_path)
            self.h_o, self.w_o = img_in.shape[:2]
        else:
            self.h_o, self.w_o = self.h_i, self.w_i

        if self.method == Pith.apd:
            print("apd")
            peak = apd(img_in, self.st_sigma, self.st_w, self.lo_w, rf = 7, percent_lo = self.percent_lo,
                       max_iter = 11, epsilon =10 ** -3, output_dir=self.output_dir)

        elif self.method == Pith.apd_pl:
                print("apd_pcl")
                peak = apd_pcl(img_in, self.st_sigma, self.st_w, self.lo_w, rf = 7, percent_lo = self.percent_lo,
                               max_iter = 11, epsilon =10 ** -3, output_dir=self.output_dir)


        elif self.method == Pith.apd_dl:
            try:
                print("apd_dl")
                peak = apd_dl(img_in, self.output_dir if self.output_dir is not None else Path("/tmp"),
                              self.weigths_path)

            except FileNotFoundError:
                peak = None

        else:
            raise ValueError(f"method {self.method} not found")

        if peak is None:
            return False
        # 3.0 export results
        radius_size_pith_region = np.maximum(self.w_o, self.w_o) // 100
        self._save_results(peak, r = radius_size_pith_region)

        return True

    def _save_results(self, peak: np.array, num : int = 8, r : int = 3):
        "Generate a polygon centered at peak. Peak has dimension 2. The radius is 10 px"
        # Crear un array de ángulos entre 0 y 2*pi
        theta = np.linspace(0, 2 * np.pi, num)
        y_c, x_c = peak
        # Calcular las coordenadas x e y
        xx = x_c + r * np.sin(theta)
        yy = y_c + r * np.cos(theta)
        if self.resize_factor == 0:
            shapes = [[[int(x), int(y)] for x, y in zip(xx, yy)]]
        else:
            shapes = [[[int(x * (self.w_i/self.w_o)), int(y * (self.h_i /self.h_o))] for x, y in zip(xx, yy)]]

        al = AL_LateWood_EarlyWood(None,
                                   write_file_path=str(self.output_filename),
                                   image_path=str(self.filename)
                                   )

        al.write_list_of_points_to_labelme_json(shapes)
        return


def test_apd():
    filename = Path("./input/A4/A4.jpg")
    output_filename = Path("./input/A4/A4.json")
    apd = APD(filename, output_filename, method=Pith.apd)
    apd.run()

    output_filename = Path("./input/A4/A4_dl.json")
    apd = APD(filename, output_filename, method=Pith.apd_dl)
    apd.run()

    output_filename = Path("./input/A4/A4_pcl.json")
    apd = APD(filename, output_filename, method=Pith.apd_pl)
    apd.run()

    return

if __name__ == "__main__":
    test_apd()
