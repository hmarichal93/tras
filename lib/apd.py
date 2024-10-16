import sys
import cv2
import numpy as np

from pathlib import Path

# Add the desired path to sys.path
root = Path(__file__).parent.parent

desired_path = Path(f"{root}/automatic_methods/pith_detection/apd")
sys.path.append(str(desired_path))

from lib.automatic_wood_pith_detector import apd, apd_pcl, apd_dl
from lib.image import load_image, resize_image_using_pil_lib, write_image, Color
from backend.labelme_layer import LabelmeObject, AL_LateWood_EarlyWood
from ui.common import Pith
class APD:

    def __init__(self, filename, output_filename, percent_lo=0.7, st_w=3, method=0, new_shape=640, lo_w=3,
                 st_sigma=1.2, weigths_path=None):
        self.filename = filename
        self.output_filename = output_filename
        self.percent_lo = percent_lo
        self.st_w = st_w
        self.method = method
        self.new_shape = new_shape
        self.lo_w = lo_w
        self.st_sigma = st_sigma
        self.weigths_path = weigths_path

    def run(self):

        img_in = load_image(self.filename)
        o_height, o_width = img_in.shape[:2]
        # 1.1 resize image
        if self.new_shape > 0:
            self.h_i, self.w_i = img_in.shape[:2]
            img_in = resize_image_using_pil_lib(img_in, height_output=self.new_shape, width_output=self.new_shape)
            self.h_o, self.w_o = img_in.shape[:2]

        if self.method == Pith.apd:
            print("apd")
            peak = apd(img_in, self.st_sigma, self.st_w, self.lo_w, rf = 7, percent_lo = self.percent_lo,
                       max_iter = 11, epsilon =10 ** -3)

        elif self.method == Pith.apd_pl:
            print("apd_pcl")
            peak = apd_pcl(img_in, self.st_sigma, self.st_w, self.lo_w, rf = 7, percent_lo = self.percent_lo,
                           max_iter = 11, epsilon =10 ** -3)

        elif self.method == Pith.apd_dl:
            print("apd_dl")
            peak = apd_dl(img_in, Path("/tmp"), self.weigths_path)[::-1]

        else:
            raise ValueError(f"method {self.method} not found")

        # 3.0 save results
        self._save_results(peak)

        return

    def _save_results(self, peak: np.array, num : int = 8, r : int = 3):
        "Generate a polygon centered at peak. Peak has dimension 2. The radius is 10 px"
        # Crear un array de ángulos entre 0 y 2*pi
        theta = np.linspace(0, 2 * np.pi, num)
        y_c, x_c = peak
        # Calcular las coordenadas x e y
        xx = x_c + r * np.sin(theta)
        yy = y_c + r * np.cos(theta)
        if self.new_shape == 0:
            shapes = [[[int(x), int(y)] for x, y in zip(xx, yy)]]
        else:
            shapes = [[[int(x * (self.w_i/self.w_o)), int(y * (self.h_i /self.h_o))] for x, y in zip(xx, yy)]]
        al = AL_LateWood_EarlyWood(None,
                                   write_file_path=str(self.output_filename),
                                   image_path=str(self.filename)
                                   )

        al.write_list_of_points_to_labelme_json(shapes)
        return



