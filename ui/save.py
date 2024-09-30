import os

import cv2
import streamlit as st
import numpy as np

from streamlit_option_menu import option_menu
from pathlib import Path
from copy import deepcopy

from lib.image import LabelMeInterface as UserInterface, Color as ColorCV2, Drawing, load_image
from ui.common import Context, Shapes, Color
from lib.io import load_json, write_json, bytesio_to_dict, read_file_as_binary





class VisualizationShape:
    def __init__(self, shapes_list, thickness, color, stroke, fill, opacity):
        self.shapes_list = shapes_list
        self.thickness = thickness
        self.color = color
        self.stroke = stroke
        self.fill = fill
        self.opacity = opacity

    def __str__(self):
        return f"({self.thickness=} {self.color=} {self.stroke=} {self.fill=} {self.opacity=})"

    def __repr__(self):
        return f"({self.thickness=} {self.color=} {self.stroke=} {self.fill=} {self.opacity=})"

class ViewContext(Context):
    def init_specific_ui_components(self):
        pass

    def update_config(self):
        pass





class UI:

    def __init__(self, runtime_config_path):
        CTX = ViewContext(runtime_config_path)
        CTX.init_specific_ui_components()
        self.CTX = CTX

    def download_results(self):
        #1.0 zip the results in self.CTX.output_dir
        os.system(f"cd {self.CTX.output_dir} && zip -r results.zip .")
        #2.0 download the zip file
        zip_file = read_file_as_binary(str(self.CTX.output_dir / "results.zip"))

        #2.2 download the zip file
        st.download_button(label="Download Results", data=zip_file, file_name="results.zip", mime="application/zip")

        return






def main(runtime_config_path):
    ui = UI(runtime_config_path)

    ui.download_results()

    ui.CTX.save_config()

    return

