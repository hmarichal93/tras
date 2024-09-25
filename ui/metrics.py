import os

import cv2
import streamlit as st
import numpy as np
import pandas as pd

from streamlit_option_menu import option_menu
from streamlit_image_zoom import image_zoom

from pathlib import Path
from copy import deepcopy

from lib.image import LabelMeInterface as UserInterface, Color as ColorCV2, Drawing, load_image
from ui.common import Context, Shapes, Color
from lib.io import load_json, write_json, bytesio_to_dict
from lib.metrics import  export_results


class ViewContext(Context):
    def init_specific_ui_components(self):
        config = self.config["image"]
        self.image_path = self.output_dir / config["image_path"]
        self.units_mode = config["scale"]["unit"]
        self.pixels_length = config["scale"]["pixels_length"]
        self.know_distance = config["scale"]["know_distance"]
        self.dpi = config["scale"]["dpi"]
        self.tree_planting_date = config["metadata"]["tree_planting_date"]


        config_manual = self.config["manual"]

        self.ew_annotation_file = config_manual["annotations_files"]["early_wood"]
        self.ew_annotation_file = None if len(self.ew_annotation_file) == 0 else self.ew_annotation_file
        self.lw_annotation_file = config_manual["annotations_files"]["late_wood"]
        self.lw_annotation_file = None if len(self.lw_annotation_file) == 0 else self.lw_annotation_file
        self.knot_annotation_file = config_manual["annotations_files"]["knot"]
        self.knot_annotation_file = None if len(self.knot_annotation_file) == 0 else self.knot_annotation_file
        self.cw_annotation_file = config_manual["annotations_files"]["compression_wood"]
        self.cw_annotation_file = None if len(self.cw_annotation_file) == 0 else self.cw_annotation_file

        self.output_dir_metrics = self.output_dir  / "metrics"
        self.output_dir_metrics.mkdir(parents=True, exist_ok=True)



    def update_config(self):
        pass

import base64

def encode_image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
    return encoded_string
class UI:

    def __init__(self, runtime_config_path):
        CTX = ViewContext(runtime_config_path)
        CTX.init_specific_ui_components()
        self.CTX = CTX

    def run_metrics(self):
        enabled = self.CTX.lw_annotation_file is not None
        run_button = st.button("Run", disabled = not enabled)
        if run_button:
            metadata = dict(
                unit = self.CTX.units_mode,
                pixels_millimeter_relation =  self.CTX.know_distance / self.CTX.pixels_length ,
                plantation_date = True,
                year = self.CTX.tree_planting_date['year']

            )
            if Path(self.CTX.lw_annotation_file).exists():
                lw_file_path = self.CTX.output_dir_metrics / "latewood_read.json"
                os.system(f"cp {self.CTX.lw_annotation_file} {lw_file_path}")
                lw_file_path = self.CTX.output_dir_metrics / "latewood.json"
            else:
                lw_file_path = None

            if Path(self.CTX.ew_annotation_file).exists():
                ew_file_path = self.CTX.output_dir_metrics / "earlywood_read.json"
                os.system(f"cp {self.CTX.ew_annotation_file} {ew_file_path}")
                ew_file_path = self.CTX.output_dir_metrics / "earlywood.json"
            else:
                ew_file_path = None

            export_results(labelme_latewood_path= lw_file_path,
                           labelme_earlywood_path= ew_file_path,
                           image_path=self.CTX.image_path,
                           metadata=metadata,
                           draw=True,
                           output_dir=self.CTX.output_dir_metrics)

            dataframe_file = self.CTX.output_dir_metrics / "measurements.csv"
            #display dataframe_file

            st.write(f"Results are saved in {self.CTX.output_dir_metrics}")

            rings_image_path = self.CTX.output_dir_metrics / "rings.png"
            import cv2
            image = cv2.cvtColor(cv2.imread(rings_image_path), cv2.COLOR_BGR2RGB)

            # Display image with default settings
            # image_zoom(image)

            # Display image with custom settings
            image_zoom(image, mode="scroll", size=(800, 600), keep_aspect_ratio=True, zoom_factor=4.0, increment=0.2)
            #st.image(str(rings_image_path), caption="Rings Image")

            df = pd.read_csv(dataframe_file)
            df['Year'] = df['Year'].astype(int)

            ring_images = [str(image_path) for image_path in self.CTX.output_dir_metrics.glob("*_ring_properties*.png")]
            from natsort import natsorted
            ring_images = natsorted(ring_images)
            base64_images = []
            #get python file path
            script_path = os.path.abspath(__file__)
            root = Path(script_path).parent.parent
            static_files_dir = Path(root) / "static"
            static_files_dir.mkdir(parents=True, exist_ok=True)
            for image_path in ring_images:
                #copy image_path to static_files_dir
                os.system(f"cp {image_path} {static_files_dir}")
                base64_images.append(f"app/static/{Path(image_path).name}")

            df['image'] = base64_images
            #put 'image' column as first
            cols = df.columns.tolist()
            cols = cols[-1:] + cols[:-1]
            df = df[cols]

            st.data_editor(df,
                           column_config= {
                               'image': st.column_config.ImageColumn('Preview Ring', help="Preview Ring")
                           },
                           hide_index = True
            )

            area_bar_plot_filepath = self.CTX.output_dir_metrics / "area_bar_plot.png"
            st.image(str(area_bar_plot_filepath), caption="Area Bar Plot")

            width_bar_plot_filepath = self.CTX.output_dir_metrics / "width_bar_plot.png"
            st.image(str(width_bar_plot_filepath), caption="Width Bar Plot")

            radius_plot_filepath = self.CTX.output_dir_metrics / "radius_plot.png"
            st.image(str(radius_plot_filepath), caption="Radius Plot")





def main(runtime_config_path):
    ui = UI(runtime_config_path)
    st.divider()
    selected = ui.run_metrics()
    st.divider()


    ui.CTX.save_config()
    return

def annotate_pith():
    #TODO: Implement
    pass
