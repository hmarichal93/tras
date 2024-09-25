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
from lib.metrics import  export_results, Table


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

        config_metric = self.config["metric"]
        self.main_label = config_metric["main_label"]
        self.ring_area = config_metric["ring_area"]
        self.ew_area = config_metric["ew_area"]
        self.lw_area = config_metric["lw_area"]
        self.eccentricity_module = config_metric["eccentricity_module"]
        self.perimeter = config_metric["perimeter"]
        self.ew_lw_label = config_metric["ew_lw_label"]
        self.cumulative_area = config_metric["cumulative_area"]
        self.cumulative_ew_area = config_metric["cumulative_ew_area"]

        self.lw_width = config_metric["lw_width"]
        self.eccentricity_phase = config_metric["eccentricity_phase"]
        self.ring_similarity_factor = config_metric["ring_similarity_factor"]
        self.year = config_metric["year"]
        self.cumulative_radius = config_metric["cumulative_radius"]
        self.cumulative_ew_radius = config_metric["cumulative_ew_radius"]
        self.lw_ratio = config_metric["lw_ratio"]
        self.annual_ring_width = config_metric["annual_ring_width"]
        self.ew_width = config_metric["ew_width"]
        self.lw_width_ratio = config_metric["lw_width_ratio"]



    def update_config(self):
        config_metric = self.config["metric"]
        config_metric["main_label"] = self.main_label
        config_metric["ring_area"] = self.ring_area
        config_metric["ew_area"] = self.ew_area
        config_metric["lw_area"] = self.lw_area
        config_metric["eccentricity_module"] = self.eccentricity_module
        config_metric["perimeter"] = self.perimeter
        config_metric["ew_lw_label"] = self.ew_lw_label
        config_metric["cumulative_area"] = self.cumulative_area
        config_metric["cumulative_ew_area"] = self.cumulative_ew_area
        config_metric["lw_width"] = self.lw_width
        config_metric["eccentricity_phase"] = self.eccentricity_phase
        config_metric["ring_similarity_factor"] = self.ring_similarity_factor
        config_metric["year"] = self.year
        config_metric["cumulative_radius"] = self.cumulative_radius
        config_metric["cumulative_ew_radius"] = self.cumulative_ew_radius
        config_metric["lw_ratio"] = self.lw_ratio
        config_metric["annual_ring_width"] = self.annual_ring_width
        config_metric["ew_width"] = self.ew_width
        config_metric["lw_width_ratio"] = self.lw_width_ratio



import base64

def encode_image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
    return encoded_string

def select_columns_to_display(CTX, table: Table):
    "select columns to display"
    columns = []
    if CTX.main_label:
        columns.append(table.main_label)
    if CTX.ring_area:
        columns.append(table.ring_area)
    if CTX.ew_area:
        columns.append(table.ew_area)
    if CTX.lw_area:
        columns.append(table.lw_area)
    if CTX.eccentricity_module:
        columns.append(table.eccentricity_module)
    if CTX.perimeter:
        columns.append(table.perimeter)
    if CTX.ew_lw_label:
        columns.append(table.ew_lw_label)
    if CTX.cumulative_area:
        columns.append(table.cumulative_area)
    if CTX.cumulative_ew_area:
        columns.append(table.cumulative_ew_area)
    if CTX.lw_width:
        columns.append(table.lw_width)
    if CTX.eccentricity_phase:
        columns.append(table.eccentricity_phase)
    if CTX.ring_similarity_factor:
        columns.append(table.ring_similarity_factor)
    if CTX.year:
        columns.append(table.year)
    if CTX.cumulative_radius:
        columns.append(table.cumulative_radius)
    if CTX.cumulative_ew_radius:
        columns.append(table.cumulative_ew_radius)
    if CTX.lw_ratio:
        columns.append(table.lw_ratio)
    if CTX.annual_ring_width:
        columns.append(table.annual_ring_width)
    if CTX.ew_width:
        columns.append(table.ew_width)
    if CTX.lw_width_ratio:
        columns.append(table.lw_width_ratio)
    return columns



class UI:

    def __init__(self, runtime_config_path):
        CTX = ViewContext(runtime_config_path)
        CTX.init_specific_ui_components()
        self.CTX = CTX
        self.df = None

    def options(self):
        st.subheader("Columns to display in the table")
        table = Table(self.CTX.units_mode)
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            main_label = st.checkbox(table.main_label, self.CTX.main_label)
            self.CTX.main_label = True if main_label else False
            ring_area = st.checkbox(table.ring_area, self.CTX.ring_area)
            self.CTX.ring_area = True if ring_area else False
            ew_area = st.checkbox(table.ew_area, self.CTX.ew_area)
            self.CTX.ew_area = True if ew_area else False
            lw_area = st.checkbox(table.lw_area, self.CTX.lw_area)
            self.CTX.lw_area = True if lw_area else False
            eccentricity_module = st.checkbox(table.eccentricity_module, self.CTX.eccentricity_module)
            self.CTX.eccentricity_module = True if eccentricity_module else False
            perimeter = st.checkbox(table.perimeter, self.CTX.perimeter)
            self.CTX.perimeter = True if perimeter else False



        with col2:
            ew_lw_label = st.checkbox(table.ew_lw_label, self.CTX.ew_lw_label)
            self.CTX.ew_lw_label = True if ew_lw_label else False
            cumulative_area = st.checkbox(table.cumulative_area, self.CTX.cumulative_area)
            self.CTX.cumulative_area = True if cumulative_area else False
            cumulative_ew_area = st.checkbox(table.cumulative_ew_area, self.CTX.cumulative_ew_area)
            self.CTX.cumulative_ew_area = True if cumulative_ew_area else False
            lw_width = st.checkbox(table.lw_width, self.CTX.lw_width)
            self.CTX.lw_width = True if lw_width else False
            eccentricity_phase = st.checkbox(table.eccentricity_phase, self.CTX.eccentricity_phase)
            self.CTX.eccentricity_phase = True if eccentricity_phase else False
            ring_similarity_factor = st.checkbox(table.ring_similarity_factor, self.CTX.ring_similarity_factor)
            self.CTX.ring_similarity_factor = True if ring_similarity_factor else False



        with col3:
            year = st.checkbox(table.year, self.CTX.year)
            self.CTX.year = True if year else False
            cumulative_radius = st.checkbox(table.cumulative_radius, self.CTX.cumulative_radius)
            self.CTX.cumulative_radius = True if cumulative_radius else False
            cumulative_ew_radius = st.checkbox(table.cumulative_ew_radius, self.CTX.cumulative_ew_radius)
            self.CTX.cumulative_ew_radius = True if cumulative_ew_radius else False
            lw_ratio = st.checkbox(table.lw_ratio, self.CTX.lw_ratio)
            self.CTX.lw_ratio = True if lw_ratio else False

        with col4:
            annual_ring_width = st.checkbox(table.annual_ring_width, self.CTX.annual_ring_width)
            self.CTX.annual_ring_width = True if annual_ring_width else False
            ew_width = st.checkbox(table.ew_width, self.CTX.ew_width)
            self.CTX.ew_width = True if ew_width else False
            lw_width_ratio = st.checkbox(table.lw_width_ratio, self.CTX.lw_width_ratio)
            self.CTX.lw_width_ratio = True if lw_width_ratio else False




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

        self.dataframe_file = self.CTX.output_dir_metrics / "measurements.csv"
        if not Path(self.dataframe_file).exists():
            return None
        #display dataframe_file

        st.write(f"Results are saved in {self.CTX.output_dir_metrics}")

        rings_image_path = self.CTX.output_dir_metrics / "rings.png"
        import cv2
        image = cv2.cvtColor(cv2.imread(rings_image_path), cv2.COLOR_BGR2RGB)

        # Display image with default settings
        # image_zoom(image)

        # Display image with custom settings
        #st.image(str(rings_image_path), caption="Rings Image")

        self.df = pd.read_csv(self.dataframe_file)
        self.df['Year'] = self.df['Year'].astype(int)

        #select columns to display
        columns = select_columns_to_display(self.CTX, Table(self.CTX.units_mode))
        self.df = self.df[columns]

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

        self.df['image'] = base64_images
        #put 'image' column as first
        cols = self.df.columns.tolist()
        cols = cols[-1:] + cols[:-1]
        self.df = self.df[cols]



        st.data_editor(self.df,
                       column_config= {
                           'image': st.column_config.ImageColumn('Preview Ring', help="Preview Ring")
                       },
                       hide_index = True
        )

        image_zoom(image, mode="scroll", size=(800, 600), keep_aspect_ratio=True, zoom_factor=4.0, increment=0.2)

        area_bar_plot_filepath = self.CTX.output_dir_metrics / "area_bar_plot.png"
        st.image(str(area_bar_plot_filepath), caption="Area Bar Plot")

        width_bar_plot_filepath = self.CTX.output_dir_metrics / "width_bar_plot.png"
        st.image(str(width_bar_plot_filepath), caption="Width Bar Plot")

        radius_plot_filepath = self.CTX.output_dir_metrics / "radius_plot.png"
        st.image(str(radius_plot_filepath), caption="Radius Plot")





def main(runtime_config_path):
    ui = UI(runtime_config_path)

    st.divider()
    ui.options()

    st.divider()
    selected = ui.run_metrics()
    st.divider()


    ui.CTX.save_config()
    return

def annotate_pith():
    #TODO: Implement
    pass
